import logging
from argparse import ArgumentParser
from pathlib import Path
import math

import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.data.av_utils import read_audio_frames

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='medium_w_loudness_16k')
    parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)

    parser.add_argument('--mask_away_clip', action='store_true')

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    parser.add_argument('--start_time', type=float, default=-1.0)
    parser.add_argument('--mask_level', type=int, default=None)
    parser.add_argument('--ref_audio', type=Path, default=None)
    parser.add_argument('--ref_start_pos_in_sec', type=float, default=None)
    parser.add_argument('--ref_crop_start_in_sec', type=float, default=None)
    parser.add_argument('--ref_crop_end_in_sec', type=float, default=None)

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    # model.download_if_needed()
    seq_cfg = model.seq_cfg

    if args.video:
        video_path: Path = Path(args.video).expanduser()
    else:
        video_path = None
    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=True)
    feature_utils = feature_utils.to(device, dtype).eval()

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len, seq_cfg.latent_downsample_rate)

    if video_path is not None:
        log.info(f'Using video {video_path}')
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)

        # Load the original audio
        audio_input_orig = read_audio_frames(video_path, seq_cfg.sampling_rate, 0.0, duration).unsqueeze(0)
        down_sample_rate = seq_cfg.spectrogram_frame_rate * seq_cfg.latent_downsample_rate
        expected_length = math.ceil(audio_input_orig.shape[1] / down_sample_rate) * down_sample_rate
        audio_input = torch.zeros([1, expected_length], device=device, dtype=dtype)
        audio_input[:, :audio_input_orig.shape[1]] = audio_input_orig

        # Load the reference audio
        if args.ref_audio is not None:
            audio_ref_orig = read_audio_frames(args.ref_audio, seq_cfg.sampling_rate, args.ref_crop_start_in_sec, args.ref_crop_end_in_sec).unsqueeze(0)
            expected_length = math.ceil(audio_ref_orig.shape[1] / down_sample_rate) * down_sample_rate
            audio_ref = torch.zeros([1, expected_length], device=device, dtype=dtype)
            audio_ref[:, :audio_ref_orig.shape[1]] = audio_ref_orig
            ref_start_norm_pos = args.ref_start_pos_in_sec / duration if args.ref_start_pos_in_sec is not None else 0.0
            audio_ref_feature = feature_utils.get_extra_condition(audio_ref)
            extra_cond = net.create_extra_condition(audio_ref_feature, ref_start_norm_pos)
        else:
            extra_cond = audio_input
    else:
        log.info('No video provided -- text-to-audio mode')
        clip_frames = sync_frames = audio_input = None

    if args.mask_level is not None:
        if args.mask_level < 0 or args.mask_level > net.mask_level_max:
            raise ValueError(f'Mask level should be in [0, {net.mask_level_max}]')

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    if args.start_time <= 0.0 or args.start_time > 1.0:
        audios = generate(clip_frames,
                        sync_frames, [prompt],
                        negative_text=[negative_prompt],
                        feature_utils=feature_utils,
                        net=net,
                        fm=fm,
                        rng=rng,
                        cfg_strength=cfg_strength,
                        extra_cond=extra_cond,
                        mask_level=args.mask_level)
    else:
        audios = generate(clip_frames,
                        sync_frames, [prompt],
                        negative_text=[negative_prompt],
                        feature_utils=feature_utils,
                        net=net,
                        fm=fm,
                        rng=rng,
                        cfg_strength=cfg_strength,
                        extra_cond=extra_cond,
                        input_audio=audio_input,
                        start_time=args.start_time,
                        mask_level=args.mask_level)

    audio = audios.float().cpu()[0]
    if video_path is not None:
        save_path = output_dir / f'{video_path.stem}.flac'
    else:
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        save_path = output_dir / f'{safe_filename}.flac'
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    log.info(f'Audio saved to {save_path}')
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info(f'Video saved to {output_dir / video_save_path}')

    log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
