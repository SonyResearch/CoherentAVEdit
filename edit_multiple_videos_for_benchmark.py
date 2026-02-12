import logging
from argparse import ArgumentParser
from pathlib import Path
import math
import pandas as pd
import os
import numpy as np

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


def convert_score_to_mask_level(score, max_mask_level, min_mask_level, min_th, max_th):

    if max_mask_level == min_mask_level:
        return min_mask_level
    elif max_mask_level - min_mask_level == 1:
        return min_mask_level if score < (min_th + max_th) / 2.0 else max_mask_level

    # For 1-5
    if score < min_th:
        mask_level = min_mask_level
    elif score >= max_th:
        mask_level = max_mask_level
    else:
        mask_level = min_mask_level + round(max(0.0, min(1.0, (score - min_th) / (max_th - min_th))) * (max_mask_level - min_mask_level))

    log.info(f'Converted score {score} to mask level {mask_level} (max level: {max_mask_level})')
    return mask_level


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='medium_w_loudness_16k')
    parser.add_argument('--model_path', type=Path, help='Path to the model weights', default="./weights/mmaudio_medium_w_loudness_16k.pth")
    parser.add_argument('--video_dir', type=Path, help='Path to the video file')
    parser.add_argument('--csv_file', type=Path, help='Path to the csv file')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--cfg_strength', type=float, default=7.5)
    parser.add_argument('--cfg_strength_extra', type=float, default=None)
    parser.add_argument('--num_steps', type=int, default=100)

    parser.add_argument('--mask_away_clip', action='store_true')

    parser.add_argument('--output_dir', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    parser.add_argument('--start_time', type=float, default=-1.0)
    parser.add_argument('--mask_level', type=int, default=0)
    parser.add_argument('--med_filter_size', type=int, default=None)
    parser.add_argument('--score_file', type=Path, help='Path to the csv file', default=None)
    parser.add_argument('--score_name', type=str, default="score_sa_tv")
    parser.add_argument('--score_min_th', type=float, default=0.02)
    parser.add_argument('--score_max_th', type=float, default=0.32)
    parser.add_argument('--max_mask_level', type=int, default=None)
    parser.add_argument('--min_mask_level', type=int, default=None)
    parser.add_argument('--save_extra', action='store_true')

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    # model.download_if_needed()
    seq_cfg = model.seq_cfg

    video_dir: Path = Path(args.video_dir).expanduser()
    model_path: Path = Path(args.model_path).expanduser()
    csv_file: Path = Path(args.csv_file).expanduser()
    negative_prompt: str = args.negative_prompt
    output_dir: Path = args.output_dir.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    cfg_strength_extra: float = cfg_strength / 2.0 if args.cfg_strength_extra is None else args.cfg_strength_extra
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
    net.load_weights(torch.load(model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model_path}')

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

    max_mask_level = args.max_mask_level if args.max_mask_level is not None else net.mask_level_max
    min_mask_level = args.min_mask_level if args.min_mask_level is not None else 1
    assert -1 <= args.mask_level <= max_mask_level, f'Mask level should be in [0, {max_mask_level}] or -1 for adaptive masking'

    if args.mask_level == -1:
        score_df = pd.read_csv(args.score_file)
        if score_df.empty:
            raise ValueError("Score CSV is empty. Please compute scores first.")

    # load file list
    filelist = pd.read_csv(csv_file)

    for index, row in filelist.iterrows():
        mp4_name = row['name']
        mp4_path = video_dir / f'{mp4_name}.mp4'

        # Load the original video
        video_info = load_video(mp4_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)

        # Load the original audio
        audio_input_orig = read_audio_frames(mp4_path, seq_cfg.sampling_rate, 0.0, duration).unsqueeze(0)
        down_sample_rate = seq_cfg.spectrogram_frame_rate * seq_cfg.latent_downsample_rate
        expected_length = math.ceil(audio_input_orig.shape[1] / down_sample_rate) * down_sample_rate
        audio_input = torch.zeros([1, expected_length], device=device, dtype=dtype)
        audio_input[:, :audio_input_orig.shape[1]] = audio_input_orig

        # Set parameters
        prompt = row['trg']
        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len, seq_cfg.latent_downsample_rate)
        if args.mask_level == -1:
            if args.score_file is None or args.score_name is None:
                raise ValueError('score_file and score_name must be specified when mask_level is -1 (adaptive masking)')
            if args.score_name not in score_df.columns:
                raise ValueError(f'score_name {args.score_name} not found in the score file columns: {score_df.columns.tolist()}')
            score = score_df.loc[score_df['name'] == mp4_name, args.score_name].values[0]
            if pd.isna(score):
                log.warning(f'No score found for {mp4_name}, using default mask level 0')
                mask_level = 0
            else:
                log.info(f'Using score {score} for {mp4_name}')
            mask_level = convert_score_to_mask_level(score, max_mask_level, min_mask_level, args.score_min_th, args.score_max_th)
        else:
            mask_level = args.mask_level

        log.info(f'Prompt: {prompt}')
        log.info(f'Negative prompt: {negative_prompt}')

        audios = generate(clip_frames,
                        sync_frames, [prompt],
                        negative_text=[negative_prompt],
                        feature_utils=feature_utils,
                        net=net,
                        fm=fm,
                        rng=rng,
                        cfg_strength=cfg_strength,
                        cfg_strength_extra=cfg_strength_extra,
                        extra_cond=audio_input,
                        mask_level=mask_level,
                        med_filter_size=args.med_filter_size,
                        return_extra=args.save_extra)
        
        if args.save_extra:
            audios, extra_spec = audios
            extra_save_path = output_dir / f'{mp4_path.stem}_extra.npy'
            extra_spec = extra_spec.float().cpu()[0].numpy()
            np.save(extra_save_path, extra_spec)

        audio = audios.float().cpu()[0]
        save_path = output_dir / f'{mp4_path.stem}.flac'
        torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

        log.info(f'Audio saved to {save_path}')
        if not skip_video_composite:
            video_save_path = output_dir / f'{mp4_path.stem}.mp4'
            make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
            log.info(f'Video saved to {video_save_path}')

        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))


if __name__ == '__main__':
    main()
