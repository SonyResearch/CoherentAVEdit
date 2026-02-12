# [Coherent Audio-Visual Editing via Conditional Audio Generation Following Video Edits](https://arxiv.org/abs/2512.07209)

[Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Yuki Mitsufuji](https://www.yukimitsufuji.com/)


## Highlight

We introduce a novel pipeline for joint audio-visual editing that enhances the coherence between edited video and its accompanying audio. Our approach first applies state-of-the-art video editing techniques to produce the target video, then performs audio editing to align with the visual changes. To achieve this, we present a new video-to-audio generation model that conditions on the source audio, target video, and a text prompt. We extend the model architecture to incorporate conditional audio input and propose a data augmentation strategy that improves training efficiency. Furthermore, our model dynamically adjusts the influence of the source audio based on the complexity of the edits, preserving the original audio structure where possible. 

## Setup

The setup process is the same as in [MMAudio](https://github.com/hkchengrex/MMAudio).

**Pretrained models:**

The pretrained model is available at https://huggingface.co/masato-a-ishii/CoherentAVEdit/tree/main

## Audio generation after video edits

In the following, we assume the pretrained model is stored in ./weights/

Before running the script, you need to first apply some video editing method to source videos and store them somewhere accessible.

### For a single video

```bash
python edit_single_video.py --duration=<duration> --video=<path-to-video> --prompt="your prompt" --mask_level=<mask level> --output=<path-to-output-dir>
```

### For multiple videos (evaluation with AvED-benchmark dataset)

First, you need to prepare for the following things:
- CSV_FILE: a CSV file used in AvED-Bench. Please refer to https://github.com/GenjiB/AVED/tree/main
- SOURCE_DIR: a directory containing the source videos of AvED-Bench.
- EDITED_VIDEO_DIR: a directory containing the edited videos.

Then, run the following.
```bash
# Hyperparameters for editing
MASK_LEVEL=-1 # This means adaptive conditioning.
MAX_MASK_LEVEL=4 # l_max in the paper.

# Create a score file
python compute_IB_scores.py --csv-file $CSV_FILE --source-dir $SOURCE_DIR --target-dir $EDITED_VIDEO_DIR --output-csv <path-to-score-file>

# Audio editing
python edit_multiple_videos_for_benchmark.py --video_dir $EDITED_VIDEO_DIR --csv_file $CSV_FILE --score_file <path-to-score-file> --output <path-to-output-dir> --mask_level $MASK_LEVEL --min_mask_level 1 --max_mask_level $MAX_MASK_LEVEL
```

<!-- ## Training

See [TRAINING.md](docs/TRAINING.md).

## Training Datasets

Our model was trained on several datasets, including [AudioSet](https://research.google.com/audioset/), [Freesound](https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md), [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), [AudioCaps](https://audiocaps.github.io/), and [WavCaps](https://github.com/XinhaoMei/WavCaps). These datasets are subject to specific licenses, which can be accessed on their respective websites. We do not guarantee that the pre-trained models are suitable for commercial use. Please use them at your own risk. -->

## Citation

```bibtex
@article{ishii2025coherent,
  title={Coherent Audio-Visual Editing via Conditional Audio Generation Following Video Edits},
  author={Ishii, Masato and Hayakawa, Akio and Shibuya, Takashi and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2512.07209},
  year={2025}
}
```

## Acknowledgement

This repository is based on [MMAudio](https://github.com/hkchengrex/MMAudio).

Many thanks to:
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) for the 16kHz BigVGAN pretrained model and the VAE architecture
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Synchformer](https://github.com/v-iashin/Synchformer) 
- [EDM2](https://github.com/NVlabs/edm2) for the magnitude-preserving VAE network architecture
- [ImageBind](https://github.com/facebookresearch/ImageBind) (by Meta / FAIR) for computing ImageBind scores
  - We modified the following two things:
    - set num_crops in data.load_and_transform_video_data to 1.
    - add "reduce" argument to models.imagebind_model.forward so that we can choose whether the model returns a feature tensor reduced over temporal dimension or not.

