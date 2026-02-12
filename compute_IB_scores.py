from transformers import AutoProcessor, ClapModel
import torch
import torchaudio
import pandas as pd
import os
import ffmpeg
import matplotlib.pyplot as plt
import sys
import argparse

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


parser = argparse.ArgumentParser(description="Compute ImageBind scores for AvED-Bench edits.")
parser.add_argument(
    "--csv-file",
    default="/group2/ct/m-ishii/AvED-Bench/avedit_dataset_v3.csv",
    help="Path to the input CSV file.",
)
parser.add_argument(
    "--source-dir",
    required=True,
    help="Directory containing source videos (mp4).",
)
parser.add_argument(
    "--target-dir",
    required=True,
    help="Directory containing target videos (mp4).",
)
parser.add_argument(
    "--output-csv",
    default=None,
    help="Optional output CSV filename. Defaults to '<output_prefix>_all.csv'.",
)
args = parser.parse_args()

# input
csv_file = args.csv_file
source_dir = args.source_dir
target_dir = args.target_dir

output_prefix = f"IB_mean_scores_{os.path.basename(os.path.normpath(target_dir))}"

# Load ImageBind model
device = torch.device("cuda")
ib_model = imagebind_model.imagebind_huge(pretrained=True)
ib_model.eval()
ib_model.to(device)

df = pd.read_csv(csv_file)
score = {"name": [], "prompt": [], "score_sa_tv": []}
for index, row in df.iterrows():
    source_audio_name = os.path.join(source_dir, row['name'] + ".mp4")
    target_video_name = os.path.join(target_dir, row['name'] + ".mp4")
    prompt = row['trg']

    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([row['src'], row['trg']], device),
        # ModalityType.TEXT: data.load_and_transform_text([row['src_category'], row['trg_category']], device),
        ModalityType.AUDIO: data.load_and_transform_audio_data([source_audio_name], device, clips_per_video=5),
        ModalityType.VISION: data.load_and_transform_video_data([source_audio_name, target_video_name], device),
    }

    embeddings = ib_model(inputs, reduce=False)
    score_ib_sa_tv = torch.cosine_similarity(embeddings[ModalityType.AUDIO][0], embeddings[ModalityType.VISION][1], dim=-1)

    score_ib_sa_tv = score_ib_sa_tv.mean().item()

    score["name"].append(row['name'])
    score["prompt"].append(prompt)
    score["score_sa_tv"].append(score_ib_sa_tv)


# save scores to a CSV file
output_csv = args.output_csv or f"{output_prefix}_all.csv"
score_df = pd.DataFrame(score)
score_df.to_csv(output_csv, index=False)
print(f"Scores saved to {output_csv}")

