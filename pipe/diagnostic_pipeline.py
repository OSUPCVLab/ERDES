"""
This module implements a diagnostic pipeline that uses two pretrained models:
1. RD Classifier: Predicts whether a patient has retinal detachment
2. Macula Classifier: If RD is detected, predicts whether the macula is detached or intact
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
import torch.nn as nn
import torchvision.io as io
from pathlib import Path
from tqdm import tqdm

from src.models.components.factory import build_3d_architecture
from src.data.components.utils import resize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DiagnosticPipeline:
    def __init__(
        self,
        model_name: str,
        rd_checkpoint_path: str,
        macula_checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        video_size: Tuple[int, int, int] = (96, 128, 128),
    ):
        logging.info("Initializing diagnostic pipeline...")
        self.device = device
        self.video_size = video_size
        self.resize_tf = resize(video_size)

        def strip_net_prefix(state_dict):
            return {
                k.replace("net.", "", 1) if k.startswith("net.") else k: v
                for k, v in state_dict.items()
            }

        logging.info("Loading RD model checkpoint...")
        self.rd_model = build_3d_architecture(model_name, num_classes=1)
        rd_state = torch.load(rd_checkpoint_path, map_location=device)
        stripped_rd_state = strip_net_prefix(rd_state['state_dict'])
        self.rd_model.load_state_dict(stripped_rd_state, strict=True)
        self.rd_model = self.rd_model.to(device)
        self.rd_model.eval()

        logging.info("Loading Macula model checkpoint...")
        self.macula_model = build_3d_architecture(model_name, num_classes=1)
        macula_state = torch.load(macula_checkpoint_path, map_location=device)
        stripped_macula_state = strip_net_prefix(macula_state['state_dict'])
        self.macula_model.load_state_dict(stripped_macula_state, strict=True)
        self.macula_model = self.macula_model.to(device)
        self.macula_model.eval()

        logging.info("Pipeline initialization complete.")

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        logging.info(f"Preprocessing video: {video_path}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        video, _, _ = io.read_video(video_path, pts_unit='sec')
        video = video.float()
        video = video.permute(3, 0, 1, 2)

        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)

        video = self.resize_tf(video)
        video = video / 255.0
        video = video.unsqueeze(0)

        return video

    def predict_single(self, video_path: str) -> Dict[str, Union[bool, Optional[bool]]]:
        logging.info(f"Running prediction on video: {video_path}")
        video = self.preprocess_video(video_path)
        video = video.to(self.device)

        with torch.no_grad():
            rd_pred = torch.sigmoid(self.rd_model(video))
            has_rd = bool(rd_pred.item() > 0.5)
            logging.info(f"RD prediction: {'Positive' if has_rd else 'Negative'}")

            macula_detached = None
            if has_rd:
                macula_pred = torch.sigmoid(self.macula_model(video))
                macula_detached = bool(macula_pred.item() <= 0.5)
                logging.info(f"Macula prediction: {'Detached' if macula_detached else 'Intact'}")

        return {
            "has_rd": has_rd,
            "macula_detached": macula_detached
        }

    def predict_batch(self, csv_path: str, video_column: str = "path") -> pd.DataFrame:
        logging.info(f"Loading input CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df['has_rd'] = False
        df['macula_detached'] = None

        logging.info(f"Processing {len(df)} videos...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            video_path = row[video_column]
            try:
                predictions = self.predict_single(video_path)
                df.at[idx, 'has_rd'] = predictions['has_rd']
                df.at[idx, 'macula_detached'] = predictions['macula_detached']
            except Exception as e:
                logging.error(f"Error processing {video_path}: {e}")
                continue

        logging.info("Batch prediction complete.")
        return df

def create_pipeline(
    model_name: str,
    experiment_root: str = "logs",
    video_size: Tuple[int, int, int] = (96, 128, 128),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> DiagnosticPipeline:
    logging.info("Creating pipeline and locating best checkpoints...")
    rd_exp_path = os.path.join(experiment_root, "train/run/", "rd", model_name)
    macula_exp_path = os.path.join(experiment_root, "train/run/", "md", model_name)

    def find_best_checkpoint(exp_path: str) -> str:
        checkpoints = list(Path(exp_path).rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {exp_path}")
        best_ckpt = next((ckpt for ckpt in checkpoints if "best" in str(ckpt)), checkpoints[-1])
        return str(best_ckpt)

    rd_ckpt = find_best_checkpoint(rd_exp_path)
    macula_ckpt = find_best_checkpoint(macula_exp_path)

    logging.info("Checkpoints located. Initializing pipeline...")
    return DiagnosticPipeline(
        model_name=model_name,
        rd_checkpoint_path=rd_ckpt,
        macula_checkpoint_path=macula_ckpt,
        video_size=video_size,
        device=device
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run retinal diagnostic pipeline on video data")
    parser.add_argument("--model", type=str, default="unet3d", help="Model architecture to use")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input video file or CSV containing video paths")
    parser.add_argument("--video-column", type=str, default="path",
                        help="Name of column containing video paths in CSV")
    parser.add_argument("--output", type=str, help="Path to save CSV results (for batch mode)")
    
    args = parser.parse_args()

    pipeline = create_pipeline(args.model)

    if args.input.endswith('.csv'):
        results = pipeline.predict_batch(args.input, args.video_column)
        if args.output:
            results.to_csv(args.output, index=False)
        print(results)
    else:
        result = pipeline.predict_single(args.input)
        print(f"Results for {args.input}:")
        print(f"Retinal Detachment: {'Yes' if result['has_rd'] else 'No'}")
        if result['has_rd']:
            print(f"Macula Status: {'Detached' if result['macula_detached'] else 'Intact'}")
