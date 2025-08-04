"""
This module implements a diagnostic pipeline that uses two pretrained models:
1. RD Classifier: Predicts whether a patient has retinal detachment
2. Macula Classifier: If RD is detected, predicts whether the macula is detached or intact
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
import torch.nn as nn
import torchvision.io as io
from pathlib import Path

from src.models.components.factory import build_3d_architecture
from src.data.components.utils import resize

class DiagnosticPipeline:
    def __init__(
        self,
        model_name: str,
        rd_checkpoint_path: str,
        macula_checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        video_size: Tuple[int, int, int] = (96, 128, 128),  # Default size from model configs
    ):
        """Initialize the diagnostic pipeline with two models.

        Args:
            model_name: Name of the architecture to use (e.g., "unet3d", "swinunetr", etc.)
            rd_checkpoint_path: Path to the RD classifier checkpoint
            macula_checkpoint_path: Path to the macula classifier checkpoint
            device: Device to run inference on
            video_size: Size to resize videos to (D, H, W)
        """
        self.device = device
        self.video_size = video_size
        self.resize_tf = resize(video_size)
        
        # Initialize RD classifier
        self.rd_model = build_3d_architecture(model_name, num_classes=1)
        rd_state = torch.load(rd_checkpoint_path, map_location=device)
        self.rd_model.load_state_dict(rd_state['state_dict'])
        self.rd_model = self.rd_model.to(device)
        self.rd_model.eval()

        # Initialize Macula classifier
        self.macula_model = build_3d_architecture(model_name, num_classes=1)
        macula_state = torch.load(macula_checkpoint_path, map_location=device)
        self.macula_model.load_state_dict(macula_state['state_dict'])
        self.macula_model = self.macula_model.to(device)
        self.macula_model.eval()

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess a video for inference.

        Args:
            video_path: Path to the video file

        Returns:
            Preprocessed video tensor ready for inference
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Read video: returns (T, H, W, C)
        video, _, _ = io.read_video(video_path, pts_unit='sec')

        # Convert to float tensor and permute to [C, D, H, W]
        video = video.float()  # [T, H, W, C]
        video = video.permute(3, 0, 1, 2)  # [C, D, H, W]

        # If video has 3 channels, convert to grayscale by averaging
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)  # [1, D, H, W]

        # Resize and normalize
        video = self.resize_tf(video)
        video = video / 255.0

        # Add batch dimension
        video = video.unsqueeze(0)
        return video

    def predict_single(self, video_path: str) -> Dict[str, Union[bool, Optional[bool]]]:
        """Run the diagnostic pipeline on a single video.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing:
                - has_rd: Whether retinal detachment is detected
                - macula_detached: Whether macula is detached (None if no RD detected)
        """
        # Preprocess video
        video = self.preprocess_video(video_path)
        video = video.to(self.device)

        with torch.no_grad():
            # First predict RD status
            rd_pred = torch.sigmoid(self.rd_model(video))
            has_rd = bool(rd_pred.item() > 0.5)

            # If RD detected, check macula status
            macula_detached = None
            if has_rd:
                macula_pred = torch.sigmoid(self.macula_model(video))
                macula_detached = bool(macula_pred.item() <= 0.5)  # 0 means detached, 1 means intact

        return {
            "has_rd": has_rd,
            "macula_detached": macula_detached
        }

    def predict_batch(self, csv_path: str, video_column: str = "path") -> pd.DataFrame:
        """Run the diagnostic pipeline on a batch of videos listed in a CSV file.

        Args:
            csv_path: Path to CSV file containing video paths
            video_column: Name of the column containing video paths

        Returns:
            DataFrame with predictions added as new columns
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Initialize results columns
        df['has_rd'] = False
        df['macula_detached'] = None

        # Process each video
        for idx, row in df.iterrows():
            video_path = row[video_column]
            
            # Make video path absolute if it's relative
            if not os.path.isabs(video_path):
                video_path = os.path.join(os.path.dirname(csv_path), video_path)
            
            # Get predictions
            predictions = self.predict_single(video_path)
            
            # Update DataFrame
            df.at[idx, 'has_rd'] = predictions['has_rd']
            df.at[idx, 'macula_detached'] = predictions['macula_detached']

        return df

def create_pipeline(
    model_name: str,
    experiment_root: str = "logs",
    video_size: Tuple[int, int, int] = (96, 128, 128),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> DiagnosticPipeline:
    """Helper function to create a diagnostic pipeline using the latest checkpoints.

    Args:
        model_name: Name of the architecture to use
        experiment_root: Root directory containing experiment logs
        video_size: Size to resize videos to
        device: Device to run inference on

    Returns:
        Initialized DiagnosticPipeline
    """
    # Find latest checkpoints
    rd_exp_path = os.path.join(experiment_root, "non_rd_vs_rd", model_name)
    macula_exp_path = os.path.join(experiment_root, "macula_detached_vs_intact", model_name)

    # Helper function to find best checkpoint
    def find_best_checkpoint(exp_path: str) -> str:
        checkpoints = list(Path(exp_path).rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {exp_path}")
        
        # Find checkpoint with "best" in name or take the last one
        best_ckpt = next((ckpt for ckpt in checkpoints if "best" in str(ckpt)), checkpoints[-1])
        return str(best_ckpt)

    rd_ckpt = find_best_checkpoint(rd_exp_path)
    macula_ckpt = find_best_checkpoint(macula_exp_path)

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

    # Create pipeline
    pipeline = create_pipeline(args.model)

    # Process input
    if args.input.endswith('.csv'):
        # Batch mode
        results = pipeline.predict_batch(args.input, args.video_column)
        if args.output:
            results.to_csv(args.output, index=False)
        print(results)
    else:
        # Single video mode
        result = pipeline.predict_single(args.input)
        print(f"Results for {args.input}:")
        print(f"Retinal Detachment: {'Yes' if result['has_rd'] else 'No'}")
        if result['has_rd']:
            print(f"Macula Status: {'Detached' if result['macula_detached'] else 'Intact'}")
