"""
Diagnostic pipeline for ocular ultrasound classification.

This pipeline uses two pretrained models:
1. RD Classifier: Predicts whether a patient has retinal detachment (RD)
2. Macula Classifier: If RD is detected, predicts whether the macula is detached or intact

Pipeline flow:
    Input Video
        │
        ▼
    RD Classifier
        │
        ├── RD Positive → Macula Classifier → Detached / Intact
        │
        └── RD Negative → Done
"""

import os
import logging
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
import torchvision.io as io
from pathlib import Path
from tqdm import tqdm

from erdes.models.components.factory import build_3d_architecture
from erdes.data.components.utils import resize

# Default weights directory (relative to this file's parent directory)
DEFAULT_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DiagnosticPipeline:
    def __init__(
        self,
        rd_model_name: str,
        macula_model_name: str,
        rd_checkpoint_path: str,
        macula_checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        video_size: Tuple[int, int, int] = (96, 128, 128),
    ):
        logging.info("Initializing diagnostic pipeline...")
        self.device = device
        self.video_size = video_size
        self.resize_tf = resize(video_size)

        # Handle legacy checkpoints saved with 'src' module name
        import sys
        import erdes
        sys.modules['src'] = erdes

        def adapt_state_dict(state_dict, model):
            """Adapt checkpoint state_dict to match model's expected keys."""
            # First strip 'net.' prefix if present
            adapted = {
                k.replace("net.", "", 1) if k.startswith("net.") else k: v
                for k, v in state_dict.items()
            }

            # Get model's expected keys
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(adapted.keys())

            # If keys already match, return as-is
            if model_keys == ckpt_keys:
                return adapted

            # Check if we need to add 'model.' prefix (for ResNet3DClassifier with avg pooling)
            if any(k.startswith("model.") for k in model_keys) and not any(k.startswith("model.") for k in ckpt_keys):
                adapted = {f"model.{k}": v for k, v in adapted.items()}

            # Check if we need to add 'enc.' prefix (for other models)
            elif any(k.startswith("enc.") for k in model_keys) and not any(k.startswith("enc.") for k in ckpt_keys):
                # Split keys between encoder and classifier
                new_adapted = {}
                for k, v in adapted.items():
                    if k.startswith("fc1.") or k.startswith("fc2."):
                        new_adapted[f"cls.{k}"] = v
                    else:
                        new_adapted[f"enc.{k}"] = v
                adapted = new_adapted

            return adapted

        logging.info(f"Loading RD model checkpoint ({rd_model_name})...")
        self.rd_model = build_3d_architecture(rd_model_name, num_classes=1)
        rd_state = torch.load(rd_checkpoint_path, map_location=device, weights_only=False)
        adapted_rd_state = adapt_state_dict(rd_state['state_dict'], self.rd_model)
        self.rd_model.load_state_dict(adapted_rd_state, strict=True)
        self.rd_model = self.rd_model.to(device)
        self.rd_model.eval()

        logging.info(f"Loading Macula model checkpoint ({macula_model_name})...")
        self.macula_model = build_3d_architecture(macula_model_name, num_classes=1)
        macula_state = torch.load(macula_checkpoint_path, map_location=device, weights_only=False)
        adapted_macula_state = adapt_state_dict(macula_state['state_dict'], self.macula_model)
        self.macula_model.load_state_dict(adapted_macula_state, strict=True)
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

    def predict_single(self, video_path: str) -> Dict[str, Union[bool, Optional[str]]]:
        logging.info(f"Running prediction on video: {video_path}")
        video = self.preprocess_video(video_path)
        video = video.to(self.device)

        with torch.no_grad():
            rd_pred = torch.sigmoid(self.rd_model(video))
            has_rd = bool(rd_pred.item() >= 0.5)
            logging.info(f"RD prediction: {'Positive' if has_rd else 'Negative'}")

            diagnosis = None
            if has_rd:
                # Class 0 = Detached, Class 1 = Intact
                macula_pred = torch.sigmoid(self.macula_model(video))
                diagnosis = "Macula Intact" if macula_pred.item() >= 0.5 else "Macula Detached"
                logging.info(f"Macula prediction: {diagnosis}")

        return {
            "has_rd": has_rd,
            "diagnosis": diagnosis
        }

    def predict_batch(self, csv_path: str, video_column: str = "path") -> pd.DataFrame:
        logging.info(f"Loading input CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df['has_rd'] = False
        df['diagnosis'] = None

        logging.info(f"Processing {len(df)} videos...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            video_path = row[video_column]
            try:
                predictions = self.predict_single(video_path)
                df.at[idx, 'has_rd'] = predictions['has_rd']
                df.at[idx, 'diagnosis'] = predictions['diagnosis']
            except Exception as e:
                logging.error(f"Error processing {video_path}: {e}")
                continue

        logging.info("Batch prediction complete.")
        return df

def create_pipeline(
    rd_model_name: str = "unet3d",
    macula_model_name: Optional[str] = None,
    rd_checkpoint_path: Optional[str] = None,
    macula_checkpoint_path: Optional[str] = None,
    weights_dir: Optional[str] = None,
    video_size: Tuple[int, int, int] = (96, 128, 128),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> DiagnosticPipeline:
    """
    Create a diagnostic pipeline.

    Args:
        rd_model_name: Model architecture for RD classifier (default: unet3d)
        macula_model_name: Model architecture for macula classifier (default: same as rd_model_name)
        rd_checkpoint_path: Path to RD model checkpoint (default: weights/non_rd_vs_rd.ckpt)
        macula_checkpoint_path: Path to macula model checkpoint (default: weights/macula_detached_vs_intact.ckpt)
        weights_dir: Directory containing model weights (default: ERDES/weights/)
        video_size: Input video dimensions (T, H, W)
        device: Device to run inference on

    Returns:
        DiagnosticPipeline instance
    """
    if macula_model_name is None:
        macula_model_name = rd_model_name

    # Use default weights directory if not specified
    if weights_dir is None:
        weights_dir = DEFAULT_WEIGHTS_DIR
    else:
        weights_dir = Path(weights_dir)

    # Use default checkpoint paths if not specified
    if rd_checkpoint_path is None:
        rd_checkpoint_path = weights_dir / "non_rd_vs_rd.ckpt"
    if macula_checkpoint_path is None:
        macula_checkpoint_path = weights_dir / "macula_detached_vs_intact.ckpt"

    # Validate checkpoint paths exist
    if not Path(rd_checkpoint_path).exists():
        raise FileNotFoundError(f"RD checkpoint not found: {rd_checkpoint_path}")
    if not Path(macula_checkpoint_path).exists():
        raise FileNotFoundError(f"Macula checkpoint not found: {macula_checkpoint_path}")

    logging.info("Creating pipeline...")
    return DiagnosticPipeline(
        rd_model_name=rd_model_name,
        macula_model_name=macula_model_name,
        rd_checkpoint_path=str(rd_checkpoint_path),
        macula_checkpoint_path=str(macula_checkpoint_path),
        video_size=video_size,
        device=device
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run retinal diagnostic pipeline on video data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single video (uses default weights from weights/ folder)
  python diagnostic_pipeline.py --input video.mp4

  # Run on a CSV of videos
  python diagnostic_pipeline.py --input videos.csv --output results.csv

  # Use custom checkpoint paths
  python diagnostic_pipeline.py --input video.mp4 --rd-ckpt /path/to/rd.ckpt --macula-ckpt /path/to/macula.ckpt
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input video file or CSV containing video paths")
    parser.add_argument("--output", type=str,
                        help="Path to save CSV results (for batch mode)")
    parser.add_argument("--video-column", type=str, default="path",
                        help="Column name containing video paths in CSV (default: path)")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Directory containing model weights (default: ERDES/weights/)")
    parser.add_argument("--rd-ckpt", type=str, default=None,
                        help="Path to RD classifier checkpoint (overrides --weights-dir)")
    parser.add_argument("--macula-ckpt", type=str, default=None,
                        help="Path to macula classifier checkpoint (overrides --weights-dir)")
    parser.add_argument("--rd-model", type=str, default="unet3d",
                        help="Model architecture for RD classifier (default: unet3d)")
    parser.add_argument("--macula-model", type=str, default=None,
                        help="Model architecture for macula classifier (default: same as --rd-model)")

    args = parser.parse_args()

    pipeline = create_pipeline(
        rd_model_name=args.rd_model,
        macula_model_name=args.macula_model,
        rd_checkpoint_path=args.rd_ckpt,
        macula_checkpoint_path=args.macula_ckpt,
        weights_dir=args.weights_dir
    )

    if args.input.endswith('.csv'):
        results = pipeline.predict_batch(args.input, args.video_column)
        if args.output:
            results.to_csv(args.output, index=False)
            logging.info(f"Results saved to {args.output}")
        print(results)
    else:
        result = pipeline.predict_single(args.input)
        print(f"\nResults for {args.input}:")
        print(f"  Retinal Detachment: {'Yes' if result['has_rd'] else 'No'}")
        if result['diagnosis']:
            print(f"  Diagnosis: {result['diagnosis']}")
