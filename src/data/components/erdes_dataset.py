import os
from typing import Tuple, Union

import pandas as pd
import torch
import torchvision.io as io
from torch.utils.data import Dataset

from .utils import resize


class VideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        size: Union[int, Tuple[int, int, int]],  # Desired (D, H, W)
        video_column: str = "path",
        label_column: str = "label",
    ):
        self.df = pd.read_csv(csv_path)
        self.video_paths = self.df[video_column].tolist()
        self.labels = self.df[label_column].tolist()
        self.resize_tf = resize(size)
        self.size = size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Read video: returns (T, H, W, C)
        video, _, _ = io.read_video(video_path, pts_unit='sec')

        # Convert to float tensor and permute to [C, D, H, W]
        video = video.float() / 255.0  # [T, H, W, C]
        video = video.permute(3, 0, 1, 2)  # [C, D, H, W]

        # If video has 3 channels, convert to grayscale by averaging
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)  # [1, D, H, W]

        # Apply transforms
        video = self.resize_tf(video)

        label = torch.tensor(label, dtype=torch.float32)

        return video, label
