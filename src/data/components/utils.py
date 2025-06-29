from typing import Dict, List, Tuple, Union

import pytorchvideo.transforms.transforms as VT
import torch
import torchvision.transforms as T
from torch.nn.functional import interpolate, pad


class PadToSquare3D:
    """
    Pads a [C, D, H, W] video tensor to make H == W by adding symmetric padding.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h == w:
            return x

        diff = abs(h - w)
        pad_left = diff // 2
        pad_right = diff - pad_left

        if h < w:
            # Pad height (top, bottom)
            padding = (0, 0, pad_left, pad_right)  # Pad H axis
        else:
            # Pad width (left, right)
            padding = (pad_left, pad_right, 0, 0)  # Pad W axis

        # Pad format: (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
        # Needs to be applied on the last two dims (H, W), across all frames and channels
        x = pad(x, pad=padding, mode='constant', value=0)
        return x


class Interpolate3D:
    def __init__(self, mode: str, size: Union[int, Tuple]):
        assert mode in [
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear", 
            "area",
            "nearest-exact"
        ]
        self.mode = mode 
        self.size = size 

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor with shape [C, D, H, W]
        """
        x = interpolate(x.unsqueeze(0), size=self.size, mode=self.mode).squeeze(0)
        return x 


def resize(size):
    d, h, w = size
    tf = T.Compose(
        [
            PadToSquare3D(),  # Ensure square before resizing
            Interpolate3D(mode="trilinear", size=(d, h, w)),
        ]
    )
    return tf
