from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .building_blocks import DownTransition, InputTransition

# https://docs.monai.io/en/stable/networks.html#vnet

class VNetEncoder(nn.Module):


    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        act: Union[Tuple[str, Dict], str] = ("elu", {"inplace": True}),
        dropout_prob_down: Optional[float] = 0.5,
        dropout_prob_up: Tuple[Optional[float], float] = (0.5, 0.5),
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        """
            V-Net based on `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/pdf/1606.04797.pdf>`_.
            Adapted from `the official Caffe implementation
            <https://github.com/faustomilletari/VNet>`_. and `another pytorch implementation
            <https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py>`_.
            The model supports 2D or 3D inputs.
            Args:
                spatial_dims: spatial dimension of the input data. Defaults to 3.
                in_channels: number of input channels for the network. Defaults to 1.
                    The value should meet the condition that ``16 % in_channels == 0``.
                out_channels: number of output channels for the network. Defaults to 1.
                act: activation type in the network. Defaults to ``("elu", {"inplace": True})``.
                dropout_prob_down: dropout ratio for DownTransition blocks. Defaults to 0.5.
                dropout_prob_up: dropout ratio for UpTransition blocks. Defaults to (0.5, 0.5).
                dropout_dim: determine the dimensions of dropout. Defaults to (0.5, 0.5).

                    - ``dropout_dim = 1``, randomly zeroes some of the elements for each channel.
                    - ``dropout_dim = 2``, Randomly zeroes out entire channels (a channel is a 2D feature map).
                    - ``dropout_dim = 3``, Randomly zeroes out entire channels (a channel is a 3D feature map).
                bias: whether to have a bias term in convolution blocks. Defaults to False.
                    According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                    if a conv layer is directly followed by a batch norm layer, bias should be False.
        """
        super().__init__()

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.in_tr = InputTransition(spatial_dims, in_channels, 16, act, bias=bias)
        self.down_tr32 = DownTransition(spatial_dims, 16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(spatial_dims, 32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(spatial_dims, 64, 3, act, dropout_prob=dropout_prob_down, bias=bias)
        self.down_tr256 = DownTransition(spatial_dims, 128, 2, act, dropout_prob=dropout_prob_down, bias=bias)
        self.bottle_neck_embed_dim = 256

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        return out256

if __name__ == "__main__":
    model = VNetEncoder().to("cuda:0")
    data = torch.randn(1, 1, 96, 128, 128).to("cuda:0")

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"number of trainable parameters: {trainable_params}")
    print(model(data).shape) # [1, 256, 6, 8, 8]
