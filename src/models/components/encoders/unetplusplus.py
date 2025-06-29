# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep


# https://docs.monai.io/en/stable/networks.html#basicunetplusplus
class UNetPlusPlusEncoder(nn.Module):

    def __init__(
        self,
        spatial_dims= 3,
        in_channels = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Optional[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Optional[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Optional[float, tuple] = 0.0,
    ):
        """
        A UNet++ implementation with 1D/2D/3D supports.

        Based on:

            Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image
            Segmentation". 4th Deep Learning in Medical Image Analysis (DLMIA)
            Workshop, DOI: https://doi.org/10.48550/arXiv.1807.10165

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            deep_supervision: whether to prune the network at inference time. Defaults to False. If true, returns a list,
                whose elements correspond to outputs at different nodes.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        # print(f"BasicUNetPlusPlus features: {fea}.")

        self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.bottle_neck_embed_dim = fea[-2]


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        """
        
        x_0 = self.conv_0_0(x)
        x_1 = self.conv_1_0(x_0)
        x_2 = self.conv_2_0(x_1)
        x_3 = self.conv_3_0(x_2)
        x_4 = self.conv_4_0(x_3)

        return x_4


if __name__ == "__main__":
    model = UNetPlusPlusEncoder(
        spatial_dims= 3,
        in_channels = 1,
        features = (32, 32, 64, 128, 256, 32),
        act= ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm = ("instance", {"affine": True}),
        bias= True,
        dropout = 0.0,
    ).to("cuda:0")

    data = torch.randn(1, 1, 96, 128, 128).to("cuda:0")

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"number of trainable parameters: {trainable_params}")
    print(model(data).shape) # [1, 256, 6, 8, 8]