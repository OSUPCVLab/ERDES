from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .encoders.swinunetr import SwinUnetrEncoder
from .encoders.unet3d import Unet3DEncoder
from .encoders.unetplusplus import UNetPlusPlusEncoder
from .encoders.unetr import UnetrEncoder
from .encoders.vit import ViTEncoder
from .encoders.vnet import VNetEncoder


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.ad_avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc1 = nn.Linear(input_dim, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, num_classes) 
    def forward(self, x):
        x = self.ad_avg_pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SwinUnetrClassifier(nn.Module):
    def __init__(
        self,        
        img_size: Union[Sequence[int], int],
        in_channels: int,
        num_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[tuple, str] = "batch",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    )->None:
        """
            Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            num_classes: number of the classification output.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.
        """
        super().__init__()
        self.enc = SwinUnetrEncoder(        
            img_size=img_size,
            in_channels = in_channels,
            depths = depths,
            num_heads = num_heads,
            feature_size = feature_size,
            norm_name = norm_name,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            dropout_path_rate = dropout_path_rate,
            normalize = normalize,
            use_checkpoint = use_checkpoint,
            spatial_dims = spatial_dims,
            downsample = downsample,
            use_v2 = use_v2,
        ) 
        input_dim = self.enc.bottle_neck_embed_dim
        self.cls = ClassificationHead(
            input_dim = input_dim, 
            hidden_size = input_dim // 2,
            num_classes = num_classes,
        )
    def forward(self, x):
        x = self.enc(x)
        x = self.cls(x)
        return x


class Unet3DClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        f_maps: list = [64, 128, 256, 512, 768],
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        conv_upscale: int = 2,
        dropout_prob: float = 0.0,
        layer_order: str = "gcr",
        num_groups: int = 8,
        pool_kernel_size: int = 2,
    ):
        super().__init__()
        self.enc = Unet3DEncoder(        
            in_channels=in_channels,
            f_maps=f_maps,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            dropout_prob=dropout_prob,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=pool_kernel_size,
         )
        input_dim = f_maps[-1]
        self.cls = ClassificationHead(
            input_dim = input_dim, 
            hidden_size = input_dim // 2,
            num_classes = num_classes,
        )
    def forward(self, x):
        x = self.enc(x)
        x = self.cls(x)
        return x

class UNetPlusPlusClassifier(nn.Module):
    def __init__(
        self,        
        in_channels: int,
        num_classes: int,
        spatial_dims= 3,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Optional[Union[str, tuple]] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Optional[Union[str, tuple]] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Optional[Union[float, tuple]] = 0.0,
    ):
        super().__init__()
        self.enc = UNetPlusPlusEncoder(     
            spatial_dims= spatial_dims,
            in_channels = in_channels,
            features = features,
            act=act,
            norm=norm,
            bias = bias,
            dropout = dropout,
        )
        input_dim = self.enc.bottle_neck_embed_dim
        self.cls = ClassificationHead(
            input_dim = input_dim, 
            hidden_size = input_dim // 2,
            num_classes = num_classes,
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.cls(x)
        return x

class UnetrClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: Optional[Union[tuple, str]] = "batch",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            num_classes: number of classification output 
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
        """
        super().__init__()
        self.enc = UnetrEncoder(        
            in_channels = in_channels,
            img_size = img_size,
            feature_size = feature_size,
            hidden_size = hidden_size,
            mlp_dim = mlp_dim,
            num_heads = num_heads,
            proj_type = proj_type,
            norm_name = norm_name,
            conv_block = conv_block,
            res_block = res_block,
            dropout_rate = dropout_rate,
            spatial_dims = spatial_dims,
            qkv_bias = qkv_bias,
            save_attn= save_attn,
        ) 
        input_dim = self.enc.bottle_neck_embed_dim
        self.cls = ClassificationHead(
            input_dim = input_dim, 
            hidden_size = input_dim // 2,
            num_classes = num_classes,
        )
        
    def forward(self, x):
        x = self.enc(x)
        x = self.cls(x)
        return x

class VNetClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        spatial_dims: int = 3,
        act: Union[Tuple[str, Dict], str] = ("elu", {"inplace": True}),
        dropout_prob_down: Optional[float] = 0.0,
        dropout_prob_up: Tuple[Optional[float], float] = (0.0, 0.0),
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        self.enc = VNetEncoder(        
            spatial_dims = spatial_dims,
            in_channels = in_channels,
            act = act,
            dropout_prob_down = dropout_prob_down,
            dropout_prob_up = dropout_prob_up,
            dropout_dim = dropout_dim,
            bias = bias,
        )
        input_dim = self.enc.bottle_neck_embed_dim
        self.cls = ClassificationHead(
            input_dim = input_dim, 
            hidden_size = input_dim // 2,
            num_classes = num_classes,
        )
    def forward(self, x):
        x = self.enc(x)
        x = self.cls(x)
        return x


class ViTClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        num_classes: int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 4,
        num_heads: int = 4,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation=None,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        super().__init__()
        self.enc = ViTEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            #post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        if post_activation == "Tanh":
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
        else:
            self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        feat = self.enc(x)
        out = self.classification_head(feat[:, 0])
        return out

if __name__ == "__main__":
    model = SwinUnetrClassifier(in_channels=1, num_classes=1, img_size=(96, 128, 128)).to("cuda:0")
    data = torch.randn(1, 1, 96, 128, 128).to("cuda:0")

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"number of trainable parameters: {trainable_params}")
    print(model(data).shape)