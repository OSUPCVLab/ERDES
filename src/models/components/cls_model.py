from typing import Dict, Optional, Sequence, Tuple, Union
import math

import torch
import torch.nn as nn

from .encoders.swinunetr import SwinUnetrEncoder
from .encoders.unet3d import Unet3DEncoder
from .encoders.unetplusplus import UNetPlusPlusEncoder
from .encoders.unetr import UnetrEncoder
from .encoders.vit import ViTEncoder
from .encoders.vnet import VNetEncoder


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
        pooling: str = "avg",  # "avg", "topk"
        topk_ratio: float = 0.5,  # percentage of temporal regions to keep (0.5 = 50%)
    ):
        super().__init__()
        self.pooling = pooling
        self.topk_ratio = topk_ratio

        if pooling == "avg":
            self.pool_3d = nn.AdaptiveAvgPool3d(output_size=1)
        else:
            # For topk: pool spatial dims per frame first
            self.spatial_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Classification layers
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        if self.pooling == "avg":
            # Global average pooling over all dimensions
            x = self.pool_3d(x)  # [B, C, 1, 1, 1]
            x = x.flatten(1)  # [B, C]

        else:  # topk
            # Top-k pooling: spatial pool first, then select top-k frames by L2 norm
            x = x.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]
            x = x.reshape(B * D, C, H, W)  # [B*D, C, H, W]
            x = self.spatial_pool(x)  # [B*D, C, 1, 1]
            x = x.reshape(B, D, C)  # [B, D, C]

            # Compute frame importance as L2 norm of features
            frame_importance = x.norm(dim=2)  # [B, D]
            k = math.ceil(D * self.topk_ratio)  # adaptive k based on D
            _, topk_indices = frame_importance.topk(k, dim=1)  # [B, k]

            # Gather top-k frames
            topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)  # [B, k, C]
            x_topk = torch.gather(x, dim=1, index=topk_indices)  # [B, k, C]
            x = x_topk.mean(dim=1)  # [B, C]

        # Classification
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
        pooling: str = "avg",
        topk_ratio: float = 0.5,
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
            pooling = pooling,
            topk_ratio=topk_ratio,
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
        pooling: str = "avg",
        topk_ratio: float = 0.5,
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
            pooling = pooling,
            topk_ratio=topk_ratio,
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
        pooling: str = "avg",
        topk_ratio: float = 0.5,
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
            pooling = pooling,
            topk_ratio=topk_ratio,
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
        pooling: str = "avg",
        topk_ratio: float = 0.5,
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
            pooling = pooling,
            topk_ratio=topk_ratio,
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
        pooling: str = "avg",
        topk_ratio: float = 0.5,
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
            pooling = pooling,
            topk_ratio=topk_ratio,
        )
    def forward(self, x):
        x = self.enc(x)
        x = self.cls(x)
        return x


class ResNet3DClassifier(nn.Module):
    def __init__(
        self,
        block: str = "basic",
        layers: list = [4, 4, 4, 4],
        block_inplanes: list = [64, 128, 256, 512],
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1,
        pooling: str = "avg",
        topk_ratio: float = 0.5,
    ):
        super().__init__()
        from monai.networks.nets import ResNet
        self.pooling = pooling

        if pooling == "avg":
            # Use original MONAI ResNet with built-in GAP + FC
            self.model = ResNet(
                block=block,
                layers=layers,
                block_inplanes=block_inplanes,
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=num_classes,
            )
        else:
            # Use encoder only + custom ClassificationHead for top-k
            self.enc = ResNet(
                block=block,
                layers=layers,
                block_inplanes=block_inplanes,
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                feed_forward=False,
                num_classes=1,
            )
            input_dim = block_inplanes[-1] if block == "basic" else block_inplanes[-1] * 4
            self.cls = ClassificationHead(
                input_dim=input_dim,
                hidden_size=input_dim // 2,
                num_classes=num_classes,
                pooling=pooling,
                topk_ratio=topk_ratio,
            )

    def forward(self, x):
        if self.pooling == "avg":
            return self.model(x)
        else:
            # Manually extract features before avgpool to preserve spatial dims
            x = self.enc.conv1(x)
            x = self.enc.bn1(x)
            x = self.enc.act(x)
            x = self.enc.maxpool(x)
            x = self.enc.layer1(x)
            x = self.enc.layer2(x)
            x = self.enc.layer3(x)
            x = self.enc.layer4(x)
            # Now x is (B, C, D', H', W') - 5D feature map for top-k pooling
            x = self.cls(x)
            return x


class SENet3DClassifier(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1,
        pooling: str = "avg",
        topk_ratio: float = 0.5,
    ):
        super().__init__()
        from monai.networks.nets import SENet154
        self.pooling = pooling

        if pooling == "avg":
            # Use original MONAI SENet154 with built-in GAP + FC
            self.model = SENet154(
                pretrained=pretrained,
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                num_classes=num_classes,
            )
        else:
            # Use encoder only + custom ClassificationHead for top-k
            self.enc = SENet154(
                pretrained=pretrained,
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                num_classes=1,
            )
            input_dim = 2048  # SENet154: 512 * 4 expansion
            self.cls = ClassificationHead(
                input_dim=input_dim,
                hidden_size=input_dim // 2,
                num_classes=num_classes,
                pooling=pooling,
                topk_ratio=topk_ratio,
            )

    def forward(self, x):
        if self.pooling == "avg":
            return self.model(x)
        else:
            x = self.enc.features(x)
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
    device = "cpu"
    data = torch.randn(1, 1, 96, 128, 128).to(device)
    img_size = (96, 128, 128)

    print("=" * 70)
    print("ENCODER OUTPUT SHAPES - Showing how D (temporal) dimension changes")
    print("Input shape: [B=1, C=1, D=96, H=128, W=128]")
    print("=" * 70)

    with torch.no_grad():
        # 1. 3D U-Net - show layer by layer
        print("\n[1] 3D U-Net Classifier - Layer by layer:")
        model = Unet3DClassifier(in_channels=1, num_classes=1).to(device)
        x = data
        print(f"    Input:           {list(x.shape)} → D=96")
        for i, encoder in enumerate(model.enc.encoder):
            x = encoder(x)
            print(f"    After encoder {i}: {list(x.shape)} → D={x.shape[2]}")
        print(f"    *** Final D={x.shape[2]} (from 96 frames) ***")
        del model, x

        # 2. V-Net - show layer by layer
        print("\n[2] V-Net Classifier - Layer by layer:")
        model = VNetClassifier(in_channels=1, num_classes=1).to(device)
        x = data
        print(f"    Input:       {list(x.shape)} → D=96")
        x = model.enc.in_tr(x)
        print(f"    After in_tr: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.down_tr32(x)
        print(f"    After down1: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.down_tr64(x)
        print(f"    After down2: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.down_tr128(x)
        print(f"    After down3: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.down_tr256(x)
        print(f"    After down4: {list(x.shape)} → D={x.shape[2]}")
        print(f"    *** Final D={x.shape[2]} (from 96 frames) ***")
        del model, x

        # 3. ResNet3D - show layer by layer
        print("\n[3] ResNet3D Classifier - Layer by layer:")
        model = ResNet3DClassifier(in_channels=1, num_classes=1, pooling="topk").to(device)
        x = data
        print(f"    Input:        {list(x.shape)} → D=96")
        x = model.enc.conv1(x)
        x = model.enc.bn1(x)
        x = model.enc.act(x)
        print(f"    After conv1:  {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.maxpool(x)
        print(f"    After maxpool:{list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer1(x)
        print(f"    After layer1: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer2(x)
        print(f"    After layer2: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer3(x)
        print(f"    After layer3: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer4(x)
        print(f"    After layer4: {list(x.shape)} → D={x.shape[2]}")
        print(f"    *** Final D={x.shape[2]} (from 96 frames) ***")
        del model, x

        # 4. SENet154 - show layer by layer
        print("\n[4] SENet154 Classifier - Layer by layer:")
        model = SENet3DClassifier(in_channels=1, num_classes=1, pooling="topk").to(device)
        x = data
        print(f"    Input:        {list(x.shape)} → D=96")
        x = model.enc.layer0(x)
        print(f"    After layer0: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer1(x)
        print(f"    After layer1: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer2(x)
        print(f"    After layer2: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer3(x)
        print(f"    After layer3: {list(x.shape)} → D={x.shape[2]}")
        x = model.enc.layer4(x)
        print(f"    After layer4: {list(x.shape)} → D={x.shape[2]}")
        print(f"    *** Final D={x.shape[2]} (from 96 frames) ***")
        del model, x

        # 5. UNet++ - show layer by layer
        print("\n[5] UNet++ Classifier - Layer by layer:")
        model = UNetPlusPlusClassifier(in_channels=1, num_classes=1).to(device)
        x = data
        print(f"    Input:         {list(x.shape)} → D=96")
        x = model.enc.conv_0_0(x)
        print(f"    After conv_0_0:{list(x.shape)} → D={x.shape[2]}")
        x = model.enc.conv_1_0(x)
        print(f"    After conv_1_0:{list(x.shape)} → D={x.shape[2]}")
        x = model.enc.conv_2_0(x)
        print(f"    After conv_2_0:{list(x.shape)} → D={x.shape[2]}")
        x = model.enc.conv_3_0(x)
        print(f"    After conv_3_0:{list(x.shape)} → D={x.shape[2]}")
        x = model.enc.conv_4_0(x)
        print(f"    After conv_4_0:{list(x.shape)} → D={x.shape[2]}")
        print(f"    *** Final D={x.shape[2]} (from 96 frames) ***")
        del model, x

        # 6. Swin-UNETR - show layer by layer
        print("\n[6] Swin-UNETR Classifier - Layer by layer:")
        model = SwinUnetrClassifier(in_channels=1, num_classes=1, img_size=img_size).to(device)
        print(f"    Input:         {list(data.shape)} → D=96")
        hidden_states = model.enc.swinViT(data, model.enc.normalize)
        for i, h in enumerate(hidden_states):
            print(f"    Stage {i}:       {list(h.shape)} → D={h.shape[2]}")
        print(f"    *** Final D={hidden_states[-1].shape[2]} (from 96 frames) ***")
        del model, hidden_states

        # 7. UNETR - show layer by layer
        print("\n[7] UNETR Classifier - Layer by layer:")
        model = UnetrClassifier(in_channels=1, num_classes=1, img_size=img_size).to(device)
        print(f"    Input:         {list(data.shape)} → D=96")
        x_vit, hidden_states = model.enc.vit(data)
        print(f"    ViT output:    {list(x_vit.shape)} (tokens, not spatial)")
        for i, h in enumerate(hidden_states):
            print(f"    Hidden {i:2d}:     {list(h.shape)} (tokens)")
        enc_out = model.enc(data)
        print(f"    Final concat:  {list(enc_out.shape)} → D={enc_out.shape[2]}")
        print(f"    *** Final D={enc_out.shape[2]} (from 96 frames) ***")
        del model, x_vit, hidden_states, enc_out

        # 8. ViT - show layer by layer
        print("\n[8] ViT Classifier - Layer by layer:")
        model = ViTClassifier(in_channels=1, num_classes=1, img_size=img_size, patch_size=7).to(device)
        print(f"    Input:           {list(data.shape)} → D=96")
        x = model.enc.patch_embedding(data)
        print(f"    After patches:   {list(x.shape)} (sequence of tokens)")
        cls_token = model.enc.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        print(f"    With CLS token:  {list(x.shape)}")
        for i, blk in enumerate(model.enc.blocks):
            x = blk(x)
        x = model.enc.norm(x)
        print(f"    After blocks:    {list(x.shape)}")
        print(f"    CLS token used for classification (no spatial D dimension)")
        del model, x, cls_token

        # Summary table
        print("\n" + "=" * 70)
        print("SUMMARY TABLE:")
        print("=" * 70)

        results = []

        # Swin-UNETR
        model = SwinUnetrClassifier(in_channels=1, num_classes=1, img_size=img_size).to(device)
        enc_out = model.enc(data)
        results.append(("Swin-UNETR", enc_out.shape[2], enc_out.shape[1]))
        del model, enc_out

        # 3D U-Net
        model = Unet3DClassifier(in_channels=1, num_classes=1).to(device)
        enc_out = model.enc(data)
        results.append(("3D U-Net", enc_out.shape[2], enc_out.shape[1]))
        del model, enc_out

        # UNet++
        model = UNetPlusPlusClassifier(in_channels=1, num_classes=1).to(device)
        enc_out = model.enc(data)
        results.append(("UNet++", enc_out.shape[2], enc_out.shape[1]))
        del model, enc_out

        # UNETR
        model = UnetrClassifier(in_channels=1, num_classes=1, img_size=img_size).to(device)
        enc_out = model.enc(data)
        results.append(("UNETR", enc_out.shape[2], enc_out.shape[1]))
        del model, enc_out

        # V-Net
        model = VNetClassifier(in_channels=1, num_classes=1).to(device)
        enc_out = model.enc(data)
        results.append(("V-Net", enc_out.shape[2], enc_out.shape[1]))
        del model, enc_out

        # ResNet3D
        model = ResNet3DClassifier(in_channels=1, num_classes=1, pooling="topk").to(device)
        x = model.enc.conv1(data)
        x = model.enc.bn1(x)
        x = model.enc.act(x)
        x = model.enc.maxpool(x)
        x = model.enc.layer1(x)
        x = model.enc.layer2(x)
        x = model.enc.layer3(x)
        x = model.enc.layer4(x)
        results.append(("ResNet3D", x.shape[2], x.shape[1]))
        del model, x

        # SENet154
        model = SENet3DClassifier(in_channels=1, num_classes=1, pooling="topk").to(device)
        enc_out = model.enc.features(data)
        results.append(("SENet154", enc_out.shape[2], enc_out.shape[1]))
        del model, enc_out

        # ViT
        model = ViTClassifier(in_channels=1, num_classes=1, img_size=img_size, patch_size=7).to(device)
        enc_out = model.enc(data)
        results.append(("ViT", "N/A (tokens)", enc_out.shape[2]))
        del model, enc_out

        print(f"{'Model':<15} {'D (temporal)':<15} {'C (features)':<15} {'TopK k=5 selects'}")
        print("-" * 60)
        for name, d, c in results:
            if isinstance(d, int):
                k_select = f"{min(5, d)} of {d}"
                if min(5, d) == d:
                    k_select += " (ALL = avg)"
            else:
                k_select = "CLS token only"
            print(f"{name:<15} {str(d):<15} {str(c):<15} {k_select}")

        print("\n" + "=" * 70)
        print("CONCLUSION: 96 frames → 3-12 temporal regions after encoding")
        print("TopK selects from these regions, NOT from original 96 frames")
        print("=" * 70)