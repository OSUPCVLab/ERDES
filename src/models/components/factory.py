"""
To select the architecture based on a config file we need to ensure
we import each of the architectures into this file. Once we have that
we can use a keyword from the config file to build the model.
"""

import math
from abc import ABC, abstractmethod

import torch.nn as nn
import torchvision


# Base class for all architecture builders
class ArchitectureBuilder(ABC):
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes

    @abstractmethod
    def build(self):
        pass

# EfficientNet3D Builder
class EfficientNet3DBuilder(ArchitectureBuilder):
    def build(self):
        from monai.networks.nets import EfficientNetBN
        return EfficientNetBN(
            "efficientnet-b0",
            pretrained=False,
            spatial_dims=3,
            in_channels=1,
            num_classes=self.num_classes,
        )

# ResNet3D Builder
class ResNet3DBuilder(ArchitectureBuilder):
    def build(self):
        from monai.networks.nets import ResNet
        return ResNet(
            block="basic",
            layers=[4, 4, 4, 4],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=self.num_classes,
        )

# SENet3D Builder
class SENet3DBuilder(ArchitectureBuilder):
    def build(self):
        from monai.networks.nets import SENet154
        return SENet154(pretrained=False, spatial_dims=3, in_channels=1, num_classes=self.num_classes)

# Unet3D Builder
class Unet3DBuilder(ArchitectureBuilder):
    def build(self):
        from .cls_model import Unet3DClassifier
        return Unet3DClassifier(
            in_channels=1,
            num_classes=self.num_classes,
        )

# SwinUnetr Builder
class SwinUnetrBuilder(ArchitectureBuilder):
    def build(self):
        from .cls_model import SwinUnetrClassifier
        return SwinUnetrClassifier(
            img_size=(96, 128, 128),
            in_channels=1,
            num_classes=self.num_classes,
        )

# UNetPlusPlus Builder
class UNetPlusPlusBuilder(ArchitectureBuilder):
    def build(self):
        from .cls_model import UNetPlusPlusClassifier
        return UNetPlusPlusClassifier(
            in_channels=1,
            num_classes=self.num_classes,
        )

# VNet Builder
class VNetBuilder(ArchitectureBuilder):
    def build(self):
        from .cls_model import VNetClassifier
        return VNetClassifier(
            in_channels=1,
            num_classes=self.num_classes,
        )

# Unetr Builder
class UnetrBuilder(ArchitectureBuilder):
    def build(self):
        from .cls_model import UnetrClassifier
        img_size = (128, 128, 128)
        return UnetrClassifier(
            in_channels=1,
            num_classes=self.num_classes,
            img_size=img_size,
        )

# ViT Builder
class ViTBuilder(ArchitectureBuilder):
    def build(self):
        from .cls_model import ViTClassifier
        img_size = (128, 128, 128)
        return ViTClassifier(
            in_channels=1,
            img_size=img_size,
            patch_size=7,
            num_classes=self.num_classes,
        )

# Registry of builders
ARCHITECTURE_BUILDERS = {
    "efficentnet_b0": EfficientNet3DBuilder,
    "resnet3d": ResNet3DBuilder,
    "senet": SENet3DBuilder,
    "unet3d": Unet3DBuilder,
    "swinunetr": SwinUnetrBuilder,
    "unetplusplus": UNetPlusPlusBuilder,
    "vnet": VNetBuilder,
    "unetr": UnetrBuilder,
    "vit": ViTBuilder,
}

def build_3d_architecture(model_name: str, num_classes: int = 1):
    builder_cls = ARCHITECTURE_BUILDERS.get(model_name)
    if builder_cls is None:
        raise ValueError(
            f"specified model '{model_name}' not supported, edit build_architecture.py file"
        )
    builder = builder_cls(num_classes=num_classes)
    return builder.build()