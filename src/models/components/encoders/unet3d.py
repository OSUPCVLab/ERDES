import torch
import torch.nn as nn

from .building_blocks import DoubleConv, create_encoders

# https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d

class Unet3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        f_maps: list = [64, 128, 256, 512, 768],
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        conv_upscale: int = 2,
        dropout_prob: float = 0.1,
        layer_order: str = "gcr",
        num_groups: int = 8,
        pool_kernel_size: int = 2,
    ):
        super().__init__()
        # assert isinstance(basic_module, nn.Sequential), f"basic_module should be nn.Sequential but it is {type(basic_module)}!"
        self.encoder = create_encoders(
            in_channels=in_channels,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            dropout_prob=dropout_prob,
            basic_module=DoubleConv,
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
            is3d=True,
        )


    def forward(self, x):
        for module in self.encoder:
            x = module(x)
        return x

if __name__ == "__main__":
    model = Unet3DEncoder(
        in_channels = 1,
        f_maps = [64, 128, 256, 512, 768],
        layer_order = "gcr",
        num_groups = 8,
        conv_padding = 1,
        conv_upscale = 2,
        dropout_prob = 0.1,
        conv_kernel_size = 3,
        pool_kernel_size = 2,
    ).to("cuda:0")

    data = torch.randn(1, 1, 96, 128, 128).to("cuda:0")
    out = model(data)
    print(out.shape)

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"number of trainable parameters: {trainable_params}")
    print(model(data).shape) # [1, 256, 24, 32, 32]
