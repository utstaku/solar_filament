"""Swin-UNet: a Swin Transformer encoder with a U-Net decoder and skip connections.

Shifted-window self-attention gives each 1/4-resolution token a context of whole windows that move between
blocks, so a filament can be followed across a tile; the U-Net decoder brings the four stages (1/4 .. 1/32)
back up. Swin starts at 1/4 resolution, too coarse for 10-20 px wide filaments on its own, so two small
convolutional stages at full and half resolution supply the last two skip connections.

The encoder is timm's ImageNet Swin-T, rebuilt for 512 px tiles with 8 x 8 windows (512 / 4 / 8 = 16
windows per side at every stage); timm resizes the pretrained position tables.
"""

import timm
import torch
from torch import nn
from torch.nn import functional as F

from models.unet import conv_block

ENCODER = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"


class SwinUNet(nn.Module):
    def __init__(self, pretrained: bool = True, tile: int = 512, stem: tuple[int, int] = (16, 32)):
        super().__init__()
        self.encoder = timm.create_model(
            ENCODER,
            pretrained=pretrained,
            features_only=True,
            in_chans=2,
            img_size=tile,
            window_size=8,
            strict_img_size=False,
        )
        widths = self.encoder.feature_info.channels()  # 96, 192, 384, 768 at 1/4 .. 1/32
        self.stem_full = conv_block(2, stem[0])
        self.stem_half = conv_block(stem[0], stem[1])
        skips = [stem[0], stem[1], *widths[:-1]]  # 1/1, 1/2, 1/4, 1/8, 1/16
        decoder = [stem[0], stem[1], *widths[:-1]]
        self.decoders = nn.ModuleList()
        previous = widths[-1]
        for skip, width in zip(reversed(skips), reversed(decoder)):
            self.decoders.append(conv_block(previous + skip, width))
            previous = width
        self.head = nn.Conv2d(stem[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 0.449) / 0.226
        full = self.stem_full(x)
        half = self.stem_half(F.max_pool2d(full, 2))
        stages = [f.permute(0, 3, 1, 2) for f in self.encoder(x)]  # timm's Swin features are NHWC
        skips = [full, half, *stages[:-1]]
        y = stages[-1]
        for decoder, skip in zip(self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = decoder(torch.cat([y, skip], dim=1))
        return self.head(y)

    def pretrained_parameters(self):
        return self.encoder.parameters()
