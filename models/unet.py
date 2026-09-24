"""A plain U-Net: the ordinary convolutional encoder-decoder for segmentation, written from scratch.

No pretrained backbone, no attention. Each level is two 3x3 conv-BN-ReLU; the encoder halves the
resolution with max-pooling, the decoder doubles it bilinearly and concatenates the skip connection.
It is fully convolutional, so it can train on tiles and predict whole 2048 x 2048 frames.
"""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """`depth` down-sampling steps; channel widths base, 2*base, ... at each level. Output: logits."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(conv_block(previous, width))
            previous = width
        self.decoders = nn.ModuleList(
            conv_block(widths[level + 1] + widths[level], widths[level]) for level in reversed(range(depth))
        )
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x if level == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for decoder in self.decoders:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)
