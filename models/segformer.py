"""SegFormer: a hierarchical transformer encoder (Mix Transformer) with its light all-MLP decoder.

The encoder's self-attention sees the whole tile, so a long filament is one context rather than a chain of
local patches. The decoder predicts at 1/4 of the input resolution and is upsampled, as in the paper; whether
that loses the 10-20 px wide filaments is part of what the experiment measures.
"""

import segmentation_models_pytorch as smp
import torch
from torch import nn


class SegFormer(nn.Module):
    """MiT-B2 encoder (ImageNet), two input channels (smp adapts the RGB patch embedding), logits out."""

    def __init__(self, variant: str = "mit_b2", pretrained: bool = True):
        super().__init__()
        self.net = smp.Segformer(
            encoder_name=variant, encoder_weights="imagenet" if pretrained else None, in_channels=2, classes=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - 0.449) / 0.226)

    def pretrained_parameters(self):
        return self.net.encoder.parameters()
