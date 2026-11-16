"""ImageNet encoders with a common full-resolution U-Net decoder."""

import segmentation_models_pytorch as smp
import torch
from torch import nn

ENCODERS = {
    "efficientnet_b3": "tu-efficientnet_b3.ra2_in1k",
    "convnext_tiny": "tu-convnext_tiny.fb_in1k",
    # keeps a 1/4-resolution branch through every stage instead of only downsampling, for thin structures
    "hrnet_w18": "tu-hrnet_w18.ms_aug_in1k",
}


class PretrainedUNet(nn.Module):
    """Two grayscale-derived channels; timm adapts RGB stem weights to two channels.

    Both channels use the same scalar ImageNet normalization, so neither acquires
    an arbitrary RGB colour identity. These constants are part of the experiment.
    """

    def __init__(self, encoder: str, pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=ENCODERS[encoder],
            encoder_weights="imagenet" if pretrained else None,
            in_channels=2,
            classes=1,
            decoder_channels=(256, 128, 64, 32, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - 0.449) / 0.226)

    def pretrained_parameters(self):
        """The ImageNet-initialised part, trained with a smaller learning rate than the decoder."""
        return self.net.encoder.parameters()
