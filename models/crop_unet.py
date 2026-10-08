"""ResNet18 U-Net for gray/CLAHE/box-seed crops."""

import segmentation_models_pytorch as smp


def crop_unet(pretrained=True):
    return smp.Unet(
        encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1
    )
