import numpy as np
import pytest
import torch

from models.pretrained_unet import PretrainedUNet
from pretrained_unet.data import tubed_skeleton
from pretrained_unet.training import predict_frame, skeleton_recall_loss


@pytest.mark.parametrize("encoder", ["efficientnet_b3", "convnext_tiny"])
def test_encoder_forward_backward(encoder):
    model = PretrainedUNet(encoder, pretrained=False).eval()
    x = torch.rand(1, 2, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 1, 64, 64)
    logits.mean().backward()
    assert next(model.net.encoder.parameters()).grad is not None


def test_recall_penalizes_break_and_empty_has_no_gradient():
    skeleton = torch.zeros(2, 16, 16)
    skeleton[0, 8, 2:14] = 1
    intact = torch.full((2, 1, 16, 16), 8.0)
    broken = intact.clone()
    broken[0, 0, 8, 6:10] = -8
    assert skeleton_recall_loss(broken, skeleton) > skeleton_recall_loss(intact, skeleton) + 0.3
    logits = broken.requires_grad_()
    skeleton_recall_loss(logits, skeleton).backward()
    assert logits.grad[0, 0, 8, 7] < 0
    assert logits.grad[1].abs().sum() == 0
    assert skeleton_recall_loss(logits, torch.zeros_like(skeleton)) == 0


def test_tube_stays_in_foreground():
    mask = np.zeros((32, 32), bool)
    mask[5:27, 14:18] = True
    tube = tubed_skeleton(mask)
    assert tube.any()
    assert not (tube & ~mask).any()


@pytest.mark.parametrize("shape", [(31, 47), (97, 115)])
def test_tile_blending_preserves_pixel_predictions(shape):
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    x = torch.randn(2, *shape)
    actual = predict_frame(Identity(), x, torch.device("cpu"), tile=64, stride=48)
    expected = x[0].sigmoid().mul(255).round().numpy().astype(np.uint8)
    assert np.max(np.abs(actual.astype(float) - expected.astype(float))) <= 1


def test_skeleton_sampling_matches_baseline(tmp_path):
    import cv2
    from pycocotools import mask as mask_util

    from cnn_baseline.data import FilamentFrames
    from pretrained_unet.data import SkeletonFrames

    mask = np.zeros((128, 128), np.uint8)
    mask[20:100, 60:66] = 1
    rle = mask_util.encode(np.asfortranarray(mask))

    class Annotations:
        def rles_for_file(self, name):
            return [[rle], []]

    cv2.imwrite(str(tmp_path / "test.jpeg"), np.full((128, 128), 100, np.uint8))
    args = (["test.jpeg"], tmp_path, Annotations())
    torch.manual_seed(0)
    bx, by = FilamentFrames(*args, tile=64, tiles_per_frame=8)[0]
    torch.manual_seed(0)
    sx, sy, skeleton = SkeletonFrames(*args, tile=64, tiles_per_frame=8)[0]
    assert torch.equal(bx, sx) and torch.equal(by, sy)
    assert torch.all(skeleton <= sy)
    assert skeleton.max() == 0.5


def test_training_saves_reloadable_checkpoint(tmp_path):
    from pretrained_unet.training import fit

    class SmallNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Module()
            self.net.encoder = torch.nn.Conv2d(2, 4, 3, padding=1)
            self.net.decoder = torch.nn.Conv2d(4, 4, 3, padding=1)
            self.net.segmentation_head = torch.nn.Conv2d(4, 1, 1)

        def forward(self, x):
            return self.net.segmentation_head(self.net.decoder(self.net.encoder(x)))

    model = SmallNet()
    x, y = torch.rand(2, 2, 32, 32), torch.zeros(2, 32, 32)
    y[:, 10:20, 15:18] = 1
    skeleton = y.clone()
    history = fit(model, [(x, y, skeleton)] * 3, torch.device("cpu"), 2, 0.2, tmp_path, {})
    saved = torch.load(tmp_path / "model.pt", weights_only=True)
    assert len(history) == 2
    other = SmallNet()
    other.load_state_dict(saved["state_dict"])
    assert torch.equal(model(x), other(x))
