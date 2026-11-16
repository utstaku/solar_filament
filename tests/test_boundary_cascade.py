import json

import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader

from boundary_cascade.boundary import boundary_loss
from boundary_cascade.cascade import detector_dataset
from boundary_cascade.common import train_model
from boundary_cascade.crops import Crops, crop_input, exclusive_labels
from cnn_baseline.training import bce_dice_loss
from dataset.annotations import Annotations
from models.crop_unet import crop_unet
from models.unet import UNet


def test_boundary_loss_control_and_local_gradients():
    target = torch.zeros(1, 32, 32)
    target[:, 8:24, 8:24] = 1
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    assert torch.equal(boundary_loss(logits, target, 0), bce_dice_loss(logits, target))
    base = torch.autograd.grad(bce_dice_loss(logits, target), logits, retain_graph=True)[0]
    weighted = torch.autograd.grad(boundary_loss(logits, target, 4), logits)[0]
    assert weighted[0, 0, 8, 16].abs() > base[0, 0, 8, 16].abs()
    assert weighted[0, 0, 16, 16] == base[0, 0, 16, 16]
    assert torch.isfinite(boundary_loss(torch.zeros_like(logits), target * 0.5, 2))


def test_crop_roundtrip_and_exclusion():
    image = np.full((128, 160), 110, np.uint8)
    x, (left, top, side) = crop_input(image, [0, 0, 40, 60], size=64)
    assert x.shape == (3, 64, 64) and left == top == 0
    assert side <= min(image.shape) and np.isfinite(x).all()
    candidates = [(0.9, 0, 0, np.full((20, 20), 255, np.uint8)), (0.7, 10, 0, np.full((20, 20), 220, np.uint8))]
    (labels, areas, boxes), probs = exclusive_labels(candidates, image.shape, min_area=10)
    assert areas.tolist() == [400, 200]
    assert (labels[:, :10] == 2).sum() == 0
    assert boxes.tolist() == [[0, 0, 20, 20], [20, 0, 10, 20]]
    assert probs[0, 0] == 255 and probs[0, 20] == 220
    empty, _ = exclusive_labels([], image.shape)
    assert empty[2].shape == (0, 4)


def test_dataset_and_training(tmp_path):
    import cv2

    image = np.full((128, 128), 100, np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpeg"), image)
    mask = np.zeros_like(image)
    mask[30:90, 60:66] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    anns = Annotations({"a.jpeg": ["b", "a"]}, {"a": [rle], "b": [rle]}, 2, 0)
    config = detector_dataset(tmp_path, anns, ["a.jpeg"], [], tmp_path / "yolo")
    label = (config.parent / "labels/train/a.txt").read_text().splitlines()
    assert len(label) == 1  # never duplicate annotators as overlapping YOLO boxes
    dataset = Crops(tmp_path, anns, ["a.jpeg"], size=64)
    assert len(dataset) == 2
    x, y = dataset[0]
    assert x.shape == (3, 64, 64) and y.sum() > 0
    loader = DataLoader(dataset, batch_size=2)
    model = UNet(in_channels=3, base=4, depth=2)
    train_model(model, loader, torch.device("cpu"), 2, tmp_path)
    assert len(json.loads((tmp_path / "history.json").read_text())) == 2
    clone = UNet(in_channels=3, base=4, depth=2)
    clone.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))


def test_refiner_output():
    model = crop_unet(pretrained=False).eval()
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)
