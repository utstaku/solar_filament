import json

import cv2
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_util
from synthetic import SIZE, make_frame
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, soft_target
from cnn_baseline.postprocess import instances
from cnn_baseline.training import bce_dice_loss, fit, predict
from dataset.annotations import load_annotations
from models.unet import UNet
from scoring.metric import PanopticQuality, instance_crops


def polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """Two synthetic frames as JPEGs plus a COCO file; the first frame has two annotators."""
    root = tmp_path_factory.mktemp("data")
    images, annotations, masks = [], [], {}
    for seed, file_name in enumerate(["20110120105534Ch.jpeg", "20110130110334Ch.jpeg"]):
        image, frame_masks = make_frame(seed=seed, n_filaments=6)
        cv2.imwrite(str(root / file_name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        masks[file_name] = frame_masks
        for annotator in range(2 if seed == 0 else 1):
            entry = f"05010{annotator + 1}-{file_name[:-5]}"
            images.append({"id": entry, "file_name": file_name, "width": SIZE, "height": SIZE})
            for k, mask in enumerate(frame_masks):
                if annotator == 1 and k == 0:
                    continue  # the second annotator missed one filament -> a soft label
                annotations.append(
                    {"id": f"{entry}-{k}", "image_id": entry, "category_id": 1, "segmentation": polygon(mask)}
                )
    (root / "annotations.json").write_text(json.dumps({"images": images, "annotations": annotations}))
    return root, masks


def test_channels_and_soft_target(dataset_dir):
    root, masks = dataset_dir
    annotations = load_annotations(root / "annotations.json")
    x = channels(cv2.imread(str(root / "20110120105534Ch.jpeg"), cv2.IMREAD_GRAYSCALE))
    assert x.shape == (2, SIZE, SIZE) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1
    y = soft_target(annotations, "20110120105534Ch.jpeg")
    assert y.shape == (SIZE, SIZE) and set(np.unique(y)) <= {0.0, 0.5, 1.0}
    assert (y == 0.5).sum() > 0  # the filament only one annotator marked
    assert y[masks["20110120105534Ch.jpeg"][1] > 0].mean() > 0.9  # polygons round-trip, bar some edge pixels
    assert set(np.unique(soft_target(annotations, "20110130110334Ch.jpeg"))) == {0.0, 1.0}


def test_dataset_tiles_and_frames(dataset_dir):
    root, _ = dataset_dir
    annotations = load_annotations(root / "annotations.json")
    files = ["20110120105534Ch.jpeg", "20110130110334Ch.jpeg"]
    torch.manual_seed(0)
    train = FilamentFrames(files, root, annotations, tile=256, tiles_per_frame=6)
    x, y = collate_tiles([train[0], train[1]])
    assert x.shape == (12, 2, 256, 256) and y.shape == (12, 256, 256)
    assert (y.amax(dim=(1, 2)) > 0).sum() >= 6  # half of the tiles are centred on filaments
    loader = DataLoader(train, batch_size=2, collate_fn=collate_tiles, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape[0] == 12

    x, y = FilamentFrames(files, root, annotations, train=False)[1]
    assert x.shape == (2, SIZE, SIZE) and y.shape == (SIZE, SIZE)


def test_unet_is_fully_convolutional():
    model = UNet(in_channels=2, base=8, depth=3)
    assert model(torch.zeros(2, 2, 64, 64)).shape == (2, 1, 64, 64)
    assert model(torch.zeros(1, 2, 200, 136)).shape == (1, 1, 200, 136)  # not a multiple of 2**depth
    assert sum(p.numel() for p in UNet(in_channels=2, base=32, depth=4).parameters()) > 5e6


def test_loss_prefers_the_right_answer():
    target = torch.zeros(2, 32, 32)
    target[:, 10:20, 5:25] = 1
    right = torch.where(target > 0, 6.0, -6.0).unsqueeze(1)
    assert bce_dice_loss(right, target) < 0.1 < bce_dice_loss(-right, target)
    assert bce_dice_loss(torch.zeros(2, 1, 32, 32), target * 0.5) > 0  # soft targets are accepted


def test_overfitting_one_frame_recovers_its_filaments(dataset_dir):
    """80 steps on tiles of one frame must make its filaments detectable in the full frame (~25 s on CPU)."""
    root, masks = dataset_dir
    annotations = load_annotations(root / "annotations.json")
    torch.manual_seed(0)
    train = FilamentFrames(["20110130110334Ch.jpeg"], root, annotations, tile=96, tiles_per_frame=8)
    loader = DataLoader(train, batch_size=1, collate_fn=collate_tiles, num_workers=0)  # one step per epoch
    model = UNet(in_channels=2, base=8, depth=3)
    history = fit(model, loader, epochs=80, lr=3e-3, device=torch.device("cpu"), log=lambda *a: None)
    assert history[-1] < 0.8 * history[0]

    frame, _ = FilamentFrames(["20110130110334Ch.jpeg"], root, annotations, train=False)[0]
    prob = next(predict(model, [frame], device=torch.device("cpu")))
    assert prob.shape == (SIZE, SIZE) and prob.dtype == np.uint8
    labels, areas, boxes = instances(prob, threshold=0.5, min_area=100)
    assert len(areas) == len(boxes) and labels.max() == len(areas)
    pq = PanopticQuality()
    gt = instance_crops([mask_util.encode(np.asfortranarray(m)) for m in masks["20110130110334Ch.jpeg"]])
    pq.update(gt, labels, areas)
    # 6 filaments in the frame. How many small spurious blobs 80 CPU steps leave behind varies with the
    # torch build and machine, so bound the false positives loosely and require good masks instead.
    assert pq.tp >= 5 and pq.fp <= 4 and pq.summary()["pq"] >= 0.6, pq.summary()


def test_instances_edge_cases():
    empty = np.zeros((64, 64), np.uint8)
    labels, areas, boxes = instances(empty, 0.5, 10)
    assert labels.max() == 0 and len(areas) == 0 and boxes.shape == (0, 4)
    prob = empty.copy()
    prob[10:20, 10:40] = 255  # 300 px
    prob[40:43, 40:43] = 255  # 9 px, dropped by min_area
    labels, areas, boxes = instances(prob, 0.5, 10)
    assert len(areas) == 1 and areas[0] == 300 and tuple(boxes[0]) == (10, 10, 30, 10)
