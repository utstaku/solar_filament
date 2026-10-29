import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from pycocotools import mask as mask_util
from synthetic import SIZE, make_frame
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, soft_target
from cnn_baseline.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
    trainable,
)
from cnn_baseline.postprocess import instances, on_disk, select
from cnn_baseline.training import bce_dice_loss, fit, predict
from dataset.annotations import load_annotations
from models.unet import UNet
from preprocess.images import read_gray
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
    # the sky beyond the limb is barely in the training tiles; left in, it yields ~0.5 blobs (3 here)
    prob = on_disk(prob, read_gray(root / "20110130110334Ch.jpeg"))
    labels, areas, boxes = instances(prob, threshold=0.5, min_area=100)
    assert len(areas) == len(boxes) and labels.max() == len(areas)
    pq = PanopticQuality()
    gt = instance_crops([mask_util.encode(np.asfortranarray(m)) for m in masks["20110130110334Ch.jpeg"]])
    pq.update(gt, labels, areas)
    assert pq.tp >= 5 and pq.fp <= 1, pq.summary()  # 6 filaments in the frame


def test_on_disk_drops_only_what_is_off_the_disk():
    image, _ = make_frame(seed=0, n_filaments=0)  # disk centred at (1030, 1015), radius 905
    prob = np.zeros((SIZE, SIZE), np.uint8)
    prob[1000:1020, 1000:1100] = 200  # near the centre
    prob[1000:1020, 1900:1925] = 200  # just inside the limb (x = 1900..1924 < 1030 + 905)
    prob[20:60, 20:60] = 128  # a corner of the sky
    kept = on_disk(prob, image)
    assert kept.dtype == np.uint8 and kept[20:60, 20:60].max() == 0
    assert (kept[1000:1020, 1000:1100] == 200).all() and (kept[1000:1020, 1900:1925] == 200).all()
    _, areas, _ = instances(kept, threshold=0.5, min_area=10)
    assert sorted(areas.tolist()) == [500, 2000]


def test_instances_edge_cases():
    empty = np.zeros((64, 64), np.uint8)
    labels, areas, boxes = instances(empty, 0.5, 10)
    assert labels.max() == 0 and len(areas) == 0 and boxes.shape == (0, 4)
    prob = empty.copy()
    prob[10:20, 10:40] = 255  # 300 px
    prob[40:43, 40:43] = 255  # 9 px, dropped by min_area
    labels, areas, boxes = instances(prob, 0.5, 10)
    assert len(areas) == 1 and areas[0] == 300 and tuple(boxes[0]) == (10, 10, 30, 10)


def test_hysteresis_keeps_faint_tails_attached_to_confident_cores():
    prob = np.zeros((128, 128), np.uint8)
    prob[40:50, 10:40] = 250  # confident core
    prob[40:50, 40:90] = 80  # faint tail (0.31) touching it
    prob[100:110, 10:60] = 80  # faint blob touching nothing confident
    _, areas, boxes = instances(prob, threshold=0.5, min_area=10)
    assert len(areas) == 1 and areas[0] == 300  # single threshold: core only
    _, areas, boxes = instances(prob, threshold=0.5, min_area=10, low=0.3)
    assert len(areas) == 1 and areas[0] == 800 and tuple(boxes[0]) == (10, 40, 80, 10)  # tail included, blob not
    assert instances(prob, threshold=0.5, min_area=10, low=0.6)[1].tolist() == [300]  # low >= threshold: ignored


def test_select_renumbers_the_kept_instances():
    labels = np.zeros((32, 32), np.int32)
    labels[0:4, 0:8], labels[10:14, 0:4], labels[20:24, 0:12] = 1, 2, 3
    areas, boxes = np.array([32, 16, 48]), np.array([[0, 0, 8, 4], [0, 10, 4, 4], [0, 20, 12, 4]])
    new_labels, new_areas, new_boxes = select(labels, areas, boxes, np.array([True, False, True]))
    assert new_areas.tolist() == [32, 48] and new_boxes.tolist() == [[0, 0, 8, 4], [0, 20, 12, 4]]
    assert new_labels.max() == 2 and (new_labels[20:24, 0:12] == 2).all() and (new_labels[10:14, 0:4] == 0).all()
    assert select(labels, areas, boxes, np.zeros(3, bool))[1].size == 0


def test_component_filter_learns_to_drop_round_faint_blobs(dataset_dir):
    """Long dark components match filaments; small round ones do not. The filter must tell them apart."""
    root, all_masks = dataset_dir
    masks = all_masks["20110130110334Ch.jpeg"]
    image = cv2.imread(str(root / "20110130110334Ch.jpeg"), cv2.IMREAD_GRAYSCALE)
    rng = np.random.default_rng(0)
    tables, targets = [], []
    for frame_idx in range(6):  # six "frames" on the same image with different fake predictions
        prob = np.zeros(image.shape, np.uint8)
        for m in masks:
            prob[m > 0] = 230
        for _ in range(6):  # round blobs the network hallucinated
            cv2.circle(
                prob, (int(rng.integers(600, 1400)), int(rng.integers(600, 1400))), int(rng.integers(8, 14)), 200, -1
            )
        labels, areas, boxes = instances(prob, threshold=0.5, min_area=50)
        ctx = frame_context(f"2011012010553{frame_idx}Ch.jpeg", image, prob)
        table = component_table(ctx, labels, boxes)
        target = matched_by_any_annotator(
            [instance_crops([mask_util.encode(np.asfortranarray(m)) for m in masks])], labels, areas
        )
        assert 0 < target.sum() < len(target)
        tables.append(table)
        targets.append(target)
    table, target = pd.concat(tables, ignore_index=True), np.concatenate(targets)
    assert trainable(target)
    oof = out_of_fold_scores(table, target, n_splits=3)
    assert oof[target].mean() > 0.7 > oof[~target].mean()
    model = ComponentFilter(cutoff=0.5).fit(table, target)
    assert (model.keep(table) == target).mean() > 0.9
    subset = ["frame", "log_area", "elongation"]  # a filter on fewer features needs only those columns
    assert ComponentFilter(features=subset[1:]).fit(table[subset], target).scores(table[subset]).shape == oof.shape
    assert out_of_fold_scores(table[subset], target, n_splits=3, features=subset[1:]).shape == oof.shape
    assert not trainable(np.ones(10, bool)) and not trainable(np.zeros(100, bool))


def test_pick_device_refuses_the_cpu_on_kaggle(monkeypatch):
    """A GPU notebook that Kaggle put on a CPU machine stops at once instead of training for days."""
    from cnn_baseline.training import pick_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    assert pick_device().type in ("mps", "cpu")  # locally: fall back as before
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    with pytest.raises(RuntimeError, match="no GPU"):
        pick_device()
