import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from test_cnn_baseline import dataset_dir  # noqa: F401

from postprocess.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
    trainable,
)
from postprocess.instances import cut, instances, select
from scoring.metric import instance_crops


def test_hysteresis_keeps_faint_tails_attached_to_confident_cores():
    prob = np.zeros((128, 128), np.uint8)
    prob[40:50, 10:40] = 250  # confident core
    prob[40:50, 40:90] = 80  # faint tail (0.31) touching it
    prob[100:110, 10:60] = 80  # faint blob touching nothing confident
    _, areas, boxes = instances(prob, threshold=0.5, min_area=10)
    assert len(areas) == 1 and areas[0] == 300  # single threshold: core only
    _, areas, boxes = instances(prob, threshold=0.5, min_area=10, low=0.3)
    assert len(areas) == 1 and areas[0] == 800 and tuple(boxes[0]) == (10, 40, 80, 10)  # tail included, blob not
    assert instances(prob, threshold=0.5, min_area=10, low=0.6)[1].tolist() == [300]


def test_select_renumbers_the_kept_instances():
    labels = np.zeros((32, 32), np.int32)
    labels[0:4, 0:8], labels[10:14, 0:4], labels[20:24, 0:12] = 1, 2, 3
    areas, boxes = np.array([32, 16, 48]), np.array([[0, 0, 8, 4], [0, 10, 4, 4], [0, 20, 12, 4]])
    new_labels, new_areas, new_boxes = select(labels, areas, boxes, np.array([True, False, True]))
    assert new_areas.tolist() == [32, 48] and new_boxes.tolist() == [[0, 0, 8, 4], [0, 20, 12, 4]]
    assert new_labels.max() == 2 and (new_labels[20:24, 0:12] == 2).all() and (new_labels[10:14, 0:4] == 0).all()
    assert select(labels, areas, boxes, np.zeros(3, bool))[1].size == 0


def test_component_filter_learns_to_drop_round_faint_blobs(dataset_dir):  # noqa: F811
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
    assert not trainable(np.ones(10, bool)) and not trainable(np.zeros(100, bool))


def test_cut_keeps_the_confident_pixels_and_drops_what_is_left_too_small():
    labels = np.zeros((40, 40), np.int32)
    labels[5:15, 5:35] = 1  # a filament with a confident core in rows 8..11
    labels[25:30, 5:10] = 2  # a faint blob
    prob = np.zeros((40, 40), np.uint8)
    prob[5:15, 5:35] = 100
    prob[8:12, 5:35] = 200
    prob[25:30, 5:10] = 120
    areas = np.array([300, 25])
    boxes = np.array([[5, 5, 30, 10], [5, 25, 5, 5]])
    out, out_areas, out_boxes = cut(labels, areas, boxes, prob, 0.6)
    assert out_areas.tolist() == [120] and out_boxes.tolist() == [[5, 8, 30, 4]]
    assert set(np.unique(out)) == {0, 1}
