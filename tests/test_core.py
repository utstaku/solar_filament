import json

import cv2
import numpy as np
import pytest
from pycocotools import mask as mask_util
from synthetic import SIZE, make_frame

from solar_filament import rle, submission
from solar_filament.annotations import load_annotations
from solar_filament.classical import segment
from solar_filament.metric import PanopticQuality, instance_crops
from solar_filament.preprocess import OFF_DISK, REL_SCALE, disk_mask, flatten


@pytest.fixture(scope="module")
def frame():
    image, masks = make_frame()
    return image, masks, flatten(image)


def shapes():
    """Masks that stress the RLE format: speckle, empty, full, blobs and thin lines."""
    rng = np.random.default_rng(1)
    for i in range(40):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        kind = i % 5
        if kind == 0:
            mask[:] = rng.random(mask.shape) < 0.002
        elif kind == 2:
            mask[:] = 1
        elif kind == 3:
            centre = (int(rng.integers(200, 1800)), int(rng.integers(200, 1800)))
            cv2.ellipse(mask, centre, (int(rng.integers(5, 400)), int(rng.integers(2, 30))), 30.0, 0, 360, 1, -1)
        elif kind == 4:
            cv2.polylines(mask, [rng.integers(100, 1900, (6, 2)).astype(np.int32)], False, 1, int(rng.integers(1, 12)))
        yield mask


def test_rle_roundtrip_and_validation():
    for mask in shapes():
        counts = rle.encode(mask)
        assert (rle.decode(counts) == mask).all()
        assert rle.is_valid(counts)
        assert not rle.is_valid(counts + "-")  # character outside the alphabet
    assert not rle.is_valid("not-an-rle")
    assert not rle.is_valid(rle.encode(next(shapes()))[:-3])  # truncated: runs no longer cover the image


def test_disk_and_flatten(frame):
    image, _, flat = frame
    ys, xs = np.nonzero(disk_mask(image))
    assert abs(xs.mean() - 1030) < 3 and abs(ys.mean() - 1015) < 3
    assert abs(np.sqrt(len(xs) / np.pi) - 905) < 0.03 * 905
    assert abs(int(np.median(flat[flat < OFF_DISK])) - REL_SCALE) <= 4  # limb darkening is gone
    assert (flat[:40, :40] == OFF_DISK).all()


def test_pq_matches_reference_implementation(frame):
    """Our windowed PQ must agree with the hosts' definition computed through pycocotools IoU."""
    _, masks, flat = frame
    gt_rles = [mask_util.encode(np.asfortranarray(m)) for m in masks]
    crops = instance_crops(gt_rles)
    for ratio, close_px in [(0.90, 7), (0.93, 0), (0.93, 15)]:
        labels, areas, _ = segment(flat, ratio, 300, close_px)
        pq = PanopticQuality()
        pq.update(crops, labels, areas)

        pred_rles = [
            mask_util.encode(np.asfortranarray((labels == k).astype(np.uint8))) for k in range(1, len(areas) + 1)
        ]
        iou = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))  # (n_pred, n_gt)
        hit = iou > 0.5
        expected = (int(hit.sum()), int((~hit.any(axis=1)).sum()), int((~hit.any(axis=0)).sum()))
        assert (pq.tp, pq.fp, pq.fn) == expected
        assert pq.iou_sum == pytest.approx(float(iou[hit].sum()))
    assert pq.tp > 0  # the detector does find filaments at 12 % contrast


def test_pq_edge_cases(frame):
    _, masks, _ = frame
    labels = np.zeros((SIZE, SIZE), np.int32)
    for k, mask in enumerate(masks[:3], start=1):
        labels[(mask > 0) & (labels == 0)] = k
    areas = np.bincount(labels.ravel(), minlength=4)[1:]
    own_masks = [mask_util.encode(np.asfortranarray((labels == k).astype(np.uint8))) for k in (1, 2, 3)]

    perfect = PanopticQuality()
    perfect.update(instance_crops(own_masks), labels, areas)
    assert perfect.summary() == {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 3, "fp": 0, "fn": 0}

    no_prediction = PanopticQuality()
    no_prediction.update(instance_crops(own_masks), np.zeros_like(labels), np.zeros(0, int))
    assert (no_prediction.tp, no_prediction.fp, no_prediction.fn, no_prediction.value) == (0, 0, 3, 0.0)

    no_ground_truth = PanopticQuality()
    no_ground_truth.update([], labels, areas)
    assert (no_ground_truth.fp, no_ground_truth.fn) == (3, 0)


def test_submission_rows(frame, tmp_path):
    _, _, flat = frame
    labels, _, boxes = segment(flat, 0.93, 300, 15)
    rows = submission.rows_from_label_map("20150125172714Mh", labels, boxes)
    written = submission.write(rows, tmp_path / "submission.csv")
    assert list(written.columns) == submission.COLUMNS and len(written) == len(boxes) > 0
    assert written["filament_id"].tolist()[:2] == ["20150125172714Mh_1", "20150125172714Mh_2"]
    for k, counts in enumerate(written["segmentation_rle"], start=1):
        assert rle.is_valid(counts)
        assert (rle.decode(counts).astype(bool) == (labels == k)).all()


def test_annotations_group_annotators_by_file(tmp_path):
    square = [100.0, 100.0, 300.0, 100.0, 300.0, 140.0, 100.0, 140.0]
    coco = {
        "images": [
            {"id": "050101-20111116063134Lh", "file_name": "20111116063134Lh.jpeg"},
            {"id": "050102-20111116063134Lh", "file_name": "20111116063134Lh.jpeg"},
        ],
        "annotations": [
            {"image_id": "050101-20111116063134Lh", "segmentation": [square]},
            {"image_id": "050102-20111116063134Lh", "segmentation": [square]},
            {"image_id": "050102-20111116063134Lh", "segmentation": [[1.0, 2.0]]},  # degenerate, skipped
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco))
    annotations = load_annotations(path)
    per_annotator = annotations.rles_for_file("20111116063134Lh.jpeg")
    assert [len(rles) for rles in per_annotator] == [1, 1]
    assert int(mask_util.area(per_annotator[0][0])) == 200 * 40
    assert annotations.n_multi_polygon == 0
