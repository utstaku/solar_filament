import json

import matplotlib
import numpy as np
import pytest
from pycocotools import mask as mask_util
from synthetic import SIZE, make_frame

from dataset.annotations import load_annotations
from error_analysis import gallery, report
from error_analysis.annotators import human_ceiling, label_map
from error_analysis.instances import FrameContext, agreement, analyze_frame, classify, overlaps, tables
from preprocess.disk import REL_SCALE, disk_geometry, flatten
from scoring.metric import PanopticQuality, instance_crops

matplotlib.use("Agg")


def rect(y0, y1, x0, x1):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


@pytest.fixture(scope="module")
def scene():
    """Six filaments and a label map exercising every status; flat is darker inside the filaments."""
    gt = [
        rect(100, 140, 100, 300),  # 0: matched exactly
        rect(200, 240, 100, 300),  # 1: near miss, prediction covers 40 % of it
        rect(300, 340, 100, 400),  # 2: fragmented into three thirds
        rect(400, 440, 100, 300),  # 3: merged with 4 by one prediction
        rect(400, 440, 320, 520),  # 4
        rect(600, 640, 100, 300),  # 5: missed
    ]
    preds = [
        rect(100, 140, 100, 300),  # matched
        rect(200, 240, 100, 180),  # near miss (IoU 0.4)
        rect(300, 340, 100, 200),  # fragments
        rect(300, 340, 200, 300),
        rect(300, 340, 300, 400),
        rect(400, 440, 100, 520),  # merge (covers 3 and 4 and the gap)
        rect(800, 840, 100, 300),  # hallucination
    ]
    labels = np.zeros((SIZE, SIZE), np.int32)
    for k, mask in enumerate(preds, start=1):
        labels[mask > 0] = k
    areas = np.array([int(m.sum()) for m in preds])
    boxes = np.array(
        [
            [100, 100, 200, 40],
            [100, 200, 80, 40],
            [100, 300, 100, 40],
            [200, 300, 100, 40],
            [300, 300, 100, 40],
            [100, 400, 420, 40],
            [100, 800, 200, 40],
        ]
    )
    flat = np.full((SIZE, SIZE), REL_SCALE, np.uint8)
    for mask in gt:
        flat[mask > 0] = int(REL_SCALE * 0.85)
    prob = np.zeros((SIZE, SIZE), np.uint8)
    prob[labels > 0] = 230
    prob[labels == 7] = 140  # the hallucination is the least confident prediction
    rles = [mask_util.encode(np.asfortranarray(m)) for m in gt]
    ctx = FrameContext(name="20140609195854Bh.jpeg", flat=flat, centre=(1024.0, 1024.0), radius=900.0, prob=prob)
    return gt, rles, labels, areas, boxes, ctx


def test_classify_every_status(scene):
    _, rles, labels, areas, _, _ = scene
    _, iou = overlaps(instance_crops(rles), labels, areas)
    gt_status, pred_status = classify(iou)
    assert gt_status == ["matched", "near_miss", "fragmented", "merged", "merged", "missed"]
    assert pred_status == ["matched", "near_miss", "fragment", "fragment", "fragment", "merge", "hallucination"]
    assert iou[1, 1] == pytest.approx(0.4)


def test_tables_reproduce_the_metric(scene):
    _, rles, labels, areas, boxes, ctx = scene
    second = rles[:-1]  # a second annotator who did not draw filament 5
    gt_rows, pred_rows = analyze_frame(ctx, [("050101-x", rles), ("050102-x", second)], labels, areas, boxes)
    gt, pred = tables(gt_rows, pred_rows)
    assert len(gt) == 11 and len(pred) == 14  # 6 + 5 filaments; 7 predictions x 2 annotators

    expected = PanopticQuality()
    for r in (rles, second):
        expected.update(instance_crops(r), labels, areas)
    terms = report.pq_terms(gt, pred)
    assert (terms["tp"], terms["fp"], terms["fn"]) == (expected.tp, expected.fp, expected.fn)
    assert terms["pq"] == expected.summary()["pq"]

    first = gt[gt["annotator"] == "050101-x"].set_index("gt_id")
    assert first.loc[5, "agreement"] == 0 and first.loc[0, "agreement"] == 1  # only one annotator drew 5
    assert first.loc[0, "darkness"] == pytest.approx(0.15, abs=0.01)
    assert first.loc[2, "coverage"] == pytest.approx(1.0) and first.loc[2, "n_overlap"] == 3
    assert 0 < first.loc[0, "limb"] < 1.5 and first.loc[0, "elongation"] > 3
    hallucination = pred[(pred["status"] == "hallucination") & (pred["annotator"] == "050101-x")].iloc[0]
    assert hallucination["mean_prob"] == pytest.approx(140 / 255) and hallucination["precision"] == 0


def test_report_what_if_and_separability(scene):
    _, rles, labels, areas, boxes, ctx = scene
    gt, pred = tables(*analyze_frame(ctx, [("050101-x", rles)], labels, areas, boxes))
    table = report.what_if(gt, pred).set_index("scenario")
    t = report.pq_terms(gt, pred)
    assert table.loc["current", "pq"] == t["pq"]
    assert table.loc["no false positives", "fp"] == 0 and table.loc["no false positives", "pq"] > t["pq"]
    assert table.loc["near misses fixed", "tp"] == t["tp"] + 1 and table.loc["near misses fixed", "fn"] == t["fn"] - 1
    assert table.loc["perfect masks (SQ = 1)", "sq"] == 1.0
    sep = report.separability(pred, "mean_prob")
    assert sep["auc"] == 1.0  # the hallucination is strictly less confident than the match
    summary = report.summary(gt, pred)
    json.dumps(summary)  # must be serialisable
    assert summary["counts"]["ground_truth"]["missed"] == 1
    assert summary["matched_area_ratio_median"] == pytest.approx(1.0)
    assert set(report.breakdowns(gt, pred)) >= {"gt_by_area", "gt_by_agreement", "pred_by_mean_prob"}


def test_galleries_are_written(scene, tmp_path):
    _, rles, labels, areas, boxes, ctx = scene
    gt, pred = tables(*analyze_frame(ctx, [("050101-x", rles)], labels, areas, boxes))
    cases = gallery.worst_cases(gt, pred, k=4)
    assert set(cases) == set(gallery.CATEGORIES) and len(cases["hallucination"]) == 1

    image = np.full((SIZE, SIZE), 120, np.uint8)
    paths = gallery.save_galleries(cases, lambda frame, annotator: (image, instance_crops(rles), labels), tmp_path)
    assert {p.name for p in paths} == {
        "gallery_hallucination.png",
        "gallery_missed.png",
        "gallery_near_miss.png",
        "gallery_fragmented.png",
        "gallery_merge.png",
    }
    assert all(p.stat().st_size > 1000 for p in paths)


def test_human_ceiling(tmp_path):
    """Identical annotators agree perfectly; one who skipped a filament scores below 1 both ways."""
    _, masks = make_frame(seed=3, n_filaments=4)
    rles = [mask_util.encode(np.asfortranarray(m)) for m in masks]
    labels, areas = label_map(rles, (SIZE, SIZE))
    assert labels.max() == 4 and len(areas) == 4

    def polygon(mask):
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]

    coco = {"images": [], "annotations": []}
    for file_name, sets in {"a.jpeg": [masks, masks], "b.jpeg": [masks, masks[:3]], "c.jpeg": [masks]}.items():
        for k, subset in enumerate(sets):
            entry = f"05010{k + 1}-{file_name[:-5]}"
            coco["images"].append({"id": entry, "file_name": file_name})
            coco["annotations"] += [{"image_id": entry, "segmentation": polygon(m)} for m in subset]
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    annotations = load_annotations(tmp_path / "ann.json")

    result = human_ceiling(annotations, ["a.jpeg", "b.jpeg", "c.jpeg"], (SIZE, SIZE))
    assert result["n_frames"] == 2 and result["n_pairs"] == 4  # c.jpeg has one annotator
    pairs = result["pairs"].set_index("frame")
    assert (pairs.loc["a.jpeg", "pq"] > 0.95).all()
    assert (pairs.loc["b.jpeg", "pq"] < 0.95).all() and (pairs.loc["b.jpeg", ["fp", "fn"]].sum(axis=1) == 1).all()
    assert 0.9 < result["overall"]["pq"] < 1.0


def test_agreement_and_disk_geometry():
    _, masks = make_frame(seed=0, n_filaments=3)
    rles = [mask_util.encode(np.asfortranarray(m)) for m in masks]
    counts = agreement([rles, rles[1:], []])
    assert counts[0].tolist() == [0, 1, 1] and counts[1].tolist() == [1, 1] and len(counts[2]) == 0
    image, _ = make_frame(seed=0)
    cx, cy, radius = disk_geometry(image)
    assert abs(cx - 1030) < 3 and abs(cy - 1015) < 3 and abs(radius - 905) < 30
    assert flatten(image).shape == image.shape
