"""Per-instance tables: every ground-truth filament and every prediction of a frame, classified.

Ground truth and predictions are linked wherever they share a pixel. The shape of those links, plus
the IoU > 0.5 rule of the metric, gives each instance one status:

    ground truth                      prediction
    matched     IoU > 0.5             matched        IoU > 0.5 with some filament
    near_miss   one overlap, IoU <= 0.5, and that prediction touches nothing else
    fragmented  covered by >= 2 preds  fragment       one of several covering a filament
    merged      swallowed by a pred    merge          covers >= 2 filaments
                that covers others
    missed      no overlap             hallucination  no overlap

Scoring is per annotator-image, so one prediction yields one row per annotator of its frame.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from preprocess.disk import REL_SCALE
from scoring.metric import IOU_THRESHOLD, InstanceCrop, instance_crops


@dataclass
class FrameContext:
    """What the features of a frame's instances are computed from."""

    name: str  # file name, e.g. 20140609195854Bh.jpeg
    flat: np.ndarray  # uint8 limb-darkening-corrected frame (preprocess.disk.flatten)
    centre: tuple[float, float]  # disk centre in pixels
    radius: float  # disk radius in pixels
    prob: np.ndarray | None = None  # uint8 probability * 255 predicted by the model, if any

    @property
    def site(self) -> str:
        return self.name[14:16]

    @property
    def year(self) -> int:
        return int(self.name[:4])


def features(crop: np.ndarray, x: int, y: int, ctx: FrameContext) -> dict:
    """Shape and photometric features of one instance given its bounding-box crop (bool) at (x, y)."""
    ys, xs = np.nonzero(crop)
    h, w = crop.shape
    cx, cy = float(xs.mean()) + x, float(ys.mean()) + y
    if len(xs) > 2:
        eigenvalues = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(float)))
        elongation = float(np.sqrt(max(eigenvalues[1], 1e-9) / max(eigenvalues[0], 1e-9)))
    else:
        elongation = 1.0
    flat = ctx.flat[y : y + h, x : x + w][crop]
    out = {
        "area": len(xs),
        "cx": cx,
        "cy": cy,
        "elongation": elongation,  # ratio of the principal axes; 1 = round, 10 = thin and long
        "darkness": float(1 - flat.mean() / REL_SCALE),  # 0.12 = 12 % darker than the local background
        "limb": float(np.hypot(cx - ctx.centre[0], cy - ctx.centre[1]) / ctx.radius) if ctx.radius else np.nan,
    }
    if ctx.prob is not None:
        prob = ctx.prob[y : y + h, x : x + w][crop]
        out.update(mean_prob=float(prob.mean() / 255), max_prob=float(prob.max() / 255))
    return out


def overlaps(gt_crops: list[InstanceCrop], labels: np.ndarray, pred_areas: np.ndarray):
    """(intersection, iou) matrices of shape (n_gt, n_pred), from the label map of the predictions."""
    n_pred = len(pred_areas)
    intersection = np.zeros((len(gt_crops), n_pred), dtype=np.int64)
    for i, (x, y, crop) in enumerate(gt_crops):
        h, w = crop.shape
        intersection[i] = np.bincount(labels[y : y + h, x : x + w][crop], minlength=n_pred + 1)[1:]
    gt_areas = np.array([int(crop.sum()) for _, _, crop in gt_crops], dtype=np.int64)
    union = gt_areas[:, None] + np.asarray(pred_areas, dtype=np.int64)[None, :] - intersection
    iou = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)
    return intersection, iou


def classify(iou: np.ndarray) -> tuple[list[str], list[str]]:
    """Status of each ground-truth row and each prediction column of an IoU matrix."""
    touching = iou > 0
    gt_links, pred_links = touching.sum(axis=1), touching.sum(axis=0)
    hit = iou > IOU_THRESHOLD  # at most one per row and per column: two IoUs > 0.5 cannot share a side
    gt_status = []
    for i in range(iou.shape[0]):
        if gt_links[i] == 0:
            gt_status.append("missed")
        elif hit[i].any():
            gt_status.append("matched")
        elif gt_links[i] >= 2:
            gt_status.append("fragmented")
        else:
            gt_status.append("merged" if pred_links[int(iou[i].argmax())] >= 2 else "near_miss")
    pred_status = []
    for j in range(iou.shape[1]):
        if pred_links[j] == 0:
            pred_status.append("hallucination")
        elif hit[:, j].any():
            pred_status.append("matched")
        elif pred_links[j] >= 2:
            pred_status.append("merge")
        else:
            pred_status.append("fragment" if gt_links[int(iou[:, j].argmax())] >= 2 else "near_miss")
    return gt_status, pred_status


def agreement(rles_by_annotator: list[list[dict]]) -> list[np.ndarray]:
    """For each annotator's filaments: how many OTHER annotators drew a matching one (IoU > 0.5)."""
    counts = []
    for a, rles_a in enumerate(rles_by_annotator):
        count = np.zeros(len(rles_a), dtype=int)
        for b, rles_b in enumerate(rles_by_annotator):
            if a != b and rles_a and rles_b:
                iou = mask_util.iou(rles_a, rles_b, [0] * len(rles_b))  # (n_a, n_b)
                count += iou.max(axis=1) > IOU_THRESHOLD
        counts.append(count)
    return counts


def analyze_frame(
    ctx: FrameContext,
    annotators: list[tuple[str, list[dict]]],
    labels: np.ndarray,
    pred_areas: np.ndarray,
    pred_boxes: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Rows for every (annotator, filament) and every (annotator, prediction) of one frame.

    `annotators` pairs each annotator-image id with its filament RLEs; `labels`, `pred_areas` and
    `pred_boxes` are the predictions as `postprocess.instances` returns them.
    """
    n_annotators = len(annotators)
    agreements = agreement([rles for _, rles in annotators])
    pred_features = []
    for k, (x, y, w, h) in enumerate(pred_boxes, start=1):
        crop = labels[y : y + h, x : x + w] == k
        pred_features.append({"pred_id": k, **features(crop, int(x), int(y), ctx)})
    frame_info = {"frame": ctx.name, "site": ctx.site, "year": ctx.year, "n_annotators": n_annotators}

    gt_rows, pred_rows = [], []
    for a, (annotator, rles) in enumerate(annotators):
        crops = instance_crops(rles)
        intersection, iou = overlaps(crops, labels, pred_areas)
        gt_status, pred_status = classify(iou)
        for i, (x, y, crop) in enumerate(crops):
            best = int(iou[i].argmax()) if iou.shape[1] else -1
            gt_rows.append(
                {
                    **frame_info,
                    "annotator": annotator,
                    "gt_id": i,
                    **features(crop, x, y, ctx),
                    "agreement": int(agreements[a][i]),  # other annotators who drew this filament too
                    "status": gt_status[i],
                    "n_overlap": int((iou[i] > 0).sum()),
                    "best_iou": float(iou[i, best]) if best >= 0 else 0.0,
                    "best_pred": best + 1 if best >= 0 and iou[i, best] > 0 else 0,
                    "coverage": float(intersection[i].sum() / max(crop.sum(), 1)),  # of its pixels under any pred
                }
            )
        for j, feats in enumerate(pred_features):
            best = int(iou[:, j].argmax()) if iou.shape[0] else -1
            pred_rows.append(
                {
                    **frame_info,
                    "annotator": annotator,
                    **feats,
                    "status": pred_status[j],
                    "n_overlap": int((iou[:, j] > 0).sum()),
                    "best_iou": float(iou[best, j]) if best >= 0 else 0.0,
                    "best_gt": best if best >= 0 and iou[best, j] > 0 else -1,
                    "precision": float(intersection[:, j].sum() / max(feats["area"], 1)),  # of its pixels on any GT
                }
            )
    return gt_rows, pred_rows


def tables(gt_rows: list[dict], pred_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(gt_rows), pd.DataFrame(pred_rows)
