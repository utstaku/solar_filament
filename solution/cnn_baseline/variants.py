"""Post-processing variants compared against A0 (`predict.py`) on the saved probability maps.

A variant turns a frame's probability map into instances (labels 1..K, areas, boxes). Three kinds:

- **keep rule**: which components survive. A0 keeps those whose filter probability of "matched by any
  annotator" passes a cutoff tuned on validation. `expected_gain` gives the target that PQ actually
  rewards, so the cutoff follows from the metric instead of a sweep.
- **grow**: kept components extend into neighbouring pixels where the network is still fairly confident
  (near misses: the core is found, the faint ends are not).
- **join**: kept components close to each other become one instance (fragments: annotators draw a filament
  across the gaps where its material thins out).
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from cnn_baseline.filtering import FEATURES
from error_analysis.instances import overlaps
from scoring.metric import IOU_THRESHOLD, InstanceCrop, PanopticQuality


def label_stats(labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(areas[n], boxes[n, 4] as x, y, w, h) of the instances 1..n of a label map (empty ones: area 0)."""
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs] - 1
    areas = np.bincount(ids, minlength=n)[:n]
    x0, y0 = np.full(n, labels.shape[1]), np.full(n, labels.shape[0])
    x1, y1 = np.full(n, -1), np.full(n, -1)
    np.minimum.at(x0, ids, xs)
    np.minimum.at(y0, ids, ys)
    np.maximum.at(x1, ids, xs)
    np.maximum.at(y1, ids, ys)
    boxes = np.stack([x0, y0, x1 - x0 + 1, y1 - y0 + 1], axis=1)
    boxes[areas == 0] = 0
    return areas, boxes


def renumber(labels: np.ndarray, mapping: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply `mapping` (old label -> group, 0 stays 0), then number the groups 1..K largest first."""
    grouped = mapping[labels]
    groups = np.unique(mapping[1:])
    groups = groups[groups > 0]
    compact = np.zeros(int(mapping.max()) + 1, dtype=np.int32)
    compact[groups] = np.arange(1, len(groups) + 1)
    grouped = compact[grouped]
    areas, _ = label_stats(grouped, len(groups))
    order = np.argsort(areas, kind="stable")[::-1]
    by_size = np.zeros(len(groups) + 1, dtype=np.int32)
    by_size[order + 1] = np.arange(1, len(groups) + 1)
    out = by_size[grouped]
    areas, boxes = label_stats(out, len(groups))
    return out, areas, boxes


def grow(labels: np.ndarray, prob: np.ndarray, low: float, steps: int):
    """Extend every instance, one pixel ring per step, into unlabelled pixels with prob >= low.

    Instances never merge: a pixel reached by two instances in the same step goes to the higher label.
    """
    allowed = prob >= round(low * 255)
    grown = labels.astype(np.uint16)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        reach = cv2.dilate(grown, kernel)
        new = (grown == 0) & allowed & (reach > 0)
        if not new.any():
            break
        grown[new] = reach[new]
    n = int(labels.max())
    areas, boxes = label_stats(grown.astype(np.int32), n)
    return grown.astype(np.int32), areas, boxes


def join(labels: np.ndarray, radius: int):
    """Merge instances whose pixels come within `radius` px of each other (transitively); no pixels are added."""
    n = int(labels.max())
    parent = np.arange(n + 1)

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    _, boxes = label_stats(labels, n)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for k, (x, y, w, h) in enumerate(boxes, start=1):
        x0, y0 = max(x - radius, 0), max(y - radius, 0)
        x1, y1 = min(x + w + radius, labels.shape[1]), min(y + h + radius, labels.shape[0])
        window = labels[y0:y1, x0:x1]
        near = cv2.dilate((window == k).astype(np.uint8), kernel) > 0
        for other in np.unique(window[near]):
            if other and other != k:
                parent[root(other)] = root(k)
    mapping = np.array([root(k) for k in range(n + 1)])
    return renumber(labels, mapping)


def expected_gain(gt_crops_per_annotator: list[list[InstanceCrop]], labels: np.ndarray, areas: np.ndarray):
    """Per component: mean over the frame's annotators of IoU if it matches their filament (IoU > 0.5), else 0.

    PQ = sum(IoU) / (TP + FP/2 + FN/2). For each annotator a kept component either matches (IoU joins the
    numerator, an FN becomes a TP) or not (one more FP); either way the denominator grows by 1/2. Keeping
    it therefore raises PQ exactly when its mean gain per annotator exceeds PQ / 2.
    """
    gain = np.zeros(len(areas))
    if not len(areas) or not gt_crops_per_annotator:
        return gain
    for crops in gt_crops_per_annotator:
        if crops:
            _, iou = overlaps(crops, labels, areas)
            best = iou.max(axis=0)
            gain += np.where(best > IOU_THRESHOLD, best, 0.0)
    return gain / len(gt_crops_per_annotator)


class GainRegressor:
    """Gradient boosting that predicts `expected_gain` from the component features."""

    def __init__(self, features: list[str] = FEATURES):
        self.features = list(features)
        self.model = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)

    def fit(self, table: pd.DataFrame, gain: np.ndarray) -> "GainRegressor":
        self.model.fit(table[self.features].to_numpy(), gain)
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict(table[self.features].to_numpy()) if len(table) else np.zeros(0)


def out_of_fold_gain(table: pd.DataFrame, gain: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Predicted gain for every component from a model that never saw its frame (same folds as the filter)."""
    pred = np.zeros(len(table))
    groups = table["frame"].to_numpy()
    splits = min(n_splits, len(np.unique(groups)))
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(table, gain, groups):
        pred[test_idx] = GainRegressor().fit(table.iloc[train_idx], gain[train_idx]).predict(table.iloc[test_idx])
    return pred


def score_frames(predictions: dict, truth: dict, files: list[str]) -> pd.DataFrame:
    """Per-frame PQ terms (as `importance.frame_terms`) of ready-made predictions {frame: (labels, areas, boxes)}."""
    rows = []
    for f in files:
        labels, areas, _ = predictions[f]
        pq = PanopticQuality()
        for gt_crops in truth[f]:
            pq.update(gt_crops, labels, areas)
        rows.append({"frame": f, "tp": pq.tp, "fp": pq.fp, "fn": pq.fn, "iou_sum": pq.iou_sum})
    return pd.DataFrame(rows, columns=["frame", "tp", "fp", "fn", "iou_sum"])
