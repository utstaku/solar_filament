"""A second stage that decides, per predicted component, whether it is a filament.

The minimum-area cut is a crude false-positive filter: at 600 px it removes a fifth of the real
filaments to get rid of hallucinations. A small classifier on the component's shape, contrast,
position and the model's confidence separates them better and lets the area cut drop.

It is trained on the validation frames, the only frames whose predictions are honest, and its
cutoff is chosen on out-of-fold scores so the validation PQ it reports is not fitted to itself.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from error_analysis.instances import FrameContext, features, overlaps
from preprocess.disk import disk_geometry, flatten
from scoring.metric import IOU_THRESHOLD, InstanceCrop

FEATURES = ["log_area", "elongation", "darkness", "limb", "mean_prob", "max_prob"]
MIN_TRAINING_COMPONENTS = 60  # below this (or with one class only) the filter is not fitted


def frame_context(name: str, image: np.ndarray, prob: np.ndarray) -> FrameContext:
    cx, cy, radius = disk_geometry(image)
    return FrameContext(name=name, flat=flatten(image), centre=(cx, cy), radius=radius, prob=prob)


def component_table(ctx: FrameContext, labels: np.ndarray, boxes: np.ndarray) -> pd.DataFrame:
    """One row per predicted component with the FEATURES columns (and frame / component id)."""
    rows = []
    for k, (x, y, w, h) in enumerate(boxes, start=1):
        crop = labels[y : y + h, x : x + w] == k
        rows.append({"frame": ctx.name, "component": k, **features(crop, int(x), int(y), ctx)})
    table = pd.DataFrame(
        rows,
        columns=["frame", "component", "area", "cx", "cy", "elongation", "darkness", "limb", "mean_prob", "max_prob"],
    )
    table["log_area"] = np.log(table["area"].astype(float).clip(lower=1))
    return table


def matched_by_any_annotator(
    gt_crops_per_annotator: list[list[InstanceCrop]], labels: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    """bool per component: IoU > 0.5 with a filament of at least one annotator.

    Such a component raises PQ whenever it is kept (its IoU outweighs the annotators it misses); a
    component matching nobody can only lower it. That makes this the right target for the filter.
    """
    keep = np.zeros(len(areas), dtype=bool)
    for crops in gt_crops_per_annotator:
        if crops and len(areas):
            _, iou = overlaps(crops, labels, areas)
            keep |= (iou > IOU_THRESHOLD).any(axis=0)
    return keep


class ComponentFilter:
    """Gradient boosting on `features`; `cutoff` is the probability below which a component is dropped."""

    def __init__(self, cutoff: float = 0.5, features: list[str] = FEATURES):
        self.cutoff, self.features = cutoff, list(features)
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)

    def fit(self, table: pd.DataFrame, target: np.ndarray) -> "ComponentFilter":
        self.model.fit(table[self.features].to_numpy(), np.asarray(target, dtype=int))
        return self

    def scores(self, table: pd.DataFrame) -> np.ndarray:
        if not len(table):
            return np.zeros(0)
        return self.model.predict_proba(table[self.features].to_numpy())[:, 1]

    def keep(self, table: pd.DataFrame) -> np.ndarray:
        return self.scores(table) >= self.cutoff


def out_of_fold_scores(
    table: pd.DataFrame, target: np.ndarray, n_splits: int = 5, features: list[str] = FEATURES
) -> np.ndarray:
    """Filter probabilities for every component, each predicted by a model that never saw its frame."""
    scores = np.zeros(len(table))
    groups = table["frame"].to_numpy()
    splits = min(n_splits, len(np.unique(groups)))
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(table, target, groups):
        model = ComponentFilter(features=features).fit(table.iloc[train_idx], target[train_idx])
        scores[test_idx] = model.scores(table.iloc[test_idx])
    return scores


def trainable(target: np.ndarray) -> bool:
    return len(target) >= MIN_TRAINING_COMPONENTS and 0 < target.sum() < len(target)
