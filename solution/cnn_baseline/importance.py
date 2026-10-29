"""Which features the component filter relies on, and whether removing one changes PQ beyond noise.

PQ is a ratio of sums over the whole validation set, sum(IoU) / (TP + FP/2 + FN/2), so it is kept here
as per-frame terms (tp, fp, fn, iou_sum). Any subset or resample of frames is then scored by summing
its terms, and two post-processing variants can be compared frame by frame.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cnn_baseline.filtering import FEATURES, ComponentFilter, out_of_fold_scores
from cnn_baseline.postprocess import select
from scoring.metric import PanopticQuality

CUTOFFS = np.round(np.arange(0.2, 0.81, 0.05), 2)  # the grid of predict.py
TERMS = ["tp", "fp", "fn", "iou_sum"]


def frame_terms(raw: dict, table: pd.DataFrame, keep: np.ndarray, truth: dict, files: list[str]) -> pd.DataFrame:
    """One row per frame (in `files` order): PQ terms summed over its annotators, keeping components where `keep`.

    `raw[f]` is (labels, areas, boxes) from `instances`; `table` has one row per component in that order,
    frame after frame, as `component_table` builds it; `truth[f]` lists the ground truth per annotator.
    """
    keep = np.asarray(keep, dtype=bool)
    frames = table["frame"].to_numpy()
    rows = []
    for f in files:
        labels, areas, _ = select(*raw[f], keep[frames == f])
        pq = PanopticQuality()
        for gt_crops in truth[f]:
            pq.update(gt_crops, labels, areas)
        rows.append({"frame": f, "tp": pq.tp, "fp": pq.fp, "fn": pq.fn, "iou_sum": pq.iou_sum})
    return pd.DataFrame(rows, columns=["frame", *TERMS])


def pq_of(terms: pd.DataFrame) -> float:
    """PQ of the frames in `terms`: sums first, ratio last (the mean of per-frame PQs is a different number)."""
    tp, fp, fn, iou_sum = (float(terms[c].sum()) for c in TERMS)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return iou_sum / denominator if denominator > 0 else 0.0


def best_cutoff(
    scores: np.ndarray, raw: dict, table: pd.DataFrame, truth: dict, files: list[str], cutoffs=CUTOFFS
) -> tuple[float, pd.DataFrame]:
    """(cutoff, frame terms) of the cutoff on the filter scores with the highest PQ; ties go to the lowest cutoff."""
    best = None
    for cutoff in cutoffs:
        terms = frame_terms(raw, table, scores >= cutoff, truth, files)
        if best is None or pq_of(terms) > pq_of(best[1]):
            best = (float(cutoff), terms)
    return best


# ---------------------------------------------------------------------------------------------------
# Yours to implement: tests/test_importance.py says what each must do.
# ---------------------------------------------------------------------------------------------------


def paired_bootstrap(
    terms_a: pd.DataFrame, terms_b: pd.DataFrame, n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """(delta, lo, hi): delta = PQ(b) - PQ(a) on all frames, and its 95 % bootstrap interval.

    `terms_a` and `terms_b` hold the same frames in the same order (raise ValueError otherwise). Each
    resample draws frames with replacement and uses the *same* frames for a and b, so the variation
    shared by both variants (easy frames, hard frames) cancels in the difference.
    """

    if len(terms_a) != len(terms_b) or (terms_a["frame"].to_numpy() != terms_b["frame"].to_numpy()).any():
        raise ValueError("terms_a and terms_b must hold the same frames in the same order")
    delta = pq_of(terms_b) - pq_of(terms_a)
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in rng.integers(0, len(terms_a), size=(n, len(terms_a))):
        diffs.append(pq_of(terms_b.iloc[idx]) - pq_of(terms_a.iloc[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def permutation_importance(
    table: pd.DataFrame,
    target: np.ndarray,
    feature: str,
    score_fn: Callable[[np.ndarray], float],
    repeats: int = 5,
    seed: int = 0,
    n_splits: int = 5,
    features: list[str] = FEATURES,
) -> np.ndarray:
    """Drop of `score_fn` when `feature` is shuffled, one value per repeat (positive = the filter uses it).

    The folds are those of `out_of_fold_scores` (GroupKFold by frame). A model is fitted on each training
    part as it is; `feature` is shuffled only in the held-out part before scoring it. `score_fn` maps
    out-of-fold filter scores (one per row of `table`) to a number where higher is better, e.g. ROC AUC
    or PQ at a fixed cutoff. `table` must not be modified.
    """
    rng = np.random.default_rng(seed)
    groups = table["frame"].to_numpy()
    splits = min(n_splits, len(np.unique(groups)))
    baseline = np.zeros(len(table))
    permuted = np.zeros((repeats, len(table)))
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(table, target, groups):
        model = ComponentFilter(features=features).fit(table.iloc[train_idx], target[train_idx])
        held_out = table.iloc[test_idx]
        baseline[test_idx] = model.scores(held_out)
        for r in range(repeats):
            shuffled = held_out.copy()
            shuffled[feature] = rng.permutation(held_out[feature].to_numpy())
            permuted[r, test_idx] = model.scores(shuffled)
    return np.array([score_fn(baseline) - score_fn(scores) for scores in permuted])


def drop_column(
    table: pd.DataFrame,
    target: np.ndarray,
    feature: str,
    evaluate: Callable[[np.ndarray], object],
    n_splits: int = 5,
    features: list[str] = FEATURES,
):
    """`evaluate` applied to out-of-fold scores of a filter refitted on `features` without `feature`.

    `evaluate` is e.g. `best_cutoff` with the frames bound, so the cutoff is chosen again for the
    smaller filter. A `feature` not in `features` raises ValueError instead of silently scoring the full set.
    """
    if feature not in features:
        raise ValueError(f"{feature!r} is not one of {features}")
    kept = [f for f in features if f != feature]
    return evaluate(out_of_fold_scores(table, target, n_splits=n_splits, features=kept))
