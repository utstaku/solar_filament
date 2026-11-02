"""Which features the component filter relies on, whether a post-processing change beats noise, and what
an out-of-fold PQ predicts for the public leaderboard.

All of it works on per-frame PQ terms (`postprocess.tuning.frame_terms`): any subset or resample of frames is
scored by summing its terms, and two post-processing variants can be compared frame by frame.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from postprocess.filtering import FEATURES, ComponentFilter, out_of_fold_scores
from postprocess.tuning import TERMS, pq_of


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


def public_pq(terms: pd.DataFrame, r: float) -> float:
    """Expected public PQ if each filament an annotator drew in our training labels appears in the test labels
    only with probability `r` (the hypothesis of the public "OOF 0.45 vs Public 0.39" notebook).

    A match whose filament is missing from the test labels turns into a false positive, and a missed filament
    is missing as often: PQ' = r IoU / (r TP + FP/2 + (1 - r) TP/2 + r FN/2). r = 1 gives back PQ.
    """
    tp, fp, fn, iou_sum = (float(terms[c].sum()) for c in TERMS)
    denominator = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return r * iou_sum / denominator if denominator > 0 else 0.0


def fit_r(terms: pd.DataFrame, public_score: float) -> float:
    """The r for which `public_pq(terms, r)` equals an observed public score (bisection over (0, 1])."""
    lo, hi = 1e-3, 1.0
    if public_pq(terms, hi) < public_score:
        return 1.0  # the public score is at or above validation: no sparsity to explain
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if public_pq(terms, mid) < public_score else (lo, mid)
    return (lo + hi) / 2
