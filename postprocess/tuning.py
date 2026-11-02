"""The filter's cutoff, chosen on Panoptic Quality.

PQ is a ratio of sums over the whole validation set, sum(IoU) / (TP + FP/2 + FN/2), so it is kept here as
per-frame terms (tp, fp, fn, iou_sum). Any subset or resample of frames is then scored by summing its terms.
"""

import numpy as np
import pandas as pd

from postprocess.instances import select
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
