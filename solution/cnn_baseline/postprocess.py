"""Probability map -> filament instances: (hysteresis) threshold and connected components."""

import cv2
import numpy as np

MAX_INSTANCES = 40  # ground truth never exceeds 26 filaments per frame


def instances(prob: np.ndarray, threshold: float, min_area: int, low: float | None = None):
    """(label map with 0 = background and 1..K, areas[K], boxes[K, 4] as x, y, w, h), largest first.

    `prob` is uint8 (H, W) holding probability * 255, as `training.predict` yields it.

    With `low` (< threshold) the threshold is hysteretic: a component of `prob >= low` is kept whole
    when any of its pixels reaches `threshold`. Filaments fade towards their ends, so a single
    threshold keeps the confident core and drops the faint tails; hysteresis follows them.
    """
    high = prob >= round(threshold * 255)
    if low is not None and low < threshold:
        n_low, low_labels = cv2.connectedComponents((prob >= round(low * 255)).astype(np.uint8), connectivity=8)
        seeded = np.zeros(n_low, dtype=bool)
        seeded[np.unique(low_labels[high])] = True
        seeded[0] = False
        foreground = seeded[low_labels].astype(np.uint8)
    else:
        foreground = high.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]  # component 0 is the background
    keep = keep[np.argsort(areas[keep])[::-1][:MAX_INSTANCES]]
    relabel = np.zeros(n, dtype=np.int32)
    relabel[keep] = np.arange(1, len(keep) + 1)
    return relabel[labels], areas[keep], stats[keep, :4]


def select(labels: np.ndarray, areas: np.ndarray, boxes: np.ndarray, keep: np.ndarray):
    """Drop the instances where `keep` (bool per instance 1..K) is False; renumber the rest 1..K'."""
    keep = np.asarray(keep, dtype=bool)
    relabel = np.zeros(len(areas) + 1, dtype=np.int32)
    relabel[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    return relabel[labels], areas[keep], boxes[keep]
