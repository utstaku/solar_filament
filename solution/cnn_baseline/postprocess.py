"""Probability map -> filament instances, the simplest way: threshold and connected components."""

import cv2
import numpy as np

MAX_INSTANCES = 40  # ground truth never exceeds 26 filaments per frame


def instances(prob: np.ndarray, threshold: float, min_area: int):
    """(label map with 0 = background and 1..K, areas[K], boxes[K, 4] as x, y, w, h), largest first.

    `prob` is uint8 (H, W) holding probability * 255, as `training.predict` yields it.
    """
    foreground = (prob >= round(threshold * 255)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]  # component 0 is the background
    keep = keep[np.argsort(areas[keep])[::-1][:MAX_INSTANCES]]
    relabel = np.zeros(n, dtype=np.int32)
    relabel[keep] = np.arange(1, len(keep) + 1)
    return relabel[labels], areas[keep], stats[keep, :4]
