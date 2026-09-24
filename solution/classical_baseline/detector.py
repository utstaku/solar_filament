"""Dark-region detector used by main.py."""

import cv2
import numpy as np

from common.preprocess import REL_SCALE

MAX_AREA = 80_000  # px; the 99th percentile of ground-truth filament area is ~14k
MAX_INSTANCES = 40  # ground truth never exceeds 26 filaments per image


def segment(flat: np.ndarray, ratio: float, min_area: int, close_px: int):
    """Threshold a flattened frame, close small gaps and split it into connected components.

    `flat` comes from `preprocess.flatten`; pixels darker than `ratio` x background are candidates.
    Returns (label map with 0 = background and 1..K, areas[K], boxes[K, 4] as x, y, w, h), keeping
    at most MAX_INSTANCES components, largest first.
    """
    dark = (flat < round(ratio * REL_SCALE)).astype(np.uint8)
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero((areas >= min_area) & (areas <= MAX_AREA))
    keep = keep[keep != 0]  # component 0 is the background
    keep = keep[np.argsort(areas[keep])[::-1][:MAX_INSTANCES]]
    relabel = np.zeros(n, dtype=np.int32)
    relabel[keep] = np.arange(1, len(keep) + 1)
    return relabel[labels], areas[keep], stats[keep, :4]
