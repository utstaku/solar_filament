"""Probability map -> filament instances, the simplest way: threshold and connected components."""

import cv2
import numpy as np

from preprocess.disk import disk_mask

MAX_INSTANCES = 40  # ground truth never exceeds 26 filaments per frame
DISK_SLACK = 1.02  # keep a sliver past the estimated limb so filaments touching it are not clipped


def on_disk(prob: np.ndarray, image: np.ndarray) -> np.ndarray:
    """`prob` with everything off the solar disk set to 0.

    Training tiles are centred on the disk, so the network barely sees the sky beyond the limb and its
    output there hovers around 0.5 -- enough to pass a 0.5 threshold as large spurious blobs.
    """
    return prob * disk_mask(image, shrink=DISK_SLACK)


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
