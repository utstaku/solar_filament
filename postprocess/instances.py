"""Probability map -> filament instances: (hysteresis) threshold and connected components; label-map bookkeeping."""

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


def merge(labels: np.ndarray, links) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge the instances of each (a, b) in `links`, transitively; renumber largest first. No pixels are added."""
    n = int(labels.max())
    parent = np.arange(n + 1)

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in links:
        parent[root(int(b))] = root(int(a))
    return renumber(labels, np.array([root(k) for k in range(n + 1)]))
