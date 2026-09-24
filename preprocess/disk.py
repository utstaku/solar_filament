"""Solar-disk geometry and limb-darkening removal for full-disk H-alpha frames."""

import cv2
import numpy as np

REL_SCALE = 200  # `flatten` stores image / background as uint8, with the background level at REL_SCALE
OFF_DISK = 255  # value `flatten` assigns outside the (shrunk) disk


def disk_geometry(image: np.ndarray) -> tuple[float, float, float]:
    """(centre x, centre y, radius) of the solar disk in pixels; radius 0 when no disk is found.

    The sky around the disk is grey rather than black in many frames, hence Otsu instead of a fixed
    threshold. Centre and radius come from the blob's centroid and area, which ignores prominences.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n < 2:
        return image.shape[1] / 2, image.shape[0] / 2, 0.0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(centroids[k][0]), float(centroids[k][1]), float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))


def disk_mask(image: np.ndarray, shrink: float = 1.0) -> np.ndarray:
    """uint8 {0, 1} mask of the solar disk, its radius scaled by `shrink`."""
    cx, cy, radius = disk_geometry(image)
    mask = np.zeros_like(image, dtype=np.uint8)
    if radius > 0:
        cv2.circle(mask, (round(cx), round(cy)), int(radius * shrink), 1, -1)
    return mask


def flatten(image: np.ndarray, shrink: float = 0.96, background_sigma: float = 40, smooth_sigma: float = 2):
    """Remove limb darkening: uint8 map of REL_SCALE * image / smooth background, OFF_DISK off-disk.

    The background is smooth, so it is estimated at 1/8 resolution. It is a normalised convolution
    (blurred image / blurred disk mask), which keeps the dark sky from bleeding into the limb.
    `shrink` drops the outermost ring, where the limb itself would read as a dark structure.
    """
    size = image.shape[0]
    disk = disk_mask(image, shrink).astype(np.float32)
    image_f = image.astype(np.float32)  # uint8 arithmetic would wrap around
    small = (size // 8, size // 8)
    weight = cv2.GaussianBlur(cv2.resize(disk, small, interpolation=cv2.INTER_AREA), (0, 0), background_sigma / 8)
    blurred = cv2.GaussianBlur(
        cv2.resize(image_f * disk, small, interpolation=cv2.INTER_AREA), (0, 0), background_sigma / 8
    )
    background = cv2.resize(blurred / np.maximum(weight, 1e-3), (size, size), interpolation=cv2.INTER_LINEAR)
    # filaments are ~10-20 px wide, so a light blur only tames the chromospheric texture
    smooth = cv2.GaussianBlur(image_f, (0, 0), smooth_sigma) if smooth_sigma > 0 else image_f
    rel = np.clip(REL_SCALE * smooth / np.maximum(background, 1.0), 0, 255).astype(np.uint8)
    rel[disk == 0] = OFF_DISK
    return rel
