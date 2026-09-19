"""Synthetic full-disk frames, so everything can be tested without the competition data."""

import cv2
import numpy as np

SIZE = 2048


def make_frame(seed: int = 0, n_filaments: int = 10, contrast: float = 0.88, centre=(1030, 1015), radius: int = 905):
    """A frame with the quirks of the real ones, plus the binary mask of each filament.

    Grey (not black) sky around the disk, a bright limb ring, black corners, limb darkening, a
    textured chromosphere and filaments that are only `1 - contrast` darker than their surroundings.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.hypot(xx - centre[0], yy - centre[1]) / radius
    image = np.where(r < 1, 120 * np.sqrt(np.clip(1 - 0.5 * r**2, 0, 1)), 35).astype(np.float32)
    image[(r > 0.99) & (r < 1.01)] = 230
    image[np.hypot(xx - SIZE / 2, yy - SIZE / 2) > 1010] = 0

    masks = []
    for _ in range(n_filaments):
        start = rng.integers(450, 1550, (1, 2))
        points = (start + np.cumsum(rng.integers(-60, 60, (6, 2)), axis=0)).astype(np.int32)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.polylines(mask, [points], False, 1, thickness=int(rng.integers(8, 18)))
        image[mask > 0] *= contrast
        masks.append(mask)

    image = cv2.GaussianBlur(image + rng.normal(0, 6, image.shape).astype(np.float32), (0, 0), 1.2)
    return np.clip(image, 0, 255).astype(np.uint8), masks
