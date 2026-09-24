"""Reading the competition's JPEG frames."""

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    """uint8 (H, W). The frames are grayscale; reading them as colour would only triplicate the data."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read {path}")
    return image
