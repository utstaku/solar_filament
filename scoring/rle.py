"""COCO run-length encoding helpers. The submission stores only the compressed `counts` string."""

import numpy as np
from pycocotools import mask as mask_util

from dataset import IMAGE_SIZE


def encode(mask: np.ndarray) -> str:
    """Binary (H, W) mask -> compressed RLE counts."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def decode(counts: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Compressed RLE counts -> uint8 (size, size) mask."""
    return mask_util.decode({"size": [size, size], "counts": counts.encode("ascii")})


def encode_component(labels: np.ndarray, label: int, box: tuple[int, int, int, int]) -> str:
    """Encode one instance of a label map, touching only its bounding box (x, y, w, h)."""
    x, y, w, h = box
    mask = np.zeros(labels.shape, dtype=np.uint8, order="F")
    mask[y : y + h, x : x + w] = labels[y : y + h, x : x + w] == label
    return mask_util.encode(mask)["counts"].decode("ascii")


def is_valid(counts: str, size: int = IMAGE_SIZE) -> bool:
    """True if `counts` is well-formed and its runs cover the image exactly.

    pycocotools decodes garbage without complaint, so the string is parsed here: each number is a
    little-endian sequence of 5-bit groups (bit 5 = continuation, bit 4 of the last group = sign),
    and from the fourth number on values are deltas against the number two positions earlier.
    """
    runs, pos = [], 0
    try:
        while pos < len(counts):
            value, group, more = 0, 0, True
            while more:
                char = ord(counts[pos]) - 48
                if not 0 <= char < 64:
                    return False
                value |= (char & 0x1F) << (5 * group)
                more = bool(char & 0x20)
                pos += 1
                group += 1
                if not more and (char & 0x10):
                    value |= -1 << (5 * group)
            if len(runs) > 2:
                value += runs[-2]
            if value < 0:
                return False
            runs.append(value)
    except IndexError:  # the string ended in the middle of a number
        return False
    return sum(runs) == size * size
