"""Submission file: one row per predicted filament, `filament_id` = `<image id>_<n>`."""

from pathlib import Path

import numpy as np
import pandas as pd

from common import rle

COLUMNS = ["filament_id", "segmentation_rle"]


def rows_from_label_map(image_id: str, labels: np.ndarray, boxes: np.ndarray) -> list[dict]:
    """Rows for the instances 1..K of a label map; `boxes[k - 1]` is instance k as (x, y, w, h)."""
    return [
        {"filament_id": f"{image_id}_{k}", "segmentation_rle": rle.encode_component(labels, k, tuple(box))}
        for k, box in enumerate(boxes, start=1)
    ]


def write(rows: list[dict], path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(path, index=False)
    return frame
