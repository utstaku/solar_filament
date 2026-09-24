"""Galleries of the worst cases: crops with ground truth in green and predictions in red.

Figures need the frames, so they are drawn on Kaggle and shown in the notebook; each gallery is also
saved as a PNG that `filament run --figures` brings back.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scoring.metric import InstanceCrop

# (table, status, sort column, descending): what "worst" means for each kind of error
CATEGORIES = {
    "hallucination": ("pred", "hallucination", "area", True),
    "missed": ("gt", "missed", "agreement", True),
    "near_miss": ("gt", "near_miss", "area", True),
    "fragmented": ("gt", "fragmented", "area", True),
    "merge": ("pred", "merge", "area", True),
}

Context = Callable[[str, str], tuple[np.ndarray, list[InstanceCrop], np.ndarray]]
"""(frame, annotator) -> (gray image, that annotator's filament crops, prediction label map)."""


def worst_cases(gt: pd.DataFrame, pred: pd.DataFrame, k: int = 8) -> dict[str, pd.DataFrame]:
    out = {}
    for name, (table, status, column, descending) in CATEGORIES.items():
        df = gt if table == "gt" else pred
        rows = df[df["status"] == status]
        if column in rows.columns:
            rows = rows.sort_values(column, ascending=not descending)
        # one case per frame keeps a gallery from filling up with a single bad image
        out[name] = rows.drop_duplicates("frame").head(k)
    return out


def draw_case(
    ax, image: np.ndarray, gt_crops: list[InstanceCrop], labels: np.ndarray, row: pd.Series, margin: int = 60
):
    x0 = int(max(row["cx"] - max(row["area"] ** 0.5, 40) - margin, 0))
    y0 = int(max(row["cy"] - max(row["area"] ** 0.5, 40) - margin, 0))
    x1 = int(min(row["cx"] + max(row["area"] ** 0.5, 40) + margin, image.shape[1]))
    y1 = int(min(row["cy"] + max(row["area"] ** 0.5, 40) + margin, image.shape[0]))
    canvas = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)
    for x, y, crop in gt_crops:
        contours, _ = cv2.findContours(
            crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x - x0, y - y0)
        )
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 1)
    contours, _ = cv2.findContours(
        (labels[y0:y1, x0:x1] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(canvas, contours, -1, (255, 0, 0), 1)
    ax.imshow(canvas)
    extra = f"  p={row['mean_prob']:.2f}" if "mean_prob" in row and pd.notna(row["mean_prob"]) else ""
    ax.set_title(f"{row['frame'][:16]}  area {int(row['area'])}  dark {row['darkness']:.2f}{extra}", fontsize=8)
    ax.axis("off")


def save_galleries(cases: dict[str, pd.DataFrame], context: Context, out_dir: Path, ncols: int = 4) -> list[Path]:
    paths = []
    for name, rows in cases.items():
        if not len(rows):
            continue
        nrows = int(np.ceil(len(rows) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
            image, gt_crops, labels = context(row["frame"], row["annotator"])
            draw_case(ax, image, gt_crops, labels, row)
        fig.suptitle(f"{name}: ground truth green, prediction red", fontsize=11)
        fig.tight_layout()
        path = Path(out_dir) / f"gallery_{name}.png"
        fig.savefig(path, dpi=110)
        plt.show()
        plt.close(fig)
        paths.append(path)
    return paths
