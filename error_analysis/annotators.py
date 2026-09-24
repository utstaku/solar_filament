"""How well the human annotators agree with each other, scored exactly like a submission.

Up to three people annotated the same frame independently. Scoring one annotator's filaments against
another's with Panoptic Quality gives the ceiling a model is measured against: a submission cannot be
expected to agree with every annotator better than the annotators agree among themselves.
"""

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from dataset.annotations import Annotations
from scoring.metric import PanopticQuality, instance_crops


def label_map(rles: list[dict], shape: tuple[int, int]):
    """One annotator's filaments as a label map, the form predictions take; later filaments win overlaps."""
    labels = np.zeros(shape, dtype=np.int32)
    if rles:
        masks = mask_util.decode(rles)  # (H, W, n)
        for k in range(masks.shape[2]):
            labels[masks[:, :, k] > 0] = k + 1
    areas = np.bincount(labels.ravel(), minlength=len(rles) + 1)[1:]
    return labels, areas


def human_ceiling(annotations: Annotations, files: list[str], shape: tuple[int, int]) -> dict:
    """PQ of every annotator against every other, over the frames with at least two annotators."""
    overall = PanopticQuality()
    per_frame = []
    for file_name in files:
        entries = annotations.file_to_entries.get(file_name, [])
        if len(entries) < 2:
            continue
        rles = [annotations.entry_to_rles.get(entry, []) for entry in entries]
        crops = [instance_crops(r) for r in rles]
        for a in range(len(entries)):
            labels, areas = label_map(rles[a], shape)
            for b in range(len(entries)):
                if a == b:
                    continue
                pair = PanopticQuality()
                pair.update(crops[b], labels, areas)
                overall.update(crops[b], labels, areas)
                per_frame.append(
                    {"frame": file_name, "as_prediction": entries[a], "as_truth": entries[b], **pair.summary()}
                )
    table = pd.DataFrame(per_frame)
    return {
        "n_frames": int(table["frame"].nunique()) if len(table) else 0,
        "n_pairs": len(table),
        "overall": overall.summary(),
        "per_frame_pq_median": round(float(table["pq"].median()), 4) if len(table) else None,
        "per_frame_pq_quartiles": [round(float(q), 4) for q in table["pq"].quantile([0.25, 0.75])]
        if len(table)
        else None,
        "pairs": table,
    }
