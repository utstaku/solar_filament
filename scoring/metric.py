"""Panoptic Quality as defined by the hosts' self-evaluation notebook.

    PQ = sum of IoU over true-positive pairs / (|TP| + 0.5 |FP| + 0.5 |FN|)

A ground-truth/prediction pair is a true positive when IoU > 0.5; unmatched predictions are false
positives and unmatched ground-truth filaments are false negatives. Scores are accumulated over
annotator-images, i.e. the same prediction is compared with every annotator's set for that image.

Predictions are passed as a label map, so they cannot overlap and a filament matches at most one
prediction. Overlaps are counted inside each ground-truth bounding box only, which makes evaluating
a 2048 x 2048 frame a matter of milliseconds.
"""

import numpy as np
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5

InstanceCrop = tuple[int, int, np.ndarray]  # x, y, boolean mask of the bounding box


def instance_crops(rles: list[dict]) -> list[InstanceCrop]:
    """Ground-truth RLEs -> bounding-box crops, the form `PanopticQuality.update` consumes."""
    crops = []
    for rle in rles:
        x, y, w, h = (int(v) for v in mask_util.toBbox(rle))
        if w > 0 and h > 0:
            crops.append((x, y, mask_util.decode(rle)[y : y + h, x : x + w].astype(bool)))
    return crops


class PanopticQuality:
    def __init__(self):
        self.iou_sum, self.tp, self.fp, self.fn = 0.0, 0, 0, 0

    def update(self, gt_crops: list[InstanceCrop], labels: np.ndarray, pred_areas: np.ndarray) -> None:
        """Add one annotator-image. `labels` holds 0 for background and 1..K for the K predictions."""
        n_pred = len(pred_areas)
        if not gt_crops or n_pred == 0:
            self.fp += n_pred
            self.fn += len(gt_crops)
            return
        pred_hit = np.zeros(n_pred, dtype=bool)
        for x, y, crop in gt_crops:
            h, w = crop.shape
            intersection = np.bincount(labels[y : y + h, x : x + w][crop], minlength=n_pred + 1)[1:]
            iou = intersection / (crop.sum() + pred_areas - intersection)
            best = int(iou.argmax())
            if iou[best] > IOU_THRESHOLD:
                self.tp += 1
                self.iou_sum += float(iou[best])
                pred_hit[best] = True
            else:
                self.fn += 1
        self.fp += int((~pred_hit).sum())

    @property
    def value(self) -> float:
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denominator if denominator > 0 else 0.0

    def summary(self) -> dict:
        """PQ with its factors: SQ (mean IoU of matches) and RQ (an F1 over instances)."""
        sq = self.iou_sum / self.tp if self.tp else 0.0
        return {
            "pq": round(self.value, 4),
            "sq": round(sq, 4),
            "rq": round(self.value / sq, 4) if sq else 0.0,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }
