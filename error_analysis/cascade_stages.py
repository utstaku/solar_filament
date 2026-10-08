"""Per-GT cascade diagnostics. Categories are operational, not causal proof."""

import numpy as np


def intersection(gt, x, y, mask):
    gx, gy, crop = gt
    x0, y0 = max(gx, x), max(gy, y)
    x1, y1 = min(gx + crop.shape[1], x + mask.shape[1]), min(gy + crop.shape[0], y + mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((crop[y0 - gy : y1 - gy, x0 - gx : x1 - gx] & mask[y0 - y : y1 - y, x0 - x : x1 - x]).sum())


def best_final_iou(gt, labels, areas):
    if not len(areas):
        return 0.0
    x, y, crop = gt
    hits = np.bincount(labels[y : y + crop.shape[0], x : x + crop.shape[1]][crop], minlength=len(areas) + 1)[1:]
    return float(np.max(hits / (crop.sum() + areas - hits)))


def diagnose(gt, candidates, final, filtered=None):
    """Candidate crop coverage is an upper bound on that candidate's achievable GT IoU.

    pre-match disappearing after exclusivity is checked before proposal coverage.
    No NMS/pre-confidence detections are available: proposal misses cannot be
    attributed to confidence, NMS, or detector features separately here.
    """
    area = int(gt[2].sum())
    coverages, ious, intersections, mask_areas = [], [], [], []
    for _, x, y, prob in candidates:
        foreground = prob >= 128
        hit = intersection(gt, x, y, foreground)
        cover = intersection(gt, x, y, np.ones(prob.shape, bool)) / area
        size = int(foreground.sum())
        coverages.append(cover)
        ious.append(hit / (area + size - hit))
        intersections.append(hit)
        mask_areas.append(size)
    final_iou = best_final_iou(gt, final[0], final[1])
    pre_iou = max(ious, default=0.0)
    coverage = max(coverages, default=0.0)
    eligible = [i for i, c in enumerate(coverages) if c > 0.5]
    if final_iou > 0.5:
        stage = "matched"
    elif pre_iou > 0.5:
        good = [i for i, v in enumerate(ious) if v > 0.5]
        stage = "minimum_area" if all(mask_areas[i] < 150 for i in good) else "overlap_resolution"
    elif coverage == 0:
        stage = "no_proposal_crop_overlap"
    elif coverage <= 0.5:
        stage = "insufficient_proposal_crop_coverage"
    elif all(mask_areas[i] == 0 for i in eligible):
        stage = "refiner_empty"
    elif all(intersections[i] == 0 for i in eligible):
        stage = "refiner_no_gt_overlap"
    else:
        stage = "refiner_iou_at_most_half"
    result = {
        "stage": stage,
        "gt_area": area,
        "best_crop_coverage": coverage,
        "best_pre_iou": pre_iou,
        "final_iou": final_iou,
        "n_candidates": len(candidates),
    }
    if filtered is not None:
        result["filtered_iou"] = best_final_iou(gt, filtered[0], filtered[1])
        result["lost_to_component_filter"] = final_iou > 0.5 and result["filtered_iou"] <= 0.5
    return result


def best_box_iou(gt, boxes):
    """Detector box diagnostic, separate from segmentation matching."""
    x, y, crop = gt
    if not len(boxes):
        return 0.0
    boxes = np.asarray(boxes)
    lo = np.maximum(boxes[:, :2], [x, y])
    hi = np.minimum(boxes[:, 2:], [x + crop.shape[1], y + crop.shape[0]])
    hit = np.clip(hi - lo, 0, None).prod(axis=1)
    union = crop.size + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - hit
    return float((hit / union).max())
