"""Aggregate the per-instance tables into numbers that decide what to work on next.

Everything here works from the two CSVs alone, so it runs locally on tables fetched from Kaggle.
"""

import numpy as np
import pandas as pd

from scoring.metric import IOU_THRESHOLD

GT_STATUSES = ("matched", "near_miss", "fragmented", "merged", "missed")
PRED_STATUSES = ("matched", "near_miss", "fragment", "merge", "hallucination")


def pq_terms(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """TP, FP, FN, the IoU sum and PQ / SQ / RQ, recomputed from the tables (matches scoring.metric)."""
    matched = gt[gt["status"] == "matched"]
    tp, fn = len(matched), int((gt["status"] != "matched").sum())
    fp = int((pred["status"] != "matched").sum())
    iou_sum = float(matched["best_iou"].sum())
    return {**pq(tp, fp, fn, iou_sum), "tp": tp, "fp": fp, "fn": fn, "iou_sum": round(iou_sum, 2)}


def pq(tp: int, fp: int, fn: int, iou_sum: float) -> dict:
    denominator = tp + 0.5 * fp + 0.5 * fn
    value = iou_sum / denominator if denominator else 0.0
    sq = iou_sum / tp if tp else 0.0
    return {"pq": round(value, 4), "sq": round(sq, 4), "rq": round(value / sq, 4) if sq else 0.0}


def what_if(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """PQ if one kind of error were fixed completely: where the upside is, before doing any work."""
    t = pq_terms(gt, pred)
    tp, fp, fn, iou_sum = t["tp"], t["fp"], t["fn"], t["iou_sum"]
    sq = iou_sum / tp if tp else 0.0
    near = gt[gt["status"] == "near_miss"]
    # a near miss lifted just over the threshold turns FP + FN into one TP at IoU ~0.5
    near_gain = len(near) * (IOU_THRESHOLD + 0.01)
    rows = [
        ("current", tp, fp, fn, iou_sum),
        ("no false positives", tp, 0, fn, iou_sum),
        ("half the false positives", tp, fp / 2, fn, iou_sum),
        ("no hallucinations", tp, fp - int((pred["status"] == "hallucination").sum()), fn, iou_sum),
        ("near misses fixed", tp + len(near), fp - len(near), fn - len(near), iou_sum + near_gain),
        ("no false negatives (same SQ)", tp + fn, fp, 0, sq * (tp + fn)),
        ("perfect masks (SQ = 1)", tp, fp, fn, float(tp)),
    ]
    return pd.DataFrame([{"scenario": name, "tp": a, "fp": b, "fn": c, **pq(a, b, c, s)} for name, a, b, c, s in rows])


def status_counts(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    return {
        "ground_truth": {s: int((gt["status"] == s).sum()) for s in GT_STATUSES},
        "predictions": {s: int((pred["status"] == s).sum()) for s in PRED_STATUSES},
    }


def breakdown(df: pd.DataFrame, column: str, bins=None, labels=None) -> pd.DataFrame:
    """Share of each status per bin of `column` (quantile bins unless `bins` is given), with counts."""
    if bins is None and pd.api.types.is_numeric_dtype(df[column]):
        groups = pd.qcut(df[column], q=4, duplicates="drop")
    elif bins is not None:
        groups = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    else:
        groups = df[column]
    if groups.isna().all():  # too few distinct values to bin: one group is still a table
        groups = pd.Series("all", index=df.index, name=column)
    table = pd.crosstab(groups, df["status"], normalize="index").round(3)
    table["n"] = df.groupby(groups, observed=True).size()
    return table


def separability(pred: pd.DataFrame, feature: str = "mean_prob") -> dict:
    """How well `feature` tells matched predictions from hallucinations: AUC via the rank statistic.

    An AUC near 1 means a second-stage filter on that feature could remove hallucinations cheaply.
    """
    if feature not in pred.columns:
        return {}
    good = pred.loc[pred["status"] == "matched", feature].to_numpy()
    bad = pred.loc[pred["status"] == "hallucination", feature].to_numpy()
    if not len(good) or not len(bad):
        return {"feature": feature, "auc": np.nan}
    ranks = pd.Series(np.concatenate([good, bad])).rank().to_numpy()
    auc = (ranks[: len(good)].sum() - len(good) * (len(good) + 1) / 2) / (len(good) * len(bad))
    return {
        "feature": feature,
        "auc": round(float(auc), 3),
        "matched_median": round(float(np.median(good)), 3),
        "hallucination_median": round(float(np.median(bad)), 3),
    }


def summary(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Everything worth keeping as JSON next to the tables."""
    out = {
        "terms": pq_terms(gt, pred),
        "counts": status_counts(gt, pred),
        "what_if": what_if(gt, pred).to_dict(orient="records"),
        "separability": [separability(pred, f) for f in ("mean_prob", "max_prob", "area", "darkness", "elongation")],
        "matched_iou": {
            "median": round(float(gt.loc[gt["status"] == "matched", "best_iou"].median()), 3) if len(gt) else None,
        },
    }
    out["separability"] = [s for s in out["separability"] if s]
    matched_pred = pred[pred["status"] == "matched"]
    if len(matched_pred) and "best_gt" in pred.columns:
        # area ratio of prediction to its filament: a systematic bias is fixable with a threshold/dilation
        joined = matched_pred.merge(
            gt[gt["status"] == "matched"][["frame", "annotator", "gt_id", "area"]].rename(columns={"area": "gt_area"}),
            left_on=["frame", "annotator", "best_gt"],
            right_on=["frame", "annotator", "gt_id"],
        )
        if len(joined):
            out["matched_area_ratio_median"] = round(float((joined["area"] / joined["gt_area"]).median()), 3)
    return out


def breakdowns(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The standard slices, as tables to print or save."""
    out = {}
    if len(gt):
        out["gt_by_area"] = breakdown(gt, "area")
        out["gt_by_darkness"] = breakdown(gt, "darkness")
        out["gt_by_limb"] = breakdown(
            gt, "limb", bins=[0, 0.5, 0.8, 0.95, 1.5], labels=["centre", "mid", "near limb", "limb"]
        )
        out["gt_by_agreement"] = breakdown(
            gt, "agreement", bins=[-0.5, 0.5, 1.5, 2.5], labels=["only this annotator", "1 other", "2 others"]
        )
        out["gt_by_site"] = breakdown(gt, "site")
    if len(pred):
        out["pred_by_area"] = breakdown(pred, "area")
        out["pred_by_darkness"] = breakdown(pred, "darkness")
        if "mean_prob" in pred.columns:
            out["pred_by_mean_prob"] = breakdown(pred, "mean_prob", bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0])
        out["pred_by_limb"] = breakdown(
            pred, "limb", bins=[0, 0.5, 0.8, 0.95, 1.5], labels=["centre", "mid", "near limb", "limb"]
        )
        out["pred_by_site"] = breakdown(pred, "site")
    return out
