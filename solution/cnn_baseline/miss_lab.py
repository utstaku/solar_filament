# kaggle: kernel_sources={user}/filament-cnn-baseline-predict
# %% [markdown]
# # Miss lab: can post-processing reach what A0 leaves out?
#
# CPU only, on the validation probability maps saved by `filament-cnn-baseline-predict`. Two questions decide
# which post-processing is worth building:
#
# 1. **Near misses** (A0 covers part of a filament, IoU <= 0.5): what probability does the network give the
#    part it leaves out? Mostly 0.1-0.5 means growing along the filament can reach it; mostly < 0.1 means
#    only a better model can.
# 2. **Missed filaments the network responded to** (mean probability >= 0.5 inside them): were they dropped by
#    the component filter, by the 300 px minimum area, or never part of a component at all?
#
# Outputs: `miss_rows.csv` (one row per annotator x filament that A0 does not match), `miss_report.json`.

# %%
import json
import time
from itertools import pairwise

import cv2
import numpy as np
import pandas as pd

from cnn_baseline.data import split_files
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import (
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
)
from postprocess.instances import instances, select
from postprocess.tuning import best_cutoff
from preprocess.images import read_gray
from scoring.metric import IOU_THRESHOLD, instance_crops

SEED = 0  # must match cnn_baseline/main.py and predict.py
VALIDATION_FRACTION = 0.15
PROB_BINS = [0, 0.1, 0.2, 0.3, 0.5, 1.01]
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs and A0 (as in `postprocess_lab.py`, with its sanity gate)

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
_, val_files = split_files([p.name for p in train_dir.iterdir()], VALIDATION_FRACTION, SEED)
report = json.loads(find_input("predict_report.json").read_text())
params, a0_cutoff = report["chosen"]["params"], report["chosen"]["filter_cutoff"]
saved = np.load(find_input("val_probs.npz"))
assert sorted(saved.files) == sorted(val_files), "val_probs.npz holds other frames than this validation split"
probs = {f: saved[f] for f in val_files}
contexts = {f: frame_context(f, read_gray(train_dir / f), probs[f]) for f in val_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}

raw = {f: instances(probs[f], **params) for f in val_files}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in val_files if len(raw[f][1])], ignore_index=True
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in val_files])
oof = out_of_fold_scores(table, target)
cutoff, terms = best_cutoff(oof, raw, table, truth, val_files)
assert cutoff == a0_cutoff, "does not reproduce predict.py"
frames, keep = table["frame"].to_numpy(), oof >= cutoff
kept = {f: select(*raw[f], keep[frames == f]) for f in val_files}
log("A0 rebuilt: cutoff", cutoff, "tp/fp/fn", terms[["tp", "fp", "fn"]].sum().to_dict())

# %% [markdown]
# ## Every filament A0 does not match, per annotator
#
# The hysteresis components are recomputed here without `instances`' minimum area and 40-instance cap, so
# a missed filament can be traced to the step that removed it.

# %%


def hysteresis_components(prob: np.ndarray):
    """(label map, areas) of every hysteresis component of `prob`, however small."""
    high = prob >= round(params["threshold"] * 255)
    n, low_labels = cv2.connectedComponents((prob >= round(params["low"] * 255)).astype(np.uint8), connectivity=8)
    seeded = np.zeros(n, bool)
    seeded[np.unique(low_labels[high])] = True
    seeded[0] = False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(seeded[low_labels].astype(np.uint8), connectivity=8)
    return labels, stats[:, cv2.CC_STAT_AREA]


rows = []
for f in val_files:
    prob = probs[f]
    kept_labels, kept_areas, _ = kept[f]
    raw_labels = raw[f][0]
    all_labels, all_areas = hysteresis_components(prob)
    for annotator, crops in enumerate(truth[f]):
        for i, (x, y, crop) in enumerate(crops):
            h, w = crop.shape
            covered = kept_labels[y : y + h, x : x + w]
            inside = covered[crop]
            counts = np.bincount(inside, minlength=len(kept_areas) + 1)[1:]
            iou = counts / (crop.sum() + kept_areas - counts) if len(kept_areas) else np.zeros(0)
            if len(iou) and iou.max() > IOU_THRESHOLD:
                continue  # matched
            p = prob[y : y + h, x : x + w][crop] / 255
            uncovered = p[inside == 0]
            in_raw = raw_labels[y : y + h, x : x + w][crop]
            touching = np.unique(all_labels[y : y + h, x : x + w][crop])
            touching = touching[touching > 0]
            if inside.any():
                status = "near_miss"  # (or a fragment / merge: some kept prediction overlaps it)
            elif p.mean() < 0.5:
                status = "missed_low_prob"
            elif (in_raw > 0).any():
                status = "missed_dropped_by_filter"
            elif len(touching) and all_areas[touching].max() >= params["min_area"]:
                status = "missed_dropped_by_instance_cap"  # big enough, but beyond the 40 largest
            elif len(touching):
                status = "missed_below_min_area"
            else:
                status = "missed_no_component"
            row = {
                "frame": f,
                "annotator": annotator,
                "gt": i,
                "area": int(crop.sum()),
                "status": status,
                "best_iou": round(float(iou.max()) if len(iou) else 0.0, 3),
                "coverage": round(float((inside > 0).mean()), 3),
                "mean_prob": round(float(p.mean()), 3),
                "uncovered_px": int(uncovered.size),
            }
            hist = np.histogram(uncovered, PROB_BINS)[0] if uncovered.size else np.zeros(len(PROB_BINS) - 1)
            for (lo, hi), n in zip(pairwise(PROB_BINS), hist):
                row[f"uncovered_p{lo:.1f}_{min(hi, 1):.1f}"] = int(n)
            rows.append(row)
miss = pd.DataFrame(rows)
miss.to_csv(output_dir() / "miss_rows.csv", index=False)

# %% [markdown]
# ## Summary

# %%
# shares of the left-out pixels by probability; >= 0.5 lies in components that were dropped (rescue)
bins = [c for c in miss.columns if c.startswith("uncovered_p") and c != "uncovered_px"]
near = miss[miss["status"] == "near_miss"]
summary = {
    "rows_by_status": miss["status"].value_counts().to_dict(),
    "near_miss": {
        "rows": len(near),
        "median_coverage": round(float(near["coverage"].median()), 3),
        "uncovered_pixels_by_prob": (near[bins].sum() / max(near[bins].to_numpy().sum(), 1)).round(3).to_dict(),
        "share_of_rows_mostly_reachable": round(
            float(((near[bins[1:4]].sum(axis=1)) / near["uncovered_px"].clip(lower=1) > 0.5).mean()), 3
        ),
    },
}
print(json.dumps(summary, indent=2))
(output_dir() / "miss_report.json").write_text(json.dumps(summary, indent=2, default=int))
log("done")
