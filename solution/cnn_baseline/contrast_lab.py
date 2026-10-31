# kaggle: kernel_sources={user}/filament-cnn-baseline-cv
# %% [markdown]
# # Contrast lab: is what A0 leaves out visible in the image?
#
# CPU only, on the 707 out-of-fold maps of `filament-cnn-baseline-cv`. The miss lab found that three
# quarters of the part of a near miss A0 leaves out has network probability below 0.1: no post-processing of
# the map can reach it. The annotators drew it on the image, though. If it is clearly darker than the
# background there, growing kept filaments along the *image* contrast (not the map) could reach it; if it
# is within the noise, the annotators drew it from context and only better information helps.
#
# For every annotator x filament that A0 (filter fitted out of fold) does not match, and for the matched ones
# as a reference: the Halpha absorption contrast (1 - flat / background level, on the limb-darkening-corrected
# frame smoothed by a 2 px Gaussian, roughly what an eye integrates) of
# - the part A0 covers and the part it leaves out (near misses), or the whole filament (missed, matched),
# against a local background ring 5-20 px around the filament (no filament of any annotator, no prediction):
# its median and robust spread (1.4826 x MAD). SNR = (median contrast - background median) / spread.
#
# Outputs: `contrast_rows.csv`, `contrast_report.json`.

# %%
import json
import time

import cv2
import numpy as np
import pandas as pd

from cnn_baseline.filtering import component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from cnn_baseline.importance import best_cutoff
from cnn_baseline.postprocess import instances, select
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from preprocess.disk import OFF_DISK, REL_SCALE
from preprocess.images import read_gray
from scoring.metric import IOU_THRESHOLD, instance_crops

A0 = {"threshold": 0.5, "min_area": 300, "low": 0.3}
SMOOTH = 2.0  # px, Gaussian sigma
RING = (5, 20)  # px, inner and outer distance of the background ring
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs and A0 on the 707 OOF plain maps

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)
maps = {}
for k in range(len(folds)):
    saved = np.load(find_input(f"oof_plain_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}

raw = {f: instances(maps[f], **A0) for f in files}
raw = {f: (labels.astype(np.uint8), areas, boxes) for f, (labels, areas, boxes) in raw.items()}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
oof = out_of_fold_scores(table, target)
cutoff, terms = best_cutoff(oof, raw, table, truth, files)
frames, keep = table["frame"].to_numpy(), oof >= cutoff
log(f"A0 on {len(files)} frames: cutoff {cutoff}, tp/fp/fn {terms[['tp', 'fp', 'fn']].sum().to_dict()}")

# %% [markdown]
# ## Contrast of every annotated filament

# %%
inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING[0] + 1, 2 * RING[0] + 1))
outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING[1] + 1, 2 * RING[1] + 1))


def stats(values: np.ndarray) -> dict:
    return {"median": float(np.median(values)), "n": int(values.size)} if values.size else {"median": np.nan, "n": 0}


rows = []
for f in files:
    labels, areas, _ = select(*raw[f], keep[frames == f])
    flat = contexts[f].flat
    on_disk = flat != OFF_DISK
    dark = 1 - cv2.GaussianBlur(flat.astype(np.float32), (0, 0), SMOOTH) / REL_SCALE
    drawn = np.zeros(flat.shape, bool)  # any annotator's filament
    for crops in truth[f]:
        for x, y, crop in crops:
            drawn[y : y + crop.shape[0], x : x + crop.shape[1]] |= crop
    for annotator, crops in enumerate(truth[f]):
        for i, (x, y, crop) in enumerate(crops):
            h, w = crop.shape
            pad = RING[1] + 1
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + w + pad, flat.shape[1]), min(y + h + pad, flat.shape[0])
            gt = np.zeros((y1 - y0, x1 - x0), np.uint8)
            gt[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = crop
            ring = (cv2.dilate(gt, outer) > 0) & ~(cv2.dilate(gt, inner) > 0)
            ring &= ~drawn[y0:y1, x0:x1] & (labels[y0:y1, x0:x1] == 0) & on_disk[y0:y1, x0:x1]
            inside = (gt > 0) & on_disk[y0:y1, x0:x1]
            if ring.sum() < 30 or inside.sum() < 10:
                continue
            background = dark[y0:y1, x0:x1][ring]
            level = float(np.median(background))
            spread = float(1.4826 * np.median(np.abs(background - level))) or 1e-6
            pred = labels[y0:y1, x0:x1]
            counts = np.bincount(pred[inside], minlength=len(areas) + 1)[1:]
            iou = counts / (inside.sum() + areas - counts) if len(areas) else np.zeros(0)
            matched = bool(len(iou) and iou.max() > IOU_THRESHOLD)
            covered, uncovered = inside & (pred > 0), inside & (pred == 0)
            status = "matched" if matched else ("near_miss" if covered.any() else "missed")
            row = {
                "frame": f,
                "annotator": annotator,
                "gt": i,
                "area": int(crop.sum()),
                "status": status,
                "background": level,
                "spread": spread,
            }
            for part, mask in (("whole", inside), ("covered", covered), ("uncovered", uncovered)):
                values = dark[y0:y1, x0:x1][mask]
                s = stats(values)
                row[f"{part}_n"] = s["n"]
                row[f"{part}_contrast"] = s["median"] - level if s["n"] else np.nan
                row[f"{part}_snr"] = (s["median"] - level) / spread if s["n"] else np.nan
                row[f"{part}_share_2sigma"] = float((values - level > 2 * spread).mean()) if s["n"] else np.nan
            rows.append(row)
contrast = pd.DataFrame(rows)
contrast.to_csv(output_dir() / "contrast_rows.csv", index=False)

# %% [markdown]
# ## Summary
#
# Pixel-weighted: the share of all left-out pixels that are > 2 spreads darker than their background is
# what a contrast-based grower could reach at most.

# %%


def describe(df: pd.DataFrame, part: str) -> dict:
    n = df[f"{part}_n"].to_numpy()
    return {
        "rows": len(df),
        "median_contrast": round(float(df[f"{part}_contrast"].median()), 4),
        "median_snr": round(float(df[f"{part}_snr"].median()), 2),
        "pixel_share_2sigma": round(float((df[f"{part}_share_2sigma"] * n).sum() / max(n.sum(), 1)), 3),
    }


report = {
    "a0": {"cutoff": cutoff, **{k: int(v) for k, v in terms[["tp", "fp", "fn"]].sum().items()}},
    "background_spread_median": round(float(contrast["spread"].median()), 4),
    "matched_whole": describe(contrast[contrast["status"] == "matched"], "whole"),
    "near_miss_covered": describe(contrast[contrast["status"] == "near_miss"], "covered"),
    "near_miss_uncovered": describe(contrast[contrast["status"] == "near_miss"], "uncovered"),
    "missed_whole": describe(contrast[contrast["status"] == "missed"], "whole"),
}
print(json.dumps(report, indent=2))
(output_dir() / "contrast_report.json").write_text(json.dumps(report, indent=2))
log("done")
