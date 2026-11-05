# kaggle: kernel_sources={user}/filament-zoo-misses
# %% [markdown]
# # Neighbouring days as extra input: are they there, and do they make faint filaments stand out?
#
# CPU only. `cnn_baseline/rotation_lab.py` found that a neighbouring day's *annotations*, moved by differential
# rotation, cover only 28 % of the next day's filaments: a weak prior. A neighbouring day's *image* is a
# different thing: a filament that lasts is in both frames, while the seeing and noise are not, so averaging
# frames moved to the same time should raise a faint filament's contrast against its background. Two checks
# before training a network on it:
#
# 1. **Availability**: for every training and test frame, the nearest other frame (training or test, any
#    site) and how many lie within a day. A network that needs a neighbour must have one at test time.
# 2. **Contrast gain**: for training frames with neighbours within `MAX_DAYS`, the SNR (as in `misses.py`) of
#    every annotated filament in the frame alone and in the mean of the frame and its neighbours moved to its
#    time (`heliographic.rotate_image`), by status of the best model (`miss_rows.csv` of `zoo/misses.py`).
#
# Outputs: `neighbour_gaps.csv`, `stack_rows.csv`, `neighbours_report.json`.

# %%
import json
import time
from datetime import UTC, datetime

import cv2
import numpy as np
import pandas as pd

from cnn_baseline.heliographic import b0_angle, rotate_image
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from preprocess.disk import OFF_DISK, REL_SCALE, disk_geometry, flatten
from preprocess.images import read_gray
from scoring.metric import instance_crops

MAX_DAYS = 1.0  # neighbours used in a stack
MAX_NEIGHBOURS = 4  # nearest first
MAX_FRAMES = 150  # training frames whose stacks are measured
SMOOTH = 2.0  # px, as misses.py
RING = (5, 20)
SEED = 0
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def taken(name: str) -> datetime:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


# %% [markdown]
# ## 1. Availability

# %%
root = find_data_root()
dirs = {"train": root / "train" / "train_images", "test": root / "test" / "test_images"}
split = {p.name: s for s, d in dirs.items() for p in d.iterdir()}
names = sorted(split, key=taken)
times = np.array([taken(f).timestamp() / 86400 for f in names])
gaps = []
for i, f in enumerate(names):
    delta = np.abs(times - times[i])
    delta[i] = np.inf
    order = np.argsort(delta)
    gaps.append(
        {
            "frame": f,
            "split": split[f],
            "nearest_days": round(float(delta[order[0]]), 3),
            "nearest_split": split[names[order[0]]],
            "within_0.5": int((delta <= 0.5).sum()),
            "within_1": int((delta <= 1).sum()),
            "within_2": int((delta <= 2).sum()),
        }
    )
gaps = pd.DataFrame(gaps)
gaps.to_csv(output_dir() / "neighbour_gaps.csv", index=False)
availability = {}
for s, g in gaps.groupby("split"):
    availability[s] = {
        "frames": len(g),
        "nearest_days_quantiles": g["nearest_days"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
        **{f"share_with_one_within_{d}": round(float((g["nearest_days"] <= d).mean()), 3) for d in (0.25, 0.5, 1, 2)},
        "mean_within_1": round(float(g["within_1"].mean()), 2),
        "nearest_is_test": round(float((g["nearest_split"] == "test").mean()), 3),
    }
log(json.dumps(availability))

# %% [markdown]
# ## 2. Contrast in the frame alone and in its stack

# %%
annotations = load_annotations(next((root / "train").glob("*.json")))
misses = pd.read_csv(find_input("miss_rows.csv"))
status = {(r.frame, r.annotator, r.gt): (r.status, r.agreement) for r in misses.itertuples()}
index = {f: i for i, f in enumerate(names)}


def neighbours(f: str) -> list[str]:
    delta = np.abs(times - times[index[f]])
    delta[index[f]] = np.inf
    order = [j for j in np.argsort(delta) if delta[j] <= MAX_DAYS]
    return [names[j] for j in order[:MAX_NEIGHBOURS]]


candidates = [f for f in names if split[f] == "train" and neighbours(f)]
rng = np.random.default_rng(SEED)
picked = sorted(rng.choice(candidates, min(MAX_FRAMES, len(candidates)), replace=False))
log(f"{len(candidates)} training frames with a neighbour within {MAX_DAYS} d; measuring {len(picked)}")

inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING[0] + 1, 2 * RING[0] + 1))
outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING[1] + 1, 2 * RING[1] + 1))


def darkness(flat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1 - smoothed flat / REL_SCALE, the smoothing normalised over valid pixels only."""
    values = np.where(valid, flat, 0).astype(np.float32)
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), SMOOTH)
    return 1 - cv2.GaussianBlur(values, (0, 0), SMOOTH) / np.maximum(weight, 1e-6) / REL_SCALE


def snr(dark, valid, drawn, x, y, crop) -> float:
    h, w = crop.shape
    pad = RING[1] + 1
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, dark.shape[1]), min(y + h + pad, dark.shape[0])
    gt = np.zeros((y1 - y0, x1 - x0), np.uint8)
    gt[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = crop
    ring = (cv2.dilate(gt, outer) > 0) & ~(cv2.dilate(gt, inner) > 0) & ~drawn[y0:y1, x0:x1] & valid[y0:y1, x0:x1]
    inside = (gt > 0) & valid[y0:y1, x0:x1]
    if ring.sum() < 30 or inside.sum() < 10:
        return np.nan
    background = dark[y0:y1, x0:x1][ring]
    level = float(np.median(background))
    spread = float(1.4826 * np.median(np.abs(background - level))) or 1e-6
    return (float(np.median(dark[y0:y1, x0:x1][inside])) - level) / spread


rows = []
for n, f in enumerate(picked):
    image = read_gray(dirs["train"] / f)
    cx, cy, radius = disk_geometry(image)
    flat = flatten(image)
    valid = flat != OFF_DISK
    total, count = np.where(valid, flat, 0).astype(np.float32), valid.astype(np.float32)
    used = []
    for g in neighbours(f):
        other = read_gray(dirs[split[g]] / g)
        gx, gy, gradius = disk_geometry(other)
        if not gradius:
            continue
        days = times[index[f]] - times[index[g]]
        moved = rotate_image(
            flatten(other),
            (cx, cy),
            radius,
            b0_angle(taken(g)),
            b0_angle(taken(f)),
            days,
            0.0,
            ((gx, gy), gradius),
            fill=OFF_DISK,
        )
        ok = moved != OFF_DISK
        total += np.where(ok, moved, 0)
        count += ok
        used.append(round(float(days), 3))
    stack = np.where(count > 0, total / np.maximum(count, 1), OFF_DISK)
    stacked_valid = count > 0
    crops_per_annotator = [instance_crops(rles) for rles in annotations.rles_for_file(f)]
    drawn = np.zeros(flat.shape, bool)
    for x, y, crop in (c for crops in crops_per_annotator for c in crops):
        drawn[y : y + crop.shape[0], x : x + crop.shape[1]] |= crop
    dark_single, dark_stack = darkness(flat, valid), darkness(stack, stacked_valid)
    for a, crops in enumerate(crops_per_annotator):
        for i, (x, y, crop) in enumerate(crops):
            st, agreement = status.get((f, a, i), ("unknown", 0))
            rows.append(
                {
                    "frame": f,
                    "annotator": a,
                    "gt": i,
                    "status": st,
                    "agreement": agreement,
                    "area": int(crop.sum()),
                    "neighbours": len(used),
                    "days": used,
                    "snr_single": snr(dark_single, valid, drawn, x, y, crop),
                    "snr_stack": snr(dark_stack, stacked_valid, drawn, x, y, crop),
                }
            )
    if n % 25 == 0:
        log(f"{n}/{len(picked)} frames")
stack_rows = pd.DataFrame(rows)
stack_rows.to_csv(output_dir() / "stack_rows.csv", index=False)

# %% [markdown]
# ## Report
#
# `gain`: median of snr_stack / snr_single per filament (> 1: the stack shows it better). A wrong registration
# would smear filaments and show as gain < 1 on the matched ones, which are clearly visible in a single frame.

# %%
valid_rows = stack_rows.dropna(subset=["snr_single", "snr_stack"])
valid_rows = valid_rows[valid_rows["snr_single"] > 0]
by_status = {}
for st, g in valid_rows.groupby("status"):
    ratio = g["snr_stack"] / g["snr_single"]
    by_status[st] = {
        "rows": len(g),
        "snr_single_median": round(float(g["snr_single"].median()), 2),
        "snr_stack_median": round(float(g["snr_stack"].median()), 2),
        "gain_median": round(float(ratio.median()), 3),
        "share_gain_gt_1": round(float((ratio > 1).mean()), 3),
        "share_single_lt_1_becoming_ge_2": round(
            float(((g["snr_single"] < 1) & (g["snr_stack"] >= 2)).sum() / max(int((g["snr_single"] < 1).sum()), 1)), 3
        ),
    }
by_neighbours = {
    int(k): round(float((g["snr_stack"] / g["snr_single"]).median()), 3) for k, g in valid_rows.groupby("neighbours")
}
report = {"availability": availability, "stack": {"frames": len(picked), "by_status": by_status}}
report["stack"]["gain_by_neighbours"] = by_neighbours
print(json.dumps(report, indent=2))
(output_dir() / "neighbours_report.json").write_text(json.dumps(report, indent=2))
log("done")
