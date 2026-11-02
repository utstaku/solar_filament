# kaggle: gpu=false
# %% [markdown]
# # Rotation lab: do annotated filaments of a neighbouring day land on today's filaments?
#
# CPU only, competition data only. Before a neighbouring day's annotations can serve as a prior for a frame,
# the transform that moves them (`cnn_baseline/heliographic.py`) has to be right, including the image
# orientation, which `rotation_check.py` could not settle from the image texture.
#
# For pairs of annotated training frames up to 3 days apart, the earlier frame's filaments (all annotators
# together) are moved to the later frame's time in several ways and compared with the later frame's filaments:
# - **none**: not rotated (only mapped between the two frames' disks);
# - **rigid**: everything turns at the equator's 13.4 deg / day;
# - **differential, p = 0**: Snodgrass rates, solar north up, east on the left (the working assumption);
# - **differential, p = 90 / 180 / 270** and **reversed** (east on the right): other orientations.
# The right transform should overlap best, and its advantage should grow with the time between the frames.
#
# Measures per pair: IoU of the two unions, and the share of the later frame's filaments (per annotator)
# covered at least half by the moved ones -- what a prior could contribute to PQ.
#
# Outputs: `rotation_pairs_iou.csv`, `rotation_lab_report.json`.

# %%
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cnn_baseline.heliographic import b0_angle, rotate_mask, synodic_rate
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from preprocess.disk import disk_geometry
from preprocess.images import read_gray
from scoring.metric import instance_crops

MAX_DAYS = 3.0
MAX_PAIRS = 400
SEED = 0
VARIANTS = {
    "none": {"days_factor": 0.0, "p": 0.0, "rate": None},
    "rigid": {"days_factor": 1.0, "p": 0.0, "rate": float(synodic_rate(0.0))},
    "differential_p0": {"days_factor": 1.0, "p": 0.0, "rate": None},
    "differential_p90": {"days_factor": 1.0, "p": 90.0, "rate": None},
    "differential_p180": {"days_factor": 1.0, "p": 180.0, "rate": None},
    "differential_p270": {"days_factor": 1.0, "p": 270.0, "rate": None},
    "reversed": {"days_factor": -1.0, "p": 0.0, "rate": None},
}
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def taken(name: str) -> datetime:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


# %% [markdown]
# ## Pairs of annotated frames up to 3 days apart

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
names = sorted(p.name for p in train_dir.iterdir())
times = {f: taken(f) for f in names}
ordered = sorted(names, key=times.get)
pairs = []
for i, a in enumerate(ordered):
    for b in ordered[i + 1 :]:
        days = (times[b] - times[a]).total_seconds() / 86400
        if days > MAX_DAYS:
            break
        if days > 0:
            pairs.append((a, b, days))
rng = np.random.default_rng(SEED)
if len(pairs) > MAX_PAIRS:
    pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), MAX_PAIRS, replace=False))]
log(f"{len(names)} annotated frames, {len(pairs)} pairs within {MAX_DAYS} days")

# %% [markdown]
# ## Moving the earlier frame's filaments

# %%
cache = {}


def frame(name: str):
    """(disk centre, radius, union of all annotators' filaments, their crops per annotator), cached."""
    if name not in cache:
        cx, cy, radius = disk_geometry(read_gray(train_dir / name))
        crops = [instance_crops(rles) for rles in annotations.rles_for_file(name)]
        union = np.zeros((2048, 2048), bool)
        for x, y, crop in (c for per_annotator in crops for c in per_annotator):
            union[y : y + crop.shape[0], x : x + crop.shape[1]] |= crop
        cache[name] = ((cx, cy), radius, union, crops)
    return cache[name]


def covered_share(moved: np.ndarray, crops_per_annotator) -> float:
    """Share of the filaments (all annotators) that `moved` covers at least half."""
    shares = [
        (moved[y : y + crop.shape[0], x : x + crop.shape[1]] & crop).sum() / crop.sum() >= 0.5
        for crops in crops_per_annotator
        for x, y, crop in crops
        if crop.sum()
    ]
    return float(np.mean(shares)) if shares else np.nan


rows = []
for n, (a, b, days) in enumerate(pairs):
    centre_a, radius_a, union_a, _ = frame(a)
    centre_b, radius_b, union_b, crops_b = frame(b)
    b0_a, b0_b = b0_angle(times[a]), b0_angle(times[b])
    row = {"a": a, "b": b, "days": days, "sites": a[14:16] + b[14:16], "b0_a": b0_a, "b0_b": b0_b}
    for name, v in VARIANTS.items():
        moved = rotate_mask(
            union_a, centre_b, radius_b, b0_a, b0_b, days * v["days_factor"], v["p"], (centre_a, radius_a), v["rate"]
        )
        inter, union = (moved & union_b).sum(), (moved | union_b).sum()
        row[f"iou_{name}"] = float(inter / union) if union else np.nan
        row[f"covered_{name}"] = covered_share(moved, crops_b)
    rows.append(row)
    if len(cache) > 200:
        cache.clear()
    if n % 50 == 0:
        log(f"{n}/{len(pairs)} pairs")
result = pd.DataFrame(rows)
result.to_csv(output_dir() / "rotation_pairs_iou.csv", index=False)

# %% [markdown]
# ## Report by time between the frames

# %%
result["gap"] = pd.cut(result["days"], [0, 0.5, 1.5, 3.0]).astype(str)
report = {"pairs": len(result)}
for gap, g in result.groupby("gap"):
    report[gap] = {
        "pairs": len(g),
        "median_iou": {v: round(float(g[f"iou_{v}"].median()), 3) for v in VARIANTS},
        "mean_covered": {v: round(float(g[f"covered_{v}"].mean()), 3) for v in VARIANTS},
        "best_iou": max(VARIANTS, key=lambda v: g[f"iou_{v}"].median()),
        "differential_p0_beats_none": round(float((g["iou_differential_p0"] > g["iou_none"]).mean()), 3),
        "differential_p0_beats_rigid": round(float((g["iou_differential_p0"] > g["iou_rigid"]).mean()), 3),
    }
print(json.dumps(report, indent=2))
(output_dir() / "rotation_lab_report.json").write_text(json.dumps(report, indent=2))
log("done")
