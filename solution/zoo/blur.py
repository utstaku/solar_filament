# kaggle: kernel_sources={user}/filament-zoo-baseline,{user}/filament-zoo-boundary-bce,{user}/filament-zoo-swin-unet,{user}/filament-zoo-swin-unet-boundary
# %% [markdown]
# # How blurred are the out-of-fold maps, and does it cost instances?
#
# The networks are trained on the mean of the annotators' masks, so where they disagree the target is 1/3,
# 1/2 or 2/3 and BCE's best answer is that fraction: the maps are soft by design. That only costs PQ when it
# moves the instances cut at 0.5 / 0.3: neighbouring filaments joined by a low bridge (**merges**), or one
# filament cut into pieces (**splits**). CPU only, on the 707 out-of-fold maps of four zoo experiments:
#
# - **calibration**: the mean probability on pixels marked by none / some / all of the annotators;
# - **ambiguous share**: of the pixels at p >= 0.3 (what A0's hysteresis grows into), those below 0.7;
# - **merges / splits** of A0's instances (no join, no filter) against each annotator's filaments: a
#   prediction covering >= 30 % of two or more filaments, a filament >= 20 % covered by two or more
#   predictions; with how many of those filaments end up unmatched (false negatives).
#
# Outputs: `blur_summary.csv`, `blur_calibration.csv`.

# %%
import json
import time

import numpy as np
import pandas as pd

from cnn_baseline.data import annotator_masks
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.instances import instances
from postprocess.pipeline import A0
from scoring.metric import IOU_THRESHOLD, instance_crops
from zoo.experiment import N_FOLDS

EXPERIMENTS = ["baseline", "boundary_bce", "swin_unet", "swin_unet_boundary"]
MERGE_COVER = 0.3  # a prediction "contains" a filament when it covers this share of it
SPLIT_COVER = 0.2  # a prediction "is a piece of" a filament when it covers this share of it
LEVELS = 7  # soft target rounded to sixths: 0, 1/3, 1/2, 2/3, 1 fall on 0, 2, 3, 4, 6
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


root = find_data_root()
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("baseline_folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)


def frame_errors(prob: np.ndarray, per_annotator: list[list[dict]]) -> dict:
    labels, areas, _ = instances(prob, **A0)
    n = len(areas)
    out = {"filaments": 0, "merges": 0, "merged": 0, "merged_fn": 0, "splits": 0, "split_fn": 0, "fn": 0}
    for rles in per_annotator:
        crops = instance_crops(rles)
        out["filaments"] += len(crops)
        if not crops:
            continue
        if n == 0:
            out["fn"] += len(crops)
            continue
        cover = np.zeros((len(crops), n))  # share of filament g covered by prediction k
        matched = np.zeros(len(crops), bool)
        for g, (x, y, crop) in enumerate(crops):
            h, w = crop.shape
            inter = np.bincount(labels[y : y + h, x : x + w][crop], minlength=n + 1)[1:]
            cover[g] = inter / crop.sum()
            matched[g] = (inter / (crop.sum() + areas - inter)).max() > IOU_THRESHOLD
        out["fn"] += int((~matched).sum())
        contains = cover >= MERGE_COVER
        merging = contains.sum(axis=0) >= 2  # predictions that contain two or more filaments
        in_merge = contains[:, merging].any(axis=1)
        out["merges"] += int(merging.sum())
        out["merged"] += int(in_merge.sum())
        out["merged_fn"] += int((in_merge & ~matched).sum())
        split = (cover >= SPLIT_COVER).sum(axis=1) >= 2
        out["splits"] += int(split.sum())
        out["split_fn"] += int((split & ~matched).sum())
    return out


# %% [markdown]
# ## Each experiment over the 707 frames

# %%
summary, calibration = [], []
for name in EXPERIMENTS:
    joint = np.zeros(LEVELS * 256, np.int64)  # (soft target level, probability) pixel counts
    errors = {}
    for k in range(N_FOLDS):
        saved = np.load(find_input(f"{name}_oof_fold{k}.npz"))
        for f in saved.files:
            prob = saved[f]
            level = np.rint(annotator_masks(annotations, f).mean(axis=0) * 6).astype(np.int64)
            joint += np.bincount((level * 256 + prob).ravel(), minlength=LEVELS * 256)
            for key, value in frame_errors(prob, annotations.rles_for_file(f)).items():
                errors[key] = errors.get(key, 0) + value
    joint = joint.reshape(LEVELS, 256)
    p = np.arange(256) / 255
    for level in np.flatnonzero(joint.sum(axis=1)):
        counts = joint[level]
        calibration.append(
            {
                "name": name,
                "target": round(level / 6, 3),
                "pixels": int(counts.sum()),
                "mean_prob": round(float((counts * p).sum() / counts.sum()), 4),
                "share_ge_0.5": round(float(counts[128:].sum() / counts.sum()), 4),
            }
        )
    total = joint.sum(axis=0)
    above = total[77:].sum()  # p >= 0.3
    summary.append(
        {
            "name": name,
            "ambiguous_share": round(float(total[77:179].sum() / above), 4),  # 0.3 <= p < 0.7 among p >= 0.3
            "pixels_ge_0.3": int(above),
            **errors,
            "merged_fn_share_of_fn": round(errors["merged_fn"] / errors["fn"], 4) if errors["fn"] else 0.0,
            "split_fn_share_of_fn": round(errors["split_fn"] / errors["fn"], 4) if errors["fn"] else 0.0,
        }
    )
    log(name, summary[-1])

# %% [markdown]
# ## Results
#
# `merged_fn_share_of_fn`: of all unmatched filaments, the share that sit inside a merge, an upper bound on
# what separating merged filaments could recover (some would still miss IoU 0.5). Same for splits.

# %%
summary = pd.DataFrame(summary)
calibration = pd.DataFrame(calibration)
print(summary.to_string(index=False))
print(calibration.to_string(index=False))
summary.to_csv(output_dir() / "blur_summary.csv", index=False)
calibration.to_csv(output_dir() / "blur_calibration.csv", index=False)
log("done")
