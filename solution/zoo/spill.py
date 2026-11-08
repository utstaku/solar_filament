# kaggle: kernel_sources={user}/filament-zoo-swin-unet,{user}/filament-zoo-shapes
# %% [markdown]
# # Where do the too-big candidates spill, and how often do an annotator's filaments touch?
#
# CPU only. `shapes.py` found that a third of the unmatched filaments two or more annotators drew are covered
# by a candidate that runs well past them (`too_big`), mostly along their length. Two ways to read that call
# for different models:
#
# - the spill lies on **another filament of the same annotator**: the network joins two filaments the
#   annotator kept apart, and a "separation" target (the gap between an annotator's touching filaments) can
#   teach it where to cut;
# - the spill lies on **another annotator's filament only** or on **unannotated** pixels: the network draws
#   a filament longer than this annotator did, a disagreement in extent that a separation target cannot see.
#
# Per too-big filament, the share of its candidate's spill (candidate minus the filament) in each place, and
# its dominant place. Then how much separation target the training set holds: per annotator, the pairs of
# filaments within `GAPS` px of each other and the pixels between them.
#
# Outputs: `spill_rows.csv`, `spill_report.json`.

# %%
import json
import time

import cv2
import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import frame_context
from postprocess.pipeline import candidates
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet"
GAPS = (3, 5, 10, 20)  # px between two filaments of one annotator
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
shapes = pd.read_csv(find_input("shape_rows.csv"))
too_big = shapes[shapes["shape"] == "too_big"]
maps = {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
files = sorted(maps)
log(f"{len(files)} frames, {len(too_big)} too-big filaments")


def label_map(crops, shape) -> np.ndarray:
    labels = np.zeros(shape, np.int32)
    for i, (x, y, crop) in enumerate(crops, start=1):
        labels[y : y + crop.shape[0], x : x + crop.shape[1]][crop] = i
    return labels


# %% [markdown]
# ## Spill of the too-big candidates

# %%
rows = []
for f, group in too_big.groupby("frame"):
    ctx = frame_context(f, read_gray(train_dir / f), maps[f])
    labels = candidates(ctx)[0]
    per_annotator = [label_map(instance_crops(rles), labels.shape) for rles in annotations.rles_for_file(f)]
    for row in group.itertuples():
        own = per_annotator[row.annotator]
        spill = (labels == row.best) & (own != row.gt + 1)
        on_own = spill & (own > 0)
        others = np.zeros(labels.shape, bool)
        for b, other in enumerate(per_annotator):
            if b != row.annotator:
                others |= other > 0
        on_other = spill & ~on_own & others
        total = max(int(spill.sum()), 1)
        shares = {
            "own_other_filament": on_own.sum() / total,
            "other_annotator_only": on_other.sum() / total,
            "unannotated": (spill & ~on_own & ~on_other).sum() / total,
        }
        rows.append(
            {
                "frame": f,
                "annotator": row.annotator,
                "gt": row.gt,
                "agreement": row.agreement,
                "area": row.area,
                "spill_px": int(spill.sum()),
                "own_filaments_touched": len(np.unique(own[on_own])),
                **{k: round(float(v), 3) for k, v in shares.items()},
                "dominant": max(shares, key=shares.get),
            }
        )
spill_rows = pd.DataFrame(rows)
spill_rows.to_csv(output_dir() / "spill_rows.csv", index=False)
log("spill done")

# %% [markdown]
# ## How much separation target there is: an annotator's filaments close to each other

# %%
pairs = {g: 0 for g in GAPS}
gap_pixels = {g: 0 for g in GAPS}
frames_with_pair = {g: 0 for g in GAPS}
filaments = 0
for f in files:
    has_pair = {g: False for g in GAPS}
    for rles in annotations.rles_for_file(f):
        crops = instance_crops(rles)
        filaments += len(crops)
        if len(crops) < 2:
            continue
        shape = maps[f].shape
        drawn = label_map(crops, shape) > 0
        for g in GAPS:
            # every filament grown by g/2 in its own padded box, so two filaments g px apart overlap
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g + 1, g + 1))
            covered = np.zeros(shape, np.uint8)
            boxes = []
            for x, y, crop in crops:
                x0, y0 = max(x - g, 0), max(y - g, 0)
                x1, y1 = min(x + crop.shape[1] + g, shape[1]), min(y + crop.shape[0] + g, shape[0])
                local = np.zeros((y1 - y0, x1 - x0), np.uint8)
                local[y - y0 : y - y0 + crop.shape[0], x - x0 : x - x0 + crop.shape[1]] = crop
                grown = cv2.dilate(local, kernel) > 0
                covered[y0:y1, x0:x1] += grown
                boxes.append((x0, y0, x1, y1, grown))
            gap_pixels[g] += int(((covered >= 2) & ~drawn).sum())
            for i, (ax0, ay0, ax1, ay1, a) in enumerate(boxes):
                for bx0, by0, bx1, by1, b in boxes[i + 1 :]:
                    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
                    if (
                        x0 < x1
                        and y0 < y1
                        and (
                            a[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0] & b[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
                        ).any()
                    ):
                        pairs[g] += 1
                        has_pair[g] = True
    for g in GAPS:
        frames_with_pair[g] += has_pair[g]
log("pairs done")

# %% [markdown]
# ## Report

# %%


def spill_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "dominant": df["dominant"].value_counts(normalize=True).round(3).to_dict(),
        "mean_share": df[["own_other_filament", "other_annotator_only", "unannotated"]].mean().round(3).to_dict(),
        "share_touching_own_filament_ge_30pct": round(float((df["own_other_filament"] >= 0.3).mean()), 3),
        "median_spill_px": float(df["spill_px"].median()),
    }


report = {
    "name": NAME,
    "too_big": spill_summary(spill_rows),
    "too_big_agreement_ge_2": spill_summary(spill_rows[spill_rows["agreement"] >= 2]),
    "separation": {
        "filaments": filaments,
        **{f"gap_{g}px": {"pairs": pairs[g], "pixels": gap_pixels[g], "frames": frames_with_pair[g]} for g in GAPS},
    },
}
print(json.dumps(report, indent=2))
(output_dir() / "spill_report.json").write_text(json.dumps(report, indent=2))
log("done")
