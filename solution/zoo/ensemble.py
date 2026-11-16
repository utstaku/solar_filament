# kaggle: kernel_sources={user}/filament-zoo-swin-unet,{user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-skeleton-majority,{user}/filament-zoo-boundary-bce,{user}/filament-zoo-hrnet,{user}/filament-zoo-swin-unet-boundary-seed1
# %% [markdown]
# # Averaging the maps of several zoo runs: does an ensemble beat the best single model?
#
# CPU only. The annotators disagree on where a filament ends, and a single network's cut at 0.5 follows one
# reading; the mean of networks trained differently should be steadier where they disagree. For each set of
# runs, the mean of their out-of-fold maps (the folds are the same in every run, so each frame's maps all come
# from models that never saw it) goes through the submitted post-processing, fitted out of fold
# (`pipeline.fit`), as `compare.py` does for single runs. The test maps of a submission would be the mean of
# the same runs' full-data models, so the filter sees maps alike in both.
#
# Sets with a run whose maps are not attached are skipped. `delta`: PQ minus `swin_unet_boundary`'s on the
# same frames, with a 95 % interval from resampling frames.
#
# Outputs: `ensemble_comparison.csv`.

# %%
import time

import numpy as np
import pandas as pd

from cnn_baseline.importance import paired_bootstrap
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from error_analysis.instances import FrameContext
from postprocess import pipeline
from postprocess.filtering import frame_context
from postprocess.tuning import pq_of
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

REFERENCE = "swin_unet_boundary"
SETS = [
    [REFERENCE],
    ["swin_unet", "swin_unet_boundary"],
    ["swin_unet", "swin_unet_boundary", "swin_unet_skeleton_majority"],
    ["swin_unet", "swin_unet_boundary", "boundary_bce"],
    ["swin_unet", "swin_unet_boundary", "hrnet"],
    ["swin_unet_boundary", "swin_unet_boundary_seed1"],
    ["swin_unet", "swin_unet_boundary", "swin_unet_boundary_seed1"],
]
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def summary(terms: pd.DataFrame) -> dict:
    tp, fp, fn = (int(terms[c].sum()) for c in ["tp", "fp", "fn"])
    return {
        "pq": round(pq_of(terms), 4),
        "sq": round(float(terms["iou_sum"].sum()) / max(tp, 1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def opened(name: str) -> dict | None:
    """frame -> lazy array of the run's out-of-fold map, or None when the run is not attached."""
    try:
        saved = [np.load(find_input(f"{name}_oof_fold{k}.npz")) for k in range(N_FOLDS)]
    except FileNotFoundError:
        return None
    return {f: npz for npz in saved for f in npz.files}


root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
runs = {name: opened(name) for name in {n for s in SETS for n in s}}
log("attached:", sorted(n for n, r in runs.items() if r), "missing:", sorted(n for n, r in runs.items() if not r))
files = sorted(runs[REFERENCE])
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
base = {f: frame_context(f, read_gray(train_dir / f), np.zeros(1, np.uint8)) for f in files}

# %%
rows, terms = [], {}
for members in SETS:
    if any(runs[n] is None for n in members):
        continue
    name = "+".join(members)
    contexts = {}
    for f in files:
        mean = sum(runs[n][f][f].astype(np.uint16) for n in members) / len(members)
        contexts[f] = FrameContext(f, base[f].flat, base[f].centre, base[f].radius, np.rint(mean).astype(np.uint8))
    _, oof = pipeline.fit(contexts, truth, files)
    terms[name] = oof.terms
    rows.append(
        {
            "name": name,
            "models": len(members),
            **summary(oof.terms),
            "cutoff": oof.cutoff,
            "kept_per_frame": round(oof.kept_per_frame, 2),
        }
    )
    log(rows[-1])
    del contexts

for row in rows:
    delta, lo, hi = paired_bootstrap(terms[REFERENCE], terms[row["name"]])
    row["delta"] = round(delta, 4)
    row["ci95"] = [round(lo, 4), round(hi, 4)]
table = pd.DataFrame(rows).sort_values("pq", ascending=False)
print(table.to_string(index=False))
table.to_csv(output_dir() / "ensemble_comparison.csv", index=False)
log("done")
