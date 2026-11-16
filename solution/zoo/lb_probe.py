# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full
# %% [markdown]
# # Leaderboard probes: what do the test annotations reward that the training ones do not?
#
# CPU only. The submitted `swin_unet_boundary` (full-data model) scores 0.4223 out of fold and 0.37 on the
# leaderboard, and neither the submission file, the test maps nor the site / year / time mix explain it
# (`lb_gap.py`). So the test annotations or the leaderboard's scoring must differ. The same submission is
# written four more times with one change each, and each change's out-of-fold PQ is computed alongside:
#
# - **more**: a lower filter cutoff, about 8.5 kept filaments per frame instead of 6.6;
# - **fewer**: a higher cutoff, about 5 per frame;
# - **thick**: every kept filament grown by 2 px (into background only);
# - **thin**: every kept filament shrunk by 2 px.
#
# A change that loses out of fold but gains on the leaderboard says how the test annotations differ.
# `base` is the submitted file again, as a check.
#
# Outputs: `submission_<variant>.csv`, `lb_probe.json`.

# %%
import json
import time

import numpy as np
import pandas as pd
from scipy import ndimage

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
)
from postprocess.instances import label_stats, select
from postprocess.pipeline import candidates
from postprocess.tuning import best_cutoff
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import PanopticQuality, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
KEPT_PER_FRAME = {"more": 8.5, "fewer": 5.0}
GROW_PX = 2
CROSS = ndimage.generate_binary_structure(2, 1)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
maps = {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
files = sorted(maps)
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
raw = {f: candidates(contexts[f]) for f in files}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
)
del contexts, maps
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
scores = out_of_fold_scores(table, target)
base_cutoff, _ = best_cutoff(scores, raw, table, truth, files)
ranked = np.sort(scores)[::-1]
cutoffs = {"base": base_cutoff} | {v: float(ranked[round(n * len(files)) - 1]) for v, n in KEPT_PER_FRAME.items()}
cutoffs |= {"thick": base_cutoff, "thin": base_cutoff}
log("cutoffs", cutoffs)
filt = ComponentFilter(cutoff=base_cutoff).fit(table, target)


# %%
def grown(labels: np.ndarray, px: int) -> np.ndarray:
    """Each instance grown by `px` px into the background; where two meet, the larger label wins."""
    labels = labels.astype(np.int32)
    for _ in range(px):
        spread = ndimage.grey_dilation(labels, footprint=CROSS)
        labels = np.where(labels == 0, spread, labels)
    return labels


def shrunk(labels: np.ndarray, px: int) -> np.ndarray:
    """Each instance shrunk by `px` px: a pixel stays if its 4 neighbours carry its label."""
    labels = labels.astype(np.int32)
    for _ in range(px):
        low = ndimage.grey_erosion(labels, footprint=CROSS)
        high = ndimage.grey_dilation(labels, footprint=CROSS)
        labels = np.where((low == labels) & (high == labels), labels, 0)
    return labels


def variant(name: str, labels, areas, boxes, frame_scores):
    labels, areas, boxes = select(labels, areas, boxes, frame_scores >= cutoffs[name])
    if name in ("thick", "thin") and len(areas):
        labels = grown(labels, GROW_PX) if name == "thick" else shrunk(labels, GROW_PX)
        areas, boxes = label_stats(labels, len(areas))
        labels, areas, boxes = select(labels, areas, boxes, areas > 0)
    return labels, areas, boxes


frames = table["frame"].to_numpy()
oof = {}
for name in cutoffs:
    pq, kept = PanopticQuality(), 0
    for f in files:
        labels, areas, _ = variant(name, *raw[f], scores[frames == f])
        kept += len(areas)
        for gt in truth[f]:
            pq.update(gt, labels, areas)
    oof[name] = pq.summary() | {"kept_per_frame": round(kept / len(files), 2)}
    log(name, oof[name])
del raw

# %%
test_files = sorted(p.name for p in test_dir.iterdir())
saved = np.load(find_input(f"{NAME}_test_full.npz"))
rows = {name: [] for name in cutoffs}
for f in test_files:
    ctx = frame_context(f, read_gray(test_dir / f), saved[f])
    inst = candidates(ctx)
    frame_scores = filt.scores(component_table(ctx, inst[0], inst[2])) if len(inst[1]) else np.zeros(0)
    for name in cutoffs:
        labels, _, boxes = variant(name, *inst, frame_scores)
        rows[name] += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
test_kept = {}
for name, r in rows.items():
    submission.write(r, output_dir() / f"submission_{name}.csv")
    test_kept[name] = round(len(r) / len(test_files), 2)
log("test kept per frame", test_kept)

report = {"name": NAME, "cutoffs": cutoffs, "oof": oof, "test_kept_per_frame": test_kept}
print(json.dumps(report, indent=2))
(output_dir() / "lb_probe.json").write_text(json.dumps(report, indent=2))
log("done")
