# kaggle: gpu=true kernel_sources={user}/filament-cnn-baseline
# %% [markdown]
# # CNN baseline: better post-processing on the saved U-Net
#
# The trained network is taken from `filament-cnn-baseline` (attached as a notebook source); nothing
# is retrained. Two things the error analysis pointed at are changed after the network:
#
# 1. **Hysteresis threshold.** Near misses covered only half of their filament and fragments split
#    where the filament fades. A component of `prob >= low` is now kept whole when any pixel of it
#    reaches the high threshold, so faint tails stay attached to confident cores.
# 2. **A component filter instead of the minimum-area cut.** The 600 px cut dropped a fifth of the
#    real filaments to remove hallucinations. A small classifier on area, elongation, darkness,
#    distance from the disk centre and the network's confidence takes that decision instead, trained
#    on the validation frames with out-of-fold scoring.
#
# The configuration with the best validation PQ is applied to the test frames → `submission.csv`,
# followed by the error analysis of that configuration.

# %%
import json
import time

import numpy as np
import pandas as pd
import torch

import error_analysis.run
from cnn_baseline.data import channels, split_files
from cnn_baseline.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
    trainable,
)
from cnn_baseline.postprocess import instances, select
from cnn_baseline.training import pick_device, predict
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from models.unet import UNet
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import PanopticQuality, instance_crops

SEED = 0  # must match cnn_baseline/main.py so the validation frames are the ones the model never saw
VALIDATION_FRACTION = 0.15
HIGHS = (0.5, 0.7, 0.9)
LOWS = (None, 0.1, 0.2, 0.3)
MIN_AREAS = (150, 300, 600)
CUTOFFS = np.round(np.arange(0.2, 0.81, 0.05), 2)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def score(predictions: dict, files: list[str]) -> dict:
    pq = PanopticQuality()
    for f in files:
        labels, areas, _ = predictions[f]
        for gt_crops in truth[f]:
            pq.update(gt_crops, labels, areas)
    return pq.summary()


# %% [markdown]
# ## Model, validation and test frames

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
_, val_files = split_files([p.name for p in train_dir.iterdir()], VALIDATION_FRACTION, SEED)
test_files = sorted(p.name for p in test_dir.iterdir())

device = pick_device()
model = UNet(in_channels=2, base=32, depth=4)
model.load_state_dict(torch.load(find_input("unet.pt"), map_location="cpu"))
images = {f: read_gray(train_dir / f) for f in val_files}
images.update({f: read_gray(test_dir / f) for f in test_files})
probs = dict(
    zip(
        val_files + test_files,
        predict(model, (torch.from_numpy(channels(images[f])) for f in val_files + test_files), device),
    )
)
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}
log(f"{len(val_files)} validation and {len(test_files)} test frames predicted on {device}")
# kept on Kaggle (not fetched: only csv/json/txt come back) so CPU notebooks can retry post-processing,
# and with the test maps also write submissions without a GPU
np.savez_compressed(output_dir() / "val_probs.npz", **{f: probs[f] for f in val_files})
np.savez_compressed(output_dir() / "test_probs.npz", **{f: probs[f] for f in test_files})

# %% [markdown]
# ## Stage A: hysteresis threshold and minimum area, chosen on validation PQ

# %%
sweep = []
for high in HIGHS:
    for low in LOWS:
        for min_area in MIN_AREAS:
            params = {"threshold": high, "min_area": min_area, "low": low}
            row = {**params, **score({f: instances(probs[f], **params) for f in val_files}, val_files)}
            sweep.append(row)
            log(row)
best_a = max(sweep, key=lambda r: r["pq"])
params_a = {k: best_a[k] for k in ("threshold", "min_area", "low")}
log("stage A best:", best_a)
print(
    pd.DataFrame(sweep)
    .pivot_table(index=["threshold", "low"], columns="min_area", values="pq", dropna=False)
    .to_string()
)

# %% [markdown]
# ## Stage B: the component filter
#
# Components come from the best hysteresis setting with a low minimum area, so that the filter, not
# the area, decides. Its target is "matched a filament of at least one annotator". Out-of-fold scores
# (5 folds by frame) choose the cutoff and give an honest validation PQ.

# %%
contexts = {f: frame_context(f, images[f], probs[f]) for f in val_files}
stage_b = {}
for min_area in (150, 300):
    params_b = {**params_a, "min_area": min_area}
    raw = {f: instances(probs[f], **params_b) for f in val_files}
    table = pd.concat([component_table(contexts[f], raw[f][0], raw[f][2]) for f in val_files], ignore_index=True)
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in val_files])
    if not trainable(target):
        log(f"min_area {min_area}: {len(target)} components, filter not trainable")
        continue
    oof = out_of_fold_scores(table, target)
    rows = []
    for cutoff in CUTOFFS:
        keep = oof >= cutoff
        filtered = {}
        for f in val_files:
            mask = (table["frame"] == f).to_numpy()
            filtered[f] = select(*raw[f], keep[mask])
        rows.append(
            {
                "min_area": min_area,
                "cutoff": float(cutoff),
                "kept": int(keep.sum()),
                "of": len(keep),
                **score(filtered, val_files),
            }
        )
    best = max(rows, key=lambda r: r["pq"])
    stage_b[min_area] = {"best": best, "rows": rows, "table": table, "target": target, "raw": raw}
    log(f"stage B min_area {min_area}: {len(target)} components, {int(target.sum())} positive, best {best}")
    print(pd.DataFrame(rows)[["cutoff", "kept", "pq", "sq", "rq", "tp", "fp", "fn"]].to_string(index=False))

# %% [markdown]
# ## Choice and test predictions

# %%
report = {
    "stage_a": {"best": best_a, "sweep": sweep},
    "stage_b": {str(k): {"best": v["best"], "rows": v["rows"]} for k, v in stage_b.items()},
}
best_b_area = max(stage_b, key=lambda k: stage_b[k]["best"]["pq"]) if stage_b else None
use_filter = best_b_area is not None and stage_b[best_b_area]["best"]["pq"] > best_a["pq"]
if use_filter:
    chosen = stage_b[best_b_area]
    params = {**params_a, "min_area": best_b_area}
    component_filter = ComponentFilter(cutoff=chosen["best"]["cutoff"]).fit(chosen["table"], chosen["target"])
    log(f"using the component filter: {params}, cutoff {component_filter.cutoff}, validation PQ {chosen['best']['pq']}")
else:
    params, component_filter = params_a, None
    log(f"using stage A only: {params}, validation PQ {best_a['pq']}")
report["chosen"] = {"params": params, "filter_cutoff": component_filter.cutoff if component_filter else None}


def postprocess(frame: str):
    labels, areas, boxes = instances(probs[frame], **params)
    if component_filter is not None and len(areas):
        table = component_table(frame_context(frame, images[frame], probs[frame]), labels, boxes)
        labels, areas, boxes = select(labels, areas, boxes, component_filter.keep(table))
    return labels, areas, boxes


rows = []
for f in test_files:
    labels, _, boxes = postprocess(f)
    rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
frame = submission.write(rows, output_dir() / "submission.csv")
report["n_test_predictions"] = len(frame)
report["n_test_images_with_predictions"] = int(frame["filament_id"].str.split("_").str[0].nunique())
log(f"wrote submission.csv with {len(frame)} rows")

# %% [markdown]
# ## Error analysis of the chosen post-processing
#
# The filter's out-of-fold decisions are used on the validation frames, so this matches the honest PQ above.

# %%
if use_filter:
    keep = out_of_fold_scores(chosen["table"], chosen["target"]) >= component_filter.cutoff
    val_predictions = {
        f: select(*chosen["raw"][f], keep[(chosen["table"]["frame"] == f).to_numpy()]) for f in val_files
    }
else:
    val_predictions = {f: instances(probs[f], **params) for f in val_files}
analysis = error_analysis.run.run(
    annotations, train_dir, val_files, probs, None, params, output_dir(), log=log, predictions=val_predictions
)
report["validation"] = analysis["terms"]
report["elapsed_sec"] = round(time.time() - T0, 1)
(output_dir() / "predict_report.json").write_text(json.dumps(report, indent=2, default=float))
