# kaggle: gpu=true kernel_sources={user}/filament-cnn-baseline
# %% [markdown]
# # Where the CNN baseline loses Panoptic Quality
#
# Loads the U-Net trained by `filament-cnn-baseline` (attached as a notebook source), predicts the same
# validation frames, and classifies every ground-truth filament and every prediction: matched, near
# miss, fragmented, merged, missed / hallucination. Then: what-if bounds, breakdowns by size, contrast,
# position, site and annotator agreement, how separable hallucinations are by confidence, and galleries
# of the worst cases.
#
# Outputs: `errors_gt.csv`, `errors_pred.csv`, `error_report.json`, `breakdowns.txt`, `gallery_*.png`.
# The same analysis runs at the end of every training notebook; this one repeats it on saved weights.

# %%
import json
import time

import pandas as pd
import torch

import error_analysis.run
from cnn_baseline.data import channels, split_files
from cnn_baseline.training import pick_device, predict
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from models.unet import UNet
from postprocess.instances import instances, on_disk
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops

SEED = 0  # must match cnn_baseline/main.py so the validation frames are the ones the model never saw
VALIDATION_FRACTION = 0.15
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_AREAS = (100, 200, 400, 600, 800)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Model and validation frames

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
_, val_files = split_files([p.name for p in train_dir.iterdir()], VALIDATION_FRACTION, SEED)

device = pick_device()
model = UNet(in_channels=2, base=32, depth=4)
weights = find_input("unet.pt")
model.load_state_dict(torch.load(weights, map_location="cpu"))
log(f"loaded {weights}; {len(val_files)} validation frames; device {device}")

images = {f: read_gray(train_dir / f) for f in val_files}
probs = {
    f: on_disk(prob, images[f])
    for f, prob in zip(val_files, predict(model, (torch.from_numpy(channels(images[f])) for f in val_files), device))
}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}
log("predicted")

# %% [markdown]
# ## Post-processing chosen on PQ, as in the baseline (wider grid)

# %%
sweep = []
for threshold in THRESHOLDS:
    for min_area in MIN_AREAS:
        pq = PanopticQuality()
        for f in val_files:
            labels, areas, _ = instances(probs[f], threshold, min_area)
            for gt_crops in truth[f]:
                pq.update(gt_crops, labels, areas)
        sweep.append({"threshold": threshold, "min_area": min_area, **pq.summary()})
best = max(sweep, key=lambda row: row["pq"])
params = {"threshold": best["threshold"], "min_area": best["min_area"]}
log("best:", best)
print(pd.DataFrame(sweep).pivot(index="threshold", columns="min_area", values="pq").to_string())

# %% [markdown]
# ## Per-instance tables, report, breakdowns and the worst cases
#
# Ground truth of the annotator in question in green, predictions in red.

# %%
report = error_analysis.run.run(annotations, train_dir, val_files, probs, instances, params, output_dir(), log=log)
report["sweep"] = sweep
(output_dir() / "error_report.json").write_text(json.dumps(report, indent=2))
