# kaggle: gpu=true
# %% [markdown]
# # CNN baseline, 5-fold: out-of-fold probability maps for all 707 training frames
#
# The baseline U-Net (`main.py`, same tiles, loss, schedule and epochs) is trained five times, each time
# without one fifth of the physical frames, and predicts the frames it did not see. Post-processing can
# then be compared on 707 frames instead of 106: paired intervals about 2.6 times narrower.
#
# Every frame is predicted both as it is and averaged over the eight rotations / mirror images (D4
# test-time augmentation), from the same forward passes, so the effect of TTA is measured on the same frames.
#
# Outputs:
# - `oof_plain_fold{k}.npz`, `oof_tta_fold{k}.npz`: uint8 probability * 255 of the frames of fold k (kept on
#   Kaggle; npz files are not fetched), `folds.json`: which frames are in which fold;
# - `test_probs_plain.npz`, `test_probs_tta.npz`: the mean of the five fold models on the 180 test frames;
# - `unet_fold{k}.pt`, `cv_report.json`: losses, timings and a quick PQ of each fold, plain vs TTA, with
#   A0's hysteresis and minimum area but no component filter.

# %%
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, fold_files, worker_init
from cnn_baseline.postprocess import instances
from cnn_baseline.training import fit, pick_device, predict_tta
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from models.unet import UNet
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops

QUICK = bool(os.environ.get("FILAMENT_QUICK"))  # a few frames and epochs, to check that the notebook runs
SEED = 0
N_FOLDS = 5
TILE = 256 if QUICK else 512  # as in main.py; quick runs use a smaller network on smaller tiles
TILES_PER_FRAME = 4
FRAMES_PER_BATCH = 2 if QUICK else 4
BASE = 8 if QUICK else 32
EPOCHS = 1 if QUICK else 16
LR = 1e-3
WORKERS = 0 if QUICK else 3  # macOS starts workers by re-running this script; quick runs are local
TTA_VIEWS = 8
QUICK_PQ = {"threshold": 0.5, "min_area": 300, "low": 0.3}  # A0's instances, without its filter
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


device = pick_device()
log("device:", device)

# %% [markdown]
# ## Data and folds
#
# Folds are disjoint sets of physical frames, so the up-to-three annotation sets of a frame stay together.

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
files = sorted(p.name for p in train_dir.iterdir())
test_files = sorted(p.name for p in test_dir.iterdir())
if QUICK:
    files, test_files = files[:20], test_files[:3]
folds = fold_files(files, N_FOLDS, SEED)
(output_dir() / "folds.json").write_text(json.dumps({"seed": SEED, "folds": folds}, indent=1))
log(f"{len(files)} training frames in {N_FOLDS} folds of {[len(f) for f in folds]}; {len(test_files)} test frames")


def frames_of(directory, names):
    for name in names:
        yield torch.from_numpy(channels(read_gray(directory / name)))


def quick_pq(maps: dict) -> dict:
    pq = PanopticQuality()
    for f, prob in maps.items():
        labels, areas, _ = instances(prob, **QUICK_PQ)
        for rles in annotations.rles_for_file(f):
            pq.update(instance_crops(rles), labels, areas)
    return {**pq.summary(), "iou_sum": round(pq.iou_sum, 4)}


# %% [markdown]
# ## Five folds
#
# Each fold: train on the other four, predict its own frames (plain and TTA) and the test frames. Test maps
# are summed as uint16 so all five fit in memory; their mean is written at the end.

# %%
report = {"seed": SEED, "epochs": EPOCHS, "tta_views": TTA_VIEWS, "quick_pq_params": QUICK_PQ, "folds": []}
test_sum = {"plain": {}, "tta": {}}
for k, held_out in enumerate(folds):
    start = time.time()
    torch.manual_seed(SEED + k)
    random.seed(SEED + k)
    train_files = sorted(set(files) - set(held_out))
    loader = DataLoader(
        FilamentFrames(train_files, train_dir, annotations, tile=TILE, tiles_per_frame=TILES_PER_FRAME),
        batch_size=FRAMES_PER_BATCH,
        shuffle=True,
        num_workers=WORKERS,
        collate_fn=collate_tiles,
        worker_init_fn=worker_init,
        pin_memory=device.type == "cuda",
        persistent_workers=WORKERS > 0,
        drop_last=True,
    )
    model = UNet(in_channels=2, base=BASE, depth=4)
    log(f"fold {k}: training on {len(train_files)} frames, holding out {len(held_out)}")
    history = fit(model, loader, epochs=EPOCHS, lr=LR, device=device, log=log)
    torch.save(model.state_dict(), output_dir() / f"unet_fold{k}.pt")
    trained = time.time()

    oof = {"plain": {}, "tta": {}}
    for f, (plain, tta) in zip(held_out, predict_tta(model, frames_of(train_dir, held_out), device, TTA_VIEWS)):
        oof["plain"][f], oof["tta"][f] = plain, tta
    for kind, maps in oof.items():
        np.savez_compressed(output_dir() / f"oof_{kind}_fold{k}.npz", **maps)
    scores = {kind: quick_pq(maps) for kind, maps in oof.items()}
    del oof

    for f, (plain, tta) in zip(test_files, predict_tta(model, frames_of(test_dir, test_files), device, TTA_VIEWS)):
        for kind, prob in (("plain", plain), ("tta", tta)):
            test_sum[kind][f] = test_sum[kind].get(f, 0) + prob.astype(np.uint16)
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["folds"].append(
        {
            "fold": k,
            "train_frames": len(train_files),
            "held_out_frames": len(held_out),
            "loss": [round(v, 4) for v in history],
            "quick_pq": scores,
            "train_sec": round(trained - start, 1),
            "predict_sec": round(time.time() - trained, 1),
        }
    )
    log(f"fold {k} done:", report["folds"][-1]["quick_pq"])
    (output_dir() / "cv_report.json").write_text(json.dumps(report, indent=2))

# %% [markdown]
# ## Test maps and the pooled quick PQ

# %%
for kind, sums in test_sum.items():
    mean = {f: np.round(total / N_FOLDS).astype(np.uint8) for f, total in sums.items()}
    np.savez_compressed(output_dir() / f"test_probs_{kind}.npz", **mean)

pooled = {}
for kind in ("plain", "tta"):
    terms = PanopticQuality()
    for fold in report["folds"]:
        s = fold["quick_pq"][kind]
        terms.tp, terms.fp, terms.fn = terms.tp + s["tp"], terms.fp + s["fp"], terms.fn + s["fn"]
        terms.iou_sum += s["iou_sum"]
    pooled[kind] = terms.summary()
report["pooled_quick_pq"] = pooled
report["elapsed_sec"] = round(time.time() - T0, 1)
(output_dir() / "cv_report.json").write_text(json.dumps(report, indent=2))
log("pooled quick PQ over all frames (no filter):", pooled)
