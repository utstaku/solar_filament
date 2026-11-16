# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true kernel_sources={user}/filament-zoo-swin-unet-boundary
# %% [markdown]
# # A filter that looks at the candidate: a small CNN on crops, alone and stacked with the component filter
#
# `filter_bound.py` found that a perfect filter would raise the best model's out-of-fold PQ from 0.42 to 0.50,
# while the gradient-boosting filter (`filtering.ComponentFilter`, six numbers per candidate: area,
# elongation, darkness, limb distance, mean and max probability) sits at ROC AUC 0.84 and its true candidates
# dropped look like its false ones on all six. Those numbers leave out what is around the candidate: its
# shape, the texture it lies in, its neighbours. Here a ResNet-18 (ImageNet weights) sees a square crop
# around each candidate, resized to `SIZE` px, in four channels: the flattened frame, the network's map, the
# candidate's mask and the other candidates' masks.
#
# On the 707 out-of-fold maps of `zoo/swin_unet_boundary.py` with the submitted candidates (A0, join), every
# score is out of fold: the CNN and the component filter use the same frame groups (`GroupKFold`, 5), and
# the stack (the component filter with the CNN's score as a seventh feature) is fitted out of fold on the
# CNN's out-of-fold scores. For the component filter, the CNN, the stack and a perfect filter: ROC AUC and PQ
# at the PQ-best cutoff.
#
# Outputs: `crop_scores.csv`, `crop_filter_report.json`.

# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "timm==1.0.29"], check=True)

# %%
import json
import random
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F

from cnn_baseline.training import pick_device
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import FEATURES, component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.pipeline import candidates
from postprocess.tuning import best_cutoff, frame_terms, pq_of
from preprocess.disk import OFF_DISK, REL_SCALE
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
SIZE = 128  # px of the resized crop
PAD = 32  # px of context around the candidate's box
MIN_SIDE = 96  # px: small candidates keep at least this much context
EPOCHS = 12
BATCH = 128
LR = 3e-4
SEED = 0
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## Candidates, their targets and the component filter's out-of-fold scores

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
maps = {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
files = sorted(maps)
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
raw = {f: candidates(contexts[f]) for f in files}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
gbm_scores = out_of_fold_scores(table, target)
log(f"{len(files)} frames, {len(table)} candidates, {target.mean():.3f} true")

# %% [markdown]
# ## Crops

# %%


def frame_crops(f: str) -> np.ndarray:
    """uint8 (K, 4, SIZE, SIZE): flattened frame, map, this candidate, the other candidates."""
    ctx, (labels, _, boxes) = contexts[f], raw[f]
    flat = np.where(ctx.flat == OFF_DISK, 0, ctx.flat).astype(np.float32)
    flat = np.clip(flat / REL_SCALE * 160, 0, 255).astype(np.uint8)  # background at 160, dark filaments lower
    out = np.zeros((len(boxes), 4, SIZE, SIZE), np.uint8)
    for k, (x, y, w, h) in enumerate(boxes, start=1):
        side = max(int(max(w, h)) + 2 * PAD, MIN_SIDE)
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = cx - side // 2, cy - side // 2
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + side, labels.shape[1]), min(y0 + side, labels.shape[0])
        window = np.zeros((4, side, side), np.uint8)
        region = labels[sy0:sy1, sx0:sx1]
        ty, tx = sy0 - y0, sx0 - x0
        window[0, ty : ty + region.shape[0], tx : tx + region.shape[1]] = flat[sy0:sy1, sx0:sx1]
        window[1, ty : ty + region.shape[0], tx : tx + region.shape[1]] = ctx.prob[sy0:sy1, sx0:sx1]
        window[2, ty : ty + region.shape[0], tx : tx + region.shape[1]] = (region == k) * 255
        window[3, ty : ty + region.shape[0], tx : tx + region.shape[1]] = ((region > 0) & (region != k)) * 255
        for c in range(4):
            out[k - 1, c] = cv2.resize(window[c], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out


crops = np.concatenate([frame_crops(f) for f in files if len(raw[f][1])])
assert len(crops) == len(table)
log(f"crops {crops.shape}, {crops.nbytes / 1e9:.2f} GB")

# %% [markdown]
# ## The CNN, out of fold on the component filter's frame groups

# %%
device = pick_device()


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random flips and quarter turns per batch: a filament has no preferred direction."""
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    return torch.rot90(x, random.randrange(4), dims=(-2, -1))


def to_input(batch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(batch).to(device).float().div_(255).sub_(0.5).div_(0.25)


@torch.no_grad()
def predict(model, idx: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), 2 * BATCH):
        x = to_input(crops[idx[i : i + 2 * BATCH]])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            views = [x, x.flip(-1), x.flip(-2), torch.rot90(x, 1, dims=(-2, -1))]
            logits = torch.stack([model(v).float().squeeze(1) for v in views]).mean(0)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train(idx: np.ndarray):
    model = timm.create_model("resnet18.a1_in1k", pretrained=True, in_chans=4, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(idx) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    y_all = torch.from_numpy(target.astype(np.float32))
    for epoch in range(EPOCHS):
        model.train()
        order, total = np.random.permutation(idx), 0.0
        for i in range(0, len(order), BATCH):
            batch = order[i : i + BATCH]
            x, y = augment(to_input(crops[batch])), y_all[batch].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x).float().squeeze(1), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += loss.item() * len(batch)
        log(f"  epoch {epoch + 1}/{EPOCHS} loss {total / len(order):.4f}")
    return model


cnn_scores = np.zeros(len(table))
groups = table["frame"].to_numpy()
for k, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(table, target, groups)):
    model = train(train_idx)
    cnn_scores[test_idx] = predict(model, test_idx)
    log(f"fold {k}: AUC {roc_auc_score(target[test_idx], cnn_scores[test_idx]):.4f}")

# %% [markdown]
# ## The stack and the PQ of each filter

# %%
stacked_table = table.assign(cnn=cnn_scores)
stack_scores = out_of_fold_scores(stacked_table, target, features=[*FEATURES, "cnn"])
results = {}
for name, scores in {"component_filter": gbm_scores, "cnn": cnn_scores, "stack": stack_scores}.items():
    cutoff, terms = best_cutoff(scores, raw, table, truth, files)
    results[name] = {
        "auc": round(float(roc_auc_score(target, scores)), 4),
        "cutoff": cutoff,
        "pq": round(pq_of(terms), 4),
        **{c: int(terms[c].sum()) for c in ["tp", "fp", "fn"]},
    }
    log(name, results[name])
perfect = frame_terms(raw, table, target, truth, files)
results["perfect"] = {"pq": round(pq_of(perfect), 4), **{c: int(perfect[c].sum()) for c in ["tp", "fp", "fn"]}}
table.assign(target=target, gbm=gbm_scores, cnn=cnn_scores, stack=stack_scores).to_csv(
    output_dir() / "crop_scores.csv", index=False
)
report = {"name": NAME, "candidates": len(table), "epochs": EPOCHS, "size": SIZE, "results": results}
print(json.dumps(report, indent=2))
(output_dir() / "crop_filter_report.json").write_text(json.dumps(report, indent=2))
log("done")
