# kaggle: gpu=true
# %% [markdown]
# # CNN baseline: a plain U-Net from scratch
#
# 1. Train a small U-Net on 512 px tiles (input: raw frame + limb-darkening-corrected frame;
#    target: fraction of annotators marking each pixel).
# 2. Predict whole frames of a held-out split, turn probabilities into instances with a threshold and
#    connected components, and pick the threshold / minimum area that maximise Panoptic Quality.
# 3. Predict the test frames with those settings → `submission.csv`.
#
# Outputs: `submission.csv`, `cnn_report.json`, `unet.pt`.

# %%
import json
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, split_files, worker_init
from cnn_baseline.postprocess import instances
from cnn_baseline.training import fit, pick_device, predict
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from models.unet import UNet
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import PanopticQuality, instance_crops

QUICK = False  # True: a few frames and epochs, to check that the whole notebook runs
SEED = 0
VALIDATION_FRACTION = 0.15
TILE = 512
TILES_PER_FRAME = 4
FRAMES_PER_BATCH = 4  # -> 16 tiles per step
EPOCHS = 2 if QUICK else 16
LR = 1e-3
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_AREAS = (50, 100, 200, 400)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


torch.manual_seed(SEED)
random.seed(SEED)
device = pick_device()
log("device:", device)

# %% [markdown]
# ## Data
#
# The split is by physical frame, so the up-to-three annotation sets of one frame never straddle it.

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
test_dir = root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
files = sorted(p.name for p in train_dir.iterdir())
if QUICK:
    files = files[:24]
train_files, val_files = split_files(files, VALIDATION_FRACTION, SEED)
log(f"{len(train_files)} training frames, {len(val_files)} validation frames")

train_set = FilamentFrames(train_files, train_dir, annotations, tile=TILE, tiles_per_frame=TILES_PER_FRAME)
train_loader = DataLoader(
    train_set,
    batch_size=FRAMES_PER_BATCH,
    shuffle=True,
    num_workers=3,
    collate_fn=collate_tiles,
    worker_init_fn=worker_init,
    pin_memory=device.type == "cuda",
    persistent_workers=True,
    drop_last=True,
)

# %% [markdown]
# ## Training

# %%
model = UNet(in_channels=2, base=32, depth=4)
log(f"U-Net with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
history = fit(model, train_loader, epochs=EPOCHS, lr=LR, device=device, log=log)
torch.save(model.state_dict(), output_dir() / "unet.pt")

plt.figure(figsize=(6, 3))
plt.plot(range(1, len(history) + 1), history, marker="o")
plt.xlabel("epoch")
plt.ylabel("BCE + Dice")
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Validation: choose the post-processing on Panoptic Quality
#
# Probability maps are kept (as uint8) so every threshold / minimum-area pair is scored on the same
# predictions. PQ is accumulated per annotator-image, exactly as the hosts score a submission.


# %%
def frames_of(directory, names):
    for name in names:
        yield torch.from_numpy(channels(read_gray(directory / name)))


val_probs = dict(zip(val_files, predict(model, frames_of(train_dir, val_files), device)))
val_truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}
log(f"predicted {len(val_probs)} validation frames")

sweep = []
for threshold in THRESHOLDS:
    for min_area in MIN_AREAS:
        pq = PanopticQuality()
        for f in val_files:
            labels, areas, _ = instances(val_probs[f], threshold, min_area)
            for gt_crops in val_truth[f]:
                pq.update(gt_crops, labels, areas)
        sweep.append({"threshold": threshold, "min_area": min_area, **pq.summary()})
        log(sweep[-1])
best = max(sweep, key=lambda row: row["pq"])
params = {"threshold": best["threshold"], "min_area": best["min_area"]}
log("best:", best)

# %% [markdown]
# ## What it looks like
#
# First annotator's ground truth in green, prediction in red, on validation frames.

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 18))
for ax, f in zip(axes.ravel(), val_files):
    canvas = cv2.cvtColor(read_gray(train_dir / f), cv2.COLOR_GRAY2RGB)
    for x, y, crop in val_truth[f][0] if val_truth[f] else []:
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x, y))
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    labels, _, _ = instances(val_probs[f], **params)
    contours, _ = cv2.findContours((labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)
    ax.imshow(canvas)
    ax.set_title(f)
    ax.axis("off")
plt.tight_layout()
plt.show()
del val_probs

# %% [markdown]
# ## Test predictions

# %%
test_files = sorted(p.name for p in test_dir.iterdir())
rows = []
for f, prob in zip(test_files, predict(model, frames_of(test_dir, test_files), device)):
    labels, _, boxes = instances(prob, **params)
    rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
frame = submission.write(rows, output_dir() / "submission.csv")

report = {
    "quick": QUICK,
    "n_train_frames": len(train_files),
    "n_val_frames": len(val_files),
    "epochs": EPOCHS,
    "loss_history": [round(v, 4) for v in history],
    "sweep": sweep,
    "best": best,
    "n_test_predictions": len(frame),
    "n_test_images_with_predictions": int(frame["filament_id"].str.split("_").str[0].nunique()),
    "elapsed_sec": round(time.time() - T0, 1),
}
(output_dir() / "cnn_report.json").write_text(json.dumps(report, indent=2))
log(f"wrote submission.csv with {len(frame)} rows")
