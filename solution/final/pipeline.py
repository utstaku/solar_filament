# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true kernel_sources={user}/filament-final-export,{user}/filament-zoo-final-sweep
# %% [markdown]
# # Final submission: test frames to `submission.csv`, from the stored weights
#
# The whole pipeline of our final submission, with no network retrained. It loads three files from `weights/`
# in the repository (written by `final/export.py`): the Swin U-Net trained on all 707 training frames, the
# post-processing settings, and the component filter's training table (the filter, a small gradient boosting
# with a fixed seed, is fitted from it in a few seconds; a pickled scikit-learn model would load only in the
# version that wrote it).
#
# 1. **Input**: each 2048 x 2048 test frame becomes two channels, the frame and its limb-darkening-corrected copy
#    (`cnn_baseline.data.channels`).
# 2. **Network**: the Swin U-Net (Swin-T encoder, U-Net decoder, trained with a boundary-weighted BCE + Dice loss)
#    predicts 512 px tiles blended with a Hann window, for each of the 8 flips and quarter turns of the frame;
#    the probability map is their mean (`zoo.experiment.predict_views`).
# 3. **Candidates**: hysteresis threshold 0.6 / 0.45, components of at least 300 px.
# 4. **Join**: pieces end to end along their axes within 50 px, any pieces within 10 px.
# 5. **Filter**: gradient boosting on shape, darkness, limb distance and the network's confidence keeps a
#    filament when its chance of matching an annotator is worth it for PQ (cutoff chosen out of fold).
# 6. **Outline**: each kept filament is reduced to its pixels with probability >= 0.6.
# 7. **Submission**: one COCO RLE per filament.
#
# Out of fold on the 707 training frames this scores PQ 0.4357; the public leaderboard gave 0.37.
# Runs on a T4 GPU in about 20 minutes; on a CPU it works but takes hours. With the submitted file attached
# (Kaggle: `filament-zoo-final-sweep`), the last cell checks that the new file reproduces it.

# %%
import subprocess
import sys

# Pin the model implementations across local checks and the Kaggle runtime.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0", "timm==1.0.29"],
    check=True,
)

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cnn_baseline.data import channels
from cnn_baseline.training import pick_device
from dataset.paths import find_data_root, find_input, output_dir
from models.swin_unet import SwinUNet
from postprocess.filtering import FEATURES, ComponentFilter, frame_context
from postprocess.pipeline import Postprocess
from preprocess.images import read_gray
from scoring import rle, submission
from scoring.check import check
from zoo.experiment import TILE, predict_views

MODEL = "swin_unet_boundary_full_fp16.pt"
CONFIG = "final_config.json"
FILTER_TABLE = "final_filter_table.csv"
SUBMITTED = "submission_tta8_h60_cut0.6.csv"  # the file sent to the leaderboard, to compare with
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def stored(name: str) -> Path:
    """`weights/<name>` in the repository, or the same file attached to the Kaggle notebook."""
    local = Path("weights") / name
    return local if local.exists() else find_input(name)


device = pick_device()
model = SwinUNet(pretrained=False, tile=TILE)
model.load_state_dict({k: v.float() for k, v in torch.load(stored(MODEL), map_location="cpu").items()})
model.to(device).eval()
config = json.loads(stored(CONFIG).read_text())
table = pd.read_csv(stored(FILTER_TABLE))
post = Postprocess(
    ComponentFilter(cutoff=config["filter_cutoff"]).fit(table[FEATURES], table["target"].to_numpy()),
    params=config["params"],
    join=config["join"],
    mask_cut=config["mask_cut"],
)
log(f"model and post-processing loaded: {post.params}, cutoff {post.filter.cutoff}, mask cut {post.mask_cut}")

# %% [markdown]
# ## Every test frame

# %%
test_dir = find_data_root() / "test" / "test_images"
test_files = sorted(p.name for p in test_dir.iterdir())
rows = []
for i, f in enumerate(test_files, start=1):
    image = read_gray(test_dir / f)
    prob = predict_views(model, torch.from_numpy(channels(image)), device)
    labels, _, boxes = post(frame_context(f, image, prob))
    rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    if i % 30 == 0:
        log(f"{i}/{len(test_files)} frames, {len(rows)} filaments")
path = output_dir() / "submission.csv"
submission.write(rows, path)
check(path)

# %% [markdown]
# ## Does it reproduce the submitted file?
#
# Each submitted filament is paired with the new filament of the same frame that overlaps it most.

# %%
try:
    submitted = pd.read_csv(find_input(SUBMITTED))
except FileNotFoundError:
    submitted = None
    log("submitted file not attached; skipping the comparison")
if submitted is not None:
    new = pd.DataFrame(rows)
    ious = []
    for frame_id, group in submitted.groupby(submitted["filament_id"].str.rsplit("_", n=1).str[0]):
        mine = [
            rle.decode(c).astype(bool)
            for c in new.loc[new["filament_id"].str.startswith(frame_id + "_"), "segmentation_rle"]
        ]
        for counts in group["segmentation_rle"]:
            mask = rle.decode(counts).astype(bool)
            best = max(((mask & m).sum() / (mask | m).sum() for m in mine), default=0.0)
            ious.append(best)
    ious = np.array(ious)
    log(
        f"submitted {len(submitted)} filaments, new {len(new)}; "
        f"identical {np.mean(ious == 1):.3f}, IoU > 0.95 {np.mean(ious > 0.95):.3f}, median IoU {np.median(ious):.4f}"
    )
