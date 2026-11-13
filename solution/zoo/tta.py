# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full
# %% [markdown]
# # Flips and turns at inference: one model, steadier maps
#
# Averaging the maps of `swin_unet_boundary` and its second seed raised the out-of-fold PQ by 0.002, and with
# `swin_unet` by 0.004 (`ensemble.py`): the maps are noisy where the networks disagree. Averaging one network
# over flipped and turned copies of the frame should calm the same noise without a second model.
#
# The saved models of `zoo/swin_unet_boundary.py` (each fold's) predict their held-out frames again, and the
# full-data model (`zoo/swin_unet_boundary_full.py`) the test frames, under the 8 flips and quarter turns of
# the square; a map is the mean of 4 views (identity, horizontal and vertical flip, half turn) or of all 8.
# The submitted post-processing (`pipeline.fit`) is fitted out of fold on each kind of map, as for any model,
# and compared with the plain maps on the same frames (`delta`, 95 % interval from resampling frames).
#
# Outputs: `swin_unet_boundary_tta{4,8}_oof_fold{k}.npz`, `..._test_full.npz`, `tta_comparison.csv`,
# `submission_tta{4,8}.csv`.

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

import numpy as np
import pandas as pd
import torch

from cnn_baseline.data import channels
from cnn_baseline.importance import paired_bootstrap
from cnn_baseline.training import pick_device
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from models.swin_unet import SwinUNet
from postprocess import pipeline
from postprocess.filtering import frame_context
from postprocess.tuning import pq_of
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS, TILE, predict_tiled

NAME = "swin_unet_boundary"
# (quarter turns, flip) of each view; the first four make the 4-view mean
VIEWS = [(0, False), (0, True), (2, True), (2, False), (1, False), (1, True), (3, False), (3, True)]
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def tta(model, frame: torch.Tensor, device) -> tuple[np.ndarray, np.ndarray]:
    """uint8 maps of the frame: the mean over the first 4 views and over all 8."""
    total = torch.zeros(frame.shape[1:])
    means = []
    for i, (turns, flip) in enumerate(VIEWS, start=1):
        view = torch.rot90(frame, turns, dims=(-2, -1))
        view = view.flip(-1) if flip else view
        prob = torch.from_numpy(predict_tiled(model, view.contiguous(), device)).float()
        prob = prob.flip(-1) if flip else prob
        total += torch.rot90(prob, -turns, dims=(-2, -1))
        if i in (4, 8):
            means.append(total.div(i).round().to(torch.uint8).numpy())
    return means[0], means[1]


def load_model(path, device):
    model = SwinUNet(pretrained=False, tile=TILE)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


device = pick_device()
out = output_dir()
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
test_files = sorted(p.name for p in test_dir.iterdir())

maps = {"plain": {}, "tta4": {}, "tta8": {}}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    model = load_model(find_input(f"{NAME}_fold{k}.pt"), device)
    four, eight = {}, {}
    for f in saved.files:
        maps["plain"][f] = saved[f]
        four[f], eight[f] = tta(model, torch.from_numpy(channels(read_gray(train_dir / f))), device)
    np.savez_compressed(out / f"{NAME}_tta4_oof_fold{k}.npz", **four)
    np.savez_compressed(out / f"{NAME}_tta8_oof_fold{k}.npz", **eight)
    maps["tta4"] |= four
    maps["tta8"] |= eight
    log(f"fold {k}: {len(saved.files)} frames")
    del model

model = load_model(find_input(f"{NAME}_full.pt"), device)
test = {"tta4": {}, "tta8": {}}
for f in test_files:
    test["tta4"][f], test["tta8"][f] = tta(model, torch.from_numpy(channels(read_gray(test_dir / f))), device)
for kind, kind_maps in test.items():
    np.savez_compressed(out / f"{NAME}_{kind}_test_full.npz", **kind_maps)
del model
log("test frames done")

# %% [markdown]
# ## Out-of-fold PQ of each kind of map, and the submissions

# %%
files = sorted(maps["plain"])
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
images = {f: read_gray(train_dir / f) for f in files}
rows, terms, posts = [], {}, {}
for kind, kind_maps in maps.items():
    contexts = {f: frame_context(f, images[f], kind_maps[f]) for f in files}
    posts[kind], oof = pipeline.fit(contexts, truth, files)
    terms[kind] = oof.terms
    tp, fp, fn = (int(oof.terms[c].sum()) for c in ["tp", "fp", "fn"])
    rows.append(
        {
            "maps": kind,
            "pq": round(pq_of(oof.terms), 4),
            "sq": round(float(oof.terms["iou_sum"].sum()) / max(tp, 1), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "cutoff": oof.cutoff,
            "kept_per_frame": round(oof.kept_per_frame, 2),
        }
    )
    log(rows[-1])
    del contexts
for row in rows:
    delta, lo, hi = paired_bootstrap(terms["plain"], terms[row["maps"]])
    row["delta"], row["ci95"] = round(delta, 4), [round(lo, 4), round(hi, 4)]
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(out / "tta_comparison.csv", index=False)

for kind, kind_maps in test.items():
    sub, kept = [], 0
    for f in test_files:
        labels, _, boxes = posts[kind](frame_context(f, read_gray(test_dir / f), kind_maps[f]))
        kept += len(boxes)
        sub += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    submission.write(sub, out / f"submission_{kind}.csv")
    log(f"{kind}: {kept / len(test_files):.2f} filaments kept per test frame")
(out / "tta_report.json").write_text(json.dumps(rows, indent=2))
log("done")
