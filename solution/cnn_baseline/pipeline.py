# kaggle: kernel_sources={user}/filament-cnn-baseline
# %% [markdown]
# # The whole pipeline, one frame at a time
#
# This notebook walks one validation frame through every stage of the pipeline, on CPU, with the trained
# U-Net of `filament-cnn-baseline` (attached as a notebook source). The full runs live in their own notebooks:
#
# | stage | script (notebook) | what it produces |
# |---|---|---|
# | train | `solution/cnn_baseline/main.py` (`filament-cnn-baseline`, T4, ~40 min) | `unet.pt` |
# | 5-fold OOF | `solution/cnn_baseline/cv.py` (`filament-cnn-baseline-cv`) | OOF maps for all 707 frames |
# | test maps | `solution/cnn_baseline/predict.py` (`filament-cnn-baseline-predict`) | single-model test maps |
# | post-processing, tuned on OOF | `solution/cnn_baseline/final_join.py` (CPU) | `submission.csv` |
# | final model | Swin U-Net with boundary loss (teammate; see the report, section 4.2) | team's final submission |
#
# Every script becomes a Kaggle notebook with `uv run filament run <script>` (tests, build, push, wait, fetch
# outputs, format check). Stages below: **disk and limb correction → network → hysteresis instances →
# joining by filament physics → component features → score and RLE.**

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_baseline.data import channels, split_files
from cnn_baseline.filtering import component_table, frame_context
from cnn_baseline.pairs import join_pairs, join_rule, pair_table
from cnn_baseline.postprocess import instances, on_disk
from cnn_baseline.training import predict
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from models.unet import UNet
from preprocess.disk import disk_geometry
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import PanopticQuality, instance_crops

SEED, VALIDATION_FRACTION = 0, 0.15  # the split the network was trained with: these frames were not seen
A0 = {"threshold": 0.5, "min_area": 300, "low": 0.3}
MAX_TRIES = 20  # validation frames tried when looking for one where joining acts


def show(name: str) -> None:
    """Save the current figure next to the notebook's outputs, then display it."""
    plt.savefig(output_dir() / f"pipeline_{name}.png", dpi=100)
    plt.show()


root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
_, val_files = split_files(sorted(p.name for p in train_dir.iterdir()), VALIDATION_FRACTION, SEED)
model = UNet(in_channels=2, base=32, depth=4)
model.load_state_dict(torch.load(find_input("unet.pt"), map_location="cpu"))


def run_frame(name: str):
    """image, network probability map, frame context, A0 instances and joined instances of one frame."""
    image = read_gray(train_dir / name)
    prob = on_disk(next(predict(model, [torch.from_numpy(channels(image))], torch.device("cpu"))), image)
    a0 = instances(prob, **A0)
    ctx = frame_context(name, image, prob)
    pairs = pair_table(ctx, a0[0], 50)
    joined = join_pairs(a0[0], pairs, join_rule(pairs)) if len(pairs) else a0
    return image, prob, ctx, a0, joined


# %% [markdown]
# **Which frame.** Chosen by a fixed rule, for illustration: among the validation frames with the most
# annotators (three on the real data), the first one where joining merges at least two pieces. If none of the
# first frames tried does, the first one is shown. It is not a typical frame; the report gives 707-frame scores.

# %%
by_annotators = sorted(val_files, key=lambda f: -len(annotations.rles_for_file(f)))  # stable: keeps name order
for frame in by_annotators[:MAX_TRIES]:
    image, prob, ctx, a0, joined = run_frame(frame)
    if len(joined[1]) < len(a0[1]):
        break
else:
    frame = by_annotators[0]
    image, prob, ctx, a0, joined = run_frame(frame)
print(frame, image.shape, image.dtype, f"({len(annotations.rles_for_file(frame))} annotators)")

# %% [markdown]
# ## 1. Disk and limb darkening
#
# Limb darkening is multiplicative, so the corrected channel is the frame *divided* by a smooth background
# (a normalised convolution over the disk): a filament then reads ~0.88 of its surroundings at the centre
# and at the limb alike. The network gets both the raw and the corrected frame.

# %%
cx, cy, radius = disk_geometry(image)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, channel, title in zip(axes, channels(image), ["raw frame", "limb-darkening corrected"]):
    ax.imshow(channel, cmap="gray")
    ax.add_patch(plt.Circle((cx, cy), radius, fill=False, color="tab:orange", lw=0.8))
    ax.set_title(title)
    ax.axis("off")
show("1_disk")
print(f"disk centre ({cx:.0f}, {cy:.0f}) px, radius {radius:.0f} px")

# %% [markdown]
# ## 2. Network: probability map
#
# A 4-level U-Net (2 input channels) trained on 512 px tiles with the fraction of annotators as a soft target.
# It ran above on CPU on the full 2048 x 2048 frame; off-disk pixels are zeroed.

# %%
print("probability map", prob.shape, prob.dtype, "(probability x 255)")

# %% [markdown]
# ## 3. Instances: hysteresis, then joining
#
# A component of p >= 0.3 is kept whole when any pixel reaches 0.5, so faint tails stay attached; then pieces
# that continue each other end to end along their axes (<= 50 px) or lie within 10 px are joined, the two
# ways a filament physically breaks (material thinning along the spine; barbs).

# %%
print(f"A0: {len(a0[1])} instances; after joining: {len(joined[1])}")
truth = annotations.rles_for_file(frame)
first = np.zeros_like(prob, dtype=np.int32)
for k, (x0, y0, mask) in enumerate(instance_crops(truth[0]), start=1):
    first[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]][mask] = k
panels = [
    (prob, "magma", "network probability"),
    (np.ma.masked_equal(a0[0], 0), "tab20", f"A0 instances ({len(a0[1])})"),
    (np.ma.masked_equal(joined[0], 0), "tab20", f"after joining ({len(joined[1])})"),
    (np.ma.masked_equal(first, 0), "tab20", f"annotator 1 ({len(truth[0])})"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, (data, cmap, title) in zip(axes.flat, panels):
    ax.imshow(data, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy + radius, cy - radius)
    ax.axis("off")
show("2_instances")

# %% [markdown]
# ## 4. Component features (the filter's inputs)
#
# The filter predicts whether a component matches any annotator from these features; it is trained out of fold
# on all 707 training frames in `final_join.py`, and its cutoff follows the break-even rule
# P(match) x IoU > PQ/2. Here are the features of this frame's joined components.

# %%
table = component_table(ctx, joined[0], joined[2])
print(table[["area", "elongation", "darkness", "limb", "mean_prob", "max_prob"]].round(3).to_string())

# %% [markdown]
# ## 5. Score, as the hosts do: per annotator
#
# PQ = sum of IoU over matches / (TP + FP/2 + FN/2), a match needing IoU > 0.5, accumulated over the
# annotators of the frame. One frame is a noisy number; the report gives the 707-frame OOF scores.

# %%
for name, (labels, areas, _) in {"A0": a0, "A0 + join": joined}.items():
    pq = PanopticQuality()
    for rles in truth:
        pq.update(instance_crops(rles), labels, areas)
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in pq.summary().items()})

# %% [markdown]
# ## 6. Submission rows
#
# One row per filament: `filament_id` = `<image id>_<n>` and the pycocotools-compressed RLE of its mask.

# %%
rows = submission.rows_from_label_map(frame.rsplit(".", 1)[0], joined[0], joined[2])
print(
    len(rows),
    "rows; first:",
    {k: (v[:60] + "...") if isinstance(v, str) and len(v) > 60 else v for k, v in rows[0].items()},
)
