# kaggle: kernel_sources={user}/filament-zoo-swin-unet
# %% [markdown]
# # Misses of the best model: what are the filaments it does not match?
#
# CPU only, on the 707 out-of-fold maps of `zoo/swin_unet.py` after the submitted post-processing (join, then
# the component filter with out-of-fold scores, so no frame is filtered by a model that saw it). For every
# annotator x filament, as in `cnn_baseline/contrast_lab.py`:
#
# - **status**: matched (IoU > 0.5), near miss (partly covered), missed (not touched);
# - **agreement**: how many of the frame's annotators drew it (another annotator's filament with IoU >= 0.3);
#   a filament only one of three drew is not one a model should be expected to find;
# - **snr**: its Halpha absorption contrast against a 5-20 px background ring, in robust spreads;
# - **max_prob**: the network's highest probability inside it (< 0.1: it did not respond at all);
# - area and **r** (distance from the disk centre, in disk radii).
#
# Outputs: `miss_rows.csv`, `miss_report.json`, `gallery_missed.png`, `gallery_near_miss.png` (real frames:
# stay within the team).

# %%
import json
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.instances import select
from postprocess.pipeline import candidates
from postprocess.tuning import best_cutoff
from preprocess.disk import OFF_DISK, REL_SCALE
from preprocess.images import read_gray
from scoring.metric import IOU_THRESHOLD, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet"
AGREE_IOU = 0.3
SMOOTH = 2.0  # px, Gaussian sigma, as contrast_lab.py
RING = (5, 20)  # px, inner and outer distance of the background ring
GALLERY = 20  # filaments per gallery
SEED = 0
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## The submitted post-processing, out of fold

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
scores = out_of_fold_scores(table, target)
cutoff, terms = best_cutoff(scores, raw, table, truth, files)
frames, keep = table["frame"].to_numpy(), scores >= cutoff
kept = {f: select(*raw[f], keep[frames == f]) if len(raw[f][1]) else raw[f] for f in files}
log(f"{NAME}: cutoff {cutoff}, tp/fp/fn {terms[['tp', 'fp', 'fn']].sum().to_dict()}")

# %% [markdown]
# ## Every annotated filament

# %%
inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING[0] + 1, 2 * RING[0] + 1))
outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING[1] + 1, 2 * RING[1] + 1))


def label_map(crops, shape) -> np.ndarray:
    labels = np.zeros(shape, np.int32)
    for i, (x, y, crop) in enumerate(crops, start=1):
        labels[y : y + crop.shape[0], x : x + crop.shape[1]][crop] = i
    return labels


rows = []
for f in files:
    ctx = contexts[f]
    labels, areas, _ = kept[f]
    on_disk = ctx.flat != OFF_DISK
    dark = 1 - cv2.GaussianBlur(ctx.flat.astype(np.float32), (0, 0), SMOOTH) / REL_SCALE
    drawn = np.zeros(ctx.flat.shape, bool)
    for crops in truth[f]:
        for x, y, crop in crops:
            drawn[y : y + crop.shape[0], x : x + crop.shape[1]] |= crop
    per_annotator = [label_map(crops, ctx.flat.shape) for crops in truth[f]]
    annotator_areas = [np.array([crop.sum() for _, _, crop in crops]) for crops in truth[f]]
    n_annotators = len(truth[f])
    for a, crops in enumerate(truth[f]):
        for i, (x, y, crop) in enumerate(crops):
            h, w = crop.shape
            size = int(crop.sum())
            agreement = 1
            for b, others in enumerate(per_annotator):
                if b == a or not others.any():
                    continue
                inter = np.bincount(others[y : y + h, x : x + w][crop])[1:]
                if len(inter):
                    other_areas = annotator_areas[b][: len(inter)]
                    agreement += int((inter / (size + other_areas - inter)).max() >= AGREE_IOU)
            pred = labels[y : y + h, x : x + w][crop]
            counts = np.bincount(pred, minlength=len(areas) + 1)[1:]
            iou = counts / (size + areas - counts) if len(areas) else np.zeros(0)
            matched = bool(len(iou) and iou.max() > IOU_THRESHOLD)
            status = "matched" if matched else ("near_miss" if (pred > 0).any() else "missed")
            pad = RING[1] + 1
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + w + pad, ctx.flat.shape[1]), min(y + h + pad, ctx.flat.shape[0])
            gt = np.zeros((y1 - y0, x1 - x0), np.uint8)
            gt[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = crop
            ring = (cv2.dilate(gt, outer) > 0) & ~(cv2.dilate(gt, inner) > 0)
            ring &= ~drawn[y0:y1, x0:x1] & (labels[y0:y1, x0:x1] == 0) & on_disk[y0:y1, x0:x1]
            inside = (gt > 0) & on_disk[y0:y1, x0:x1]
            snr = np.nan
            if ring.sum() >= 30 and inside.sum() >= 10:
                background = dark[y0:y1, x0:x1][ring]
                level = float(np.median(background))
                spread = float(1.4826 * np.median(np.abs(background - level))) or 1e-6
                snr = (float(np.median(dark[y0:y1, x0:x1][inside])) - level) / spread
            ys, xs = np.nonzero(crop)
            r = np.hypot(xs.mean() + x - ctx.centre[0], ys.mean() + y - ctx.centre[1]) / ctx.radius
            rows.append(
                {
                    "frame": f,
                    "annotator": a,
                    "gt": i,
                    "box": [int(x), int(y), int(w), int(h)],
                    "n_annotators": n_annotators,
                    "agreement": agreement,
                    "status": status,
                    "area": size,
                    "r": round(float(r), 3),
                    "snr": round(float(snr), 2),
                    "max_prob": round(float(ctx.prob[y : y + h, x : x + w][crop].max()) / 255, 3),
                    "best_iou": round(float(iou.max()) if len(iou) else 0.0, 3),
                }
            )
misses = pd.DataFrame(rows)
misses.to_csv(output_dir() / "miss_rows.csv", index=False)
log(f"{len(misses)} annotator x filament rows")

# %% [markdown]
# ## Summary
#
# Unmatched filaments (near misses and missed: the false negatives) broken down by what might explain them.

# %%


def shares(df: pd.DataFrame, column: str, bins=None, labels=None) -> dict:
    values = pd.cut(df[column], bins, labels=labels, right=False) if bins is not None else df[column]
    counts = df.groupby(values, observed=False)["status"].agg(
        rows="size", unmatched=lambda s: int((s != "matched").sum())
    )
    counts["miss_rate"] = (counts["unmatched"] / counts["rows"].clip(lower=1)).round(3)
    counts["share_of_fn"] = (counts["unmatched"] / max(int((df["status"] != "matched").sum()), 1)).round(3)
    return {str(k): v for k, v in counts.to_dict(orient="index").items()}


three = misses[misses["n_annotators"] == 3]
report = {
    "name": NAME,
    "cutoff": cutoff,
    "status": misses["status"].value_counts().to_dict(),
    "agreement_of_three": shares(three, "agreement"),
    "snr": shares(misses, "snr", [-np.inf, 0.5, 1, 2, 3, np.inf], ["<0.5", "0.5-1", "1-2", "2-3", ">=3"]),
    "max_prob": shares(misses, "max_prob", [0, 0.1, 0.3, 0.5, 1.01], ["<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"]),
    "area": shares(misses, "area", [0, 500, 1000, 2000, 5000, np.inf], ["<500", "500-1k", "1-2k", "2-5k", ">=5k"]),
    "r": shares(misses, "r", [0, 0.5, 0.7, 0.85, 0.95, 2], ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">=0.95"]),
}
print(json.dumps(report, indent=2))
(output_dir() / "miss_report.json").write_text(json.dumps(report, indent=2))

# %% [markdown]
# ## Galleries: annotated filament in red, kept predictions in cyan, then the network's map
#
# Missed or near-missed filaments that at least two annotators drew, at random.

# %%


def gallery(df: pd.DataFrame, path: str, title: str) -> None:
    picked = df.sample(min(GALLERY, len(df)), random_state=SEED) if len(df) else df
    cols = 4
    fig, axes = plt.subplots(
        int(np.ceil(len(picked) / cols)) or 1, 2 * cols, figsize=(4 * cols, 2.1 * np.ceil(len(picked) / cols) + 0.4)
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for n, row in enumerate(picked.itertuples()):
        ctx, labels = contexts[row.frame], kept[row.frame][0]
        x, y, w, h = row.box
        pad = max(40, w // 2, h // 2)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, ctx.flat.shape[1]), min(y + h + pad, ctx.flat.shape[0])
        gt = label_map(truth[row.frame][row.annotator], ctx.flat.shape)[y0:y1, x0:x1] == row.gt + 1
        image_ax, prob_ax = axes[n // cols, 2 * (n % cols)], axes[n // cols, 2 * (n % cols) + 1]
        image_ax.imshow(ctx.flat[y0:y1, x0:x1], cmap="gray")
        image_ax.contour(gt, [0.5], colors="red", linewidths=0.8)
        if (labels[y0:y1, x0:x1] > 0).any():
            image_ax.contour(labels[y0:y1, x0:x1] > 0, [0.5], colors="cyan", linewidths=0.8)
        image_ax.set_title(f"{row.status} {row.agreement}/{row.n_annotators} snr {row.snr}", fontsize=7)
        prob_ax.imshow(ctx.prob[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=255)
        prob_ax.contour(gt, [0.5], colors="white", linewidths=0.6)
        prob_ax.set_title(f"max p {row.max_prob}, iou {row.best_iou}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir() / path, dpi=110)
    plt.close(fig)


agreed = misses[misses["agreement"] >= 2]
gallery(agreed[agreed["status"] == "missed"], "gallery_missed.png", f"{NAME}: missed, drawn by >= 2 annotators")
gallery(agreed[agreed["status"] == "near_miss"], "gallery_near_miss.png", f"{NAME}: near misses, >= 2 annotators")
log("done")
