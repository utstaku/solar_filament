# kaggle: kernel_sources={user}/filament-zoo-swin-unet,{user}/filament-zoo-misses
# %% [markdown]
# # How do the best model's candidates miss the annotated shapes?
#
# CPU only, on the 707 out-of-fold maps of `zoo/swin_unet.py` and the submitted candidates (A0, join).
# `filter_bound.py` showed that a perfect filter would leave 2565 of the 3312 false negatives: filaments no
# candidate matches at IoU > 0.5. Those need a better map, not a better filter, and how they fail says what to
# change in the model. Every annotator x filament no candidate matches gets a **shape** from the candidate that
# overlaps it most (`cover`: share of the filament it covers, `purity`: share of the candidate on it):
#
# - **none**: no candidate covers 5 % of it (the map stays below 0.5 there, or the piece is under 300 px);
# - **split**: two or more candidates each cover >= 20 % of it;
# - **too_big**: the best covers most of it (cover >= 0.5) but spills well past it (purity < 0.5): merged
#   with a neighbour, or drawn wider or longer than the annotators;
# - **too_small**: the best lies on it (purity >= 0.5) but covers less than half: a fragment;
# - **offset**: neither.
#
# `other_annotator` marks filaments whose best candidate matches another annotator: the network agrees with
# someone else's outline. The same shapes are given, from the candidate's side, to the false candidates the
# filter keeps (`unannotated`: no annotator drew anything under it).
#
# Outputs: `shape_rows.csv`, `false_kept_rows.csv`, `shapes_report.json`, `gallery_<shape>.png`.

# %%
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.pipeline import candidates
from postprocess.tuning import best_cutoff
from preprocess.images import read_gray
from scoring.metric import IOU_THRESHOLD, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet"
TOUCH = 0.05  # a candidate covering less of a filament does not count
SPLIT_COVER = 0.2
GALLERY = 12
SEED = 0
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


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
cutoff, _ = best_cutoff(scores, raw, table, truth, files)
frames = table["frame"].to_numpy()
kept = {f: np.zeros(0, bool) for f in files} | {f: (scores >= cutoff)[frames == f] for f in files if len(raw[f][1])}
true = {f: np.zeros(0, bool) for f in files} | {f: target[frames == f] for f in files if len(raw[f][1])}
agreement = pd.read_csv(find_input("miss_rows.csv")).set_index(["frame", "annotator", "gt"])["agreement"]
log(f"{len(files)} frames, {len(table)} candidates, cutoff {cutoff}")


def shape(cover: np.ndarray, purity: np.ndarray) -> tuple[str, int]:
    """(shape, index of the best candidate) from per-candidate cover and purity of one filament."""
    if not len(cover) or cover.max() < TOUCH:
        return "none", -1
    best = int(np.argmax(cover))
    if (cover >= SPLIT_COVER).sum() >= 2:
        return "split", best
    if cover[best] >= 0.5:
        return ("too_big" if purity[best] < 0.5 else "offset"), best
    return ("too_small" if purity[best] >= 0.5 else "offset"), best


# %% [markdown]
# ## Filaments no candidate matches

# %%
rows, false_rows = [], []
for f in files:
    labels, areas, _ = raw[f]
    n = len(areas)
    gt_union = np.zeros(labels.shape, np.int32)  # how many annotators drew each pixel
    inter_all = np.zeros(n)  # pixels of each candidate on any annotator's filament
    best_iou = np.zeros(n)
    for a, crops in enumerate(truth[f]):
        drawn = np.zeros(labels.shape, bool)
        for i, (x, y, crop) in enumerate(crops):
            h, w = crop.shape
            drawn[y : y + h, x : x + w] |= crop
            size = int(crop.sum())
            inter = np.bincount(labels[y : y + h, x : x + w][crop], minlength=n + 1)[1:].astype(float)
            iou = inter / (size + areas - inter) if n else np.zeros(0)
            if n:
                best_iou = np.maximum(best_iou, iou)
            if n and iou.max() > IOU_THRESHOLD:
                continue
            cover, purity = (inter / size, inter / np.maximum(areas, 1)) if n else (np.zeros(0), np.zeros(0))
            kind, best = shape(cover, purity)
            rows.append(
                {
                    "frame": f,
                    "annotator": a,
                    "gt": i,
                    "box": [int(x), int(y), int(w), int(h)],
                    "agreement": int(agreement.get((f, a, i), 1)),
                    "area": size,
                    "max_prob": round(float(contexts[f].prob[y : y + h, x : x + w][crop].max()) / 255, 3),
                    "shape": kind,
                    "best": best + 1,
                    "cover": round(float(cover[best]), 3) if best >= 0 else 0.0,
                    "purity": round(float(purity[best]), 3) if best >= 0 else 0.0,
                    "best_area": int(areas[best]) if best >= 0 else 0,
                    "best_kept": bool(kept[f][best]) if best >= 0 else False,
                    "other_annotator": bool(true[f][best]) if best >= 0 else False,
                }
            )
        gt_union += drawn
    for k in range(n):
        if true[f][k] or not kept[f][k]:
            continue
        on = gt_union[labels == k + 1]
        purity = float((on > 0).mean())
        false_rows.append(
            {
                "frame": f,
                "component": k + 1,
                "area": int(areas[k]),
                "purity": round(purity, 3),
                "annotators_under": int(on.max()) if len(on) else 0,
                "best_iou": round(float(best_iou[k]), 3),
                "kind": "unannotated" if purity < TOUCH else ("too_big" if purity < 0.5 else "too_small_or_offset"),
            }
        )
shapes = pd.DataFrame(rows)
false_kept = pd.DataFrame(false_rows)
shapes.to_csv(output_dir() / "shape_rows.csv", index=False)
false_kept.to_csv(output_dir() / "false_kept_rows.csv", index=False)

# %% [markdown]
# ## Summary

# %%


def breakdown(df: pd.DataFrame, column: str) -> dict:
    out = {}
    for key, g in df.groupby(column):
        out[str(key)] = {
            "rows": len(g),
            "share": round(len(g) / len(df), 3),
            **({"agreement_ge_2": round(float((g["agreement"] >= 2).mean()), 3)} if "agreement" in g else {}),
            **({"other_annotator": round(float(g["other_annotator"].mean()), 3)} if "other_annotator" in g else {}),
            "median_area": float(g["area"].median()),
            **({"median_best_area": float(g["best_area"].median())} if "best_area" in g else {}),
            **({"median_max_prob": round(float(g["max_prob"].median()), 3)} if "max_prob" in g else {}),
            **({"median_best_iou": round(float(g["best_iou"].median()), 3)} if "best_iou" in g else {}),
        }
    return out


report = {
    "name": NAME,
    "cutoff": cutoff,
    "unmatched_filaments": len(shapes),
    "shapes": breakdown(shapes, "shape"),
    "shapes_agreement_ge_2": breakdown(shapes[shapes["agreement"] >= 2], "shape"),
    "false_kept": len(false_kept),
    "false_kept_kinds": breakdown(false_kept, "kind"),
}
print(json.dumps(report, indent=2))
(output_dir() / "shapes_report.json").write_text(json.dumps(report, indent=2))

# %% [markdown]
# ## Galleries: the filament in red, kept candidates in cyan, dropped ones in yellow, then the map
#
# Filaments at least two annotators drew, at random per shape.

# %%


def gallery(df: pd.DataFrame, path: str, title: str) -> None:
    picked = df.sample(min(GALLERY, len(df)), random_state=SEED)
    cols = 3
    n_rows = int(np.ceil(len(picked) / cols)) or 1
    fig, axes = plt.subplots(n_rows, 2 * cols, figsize=(4 * cols, 2.1 * n_rows + 0.4))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for n, row in enumerate(picked.itertuples()):
        ctx, labels = contexts[row.frame], raw[row.frame][0]
        x, y, w, h = row.box
        pad = max(40, w // 2, h // 2)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, ctx.flat.shape[1]), min(y + h + pad, ctx.flat.shape[0])
        crops = truth[row.frame][row.annotator]
        gt = np.zeros(ctx.flat.shape, bool)
        gx, gy, crop = crops[row.gt]
        gt[gy : gy + crop.shape[0], gx : gx + crop.shape[1]] = crop
        window = labels[y0:y1, x0:x1]
        keep = np.concatenate([[False], kept[row.frame]])
        image_ax, prob_ax = axes[n // cols, 2 * (n % cols)], axes[n // cols, 2 * (n % cols) + 1]
        image_ax.imshow(ctx.flat[y0:y1, x0:x1], cmap="gray")
        image_ax.contour(gt[y0:y1, x0:x1], [0.5], colors="red", linewidths=0.8)
        for mask, colour in ((keep[window], "cyan"), ((window > 0) & ~keep[window], "yellow")):
            if mask.any():
                image_ax.contour(mask, [0.5], colors=colour, linewidths=0.8)
        image_ax.set_title(f"{row.shape} agree {row.agreement} cover {row.cover} purity {row.purity}", fontsize=7)
        prob_ax.imshow(ctx.prob[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=255)
        prob_ax.contour(gt[y0:y1, x0:x1], [0.5], colors="white", linewidths=0.6)
        prob_ax.set_title(f"max p {row.max_prob}, other ann. {row.other_annotator}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir() / path, dpi=110)
    plt.close(fig)


agreed = shapes[shapes["agreement"] >= 2]
for kind in ["none", "split", "too_big", "too_small", "offset"]:
    subset = agreed[agreed["shape"] == kind]
    if len(subset):
        gallery(subset, f"gallery_{kind}.png", f"{NAME}: unmatched, {kind}, drawn by >= 2 annotators")
log("done")
