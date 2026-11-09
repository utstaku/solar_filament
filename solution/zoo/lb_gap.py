# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full
# %% [markdown]
# # Why is the leaderboard 0.05 below the out-of-fold PQ? The submission file, the test maps, time gaps
#
# CPU only. `swin_unet_boundary` scores 0.4223 out of fold and 0.37 on the public leaderboard. Site and year
# mix do not explain it (`site_gap.py`), and a random 90 frames of the out-of-fold set fall to 0.375 almost
# never. Three things that could:
#
# 1. **The submission file.** Each frame's kept filaments are written exactly as `submission.py` writes them,
#    read back from the RLE strings and scored again; the PQ must not move, and the masks must match pixel for
#    pixel.
# 2. **Test maps unlike validation maps.** Per frame: disk radius, image brightness and contrast on the disk,
#    share of the disk the network calls filament, candidates and kept filaments; per candidate: the filter's
#    features. Out-of-fold maps against the test maps of the fold-0 model (same kind of model) and of the
#    full-data model (the submitted one).
# 3. **Time to the nearest training frame.** A validation frame's nearest frame in its model's training folds
#    against a test frame's nearest training frame; out-of-fold PQ by that gap, reweighted to the test's gaps.
#
# Outputs: `lb_gap.json`, `lb_gap_frames.csv`, `lb_gap_candidates.csv`.

# %%
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess import pipeline
from postprocess.filtering import FEATURES, component_table, frame_context
from postprocess.tuning import pq_of
from preprocess.images import read_gray
from scoring import rle, submission
from scoring.metric import PanopticQuality, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
GAP_BINS = [0, 0.5, 1, 2, 4, np.inf]
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
maps, fold_of = {}, {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
    fold_of |= {f: k for f in saved.files}
files = sorted(maps)
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
post, oof = pipeline.fit(contexts, truth, files)
log(f"out-of-fold PQ {pq_of(oof.terms):.4f}")


# %% [markdown]
# ## 1. Submission round trip


# %%
def scored(gt: list, labels: np.ndarray, areas: np.ndarray) -> PanopticQuality:
    pq = PanopticQuality()
    for crops in gt:
        pq.update(crops, labels, areas)
    return pq


direct, decoded = PanopticQuality(), PanopticQuality()
mismatched_pixels, rows_written = 0, 0
for f in files:
    labels, areas, boxes = post(contexts[f])
    rows = submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    rows_written += len(rows)
    back = np.zeros(labels.shape, np.int32)
    for k, row in enumerate(rows, start=1):
        mask = rle.decode(row["segmentation_rle"]).astype(bool)
        mismatched_pixels += int((mask != (labels == k)).sum())
        back[mask] = k
    back_areas = np.bincount(back.ravel(), minlength=len(rows) + 1)[1:]
    for pq, (lab, ar) in [(direct, (labels, areas)), (decoded, (back, back_areas))]:
        part = scored(truth[f], lab, ar)
        pq.iou_sum, pq.tp, pq.fp, pq.fn = pq.iou_sum + part.iou_sum, pq.tp + part.tp, pq.fp + part.fp, pq.fn + part.fn
round_trip = {
    "rows": rows_written,
    "pq_direct_in_sample_filter": direct.summary(),
    "pq_after_rle": decoded.summary(),
    "mismatched_pixels": mismatched_pixels,
}
log(round_trip)


# %% [markdown]
# ## 2. Validation maps against test maps


# %%
def disk_mask(ctx, step: int = 4) -> np.ndarray:
    size = ctx.prob.shape[0]
    yy, xx = np.mgrid[0:size:step, 0:size:step]
    return (xx - ctx.centre[0]) ** 2 + (yy - ctx.centre[1]) ** 2 < (0.98 * ctx.radius) ** 2


def frame_stats(ctx, image: np.ndarray, split: str):
    disk = disk_mask(ctx)
    prob, img, flat = ctx.prob[::4, ::4][disk], image[::4, ::4][disk], ctx.flat[::4, ::4][disk]
    labels, areas, boxes = pipeline.candidates(ctx, post.params, post.join)
    table = component_table(ctx, labels, boxes) if len(areas) else pd.DataFrame(columns=["frame", *FEATURES])
    score = post.filter.scores(table) if len(table) else np.zeros(0)
    kept = score >= post.filter.cutoff
    row = {
        "frame": ctx.name,
        "split": split,
        "site": ctx.name[14:16],
        "radius": ctx.radius,
        "image_mean": float(img.mean()),
        "image_std": float(img.std()),
        "flat_std": float(flat.std()),
        "foreground_share": float((prob >= 128).mean()),
        "prob_mean": float(prob.mean()) / 255,
        "candidates": len(areas),
        "kept": int(kept.sum()),
        "kept_area_median": float(np.median(areas[kept])) if kept.any() else np.nan,
        "score_mean": float(score.mean()) if len(score) else np.nan,
    }
    return row, table.assign(split=split, score=score, kept=kept)


frame_rows, candidate_tables = [], []
for f in files:
    row, table = frame_stats(contexts[f], read_gray(train_dir / f), "val")
    frame_rows.append(row)
    candidate_tables.append(table)
del contexts
log("validation frames done")

test_files = sorted(p.name for p in test_dir.iterdir())
for source in ["fold0", "full"]:
    saved = np.load(find_input(f"{NAME}_test_{source}.npz"))
    for f in test_files:
        image = read_gray(test_dir / f)
        row, table = frame_stats(frame_context(f, image, saved[f]), image, f"test_{source}")
        frame_rows.append(row)
        candidate_tables.append(table)
    log(f"test {source} done")
frames = pd.DataFrame(frame_rows)
cands = pd.concat(candidate_tables, ignore_index=True)

per_frame = frames.drop(columns=["frame", "site"]).groupby("split").median().round(4)
per_frame_mean = frames[["split", "candidates", "kept", "foreground_share"]].groupby("split").mean().round(4)
print(per_frame.T.to_string())
print(per_frame_mean.T.to_string())
quantiles = {}
for split, g in cands.groupby("split"):
    quantiles[split] = {
        "candidates": len(g),
        "kept_share": round(float(g["kept"].mean()), 3),
        **{c: [round(float(v), 3) for v in g[c].quantile([0.1, 0.5, 0.9])] for c in FEATURES},
        **{f"kept_{c}": [round(float(v), 3) for v in g.loc[g["kept"], c].quantile([0.1, 0.5, 0.9])] for c in FEATURES},
    }
print(json.dumps(quantiles, indent=1))


# %% [markdown]
# ## 3. Time to the nearest training frame


# %%
def when(f: str) -> float:
    return datetime.strptime(f[:14], "%Y%m%d%H%M%S").timestamp() / 86400


train_t = np.array([when(f) for f in files])
train_fold = np.array([fold_of[f] for f in files])
val_gap = {f: float(np.abs(train_t[train_fold != fold_of[f]] - when(f)).min()) for f in files}
test_gap = {f: float(np.abs(train_t - when(f)).min()) for f in test_files}

terms = oof.terms.set_index("frame").assign(gap=pd.Series(val_gap))
terms["bin"] = pd.cut(terms["gap"], GAP_BINS, right=False).astype(str)
test_bins = pd.cut(pd.Series(test_gap), GAP_BINS, right=False).astype(str).value_counts(normalize=True)
by_gap = {}
parts = []
for key, g in terms.groupby("bin"):
    by_gap[key] = {
        "val_frames": len(g),
        "val_share": round(len(g) / len(terms), 3),
        "test_share": round(float(test_bins.get(key, 0)), 3),
        "pq": round(pq_of(g), 4),
    }
    parts.append(g[["tp", "fp", "fn", "iou_sum"]].sum() / len(g) * test_bins.get(key, 0))
total = sum(parts)
gaps = {
    "by_gap_days": by_gap,
    "oof_pq_reweighted_to_test_gaps": round(
        float(total["iou_sum"] / (total["tp"] + 0.5 * total["fp"] + 0.5 * total["fn"])), 4
    ),
    "val_gap_quantiles": [round(v, 3) for v in np.quantile(list(val_gap.values()), [0.1, 0.25, 0.5, 0.75, 0.9])],
    "test_gap_quantiles": [round(v, 3) for v in np.quantile(list(test_gap.values()), [0.1, 0.25, 0.5, 0.75, 0.9])],
}
print(json.dumps(gaps, indent=1))

# %%
report = {
    "name": NAME,
    "oof_pq": round(pq_of(oof.terms), 4),
    "round_trip": round_trip,
    "per_frame_median": per_frame.to_dict(),
    "per_frame_mean": per_frame_mean.to_dict(),
    "candidates": quantiles,
    "time_gap": gaps,
}
(output_dir() / "lb_gap.json").write_text(json.dumps(report, indent=2, default=str))
frames.assign(gap=frames["frame"].map(val_gap | test_gap)).to_csv(output_dir() / "lb_gap_frames.csv", index=False)
cands.to_csv(output_dir() / "lb_gap_candidates.csv", index=False)
log("done")
