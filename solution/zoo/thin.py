# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full
# %% [markdown]
# # Thinner masks: how much does a higher cut on the probability cost out of fold?
#
# CPU only. On the leaderboard, shrinking every kept filament by 2 px cost 0.05 where it cost 0.08 out of fold,
# and growing it cost 0.11 against 0.10 (`lb_probe.py`): the test annotations may be drawn a little thinner.
# A gentler way to thin than erosion is to keep only the confident pixels. Two kinds:
#
# - **mask cut**: the same filaments as now (A0 hysteresis 0.5 / 0.3, join, filter at its out-of-fold cutoff),
#   each reduced to its pixels with probability >= t (a filament left under 30 px is
#   dropped);
# - **hysteresis**: the candidates themselves made at a higher low threshold (and high threshold), the filter
#   refitted on them (`pipeline.fit`).
#
# Each with its out-of-fold PQ, SQ, matches and the median kept area, against the submitted setting; the mask
# cuts also as submissions (`submission_cut{t}.csv`) to try on the leaderboard.
#
# Outputs: `thin.csv`, `submission_cut*.csv`.

# %%
import time

import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess import pipeline
from postprocess.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
)
from postprocess.instances import label_stats, select
from postprocess.pipeline import A0, candidates
from postprocess.tuning import best_cutoff, pq_of
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import PanopticQuality, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
MASK_CUTS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]  # 0.3: the submitted masks
HYSTERESIS = [(0.5, 0.35), (0.5, 0.4), (0.55, 0.4), (0.6, 0.45)]
SUBMIT_CUTS = [0.4, 0.5]
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
maps = {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
files = sorted(maps)
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}


def cut(labels, areas, boxes, prob: np.ndarray, t: float):
    """The instances reduced to their pixels with probability >= t; empty ones dropped."""
    if t <= A0["low"] or not len(areas):
        return labels, areas, boxes
    labels = np.where(prob >= round(t * 255), labels, 0)
    areas, boxes = label_stats(labels, len(areas))
    return select(labels, areas, boxes, areas >= 30)


def summary(name: str, pq: PanopticQuality, kept_areas: list) -> dict:
    return {"setting": name, **pq.summary(), "kept_area_median": float(np.median(kept_areas))}


# %% [markdown]
# ## Mask cuts on the submitted filaments

# %%
raw = {f: candidates(contexts[f]) for f in files}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
scores = out_of_fold_scores(table, target)
cutoff, _ = best_cutoff(scores, raw, table, truth, files)
frames = table["frame"].to_numpy()
submitted = pipeline.Postprocess(ComponentFilter(cutoff=cutoff).fit(table, target))
rows = []
for t in MASK_CUTS:
    pq, kept_areas = PanopticQuality(), []
    for f in files:
        labels, areas, boxes = select(*raw[f], scores[frames == f] >= cutoff)
        labels, areas, _ = cut(labels, areas, boxes, maps[f], t)
        kept_areas += areas.tolist()
        for gt in truth[f]:
            pq.update(gt, labels, areas)
    rows.append(summary(f"cut{t}", pq, kept_areas))
    log(rows[-1])
del raw, table

# %% [markdown]
# ## Higher hysteresis thresholds, filter refitted

# %%
for high, low in HYSTERESIS:
    params = {**A0, "threshold": high, "low": low}
    _, oof = pipeline.fit(contexts, truth, files, params=params)
    tp, fp, fn = (int(oof.terms[c].sum()) for c in ["tp", "fp", "fn"])
    rows.append(
        {
            "setting": f"hyst{high}/{low}",
            "pq": round(pq_of(oof.terms), 4),
            "sq": round(float(oof.terms["iou_sum"].sum()) / max(tp, 1), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "kept_per_frame": round(oof.kept_per_frame, 2),
        }
    )
    log(rows[-1])
result = pd.DataFrame(rows)
print(result.to_string(index=False))
result.to_csv(output_dir() / "thin.csv", index=False)

# %% [markdown]
# ## Submissions with the mask cuts

# %%
del contexts
saved = np.load(find_input(f"{NAME}_test_full.npz"))
test_files = sorted(p.name for p in test_dir.iterdir())
subs = {t: [] for t in SUBMIT_CUTS}
for f in test_files:
    labels, areas, boxes = submitted(frame_context(f, read_gray(test_dir / f), saved[f]))
    for t in SUBMIT_CUTS:
        lab, _, box = cut(labels, areas, boxes, saved[f], t)
        subs[t] += submission.rows_from_label_map(f.rsplit(".", 1)[0], lab, box)
for t, rows_t in subs.items():
    submission.write(rows_t, output_dir() / f"submission_cut{t}.csv")
    log(f"cut {t}: {len(rows_t)} filaments")
log("done")
