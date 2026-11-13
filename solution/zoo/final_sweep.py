# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full,{user}/filament-zoo-tta
# %% [markdown]
# # One model, steadier maps, thinner masks: the two together
#
# CPU only. Two single-model changes each raised the out-of-fold PQ of `swin_unet_boundary`: the mean of 8 flips
# and turns at inference (`tta.py`, +0.004) and keeping only the pixels of a kept filament with probability
# >= 0.5 instead of 0.3 (`thin.py`, +0.005, still rising at 0.55). Here both, on plain and 8-view maps:
#
# - candidates from A0 hysteresis (0.5 / 0.3) or a higher one (0.6 / 0.45), joined, filter fitted out of fold;
# - each kept filament cut at t = 0.3 (the submitted masks) ... 0.7.
#
# `delta`: PQ minus the submitted setting's (plain maps, A0, t = 0.3) on the same frames, with a 95 % interval
# from resampling frames. Submissions of the 8-view maps at a few cuts: `submission_tta8_<hyst>_cut<t>.csv`.
#
# Outputs: `final_sweep.csv`, `submission_*.csv`.

# %%
import time

import numpy as np
import pandas as pd

from cnn_baseline.importance import paired_bootstrap
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
from postprocess.tuning import TERMS, best_cutoff, pq_of
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import PanopticQuality, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
MAPS = {"plain": NAME, "tta8": f"{NAME}_tta8"}
HYSTERESIS = {"a0": A0, "h60": {**A0, "threshold": 0.6, "low": 0.45}}
MASK_CUTS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
SUBMIT = [("tta8", "a0", 0.5), ("tta8", "a0", 0.6), ("tta8", "h60", 0.6)]
REFERENCE = ("plain", "a0", 0.3)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def cut(labels, areas, boxes, prob: np.ndarray, t: float, low: float):
    """The instances reduced to their pixels with probability >= t; ones left under 30 px dropped."""
    if t <= low or not len(areas):
        return labels, areas, boxes
    labels = np.where(prob >= round(t * 255), labels, 0)
    areas, boxes = label_stats(labels, len(areas))
    return select(labels, areas, boxes, areas >= 30)


root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
images = None
truth = None
rows, terms, posts = [], {}, {}
for kind, prefix in MAPS.items():
    maps = {}
    for k in range(N_FOLDS):
        saved = np.load(find_input(f"{prefix}_oof_fold{k}.npz"))
        maps |= {f: saved[f] for f in saved.files}
    files = sorted(maps)
    if images is None:
        images = {f: read_gray(train_dir / f) for f in files}
        truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
    contexts = {f: frame_context(f, images[f], maps[f]) for f in files}
    for hyst, params in HYSTERESIS.items():
        raw = {f: candidates(contexts[f], params) for f in files}
        table = pd.concat(
            [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
        )
        target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
        scores = out_of_fold_scores(table, target)
        cutoff, _ = best_cutoff(scores, raw, table, truth, files)
        posts[kind, hyst] = pipeline.Postprocess(ComponentFilter(cutoff=cutoff).fit(table, target), dict(params))
        frames = table["frame"].to_numpy()
        kept = {f: select(*raw[f], scores[frames == f] >= cutoff) for f in files}
        for t in MASK_CUTS:
            frame_rows, kept_areas = [], []
            for f in files:
                labels, areas, _ = cut(*kept[f], maps[f], t, params["low"])
                kept_areas += areas.tolist()
                pq = PanopticQuality()
                for gt in truth[f]:
                    pq.update(gt, labels, areas)
                frame_rows.append({"frame": f, "tp": pq.tp, "fp": pq.fp, "fn": pq.fn, "iou_sum": pq.iou_sum})
            terms[kind, hyst, t] = pd.DataFrame(frame_rows, columns=["frame", *TERMS])
            tp = int(terms[kind, hyst, t]["tp"].sum())
            rows.append(
                {
                    "maps": kind,
                    "hysteresis": hyst,
                    "cut": t,
                    "pq": round(pq_of(terms[kind, hyst, t]), 4),
                    "sq": round(float(terms[kind, hyst, t]["iou_sum"].sum()) / max(tp, 1), 4),
                    **{c: int(terms[kind, hyst, t][c].sum()) for c in ["tp", "fp", "fn"]},
                    "filter_cutoff": cutoff,
                    "kept_area_median": float(np.median(kept_areas)),
                }
            )
            log(rows[-1])
        del raw, table, kept
    del contexts, maps

for row in rows:
    delta, lo, hi = paired_bootstrap(terms[REFERENCE], terms[row["maps"], row["hysteresis"], row["cut"]])
    row["delta"], row["ci95"] = round(delta, 4), [round(lo, 4), round(hi, 4)]
table = pd.DataFrame(rows).sort_values("pq", ascending=False)
print(table.to_string(index=False))
table.to_csv(output_dir() / "final_sweep.csv", index=False)

# %%
test_files = sorted(p.name for p in test_dir.iterdir())
test_maps = {kind: np.load(find_input(f"{prefix}_test_full.npz")) for kind, prefix in MAPS.items()}
for kind, hyst, t in SUBMIT:
    post, sub = posts[kind, hyst], []
    for f in test_files:
        prob = test_maps[kind][f]
        labels, areas, boxes = post(frame_context(f, read_gray(test_dir / f), prob))
        labels, _, boxes = cut(labels, areas, boxes, prob, t, post.params["low"])
        sub += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    submission.write(sub, output_dir() / f"submission_{kind}_{hyst}_cut{t}.csv")
    log(f"{kind} {hyst} cut {t}: {len(sub) / len(test_files):.2f} filaments per test frame")
log("done")
