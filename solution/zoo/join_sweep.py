# kaggle: kernel_sources={user}/filament-zoo-swin-unet
# %% [markdown]
# # Does the join make the too-big candidates? Its reach against PQ and merged filaments
#
# CPU only, on the 707 out-of-fold maps of `zoo/swin_unet.py`. `spill.py` found that 40 % of the too-big
# candidates over filaments two or more annotators drew spill onto another filament of the same annotator,
# while an annotator's filaments rarely lie within 10 px of each other. The join (`joining.join_rule`) links
# pieces that continue each other up to `far` = 50 px apart and any two up to `near` = 10 px, so it may be the
# one merging them. For each (near, far):
#
# - PQ with the filter (out-of-fold scores, PQ-best cutoff) and with a perfect filter;
# - **merges**: kept candidates covering >= 30 % of two or more filaments of one annotator.
#
# Outputs: `join_sweep.csv`.

# %%
import time

import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.instances import instances
from postprocess.joining import join_pairs, join_rule, pair_table
from postprocess.pipeline import A0, PAIR_RADIUS
from postprocess.tuning import best_cutoff, frame_terms, pq_of
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet"
REACH = [(10, 50), (0, 0), (10, 10), (10, 20), (10, 30), (5, 50), (0, 50), (5, 30)]  # (near, far), submitted first
MERGE_COVER = 0.3
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
pieces = {f: instances(contexts[f].prob, **A0) for f in files}
pairs = {f: pair_table(contexts[f], pieces[f][0], PAIR_RADIUS) for f in files}
log(f"{len(files)} frames, pieces and pairs ready")


def merges(labels: np.ndarray, n: int, keep: np.ndarray, crops_per_annotator) -> int:
    count = 0
    for crops in crops_per_annotator:
        if len(crops) < 2 or not n:
            continue
        cover = np.zeros((len(crops), n))
        for g, (x, y, crop) in enumerate(crops):
            h, w = crop.shape
            cover[g] = np.bincount(labels[y : y + h, x : x + w][crop], minlength=n + 1)[1:] / crop.sum()
        count += int((((cover >= MERGE_COVER).sum(axis=0) >= 2) & keep).sum())
    return count


# %%
rows = []
for near, far in REACH:
    raw = {}
    for f in files:
        labels, areas, boxes = pieces[f]
        table = pairs[f]
        if near + far and len(table):
            labels, areas, boxes = join_pairs(labels, table, join_rule(table, near=near, far=far))
        raw[f] = (labels.astype(np.uint8), areas, boxes)
    table = pd.concat(
        [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    scores = out_of_fold_scores(table, target)
    cutoff, filtered = best_cutoff(scores, raw, table, truth, files)
    perfect = frame_terms(raw, table, target, truth, files)
    frames, keep = table["frame"].to_numpy(), scores >= cutoff
    merged = sum(merges(raw[f][0], len(raw[f][1]), keep[frames == f], truth[f]) for f in files if len(raw[f][1]))
    rows.append(
        {
            "near": near,
            "far": far,
            "candidates": len(table),
            "cutoff": cutoff,
            "filter_pq": round(pq_of(filtered), 4),
            **{f"filter_{c}": int(filtered[c].sum()) for c in ["tp", "fp", "fn"]},
            "perfect_pq": round(pq_of(perfect), 4),
            "kept_merges": merged,
        }
    )
    log(rows[-1])
sweep = pd.DataFrame(rows)
print(sweep.to_string(index=False))
sweep.to_csv(output_dir() / "join_sweep.csv", index=False)
log("done")
