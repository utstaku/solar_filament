# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary
# %% [markdown]
# # A filter target that follows PQ: how many annotators a candidate matches, not whether any does
#
# PQ is summed over annotator x frame: a kept candidate is matched or a false positive once per annotator.
# Kept, it adds the IoU of every annotator it matches to the numerator, and 0.5 per annotator to the
# denominator TP + FP/2 + FN/2 (a match: TP +1, FN -1; no match: FP +1). So it raises PQ when its **mean
# IoU per annotator** (unmatched annotators count 0) exceeds about PQ / 2: a candidate one annotator in three drew is about neutral, one all three drew is well worth keeping.
# The component filter (`filtering.ComponentFilter`) learns "matches any annotator", which ranks those two
# alike.
#
# On the 707 out-of-fold maps of `zoo/swin_unet_boundary.py` and the submitted candidates (A0, join), with
# the filter's features and gradient boosting of the same size, out of fold on the same frame groups:
#
# - **any**: the current classifier, P(matches any annotator);
# - **share**: regression of the share of the frame's annotators it matches;
# - **mean_iou**: regression of its mean IoU per annotator (the PQ gain above).
#
# Each is cut at the PQ-best of 33 quantiles of its own scores; also the PQ of keeping exactly the candidates
# whose true mean IoU per annotator beats each threshold (an upper bound for that target).
#
# Outputs: `filter_target.csv`, `filter_target_rows.csv`.

# %%
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from error_analysis.instances import overlaps
from postprocess.filtering import FEATURES, component_table, frame_context
from postprocess.pipeline import candidates
from postprocess.tuning import best_cutoff, frame_terms, pq_of
from preprocess.images import read_gray
from scoring.metric import IOU_THRESHOLD, instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
QUANTILES = np.linspace(0.1, 0.9, 33)
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

# %% [markdown]
# ## Per candidate: annotators matched and mean IoU per annotator

# %%
matched, iou_sum, n_annotators = [], [], []
for f in files:
    labels, areas, _ = raw[f]
    if not len(areas):
        continue
    count, total = np.zeros(len(areas)), np.zeros(len(areas))
    for crops in truth[f]:
        if crops:
            iou = overlaps(crops, labels, areas)[1].max(axis=0)
            hit = iou > IOU_THRESHOLD
            count += hit
            total += np.where(hit, iou, 0)
    matched.append(count)
    iou_sum.append(total)
    n_annotators.append(np.full(len(areas), len(truth[f])))
matched, iou_sum, n_annotators = (np.concatenate(a) for a in (matched, iou_sum, n_annotators))
table = table.assign(
    matched=matched.astype(int),
    n_annotators=n_annotators,
    share=matched / n_annotators,
    mean_iou=iou_sum / n_annotators,
)
log(f"{len(table)} candidates; matched by 0/1/2/3: {np.bincount(table['matched'], minlength=4).tolist()}")

# %% [markdown]
# ## Out-of-fold scores for each target

# %%
groups = table["frame"].to_numpy()
splits = list(GroupKFold(n_splits=5).split(table, table["matched"], groups))
x = table[FEATURES].to_numpy()
scores = {k: np.zeros(len(table)) for k in ["any", "share", "mean_iou"]}
for train_idx, test_idx in splits:
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
    clf.fit(x[train_idx], (table["matched"].to_numpy()[train_idx] > 0).astype(int))
    scores["any"][test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    for target in ["share", "mean_iou"]:
        reg = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
        reg.fit(x[train_idx], table[target].to_numpy()[train_idx])
        scores[target][test_idx] = reg.predict(x[test_idx])

rows = []
for name, s in scores.items():
    cutoff, terms = best_cutoff(s, raw, table, truth, files, cutoffs=np.quantile(s, QUANTILES))
    rows.append(
        {
            "filter": name,
            "pq": round(pq_of(terms), 4),
            **{c: int(terms[c].sum()) for c in ["tp", "fp", "fn"]},
            "kept": int((s >= cutoff).sum()),
            "kept_share": round(float((s >= cutoff).mean()), 3),
        }
    )
    log(rows[-1])
for threshold in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]:
    keep = table["mean_iou"].to_numpy() > threshold
    terms = frame_terms(raw, table, keep, truth, files)
    rows.append(
        {
            "filter": f"oracle_mean_iou>{threshold}",
            "pq": round(pq_of(terms), 4),
            **{c: int(terms[c].sum()) for c in ["tp", "fp", "fn"]},
            "kept": int(keep.sum()),
            "kept_share": round(float(keep.mean()), 3),
        }
    )
    log(rows[-1])
result = pd.DataFrame(rows)
print(result.to_string(index=False))
result.to_csv(output_dir() / "filter_target.csv", index=False)
table.assign(**{f"score_{k}": v for k, v in scores.items()}).to_csv(
    output_dir() / "filter_target_rows.csv", index=False
)
log("done")
