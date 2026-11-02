# kaggle: kernel_sources={user}/filament-cnn-baseline-predict
# %% [markdown]
# # Which features the component filter of `predict.py` relies on
#
# CPU only: the validation probability maps and the chosen post-processing come from the
# `filament-cnn-baseline-predict` output (`val_probs.npz`, `predict_report.json`); nothing is predicted.
#
# The stage-B components are rebuilt and the filter's out-of-fold PQ must reproduce the report before
# anything else is computed. Then, per feature:
#
# 1. **impurity importance** of the filter fitted on all frames (what the trees split on; biased, for reference);
# 2. **permutation importance**: the drop in ROC AUC and in PQ at the chosen cutoff when the feature is
#    shuffled in the held-out folds;
# 3. **drop-column**: the filter refitted without the feature, its cutoff chosen again, and the change in PQ
#    with a paired bootstrap interval over frames.
#
# Outputs: `filter_importance.json`, and `filter_components.csv` (one row per component: features,
# target, out-of-fold score, kept) for the analysis by distance from the disk centre.

# %%
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cnn_baseline.data import split_files
from cnn_baseline.importance import drop_column, paired_bootstrap, permutation_importance
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import (
    FEATURES,
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
)
from postprocess.instances import instances
from postprocess.tuning import best_cutoff, frame_terms, pq_of
from preprocess.images import read_gray
from scoring.metric import instance_crops

SEED = 0  # must match cnn_baseline/main.py and predict.py
VALIDATION_FRACTION = 0.15
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs: the saved maps and what `predict.py` chose

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
_, val_files = split_files([p.name for p in train_dir.iterdir()], VALIDATION_FRACTION, SEED)

report = json.loads(find_input("predict_report.json").read_text())
params, cutoff = report["chosen"]["params"], report["chosen"]["filter_cutoff"]
expected = report["stage_b"][str(params["min_area"])]["best"]
log("chosen:", params, "cutoff", cutoff, "expected out-of-fold PQ", expected["pq"])

saved = np.load(find_input("val_probs.npz"))
assert sorted(saved.files) == sorted(val_files), "val_probs.npz holds other frames than this validation split"
probs = {f: saved[f] for f in val_files}
images = {f: read_gray(train_dir / f) for f in val_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}
log(f"{len(val_files)} validation frames loaded")

# %% [markdown]
# ## Stage B rebuilt, and the sanity gate
#
# Rows are concatenated in `val_files` order as in `predict.py`: GroupKFold's folds depend on that order,
# so any other order gives other out-of-fold scores.

# %%
raw = {f: instances(probs[f], **params) for f in val_files}
table = pd.concat(
    [component_table(frame_context(f, images[f], probs[f]), raw[f][0], raw[f][2]) for f in val_files],
    ignore_index=True,
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in val_files])
oof = out_of_fold_scores(table, target)
base_cutoff, base_terms = best_cutoff(oof, raw, table, truth, val_files)
base = {"cutoff": base_cutoff, "pq": round(pq_of(base_terms), 4), **base_terms[["tp", "fp", "fn"]].sum().to_dict()}
log(f"{len(table)} components, {int(target.sum())} positive; rebuilt:", base)
assert base_cutoff == cutoff and abs(base["pq"] - expected["pq"]) < 1e-4, "does not reproduce predict.py"
assert all(int(base[k]) == expected[k] for k in ("tp", "fp", "fn")), "does not reproduce predict.py"
log("sanity gate passed")

# %% [markdown]
# ## Importances


# %%
def auc(scores):
    return roc_auc_score(target, scores)


def pq_at_cutoff(scores):
    return pq_of(frame_terms(raw, table, scores >= base_cutoff, truth, val_files))


def choose_cutoff(scores):
    return best_cutoff(scores, raw, table, truth, val_files)


impurity = ComponentFilter().fit(table, target).model.feature_importances_
rows = []
for feature, impurity_importance in zip(FEATURES, impurity):
    auc_drops = permutation_importance(table, target, feature, auc)
    pq_drops = permutation_importance(table, target, feature, pq_at_cutoff)
    dropped_cutoff, dropped_terms = drop_column(table, target, feature, choose_cutoff)
    delta, lo, hi = paired_bootstrap(base_terms, dropped_terms)
    row = {
        "feature": feature,
        "impurity": round(float(impurity_importance), 4),
        "perm_auc_drop": round(float(auc_drops.mean()), 4),
        "perm_auc_drop_sd": round(float(auc_drops.std()), 4),
        "perm_pq_drop": round(float(pq_drops.mean()), 4),
        "perm_pq_drop_sd": round(float(pq_drops.std()), 4),
        "dropped_cutoff": dropped_cutoff,
        "dropped_pq": round(pq_of(dropped_terms), 4),
        "dropped_minus_full_pq": round(delta, 4),  # negative: PQ is lost without the feature
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
    }
    rows.append(row)
    log(row)
importance = pd.DataFrame(rows)
print(f"full filter: out-of-fold AUC {auc(oof):.4f}, PQ {base['pq']} at cutoff {base_cutoff}")
print(importance.to_string(index=False))

# %% [markdown]
# ## Outputs

# %%
components = table.assign(target=target, oof_score=oof, kept=oof >= base_cutoff)
components.to_csv(output_dir() / "filter_components.csv", index=False)
result = {
    "params": params,
    "full": {**base, "oof_auc": round(float(auc(oof)), 4)},
    "features": rows,
    "n_components": len(table),
    "n_positive": int(target.sum()),
    "elapsed_sec": round(time.time() - T0, 1),
}
(output_dir() / "filter_importance.json").write_text(json.dumps(result, indent=2, default=float))
log("wrote filter_importance.json and filter_components.csv")
