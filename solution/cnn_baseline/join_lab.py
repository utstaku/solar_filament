# kaggle: kernel_sources={user}/filament-cnn-baseline-predict
# %% [markdown]
# # Join lab: which kept components belong to the same filament
#
# CPU only, on the probability maps saved by `filament-cnn-baseline-predict`. Joining A0's kept
# components within 25 px raised validation PQ by +0.005 (`postprocess_lab.py`), but distance alone cannot
# tell how two pieces relate. A filament breaks along its spine (pieces end to end, the gap along their
# axis, often still faintly dark) or shows a barb (a small piece at the side of the spine, at an angle);
# two different filaments lie side by side or in different channels (`cnn_baseline/pairs.py`).
#
# 1. **Diagnosis.** Every pair of A0's kept validation components within 50 px, with its features and
#    whether some annotator drew both pieces as one filament (`same`) or as two (`different`).
#    How well does each feature separate them, by distance? How many `same` pairs are at an angle (barbs)?
# 2. **Variants**, each against A0 with a paired bootstrap over frames:
#    - `join25`: the distance rule from the first lab (reference);
#    - `join_rule`: <= 50 px when the pieces continue each other end to end along the axis, else <= 10 px;
#    - `join_dark`: <= 25 px and the gap faintly dark (>= 3 % below background) or prob >= 0.1;
#    - `join_learned`: a pair classifier (gradient boosting, out-of-fold by frame) joins when P(same) > 0.5.
#
# Outputs: `join_pairs.csv`, `join_results.csv` / `.json`, `submission_<variant>.csv`.

# %%
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from cnn_baseline.data import split_files
from cnn_baseline.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
    trainable,
)
from cnn_baseline.importance import best_cutoff, paired_bootstrap, pq_of
from cnn_baseline.pairs import PAIR_FEATURES, aligned_end_to_end, join_pairs, pair_table, same_filament
from cnn_baseline.postprocess import instances, select
from cnn_baseline.variants import join, score_frames
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops

SEED = 0  # must match cnn_baseline/main.py and predict.py
VALIDATION_FRACTION = 0.15
PAIR_RADIUS = 50
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
_, val_files = split_files([p.name for p in train_dir.iterdir()], VALIDATION_FRACTION, SEED)
test_files = sorted(p.name for p in test_dir.iterdir())

report = json.loads(find_input("predict_report.json").read_text())
params, a0_cutoff = report["chosen"]["params"], report["chosen"]["filter_cutoff"]
expected = report["stage_b"][str(params["min_area"])]["best"]
val_saved, test_saved = np.load(find_input("val_probs.npz")), np.load(find_input("test_probs.npz"))
assert sorted(val_saved.files) == sorted(val_files), "val_probs.npz holds other frames than this validation split"
assert sorted(test_saved.files) == test_files, "test_probs.npz holds other frames than the test set"
probs = {f: val_saved[f] for f in val_files} | {f: test_saved[f] for f in test_files}
contexts = {f: frame_context(f, read_gray(train_dir / f), probs[f]) for f in val_files}
contexts |= {f: frame_context(f, read_gray(test_dir / f), probs[f]) for f in test_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}
log(f"{len(val_files)} validation and {len(test_files)} test frames loaded; A0 = {params}, cutoff {a0_cutoff}")

# %% [markdown]
# ## A0 and the sanity gate (as in `postprocess_lab.py`; scored frame by frame to fit in memory)

# %%


def compact(inst):
    labels, areas, boxes = inst
    return labels.astype(np.uint8), areas, boxes


raw = {f: compact(instances(probs[f], **params)) for f in val_files}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in val_files if len(raw[f][1])], ignore_index=True
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in val_files])
oof = out_of_fold_scores(table, target)
cutoff, _ = best_cutoff(oof, raw, table, truth, val_files)
component_filter = ComponentFilter(cutoff=cutoff).fit(table, target)
frames, keep = table["frame"].to_numpy(), oof >= cutoff


def a0_val(f):
    return select(*raw[f], keep[frames == f])


def a0_test(f):
    inst = instances(probs[f], **params)
    if not len(inst[1]):
        return inst
    return select(*inst, component_filter.keep(component_table(contexts[f], inst[0], inst[2])))


def summary(terms: pd.DataFrame) -> dict:
    tp, fp, fn = (int(terms[c].sum()) for c in ["tp", "fp", "fn"])
    sq = float(terms["iou_sum"].sum()) / tp if tp else 0.0
    pq = pq_of(terms)
    return {
        "pq": round(pq, 4),
        "sq": round(sq, 4),
        "rq": round(pq / sq, 4) if sq else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def val_terms(val) -> pd.DataFrame:
    return pd.concat([score_frames({f: val(f)}, truth, [f]) for f in val_files], ignore_index=True)


a0_terms = val_terms(a0_val)
a0_summary = summary(a0_terms)
log("A0 rebuilt:", a0_summary, "cutoff", cutoff)
assert cutoff == a0_cutoff and abs(a0_summary["pq"] - expected["pq"]) < 1e-4, "does not reproduce predict.py"
assert all(a0_summary[k] == expected[k] for k in ("tp", "fp", "fn")), "does not reproduce predict.py"
log("sanity gate passed")

# %% [markdown]
# ## 1. Diagnosis: pairs of kept components on the validation frames

# %%
val_pairs = []
for f in val_files:
    labels, areas, _ = a0_val(f)
    t = pair_table(contexts[f], labels, PAIR_RADIUS)
    t["same"], t["different"] = same_filament(truth[f], labels, areas, t)
    t["aligned"] = aligned_end_to_end(t) if len(t) else []
    val_pairs.append(t)
val_pairs = pd.concat([t for t in val_pairs if len(t)], ignore_index=True)
val_pairs.to_csv(output_dir() / "join_pairs.csv", index=False)
log(
    f"{len(val_pairs)} pairs within {PAIR_RADIUS} px; same {int(val_pairs['same'].sum())}, different {int(val_pairs['different'].sum())}"
)

bins = pd.cut(val_pairs["dist"], [0, 10, 25, 50], include_lowest=True).astype(str)
labelled = val_pairs[val_pairs["same"] | val_pairs["different"]]
diagnosis = {"counts": val_pairs.groupby(bins, observed=True)[["same", "different"]].sum().astype(int).to_dict()}
auc = {}
for feature in PAIR_FEATURES:
    auc[feature] = {
        str(b): round(roc_auc_score(g["same"], g[feature]), 3)
        for b, g in labelled.groupby(pd.cut(labelled["dist"], [0, 10, 25, 50], include_lowest=True), observed=True)
        if 0 < g["same"].sum() < len(g)
    }
diagnosis["auc_same_vs_different"] = auc
same = val_pairs[val_pairs["same"]]
diagnosis["same_at_an_angle"] = {
    "share_angle_diff_over_45deg": round(float((same["angle_diff"] > np.radians(45)).mean()), 3),
    "share_aligned_end_to_end": round(float(same["aligned"].mean()), 3),
    "different_aligned_end_to_end": round(float(val_pairs.loc[val_pairs["different"], "aligned"].mean()), 3),
}
print(pd.DataFrame(auc).T.to_string())
print(json.dumps({k: v for k, v in diagnosis.items() if k != "auc_same_vs_different"}, indent=2, default=str))

# %% [markdown]
# ## 2. Variants

# %%
results = []


def evaluate(name: str, val, test, rule: str, terms: pd.DataFrame | None = None):
    terms = val_terms(val) if terms is None else terms
    delta, lo, hi = paired_bootstrap(a0_terms, terms) if results else (0.0, 0.0, 0.0)
    rows = []
    for f in test_files:
        labels, _, boxes = test(f)
        rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    submission.write(rows, output_dir() / f"submission_{name}.csv")
    row = {
        "variant": name,
        **summary(terms),
        "delta": round(delta, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
    }
    results.append({**row, "n_test": len(rows), "rule": rule})
    log(results[-1])


def by_pairs(decide):
    """A variant that joins A0's kept pieces where `decide(pair table)` is True."""

    def run(inst, f):
        labels = inst[0]
        t = pair_table(contexts[f], labels, PAIR_RADIUS)
        return join_pairs(labels, t, decide(t)) if len(t) else inst

    return (lambda f: run(a0_val(f), f)), (lambda f: run(a0_test(f), f))


def pair_classifier():
    return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)


def rule(t):
    return ((t["dist"] <= 10) | ((t["dist"] <= 50) & aligned_end_to_end(t))).to_numpy()


def dark(t):
    return ((t["dist"] <= 25) & ((t["path_dark"] >= 0.03) | (t["path_prob_mean"] >= 0.1))).to_numpy()


evaluate("a0", a0_val, a0_test, f"filter cutoff {cutoff}", a0_terms)
evaluate("join25", lambda f: join(a0_val(f)[0], 25), lambda f: join(a0_test(f)[0], 25), "join within 25 px")
evaluate("join_rule", *by_pairs(rule), "end to end along the axis <= 50 px, otherwise <= 10 px")
evaluate("join_dark", *by_pairs(dark), "<= 25 px and gap >= 3 % darker than background or prob >= 0.1")

# learned: out-of-fold P(same) on the validation pairs, a model on all of them for the test frames
same_target = val_pairs["same"].to_numpy(bool)
if trainable(same_target):
    groups = val_pairs["frame"].to_numpy()
    oof_same = np.zeros(len(val_pairs))
    splits = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(val_pairs, same_target, groups)
    for train_idx, test_idx in splits:
        fold = pair_classifier().fit(val_pairs.iloc[train_idx][PAIR_FEATURES], same_target[train_idx])
        oof_same[test_idx] = fold.predict_proba(val_pairs.iloc[test_idx][PAIR_FEATURES])[:, 1]
    pair_model = pair_classifier().fit(val_pairs[PAIR_FEATURES], same_target)
    diagnosis["pair_classifier_oof_auc"] = round(float(roc_auc_score(same_target, oof_same)), 3)
    log("pair classifier: out-of-fold AUC", diagnosis["pair_classifier_oof_auc"])
    rows_of = {f: g.index.to_numpy() for f, g in val_pairs.groupby("frame")}

    def learned_val(f):
        inst = a0_val(f)
        if f not in rows_of:
            return inst
        t = pair_table(contexts[f], inst[0], PAIR_RADIUS)  # the same rows, in the same order, as in val_pairs
        return join_pairs(inst[0], t, oof_same[rows_of[f]] > 0.5)

    _, learned_test = by_pairs(lambda t: pair_model.predict_proba(t[PAIR_FEATURES])[:, 1] > 0.5)
    evaluate("join_learned", learned_val, learned_test, "pair classifier P(same) > 0.5 (out of fold on validation)")
else:
    log(f"join_learned skipped: {len(same_target)} pairs, {int(same_target.sum())} same -- too few to train on")

# %% [markdown]
# ## Results
#
# `delta` = PQ(variant) - PQ(A0), 95 % paired-bootstrap interval over frames (too narrow if anything:
# consecutive days are not independent).

# %%
table_out = pd.DataFrame(results)
print(table_out.drop(columns="rule").to_string(index=False))
table_out.to_csv(output_dir() / "join_results.csv", index=False)
(output_dir() / "join_results.json").write_text(
    json.dumps({"a0_params": params, "results": results, "diagnosis": diagnosis}, indent=2, default=str)
)
log("done")
