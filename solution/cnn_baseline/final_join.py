# kaggle: kernel_sources={user}/filament-cnn-baseline-cv,{user}/filament-cnn-baseline-predict
# %% [markdown]
# # Submission candidate: join, then filter, tuned on 707 out-of-fold frames
#
# CPU only. The 707-frame OOF lab (`oof_lab.py`) found that joining A0's candidates (end to end along the
# axis up to 50 px, any pieces up to 10 px) and then filtering whole filaments raises PQ by +0.010
# [+0.006, +0.015]. Its own test files must not be submitted: its test maps were the mean of five fold
# models, smoother than the single-model OOF maps the filter learned from, so the filter kept about half as
# many filaments on test (3.9 per frame against 6.9 on validation).
#
# Here the filter is trained on the single-model plain OOF maps of `filament-cnn-baseline-cv`, and applied to
# the single-model plain test maps of `filament-cnn-baseline-predict` (the v2 U-Net), so both sides look alike.
# The instances per frame on validation and test are printed as a check.
#
# Outputs: `submission.csv` (join_then_filter), `submission_a0.csv`, `final_report.json`.

# %%
import json
import time

import numpy as np
import pandas as pd

from cnn_baseline.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
)
from cnn_baseline.importance import best_cutoff, fit_r, paired_bootstrap, pq_of, public_pq
from cnn_baseline.pairs import join_pairs, join_rule, pair_table
from cnn_baseline.postprocess import instances, select
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops

A0 = {"threshold": 0.5, "min_area": 300, "low": 0.3}
PAIR_RADIUS = 50
A0_PUBLIC = 0.33  # what predict.py's A0 scored on the public leaderboard (shown to two decimals)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs: OOF plain maps for the 707 training frames, the v2 model's maps for the test frames

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)
test_files = sorted(p.name for p in test_dir.iterdir())
maps = {}
for k in range(len(folds)):
    saved = np.load(find_input(f"oof_plain_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
saved = np.load(find_input("test_probs.npz"))  # predict.py: one model, no TTA
maps |= {f: saved[f] for f in saved.files}
assert set(maps) == set(files) | set(test_files), "the maps do not cover every frame"
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
contexts |= {f: frame_context(f, read_gray(test_dir / f), maps[f]) for f in test_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
log(f"{len(files)} training and {len(test_files)} test frames")

# %% [markdown]
# ## A0 and join_then_filter (as in `oof_lab.py`)

# %%


def compact(inst):
    labels, areas, boxes = inst
    return labels.astype(np.uint8), areas, boxes


def joined(inst, f):
    labels = inst[0]
    t = pair_table(contexts[f], labels, PAIR_RADIUS)
    return join_pairs(labels, t, join_rule(t)) if len(t) else inst


def filtered(make_raw) -> dict:
    raw = {f: compact(make_raw(f)) for f in files}
    table = pd.concat(
        [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    oof = out_of_fold_scores(table, target)
    cutoff, terms = best_cutoff(oof, raw, table, truth, files)
    model = ComponentFilter(cutoff=cutoff).fit(table, target)

    def test(f):
        inst = make_raw(f)
        if not len(inst[1]):
            return inst
        return select(*inst, model.keep(component_table(contexts[f], inst[0], inst[2])))

    kept_val = int((oof >= cutoff).sum())
    return {"terms": terms, "cutoff": cutoff, "test": test, "per_val_frame": kept_val / len(files)}


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


def write(variant: dict, path: str) -> float:
    rows = []
    for f in test_files:
        labels, _, boxes = variant["test"](f)
        rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    submission.write(rows, output_dir() / path)
    return len(rows) / len(test_files)


variants = {
    "a0": filtered(lambda f: instances(maps[f], **A0)),
    "join_then_filter": filtered(lambda f: joined(instances(maps[f], **A0), f)),
}
report = {}
for name, v in variants.items():
    per_test = write(v, "submission.csv" if name == "join_then_filter" else f"submission_{name}.csv")
    report[name] = {
        **summary(v["terms"]),
        "cutoff": v["cutoff"],
        "kept_per_val_frame": round(v["per_val_frame"], 2),
        "kept_per_test_frame": round(per_test, 2),
    }
    log(name, report[name])

# %% [markdown]
# ## Comparison and the expected public score
#
# `r` is fitted so that A0 on these OOF counts gives A0's public score; the candidate's expected public
# score follows (`importance.public_pq`). Both rest on the assumption that the test labels are sparser by a
# constant factor, and on A0's public score being known only to two decimals (0.325-0.335).

# %%
delta, lo, hi = paired_bootstrap(variants["a0"]["terms"], variants["join_then_filter"]["terms"])
report["delta_vs_a0"] = {"delta": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)]}
report["expected_public"] = {}
for public in (A0_PUBLIC - 0.005, A0_PUBLIC, A0_PUBLIC + 0.005):
    r = fit_r(variants["a0"]["terms"], public)
    report["expected_public"][f"a0_at_{public:.3f}"] = {
        "r": round(r, 3),
        "join_then_filter": round(public_pq(variants["join_then_filter"]["terms"], r), 4),
    }
print(json.dumps(report, indent=2))
(output_dir() / "final_report.json").write_text(json.dumps(report, indent=2))
log("done")
