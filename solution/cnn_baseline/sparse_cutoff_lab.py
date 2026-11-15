# kaggle: kernel_sources={user}/filament-cnn-baseline-cv,{user}/filament-cnn-baseline-predict
# %% [markdown]
# # Sparse test labels: does the component filter's best cutoff move?
#
# CPU only. Every submission so far fits one factor: the public score is what the OOF counts give if each
# filament an annotator drew in training appears in the test labels only with probability r ≈ 0.73-0.77
# (`importance.public_pq`, `fit_r`). If that is right, a kept component matches a test annotator less often
# than on OOF, so by the break-even rule (keep when P(match) x IoU > PQ / 2) the filter should keep fewer.
#
# Here the join_then_filter pipeline of `final_join.py` is rebuilt on the same inputs, and its filter cutoff is
# chosen two ways on the 707 OOF frames: by OOF PQ (as submitted) and by the expected public PQ at r. If the
# cutoff moves clearly, the same shift is a candidate for the Swin pipeline; if not, the submitted cutoff stays.
#
# Outputs: `sparse_cutoff_report.json`, `sparse_cutoff_sweep.csv`, `submission_sparse.csv` (U-Net, r-best cutoff).

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
from cnn_baseline.importance import best_cutoff, fit_r, frame_terms, pq_of, public_pq
from cnn_baseline.pairs import join_pairs, join_rule, pair_table
from cnn_baseline.postprocess import instances, select
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops

QUICK = False  # True: a few frames, to check that the whole notebook runs
A0 = {"threshold": 0.5, "min_area": 300, "low": 0.3}
PAIR_RADIUS = 50
A0_PUBLIC = 0.33  # predict.py's A0 on the public leaderboard, shown to two decimals
FINE_CUTOFFS = np.round(np.arange(0.2, 0.801, 0.025), 3)
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs (as in `final_join.py`): single-model OOF maps and the v2 model's test maps

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)
test_files = sorted(p.name for p in test_dir.iterdir())
if QUICK:
    files, test_files = files[:12], test_files[:2]
maps = {}
for k in range(len(folds)):
    saved = np.load(find_input(f"oof_plain_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files if f in files}
saved = np.load(find_input("test_probs.npz"))
maps |= {f: saved[f] for f in saved.files if f in test_files}
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
contexts |= {f: frame_context(f, read_gray(test_dir / f), maps[f]) for f in test_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
log(f"{len(files)} training and {len(test_files)} test frames")

# %% [markdown]
# ## A0 and join_then_filter candidates, filter scores out of fold

# %%


def compact(inst):
    labels, areas, boxes = inst
    return labels.astype(np.uint8), areas, boxes


def joined(inst, f):
    t = pair_table(contexts[f], inst[0], PAIR_RADIUS)
    return join_pairs(inst[0], t, join_rule(t)) if len(t) else inst


def scored(make_raw) -> dict:
    raw = {f: compact(make_raw(f)) for f in files}
    table = pd.concat(
        [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    return {
        "make_raw": make_raw,
        "raw": raw,
        "table": table,
        "target": target,
        "oof": out_of_fold_scores(table, target),
    }


a0 = scored(lambda f: instances(maps[f], **A0))
jtf = scored(lambda f: joined(instances(maps[f], **A0), f))
_, a0_terms = best_cutoff(a0["oof"], a0["raw"], a0["table"], truth, files)  # A0 as submitted (OOF-best cutoff)
rs = {f"{p:.3f}": fit_r(a0_terms, p) for p in (A0_PUBLIC - 0.005, A0_PUBLIC, A0_PUBLIC + 0.005)}
log("r fitted from A0's public score:", {k: round(v, 3) for k, v in rs.items()})

# %% [markdown]
# ## The sweep: OOF PQ and expected public PQ for every cutoff

# %%
rows = []
for c in FINE_CUTOFFS:
    terms = frame_terms(jtf["raw"], jtf["table"], jtf["oof"] >= c, truth, files)
    row = {"cutoff": float(c), "oof_pq": pq_of(terms), "kept_per_frame": float((jtf["oof"] >= c).sum()) / len(files)}
    row |= {f"public_at_{k}": public_pq(terms, r) for k, r in rs.items()}
    rows.append(row)
sweep = pd.DataFrame(rows)
sweep.round(4).to_csv(output_dir() / "sparse_cutoff_sweep.csv", index=False)
print(sweep.round(4).to_string(index=False))

oof_best = sweep.loc[sweep["oof_pq"].idxmax()]
report = {"r": {k: round(v, 4) for k, v in rs.items()}, "oof_best": oof_best.round(4).to_dict(), "public_best": {}}
for k in rs:
    best = sweep.loc[sweep[f"public_at_{k}"].idxmax()]
    report["public_best"][k] = {
        "cutoff": float(best["cutoff"]),
        "shift": round(float(best["cutoff"] - oof_best["cutoff"]), 3),
        "expected_public": round(float(best[f"public_at_{k}"]), 4),
        "expected_public_at_oof_cutoff": round(float(oof_best[f"public_at_{k}"]), 4),
        "expected_gain": round(float(best[f"public_at_{k}"] - oof_best[f"public_at_{k}"]), 4),
        "oof_pq_cost": round(float(best["oof_pq"] - oof_best["oof_pq"]), 4),
        "kept_per_frame": round(float(best["kept_per_frame"]), 2),
    }

# %% [markdown]
# ## Test files at the r-best cutoff (central r), with the density check of `final_join.py`

# %%
central = report["public_best"][f"{A0_PUBLIC:.3f}"]
model = ComponentFilter(cutoff=central["cutoff"]).fit(jtf["table"], jtf["target"])
out = []
for f in test_files:
    inst = jtf["make_raw"](f)
    if len(inst[1]):
        inst = select(*inst, model.keep(component_table(contexts[f], inst[0], inst[2])))
    out += submission.rows_from_label_map(f.rsplit(".", 1)[0], inst[0], inst[2])
submission.write(out, output_dir() / "submission_sparse.csv")
report["test_kept_per_frame_at_public_best"] = round(len(out) / len(test_files), 2)
print(json.dumps(report, indent=2))
(output_dir() / "sparse_cutoff_report.json").write_text(json.dumps(report, indent=2))
log("done")
