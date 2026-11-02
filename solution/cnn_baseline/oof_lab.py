# kaggle: kernel_sources={user}/filament-cnn-baseline-cv
# %% [markdown]
# # OOF lab: post-processing compared on all 707 training frames
#
# CPU only. `filament-cnn-baseline-cv` trained the baseline U-Net five times and saved, for every training
# frame, the probability map of the fold model that never saw it -- plain and averaged over the eight D4
# views (TTA) -- plus the five models' mean on the test frames. Every variant is scored on the same 707
# frames and compared with A0 by a paired bootstrap over frames (intervals ~2.6x narrower than on 106).
#
# - **a0_plain / a0_tta**: A0's post-processing (hysteresis 0.5 / 0.3, min area 300, GBM component filter
#   with an out-of-fold cutoff), on plain and TTA maps. The better one is the base of the rest.
# - **join_rule**: A0's kept pieces joined end to end along the axis up to 50 px, any pieces up to 10 px.
# - **rescue**: the same joins, also bringing back dropped pieces that continue a kept one.
# - **join_then_filter**: candidates joined first, the filter refitted on whole filaments.
# - **strict / strict_join**: the keep rule of the public "OOF 0.45 vs Public 0.39" notebook (p >= 0.35
#   linked, peak >= 0.99, mean >= 0.85, area >= 200) instead of the GBM, without and with the joins.
#
# Outputs: `oof_results.csv` / `.json`, `submission_<variant>.csv` (test: the five fold models' mean map).

# %%
import dataclasses
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
from cnn_baseline.importance import best_cutoff, paired_bootstrap, pq_of
from cnn_baseline.pairs import join_pairs, join_rule, pair_table, rescue
from cnn_baseline.postprocess import instances, select
from cnn_baseline.variants import score_frames, strict_keep
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops

A0 = {"threshold": 0.5, "min_area": 300, "low": 0.3}  # predict.py's choice on the 106 validation frames
PAIR_RADIUS = 50
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Inputs
#
# The flattened frames are computed once; each map kind gets its own context (the filter's confidence
# features come from the map).

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)
test_files = sorted(p.name for p in test_dir.iterdir())
assert sorted(p.name for p in train_dir.iterdir()) == files, "the folds do not cover the training frames"
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
base_contexts = {f: frame_context(f, read_gray(train_dir / f), None) for f in files}
base_contexts |= {f: frame_context(f, read_gray(test_dir / f), None) for f in test_files}
log(f"{len(files)} training frames in {len(folds)} folds, {len(test_files)} test frames")


def load_maps(kind: str) -> dict:
    maps = {}
    for k in range(len(folds)):
        saved = np.load(find_input(f"oof_{kind}_fold{k}.npz"))
        maps |= {f: saved[f] for f in saved.files}
    saved = np.load(find_input(f"test_probs_{kind}.npz"))
    maps |= {f: saved[f] for f in saved.files}
    assert set(maps) == set(files) | set(test_files), f"{kind} maps do not cover every frame"
    return maps


def with_maps(maps: dict) -> dict:
    return {f: dataclasses.replace(ctx, prob=maps[f]) for f, ctx in base_contexts.items()}


# %% [markdown]
# ## Machinery (as in `join_lab.py`): variants are scored and written frame by frame

# %%


def compact(inst):
    labels, areas, boxes = inst
    return labels.astype(np.uint8), areas, boxes


def table_of(raw: dict, contexts: dict, names: list[str]) -> pd.DataFrame:
    tables = [component_table(contexts[f], raw[f][0], raw[f][2]) for f in names if len(raw[f][1])]
    return pd.concat(tables, ignore_index=True)


def filtered(make_raw, contexts: dict) -> dict:
    """Candidates from `make_raw(frame)`, the GBM filter fitted out of fold on all 707 frames."""
    raw = {f: compact(make_raw(f)) for f in files}
    table = table_of(raw, contexts, files)
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    oof = out_of_fold_scores(table, target)
    cutoff, _ = best_cutoff(oof, raw, table, truth, files)
    model = ComponentFilter(cutoff=cutoff).fit(table, target)
    frames, keep = table["frame"].to_numpy(), oof >= cutoff
    log(f"filter: {len(table)} candidates, {int(target.sum())} matched, cutoff {cutoff}")

    def val_keep(f):
        return keep[frames == f]

    def test_candidates(f):
        inst = make_raw(f)
        if not len(inst[1]):
            return inst, np.zeros(0, bool)
        return inst, model.keep(component_table(contexts[f], inst[0], inst[2]))

    def test(f):
        inst, k = test_candidates(f)
        return select(*inst, k)

    return {
        "val": lambda f: select(*raw[f], val_keep(f)),
        "test": test,
        "rule": f"filter cutoff {cutoff}",
        "raw": raw,
        "val_keep": val_keep,
        "test_candidates": test_candidates,
    }


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
    return pd.concat([score_frames({f: val(f)}, truth, [f]) for f in files], ignore_index=True)


results, reference = [], {}


def evaluate(name: str, val, test, rule: str, kind: str) -> pd.DataFrame:
    terms = val_terms(val)
    row = {"variant": name, "maps": kind, **summary(terms)}
    for ref_name, ref_terms in reference.items():
        delta, lo, hi = paired_bootstrap(ref_terms, terms)
        row |= {f"d_{ref_name}": round(delta, 4), f"lo_{ref_name}": round(lo, 4), f"hi_{ref_name}": round(hi, 4)}
    rows = []
    for f in test_files:
        labels, _, boxes = test(f)
        rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    submission.write(rows, output_dir() / f"submission_{name}.csv")
    results.append({**row, "n_test": len(rows), "rule": rule})
    log(results[-1])
    return terms


# %% [markdown]
# ## A0 on plain and TTA maps
#
# The 707-frame reference is A0 on the plain maps; the better of the two kinds is the base for the rest.

# %%
best = None  # (pq, kind, maps, contexts, a0) of the better kind; only it stays in memory
for map_kind in ("plain", "tta"):
    kind_maps = load_maps(map_kind)
    kind_contexts = with_maps(kind_maps)
    kind_a0 = filtered(lambda f, m=kind_maps: instances(m[f], **A0), kind_contexts)
    terms = evaluate(f"a0_{map_kind}", kind_a0["val"], kind_a0["test"], kind_a0["rule"], map_kind)
    if map_kind == "plain":
        reference["a0_plain"] = terms
    if best is None or pq_of(terms) > best[0]:
        best = (pq_of(terms), map_kind, kind_maps, kind_contexts, kind_a0)
    del kind_maps, kind_contexts, kind_a0
_, kind, maps, contexts, a0 = best
del best
log(f"base for the variants: {kind} maps")

# %% [markdown]
# ## Variants on the better maps

# %%


def joined(inst, f):
    labels = inst[0]
    t = pair_table(contexts[f], labels, PAIR_RADIUS)
    return join_pairs(labels, t, join_rule(t)) if len(t) else inst


def rescued_test(f):
    inst, keep = a0["test_candidates"](f)
    return rescue(contexts[f], inst[0], keep, PAIR_RADIUS) if len(keep) else inst


evaluate(
    "join_rule",
    lambda f: joined(a0["val"](f), f),
    lambda f: joined(a0["test"](f), f),
    "A0, then end to end along the axis <= 50 px, any pieces <= 10 px",
    kind,
)
evaluate(
    "rescue",
    lambda f: rescue(contexts[f], a0["raw"][f][0], a0["val_keep"](f), PAIR_RADIUS),
    rescued_test,
    "the joins of join_rule, also bringing back dropped pieces that continue a kept one",
    kind,
)
then_filter = filtered(lambda f: joined(instances(maps[f], **A0), f), contexts)
evaluate(
    "join_then_filter", then_filter["val"], then_filter["test"], "join_rule first, then " + then_filter["rule"], kind
)
del then_filter
evaluate(
    "strict",
    lambda f: strict_keep(maps[f]),
    lambda f: strict_keep(maps[f]),
    "p >= 0.35 linked, peak >= 0.99, mean >= 0.85, area >= 200 (no GBM)",
    kind,
)
evaluate(
    "strict_join",
    lambda f: joined(strict_keep(maps[f]), f),
    lambda f: joined(strict_keep(maps[f]), f),
    "strict, then the joins of join_rule",
    kind,
)

# %% [markdown]
# ## Results
#
# `d_a0_plain` = PQ(variant) - PQ(A0 on plain maps) over the 707 frames, with its 95 % paired-bootstrap
# interval (`lo_`, `hi_`); still somewhat too narrow, since frames of consecutive days are not independent.

# %%
table = pd.DataFrame(results)
print(table.drop(columns="rule").to_string(index=False))
table.to_csv(output_dir() / "oof_results.csv", index=False)
(output_dir() / "oof_results.json").write_text(json.dumps({"a0": A0, "base": kind, "results": results}, indent=2))
log("done")
