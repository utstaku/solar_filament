# kaggle: kernel_sources=iceqbie/filament-cnn-baseline-cv,iceqbie/filament-cnn-baseline-predict
# %% [markdown]
# # Ensemble: the swin U-Net and the CNN U-Net, averaged before post-processing
#
# CPU only, nothing is retrained. The team's best single model (`swin_unet_boundary`, OOF PQ 0.4357) and our
# U-Net (OOF 0.396 with join_then_filter) are different kinds of network: a transformer that sees long-range
# context and a CNN that sees local texture. Where they disagree is mostly the faint ends and boundaries,
# so averaging their probability maps should cancel part of each model's errors there.
#
# `p = w * p_swin + (1 - w) * p_unet` for each weight `w`, then two candidate makers (A0 hysteresis, and the
# swin recipe: hysteresis 0.6/0.45 with masks cut at 0.6), then join_then_filter as in `final_join.py`
# (the GBM filter is trained on the blended OOF maps, so it learns the blend's calibration).
# Everything is scored on the 707 out-of-fold frames, and each variant is compared with `w = 1` (the swin
# alone, same post-processing) by a paired bootstrap over frames.
#
# Calibration check (lesson of `final_join.py`): OOF and test maps must look alike, or the filter keeps a
# different number of filaments on test. The kept instances per frame are printed for both.
#
# Outputs: `submission.csv` (best variant), `submission_swin_only.csv`, `ensemble_report.json`.
#
# **To run it (owner of the swin notebooks):** the U-Net maps come from iceqbie's notebooks, which must be
# shared with you. Append your swin notebook(s) to `kernel_sources` in the first line (comma-separated,
# `<you>/<slug>`), set `SWIN_OOF` / `SWIN_TEST` below to your file names, set `QUICK = True` for a first try,
# then `uv run filament run solution/ensemble/main.py`. Check first that `w1.0_swin` reproduces your OOF PQ.

# %%
import json
import time
from dataclasses import replace

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
from cnn_baseline.pairs import join_pairs, join_rule, pair_table
from cnn_baseline.postprocess import instances, select
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, input_root, output_dir
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops

QUICK = False  # True: a few frames, to check that the whole notebook runs
WEIGHTS = (1.0, 0.8, 0.7, 0.6, 0.5)  # weight of the swin map; 1.0 is the swin alone
CANDIDATES = {
    "a0": {"threshold": 0.5, "min_area": 300, "low": 0.3, "cut": None},
    "swin": {"threshold": 0.6, "min_area": 300, "low": 0.45, "cut": 0.6},
}
PAIR_RADIUS = 50
# The swin maps: uint8 probability * 255 in .npz files keyed by image file name (as `cv.py` saves ours).
# Set the glob patterns to your file names; float maps need `* 255` and `.astype(np.uint8)` in `load_maps`.
SWIN_OOF = "swin_oof*.npz"
SWIN_TEST = "swin_test*.npz"
# Our U-Net: OOF of the five fold models; test of the v2 single model (as in final_join.py, so both are one model)
UNET_OOF = "oof_plain_fold*.npz"
UNET_TEST = "test_probs.npz"
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def load_maps(pattern: str) -> dict[str, np.ndarray]:
    maps = {}
    for path in sorted(input_root().glob(f"**/{pattern}")):
        saved = np.load(path)
        maps |= {f: saved[f] for f in saved.files}
    if not maps:
        raise FileNotFoundError(f"{pattern} not found under {input_root()}")
    return maps


# %% [markdown]
# ## Inputs

# %%
root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)
test_files = sorted(p.name for p in test_dir.iterdir())
if QUICK:
    files, test_files = files[:12], test_files[:2]

unet = load_maps(UNET_OOF) | load_maps(UNET_TEST)
swin = load_maps(SWIN_OOF) | load_maps(SWIN_TEST)
for name, maps in (("unet", unet), ("swin", swin)):
    missing = (set(files) | set(test_files)) - set(maps)
    assert not missing, f"{name} maps miss {len(missing)} frames, e.g. {sorted(missing)[:3]}"
# flattened image and disk geometry once per frame; the probability is swapped in per weight
base = {f: frame_context(f, read_gray(train_dir / f), unet[f]) for f in files}
base |= {f: frame_context(f, read_gray(test_dir / f), unet[f]) for f in test_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
log(f"{len(files)} training and {len(test_files)} test frames")

# %% [markdown]
# ## Candidates, join, filter

# %%


def cut_masks(inst, prob: np.ndarray, cut: float | None):
    """Keep each instance's connectivity from hysteresis but its shape from `prob >= cut` (thinner masks)."""
    labels, areas, _ = inst
    if cut is None or not len(areas):
        return inst
    labels = np.where(prob >= round(cut * 255), labels, 0)
    kept = np.bincount(labels.ravel(), minlength=len(areas) + 1)[1:]
    keep = kept > 0
    relabel = np.zeros(len(areas) + 1, dtype=np.int32)
    relabel[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    labels = relabel[labels]
    new_boxes = []
    for k in range(1, int(keep.sum()) + 1):
        rows, cols = np.nonzero(labels == k)
        new_boxes.append([cols.min(), rows.min(), cols.max() - cols.min() + 1, rows.max() - rows.min() + 1])
    return labels, kept[keep], np.array(new_boxes, dtype=np.int32).reshape(-1, 4)


def candidates(prob: np.ndarray, threshold: float, min_area: int, low: float, cut: float | None):
    return cut_masks(instances(prob, threshold, min_area, low), prob, cut)


def compact(inst):
    labels, areas, boxes = inst
    return labels.astype(np.uint8), areas, boxes


def run_variant(contexts: dict, params: dict) -> dict:
    """join_then_filter on the candidates of `params`, the filter trained and scored on the OOF frames."""

    def make_raw(f):
        inst = candidates(contexts[f].prob, **params)
        t = pair_table(contexts[f], inst[0], PAIR_RADIUS)
        return join_pairs(inst[0], t, join_rule(t)) if len(t) else inst

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

    return {"terms": terms, "cutoff": cutoff, "test": test, "per_val_frame": int((oof >= cutoff).sum()) / len(files)}


def summary(terms: pd.DataFrame) -> dict:
    tp, fp, fn = (int(terms[c].sum()) for c in ["tp", "fp", "fn"])
    sq = float(terms["iou_sum"].sum()) / tp if tp else 0.0
    return {"pq": round(pq_of(terms), 4), "sq": round(sq, 4), "tp": tp, "fp": fp, "fn": fn}


def test_rows(variant: dict) -> list:
    rows = []
    for f in test_files:
        labels, _, boxes = variant["test"](f)
        rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    return rows


# %% [markdown]
# ## Every weight × candidate maker, scored on the OOF frames
#
# One weight at a time, so only one set of blended maps is in memory.

# %%
report, best = {}, None
for w in WEIGHTS:
    contexts = {
        f: replace(base[f], prob=np.round(w * swin[f] + (1 - w) * unet[f].astype(np.float32)).astype(np.uint8))
        for f in files + test_files
    }
    for cname, params in CANDIDATES.items():
        key = f"w{w:.1f}_{cname}"
        v = run_variant(contexts, params)
        rows = test_rows(v)
        report[key] = {
            **summary(v["terms"]),
            "cutoff": v["cutoff"],
            "kept_per_val_frame": round(v["per_val_frame"], 2),
            "kept_per_test_frame": round(len(rows) / len(test_files), 2),
        }
        report[key]["_terms"] = v["terms"]
        log(key, {k: x for k, x in report[key].items() if k != "_terms"})
        if w == 1.0 and cname == "swin":
            submission.write(rows, output_dir() / "submission_swin_only.csv")
        if best is None or report[key]["pq"] > report[best[0]]["pq"]:
            best = (key, rows)
    del contexts

# %% [markdown]
# ## Comparison with the swin alone (same post-processing), paired over frames

# %%
reference = "w1.0_swin"
for key, row in report.items():
    if key != reference:
        delta, lo, hi = paired_bootstrap(report[reference]["_terms"], row["_terms"])
        row["delta_vs_swin_only"] = {"delta": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)]}
submission.write(best[1], output_dir() / "submission.csv")
out = {k: {m: x for m, x in v.items() if m != "_terms"} for k, v in report.items()}
out["best"] = best[0]
print(json.dumps(out, indent=2))
(output_dir() / "ensemble_report.json").write_text(json.dumps(out, indent=2))
log("done; best", best[0])
