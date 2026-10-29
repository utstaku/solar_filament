# kaggle: kernel_sources={user}/filament-cnn-baseline-predict
# %% [markdown]
# # Post-processing lab: variants of A0 on the saved probability maps
#
# CPU only. `filament-cnn-baseline-predict` saved the U-Net's probability maps of the 106 validation and
# 180 test frames (`val_probs.npz`, `test_probs.npz`) and its chosen post-processing (A0: hysteresis
# 0.5 / 0.3, min area 300, GBM filter at an out-of-fold cutoff). A0 is rebuilt here and must reproduce
# its validation PQ; every variant is then scored on the same frames and compared with A0 by a paired
# bootstrap over frames.
#
# Variants (see `cnn_baseline/variants.py`):
# - **epq**: keep a component when its predicted gain per annotator (matched IoU) exceeds PQ / 2 -- the
#   rule PQ itself implies, no cutoff sweep. **epq_tuned** sweeps that threshold instead (selected on
#   the same frames, so optimistic).
# - **low0.2 / low0.1**: a lower hysteresis threshold (longer tails, more fragments linked), filter refitted.
# - **grow**: kept components extend into pixels with prob >= 0.2 / 0.15, at most 10 / 20 px.
# - **join**: kept components within 10 / 25 px become one instance; **join10_pre** joins before the filter.
#
# Outputs: `lab_results.csv` / `.json`, `submission_<variant>.csv` for every variant, and galleries of
# the components A0 keeps near the disk centre that no annotator drew (`gallery_kept_fakes.png`) next to
# kept real ones (`gallery_kept_real.png`).

# %%
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnn_baseline.data import split_files
from cnn_baseline.filtering import (
    ComponentFilter,
    component_table,
    frame_context,
    matched_by_any_annotator,
    out_of_fold_scores,
)
from cnn_baseline.importance import best_cutoff, paired_bootstrap, pq_of
from cnn_baseline.postprocess import instances, select
from cnn_baseline.variants import GainRegressor, expected_gain, grow, join, out_of_fold_gain, score_frames
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from error_analysis.gallery import draw_case
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops

SEED = 0  # must match cnn_baseline/main.py and predict.py
VALIDATION_FRACTION = 0.15
GAIN_THRESHOLDS = np.round(np.arange(0.05, 0.401, 0.025), 3)
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
images = {f: read_gray(train_dir / f) for f in val_files}
contexts = {f: frame_context(f, images[f], probs[f]) for f in val_files}
contexts |= {f: frame_context(f, read_gray(test_dir / f), probs[f]) for f in test_files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in val_files}
log(f"{len(val_files)} validation and {len(test_files)} test frames loaded; A0 = {params}, cutoff {a0_cutoff}")

# %% [markdown]
# ## Machinery
#
# Rows of a component table are concatenated in `val_files` order, as in `predict.py`, so the
# out-of-fold folds (GroupKFold by frame) are the same ones. A variant is a pair of functions
# frame -> instances (validation, test); it is scored and written frame by frame, so only one
# frame's full-size label maps are in memory at a time (a first version that kept them all ran out of RAM).

# %%


def compact(inst):
    """The label map as uint8 (at most MAX_INSTANCES = 40 per frame): 4 MB a frame instead of 16."""
    labels, areas, boxes = inst
    return labels.astype(np.uint8), areas, boxes


def table_of(raw: dict, files: list[str]) -> pd.DataFrame:
    tables = [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])]
    return pd.concat(tables, ignore_index=True)


def filtered(make_raw) -> dict:
    """Instances from `make_raw(frame)`, then the A0 filter refitted on them with its cutoff chosen again."""
    raw = {f: compact(make_raw(f)) for f in val_files}
    table = table_of(raw, val_files)
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in val_files])
    oof = out_of_fold_scores(table, target)
    cutoff, _ = best_cutoff(oof, raw, table, truth, val_files)
    model = ComponentFilter(cutoff=cutoff).fit(table, target)
    frames, keep = table["frame"].to_numpy(), oof >= cutoff

    def test(f):
        inst = make_raw(f)
        if not len(inst[1]):
            return inst
        return select(*inst, model.keep(component_table(contexts[f], inst[0], inst[2])))

    return {
        "val": lambda f: select(*raw[f], keep[frames == f]),
        "test": test,
        "rule": f"filter cutoff {cutoff}",
        "raw": raw,
        "table": table,
        "target": target,
        "oof": oof,
    }


def summary(terms: pd.DataFrame) -> dict:
    tp, fp, fn = (int(terms[c].sum()) for c in ["tp", "fp", "fn"])
    iou_sum = float(terms["iou_sum"].sum())
    sq = iou_sum / tp if tp else 0.0
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


results = []


def evaluate(name: str, variant: dict, terms: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score a variant against A0, write its submission, and record the result row."""
    terms = val_terms(variant["val"]) if terms is None else terms
    delta, lo, hi = paired_bootstrap(a0_terms, terms) if results else (0.0, 0.0, 0.0)
    test_rows = []
    for f in test_files:
        labels, _, boxes = variant["test"](f)
        test_rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
    submission.write(test_rows, output_dir() / f"submission_{name}.csv")
    row = {
        "variant": name,
        **summary(terms),
        "delta": round(delta, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "n_test": len(test_rows),
        "rule": variant["rule"],
    }
    results.append(row)
    log(row)
    return terms


# %% [markdown]
# ## A0, and the sanity gate

# %%
a0 = filtered(lambda f: instances(probs[f], **params))
a0_terms = val_terms(a0["val"])
a0_summary = summary(a0_terms)
log("A0 rebuilt:", a0_summary, a0["rule"])
assert a0["rule"] == f"filter cutoff {a0_cutoff}", "does not reproduce predict.py"
assert abs(a0_summary["pq"] - expected["pq"]) < 1e-4, "does not reproduce predict.py"
assert all(int(a0_summary[k]) == expected[k] for k in ("tp", "fp", "fn")), "does not reproduce predict.py"
log("sanity gate passed")
evaluate("a0", a0, a0_terms)

# %% [markdown]
# ## Variants
#
# `delta` = PQ(variant) - PQ(A0) on the validation frames, with its 95 % paired-bootstrap interval. Frames of
# consecutive days are not independent, so the intervals are, if anything, too narrow.

# %%
# keep rule from the metric: same components as A0, a regression on the gain per annotator
gain = np.concatenate([expected_gain(truth[f], a0["raw"][f][0], a0["raw"][f][1]) for f in val_files])
oof_gain = out_of_fold_gain(a0["table"], gain)
regressor = GainRegressor().fit(a0["table"], gain)
a0_test_raw = {f: compact(instances(probs[f], **params)) for f in test_files}
a0_test_table = table_of(a0_test_raw, test_files)
test_gain = regressor.predict(a0_test_table)
val_frames, test_frames = a0["table"]["frame"].to_numpy(), a0_test_table["frame"].to_numpy()


def gain_rule(threshold: float, rule: str) -> dict:
    return {
        "val": lambda f: select(*a0["raw"][f], (oof_gain > threshold)[val_frames == f]),
        "test": lambda f: select(*a0_test_raw[f], (test_gain > threshold)[test_frames == f]),
        "rule": rule,
    }


half_pq = round(a0_summary["pq"] / 2, 4)
evaluate("epq", gain_rule(half_pq, f"predicted gain > PQ/2 = {half_pq}"))
sweep = {t: pq_of(val_terms(gain_rule(t, "")["val"])) for t in GAIN_THRESHOLDS}
best_t = max(sweep, key=sweep.get)
log("gain threshold sweep:", {float(t): round(v, 4) for t, v in sweep.items()})
evaluate("epq_tuned", gain_rule(best_t, f"predicted gain > {best_t} (swept on validation)"))

for low in (0.2, 0.1):
    evaluate(f"low{low}", filtered(lambda f, low=low: instances(probs[f], **{**params, "low": low})))

for low, steps in ((0.2, 10), (0.15, 20)):
    evaluate(
        f"grow{low}_s{steps}",
        {
            "val": lambda f, low=low, steps=steps: grow(a0["val"](f)[0], probs[f], low, steps),
            "test": lambda f, low=low, steps=steps: grow(a0["test"](f)[0], probs[f], low, steps),
            "rule": f"A0, then grow into prob >= {low} for <= {steps} px",
        },
    )
for radius in (10, 25):
    evaluate(
        f"join{radius}",
        {
            "val": lambda f, radius=radius: join(a0["val"](f)[0], radius),
            "test": lambda f, radius=radius: join(a0["test"](f)[0], radius),
            "rule": f"A0, then join within {radius} px",
        },
    )
join_pre = filtered(lambda f: join(instances(probs[f], **params)[0], 10))
join_pre["rule"] = "join within 10 px, then the filter (refitted), " + join_pre["rule"]
evaluate("join10_pre", join_pre)
del join_pre

table = pd.DataFrame(results)
print(table.drop(columns="rule").to_string(index=False))
table.to_csv(output_dir() / "lab_results.csv", index=False)
(output_dir() / "lab_results.json").write_text(
    json.dumps(
        {"a0_params": params, "results": results, "gain_sweep": {str(t): v for t, v in sweep.items()}},
        indent=2,
        default=float,
    )
)

# %% [markdown]
# ## What A0 keeps near the disk centre
#
# Components inside 0.8 R that the filter keeps, split by whether any annotator drew them. Ground truth of
# all annotators in green, A0's kept instances in red. Sorted by the filter's confidence: the fakes on top
# are the ones the filter cannot tell apart.

# %%
kept = a0["table"].assign(target=a0["target"], oof=a0["oof"])
kept = kept[(kept["oof"] >= a0_cutoff) & (kept["limb"] < 0.8)].sort_values("oof", ascending=False)


def gallery(rows: pd.DataFrame, path: str, title: str, ncols: int = 6):
    if not len(rows):
        log(f"{path}: nothing to draw")
        return
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        f = row["frame"]
        draw_case(ax, images[f], [c for crops in truth[f] for c in crops], a0["val"](f)[0], row)
        ax.set_title(
            f"{f[:16]} limb {row['limb']:.2f} score {row['oof']:.2f}\narea {int(row['area'])} elong {row['elongation']:.1f} dark {row['darkness']:.2f}",
            fontsize=7,
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir() / path, dpi=110)
    plt.show()
    plt.close(fig)


fakes = kept[~kept["target"]].groupby("frame").head(2).head(24)
reals = kept[kept["target"]].groupby("frame").head(1)
reals = reals.sample(min(12, len(reals)), random_state=0)
gallery(fakes, "gallery_kept_fakes.png", "kept by A0 inside 0.8 R, drawn by no annotator (most confident first)")
gallery(reals, "gallery_kept_real.png", "kept by A0 inside 0.8 R and drawn by an annotator (random)")
log(f"{(~kept['target']).sum()} kept fakes and {kept['target'].sum()} kept real components inside 0.8 R; done")
