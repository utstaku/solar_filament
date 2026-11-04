# kaggle: kernel_sources={user}/filament-zoo-baseline,{user}/filament-zoo-boundary-bce,{user}/filament-zoo-annotator-sampling,{user}/filament-zoo-edgeattnet,{user}/filament-zoo-hrnet,{user}/filament-zoo-segformer,{user}/filament-zoo-swin-unet,{user}/filament-zoo-swin-unet-boundary
# %% [markdown]
# # Zoo comparison: every experiment's 707 out-of-fold maps through the same post-processing
#
# CPU only. For each experiment whose notebook has finished (missing ones are skipped):
#
# - **raw**: A0's instances (hysteresis 0.5 / 0.3, 300 px), no filter: what the network alone gives;
# - **post**: `postprocess.pipeline` (join, then the component filter with its cutoff chosen on
#   out-of-fold scores), fitted on the experiment's own maps: what it would submit.
#
# Each is compared with the baseline on the same frames by a paired bootstrap over frames.
# Outputs: `zoo_comparison.json`, `zoo_comparison.csv`.

# %%
import json
import time

import numpy as np
import pandas as pd

from cnn_baseline.importance import paired_bootstrap
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from error_analysis.instances import FrameContext
from postprocess import pipeline
from postprocess.filtering import frame_context
from postprocess.instances import instances
from postprocess.tuning import TERMS, pq_of
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops
from zoo.experiment import EXPERIMENTS, N_FOLDS

T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %% [markdown]
# ## Experiments that finished all five folds

# %%
available = {}
for name in EXPERIMENTS:
    try:
        paths = [find_input(f"{name}_oof_fold{k}.npz") for k in range(N_FOLDS)]
        report = json.loads(find_input(f"{name}_report.json").read_text())
    except FileNotFoundError as missing:
        log(f"{name}: skipped ({missing})")
        continue
    available[name] = (paths, report)
log("comparing", list(available))
assert "baseline" in available, "the baseline is what every experiment is compared with"

# %% [markdown]
# ## Frames, truth and the parts of each frame's context that do not depend on the model

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
folds = json.loads(find_input("baseline_folds.json").read_text())["folds"]
files = sorted(f for fold in folds for f in fold)
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
base = {f: frame_context(f, read_gray(train_dir / f), np.zeros(1, np.uint8)) for f in files}
log(f"{len(files)} frames")


def frame_terms(predictions: dict) -> pd.DataFrame:
    rows = []
    for f in files:
        labels, areas, _ = predictions[f]
        pq = PanopticQuality()
        for gt_crops in truth[f]:
            pq.update(gt_crops, labels, areas)
        rows.append({"frame": f, "tp": pq.tp, "fp": pq.fp, "fn": pq.fn, "iou_sum": pq.iou_sum})
    return pd.DataFrame(rows, columns=["frame", *TERMS])


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


# %% [markdown]
# ## Each experiment: raw and post-processed PQ on the 707 frames

# %%
terms, rows = {}, []
for name, (paths, report) in available.items():
    maps = {}
    for path in paths:
        saved = np.load(path)
        maps |= {f: saved[f] for f in saved.files}
    assert set(maps) == set(files), f"{name}: the maps do not cover every frame"
    contexts = {f: FrameContext(f, base[f].flat, base[f].centre, base[f].radius, maps[f]) for f in files}
    raw = frame_terms({f: instances(maps[f], **pipeline.A0) for f in files})
    try:
        _, oof = pipeline.fit(contexts, truth, files)
    except ValueError as degenerate:  # no candidate (or every one) matches: nothing for the filter to learn
        log(f"{name}: post = raw ({degenerate})")
        oof = pipeline.OutOfFold(raw, None, float("nan"))
    terms[name] = {"raw": raw, "post": oof.terms}
    train_hours = sum(fold["train_sec"] + fold["predict_sec"] for fold in report["folds"]) / 3600
    rows.append(
        {
            "name": name,
            **{f"raw_{k}": v for k, v in summary(raw).items()},
            **{f"post_{k}": v for k, v in summary(oof.terms).items()},
            "cutoff": oof.cutoff,
            "kept_per_frame": round(oof.kept_per_frame, 2),
            "gpu_hours": round(train_hours, 2),
            "peak_gpu_gb": max((fold["peak_gpu_gb"] or 0) for fold in report["folds"]),
        }
    )
    log(name, rows[-1])
    del maps, contexts

# %% [markdown]
# ## Against the baseline, frame by frame
#
# `delta` is the experiment's PQ minus the baseline's on all 707 frames, with a 95 % interval from
# resampling frames (the same frames for both). Neighbouring days are not independent, so the
# intervals are still somewhat narrow.

# %%
for row in rows:
    for kind in ("raw", "post"):
        delta, lo, hi = paired_bootstrap(terms["baseline"][kind], terms[row["name"]][kind])
        row[f"{kind}_delta"] = round(delta, 4)
        row[f"{kind}_ci95"] = [round(lo, 4), round(hi, 4)]
table = pd.DataFrame(rows).sort_values("post_pq", ascending=False)
print(
    table[
        ["name", "raw_pq", "raw_delta", "post_pq", "post_delta", "post_ci95", "post_sq", "post_rq", "gpu_hours"]
    ].to_string(index=False)
)
table.to_csv(output_dir() / "zoo_comparison.csv", index=False)
(output_dir() / "zoo_comparison.json").write_text(json.dumps(rows, indent=2, default=float))
log("done")
