# kaggle: kernel_sources={user}/filament-zoo-swin-unet
# %% [markdown]
# # How much does the component filter cost? Its errors and the PQ of a perfect filter
#
# CPU only, on the 707 out-of-fold maps of `zoo/swin_unet.py` with the submitted post-processing (candidates,
# join, the filter with out-of-fold scores and its PQ-best cutoff). `misses.py` found 91 % of the false
# negatives inside a region the network gave p >= 0.5; this splits them into those a candidate would have
# matched but the filter dropped, and those no candidate matches (the cut of the map is wrong). For the
# candidates of a few hysteresis settings:
#
# - PQ keeping every candidate, at the filter's cutoff, and with a **perfect filter** (keep exactly the
#   candidates that match some annotator, `matched_by_any_annotator`): the upper bound of any filter;
# - the filter's errors at its cutoff: true candidates dropped, false ones kept, with their scores and
#   features next to the right decisions, and the filter's ROC AUC.
#
# Outputs: `filter_bound.csv`, `filter_errors.csv`, `filter_bound_report.json`.

# %%
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import FEATURES, component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.pipeline import A0, candidates
from postprocess.tuning import best_cutoff, frame_terms, pq_of
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet"
SETTINGS = {  # name: (hysteresis params, join)
    "A0_join": (A0, True),
    "A0": (A0, False),
    "low_0.4_0.2_join": ({"threshold": 0.4, "min_area": 300, "low": 0.2}, True),
    "high_0.6_0.4_join": ({"threshold": 0.6, "min_area": 300, "low": 0.4}, True),
}
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
log(f"{len(files)} frames")


def summary(terms: pd.DataFrame) -> dict:
    return {"pq": round(pq_of(terms), 4), **{c: int(terms[c].sum()) for c in ["tp", "fp", "fn"]}}


# %% [markdown]
# ## Every candidate setting: no filter, the filter, a perfect filter

# %%
bounds, errors = [], None
for setting, (params, join) in SETTINGS.items():
    raw = {f: candidates(contexts[f], params, join) for f in files}
    table = pd.concat(
        [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    scores = out_of_fold_scores(table, target)
    cutoff, filtered = best_cutoff(scores, raw, table, truth, files)
    keep = scores >= cutoff
    every = frame_terms(raw, table, np.ones(len(table), bool), truth, files)
    perfect = frame_terms(raw, table, target, truth, files)
    bounds.append(
        {
            "setting": setting,
            "candidates": len(table),
            "true_candidates": int(target.sum()),
            "auc": round(float(roc_auc_score(target, scores)), 4),
            "cutoff": cutoff,
            "true_dropped": int((target & ~keep).sum()),
            "false_kept": int((~target & keep).sum()),
            **{f"all_{k}": v for k, v in summary(every).items()},
            **{f"filter_{k}": v for k, v in summary(filtered).items()},
            **{f"perfect_{k}": v for k, v in summary(perfect).items()},
        }
    )
    log(setting, bounds[-1])
    if setting == "A0_join":
        errors = table.assign(target=target, score=np.round(scores, 4), kept=keep)
bounds = pd.DataFrame(bounds)
bounds.to_csv(output_dir() / "filter_bound.csv", index=False)
errors.to_csv(output_dir() / "filter_errors.csv", index=False)

# %% [markdown]
# ## The filter's errors on the submitted candidates
#
# Medians of the features per decision: a dropped true candidate that looks like the kept true ones (and
# unlike the dropped false ones) is a feature the filter lacks, not noise.

# %%
errors["decision"] = np.select(
    [errors["target"] & errors["kept"], errors["target"], errors["kept"]],
    ["true_kept", "true_dropped", "false_kept"],
    "false_dropped",
)
columns = ["area", *FEATURES, "score"]
by_decision = {
    d: {"rows": len(g), **{c: round(float(g[c].median()), 3) for c in columns}} for d, g in errors.groupby("decision")
}
score_bins = pd.cut(errors["score"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01], right=False)
by_score = errors.groupby(score_bins, observed=False)["target"].agg(rows="size", true="sum")
by_score["true_share"] = (by_score["true"] / by_score["rows"].clip(lower=1)).round(3)
report = {
    "name": NAME,
    "bounds": bounds.to_dict(orient="records"),
    "by_decision": by_decision,
    "by_score": {str(k): v for k, v in by_score.to_dict(orient="index").items()},
}
print(json.dumps(report, indent=2))
(output_dir() / "filter_bound_report.json").write_text(json.dumps(report, indent=2))
log("done")
