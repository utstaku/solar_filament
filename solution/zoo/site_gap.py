# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full
# %% [markdown]
# # Why is the leaderboard 0.05 below the out-of-fold PQ? Sites, years and annotators of train and test
#
# CPU only. The submitted `swin_unet_boundary` scores 0.4223 out of fold and 0.37 on the public leaderboard;
# every model so far kept that gap. If the test frames come from other sites or years than the training mix,
# or have other numbers of annotators, the out-of-fold PQ reweighted to the test's mix should move toward
# the leaderboard, and the sites that pull it down are where to work.
#
# - Out-of-fold PQ (submitted post-processing, `pipeline.fit`) per site, per year and per number of
#   annotators, with their share of training frames;
# - the test frames' share per site and year (from the file name), and the out-of-fold PQ reweighted to it;
# - filaments kept per frame by site, out of fold ((tp + fp) / annotators: every annotator sees the same
#   predictions) and on the test maps of the full-data model.
#
# Outputs: `site_gap.json`, `site_frames.csv`.

# %%
import json
import time

import numpy as np
import pandas as pd

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess import pipeline
from postprocess.filtering import frame_context
from postprocess.tuning import pq_of
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


root = find_data_root()
train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
maps = {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
files = sorted(maps)
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
post, oof = pipeline.fit(contexts, truth, files)
log(f"out-of-fold PQ {pq_of(oof.terms):.4f}")

saved = np.load(find_input(f"{NAME}_test_full.npz"))
test_files = sorted(saved.files)
test_kept = {}
for f in test_files:
    test_kept[f] = len(post(frame_context(f, read_gray(test_dir / f), saved[f]))[1])

# %%
frames = oof.terms.set_index("frame").assign(
    site=lambda d: d.index.str[14:16],
    year=lambda d: d.index.str[:4].astype(int),
    annotators=[len(truth[f]) for f in oof.terms["frame"]],
    filaments=[sum(len(c) for c in truth[f]) / max(len(truth[f]), 1) for f in oof.terms["frame"]],
)
test = pd.DataFrame({"frame": test_files, "kept": [test_kept[f] for f in test_files]}).assign(
    site=lambda d: d["frame"].str[14:16], year=lambda d: d["frame"].str[:4].astype(int)
)


def by(column: str) -> dict:
    out = {}
    for key, g in frames.groupby(column):
        out[str(key)] = {
            "frames": len(g),
            "share": round(len(g) / len(frames), 3),
            "pq": round(pq_of(g), 4),
            "filaments_per_annotator": round(float(g["filaments"].mean()), 2),
        }
    return out


def reweighted(column: str) -> float:
    """Out-of-fold PQ with each group's terms scaled to its share of the test frames (groups absent in
    training are left out)."""
    test_share = test[column].value_counts(normalize=True)
    parts = []
    for key, g in frames.groupby(column):
        if key in test_share.index:
            parts.append(g[["tp", "fp", "fn", "iou_sum"]].sum() / len(g) * test_share[key])
    total = sum(parts)
    return float(total["iou_sum"] / (total["tp"] + 0.5 * total["fp"] + 0.5 * total["fn"]))


report = {
    "name": NAME,
    "oof_pq": round(pq_of(oof.terms), 4),
    "train_by_site": by("site"),
    "train_by_year": by("year"),
    "train_by_annotators": by("annotators"),
    "test_share_by_site": test["site"].value_counts(normalize=True).round(3).to_dict(),
    "test_share_by_year": {str(k): v for k, v in test["year"].value_counts(normalize=True).round(3).items()},
    "test_sites_not_in_train": sorted(set(test["site"]) - set(frames["site"])),
    "test_years_not_in_train": sorted(int(y) for y in set(test["year"]) - set(frames["year"])),
    "oof_pq_reweighted_to_test_sites": round(reweighted("site"), 4),
    "oof_pq_reweighted_to_test_years": round(reweighted("year"), 4),
    "val_kept_per_frame_by_site": {
        str(s): round(float(((g["tp"] + g["fp"]) / g["annotators"]).mean()), 2) for s, g in frames.groupby("site")
    },
    "test_kept_per_frame_by_site": {str(s): round(float(g["kept"].mean()), 2) for s, g in test.groupby("site")},
}
print(json.dumps(report, indent=2))
(output_dir() / "site_gap.json").write_text(json.dumps(report, indent=2))
frames.to_csv(output_dir() / "site_frames.csv")
log("done")
