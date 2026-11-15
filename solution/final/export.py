# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary-full,{user}/filament-zoo-tta
# %% [markdown]
# # Final submission, step 1: the files the pipeline loads instead of retraining
#
# CPU only. The final pipeline (`final/pipeline.py`) must reproduce the submission from files in the
# repository, without training anything. This writes them:
#
# - `swin_unet_boundary_full_fp16.pt`: the Swin U-Net trained on all 707 frames (`zoo/swin_unet_boundary_full.py`),
#   its weights stored as float16 so the file stays under GitHub's 100 MB limit (inference runs in float16 anyway);
# - `final_filter_table.csv`: the component filter's training set, one row per candidate (hysteresis 0.6 / 0.45,
#   joined) of the 707 out-of-fold 8-view maps (`zoo/tta.py`): its features and whether it matched an annotator;
# - `final_config.json`: the post-processing (`postprocess.pipeline.Postprocess`): hysteresis, join, the filter's
#   out-of-fold cutoff and the mask cut at 0.6 (`zoo/final_sweep.py`, OOF PQ 0.4357).
#
# The pipeline fits the filter (gradient boosting with a fixed seed, a few seconds) from the table rather than
# unpickling it: a pickled scikit-learn model only loads in the version that wrote it.

# %%
import json

import numpy as np
import pandas as pd
import torch

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import FEATURES, component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.pipeline import A0, candidates
from postprocess.tuning import best_cutoff, pq_of
from preprocess.images import read_gray
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS

NAME = "swin_unet_boundary"
PARAMS = {**A0, "threshold": 0.6, "low": 0.45}
MASK_CUT = 0.6
VIEWS = 8

state = torch.load(find_input(f"{NAME}_full.pt"), map_location="cpu")
torch.save(
    {k: v.half() if v.is_floating_point() else v for k, v in state.items()}, output_dir() / f"{NAME}_full_fp16.pt"
)

root = find_data_root()
train_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
maps = {}
for k in range(N_FOLDS):
    saved = np.load(find_input(f"{NAME}_tta{VIEWS}_oof_fold{k}.npz"))
    maps |= {f: saved[f] for f in saved.files}
files = sorted(maps)
contexts = {f: frame_context(f, read_gray(train_dir / f), maps[f]) for f in files}
truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
# as `pipeline.fit` does: candidates, their features and targets, the cutoff with the best out-of-fold PQ
raw = {f: candidates(contexts[f], PARAMS) for f in files}
table = pd.concat(
    [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
)
target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
cutoff, terms = best_cutoff(out_of_fold_scores(table, target), raw, table, truth, files)
table[FEATURES].assign(target=target.astype(int)).to_csv(output_dir() / "final_filter_table.csv", index=False)

config = {
    "model": f"{NAME}_full_fp16.pt",
    "views": VIEWS,
    "params": PARAMS,
    "join": True,
    "filter_cutoff": cutoff,
    "mask_cut": MASK_CUT,
    "oof_pq_before_cut": round(pq_of(terms), 4),
    "candidates": len(table),
}
print(json.dumps(config, indent=2))
(output_dir() / "final_config.json").write_text(json.dumps(config, indent=2))
