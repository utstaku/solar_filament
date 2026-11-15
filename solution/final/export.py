# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary-full,{user}/filament-zoo-tta
# %% [markdown]
# # Final submission, step 1: the files the pipeline loads instead of retraining
#
# CPU only. The final pipeline (`final/pipeline.py`) must reproduce the submission from files in the
# repository, without training anything. This writes them:
#
# - `swin_unet_boundary_full_fp16.pt`: the Swin U-Net trained on all 707 frames (`zoo/swin_unet_boundary_full.py`),
#   its weights stored as float16 so the file stays under GitHub's 100 MB limit (inference runs in float16 anyway);
# - `final_postprocess.pkl`: the fitted post-processing (`postprocess.pipeline.Postprocess`): hysteresis 0.6 / 0.45,
#   join, the component filter fitted on the 707 out-of-fold 8-view maps (`zoo/tta.py`) with its out-of-fold
#   cutoff, and the kept filaments cut at probability 0.6 (`zoo/final_sweep.py`, OOF PQ 0.4357);
# - `final_config.json`: what was used, with the out-of-fold PQ before the cut.

# %%
import dataclasses
import json
import pickle

import numpy as np
import sklearn
import torch

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess import pipeline
from postprocess.filtering import frame_context
from postprocess.pipeline import A0
from postprocess.tuning import pq_of
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
post, oof = pipeline.fit(contexts, truth, files, params=PARAMS)
post = dataclasses.replace(post, mask_cut=MASK_CUT)
with open(output_dir() / "final_postprocess.pkl", "wb") as f:
    pickle.dump(post, f)

config = {
    "model": f"{NAME}_full_fp16.pt",
    "views": VIEWS,
    "params": PARAMS,
    "join": post.join,
    "filter_cutoff": post.filter.cutoff,
    "mask_cut": MASK_CUT,
    "oof_pq_before_cut": round(pq_of(oof.terms), 4),
    "kept_per_frame": round(oof.kept_per_frame, 2),
    "scikit_learn": sklearn.__version__,
    "torch": torch.__version__,
}
print(json.dumps(config, indent=2))
(output_dir() / "final_config.json").write_text(json.dumps(config, indent=2))
