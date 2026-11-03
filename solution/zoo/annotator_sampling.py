# kaggle: gpu=true accelerator=NvidiaTeslaT4
# %% [markdown]
# # Zoo, annotator sampling: five folds, out-of-fold maps for all 707 frames
#
# the plain U-Net trained on one annotation set per visit, drawn at random, instead of their mean:
# a filament only one annotator drew keeps a full target instead of a 1/3 one.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `annotator_sampling_oof_fold{k}.npz`,
# `annotator_sampling_test_fold{k}.npz`, `annotator_sampling_fold{k}.pt`, `annotator_sampling_folds.json`, `annotator_sampling_report.json`.

# %%
from zoo.experiment import run

run("annotator_sampling")
