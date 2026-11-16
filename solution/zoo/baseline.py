# kaggle: gpu=true accelerator=NvidiaTeslaT4
# %% [markdown]
# # Zoo, baseline: five folds, out-of-fold maps for all 707 frames
#
# the plain U-Net of `cnn_baseline` (BCE + Dice on the mean of the annotators): the reference every
# other experiment is compared with, made the same way (same folds, tiled inference).
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `baseline_oof_fold{k}.npz`,
# `baseline_test_fold{k}.npz`, `baseline_fold{k}.pt`, `baseline_folds.json`, `baseline_report.json`.

# %%
from zoo.experiment import run

run("baseline")
