# kaggle: gpu=true accelerator=NvidiaTeslaT4
# %% [markdown]
# # Zoo, boundary bce: five folds, out-of-fold maps for all 707 frames
#
# the plain U-Net with the BCE of pixels within 3 px of a filament's border counted 5 times
# (`boundary_bce_dice_loss`), aimed at near misses that lose a filament's border and faint ends.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `boundary_bce_oof_fold{k}.npz`,
# `boundary_bce_test_fold{k}.npz`, `boundary_bce_fold{k}.pt`, `boundary_bce_folds.json`, `boundary_bce_report.json`.

# %%
from zoo.experiment import run

run("boundary_bce")
