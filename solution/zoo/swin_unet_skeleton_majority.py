# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin_unet_skeleton_majority: five folds, out-of-fold maps for all 707 frames
#
# The Swin-UNet trained with BCE + Dice plus Skeleton Recall as `zoo/swin_unet_skeleton`, with two changes after
# that run lost 0.09 PQ (fewer matches, more false positives, lower IoU): the skeleton is that of the pixels
# a majority of the annotators drew (`level=0.5`), so filaments one annotator in three drew no longer pull
# the map up, and the term weighs 0.2 instead of 1.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_skeleton_majority_oof_fold{k}.npz`,
# `swin_unet_skeleton_majority_test_fold{k}.npz`, `swin_unet_skeleton_majority_fold{k}.pt`, `swin_unet_skeleton_majority_folds.json`,
# `swin_unet_skeleton_majority_report.json`.

# %%
import subprocess
import sys

# Pin the model implementations across local checks and the Kaggle runtime.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0", "timm==1.0.29"],
    check=True,
)

# %%
from zoo.experiment import run

run("swin_unet_skeleton_majority")
