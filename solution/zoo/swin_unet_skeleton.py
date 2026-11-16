# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin_unet_skeleton: five folds, out-of-fold maps for all 707 frames
#
# The Swin-UNet (`models/swin_unet.py`) trained with BCE + Dice plus Skeleton Recall
# (`training.skeleton_recall_bce_dice_loss`): `zoo/shapes.py` found that a third of the unmatched filaments
# two or more annotators drew come out as fragments or in pieces, cut where a faint stretch falls below 0.5;
# the recall of the annotated centre line puts weight on those few pixels.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_skeleton_oof_fold{k}.npz`,
# `swin_unet_skeleton_test_fold{k}.npz`, `swin_unet_skeleton_fold{k}.pt`, `swin_unet_skeleton_folds.json`,
# `swin_unet_skeleton_report.json`.

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

run("swin_unet_skeleton")
