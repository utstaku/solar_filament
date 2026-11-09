# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin_unet_boundary_seed1: five folds, out-of-fold maps for all 707 frames
#
# `zoo/swin_unet_boundary.py` again with other training seeds (`TRAIN_SEEDS`) and the same folds, so its
# out-of-fold maps can be averaged with the first run's: what a two-seed ensemble gives.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_boundary_seed1_oof_fold{k}.npz`,
# `swin_unet_boundary_seed1_test_fold{k}.npz`, `swin_unet_boundary_seed1_fold{k}.pt`, `swin_unet_boundary_seed1_folds.json`,
# `swin_unet_boundary_seed1_report.json`.

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

run("swin_unet_boundary_seed1")
