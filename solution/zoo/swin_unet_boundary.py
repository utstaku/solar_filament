# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin_unet_boundary: five folds, out-of-fold maps for all 707 frames
#
# The Swin-UNet (`models/swin_unet.py`) trained with the boundary-weighted BCE + Dice of
# `boundary_bce`: the two changes that raised the out-of-fold PQ, together.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_boundary_oof_fold{k}.npz`,
# `swin_unet_boundary_test_fold{k}.npz`, `swin_unet_boundary_fold{k}.pt`, `swin_unet_boundary_folds.json`,
# `swin_unet_boundary_report.json`.

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

run("swin_unet_boundary")
