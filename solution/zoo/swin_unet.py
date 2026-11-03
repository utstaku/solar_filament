# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin unet: five folds, out-of-fold maps for all 707 frames
#
# Swin-T (ImageNet) as the encoder of a U-Net, with two convolutional stages at full and half
# resolution for the skip connections Swin does not have (`models/swin_unet.py`).
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_oof_fold{k}.npz`,
# `swin_unet_test_fold{k}.npz`, `swin_unet_fold{k}.pt`, `swin_unet_folds.json`, `swin_unet_report.json`.

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

run("swin_unet")
