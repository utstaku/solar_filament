# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, hrnet: five folds, out-of-fold maps for all 707 frames
#
# HRNet-W18 (ImageNet) as the encoder of a U-Net: it keeps a high-resolution branch through every
# stage instead of only downsampling, for the 10-20 px wide filaments.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `hrnet_oof_fold{k}.npz`,
# `hrnet_test_fold{k}.npz`, `hrnet_fold{k}.pt`, `hrnet_folds.json`, `hrnet_report.json`.

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

run("hrnet")
