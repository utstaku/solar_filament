# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin_unet on all 707 training frames: the model to submit
#
# The out-of-fold run (`zoo/swin_unet.py`) chose it; this trains it once more on every training frame with
# the same schedule and predicts the test frames. Outputs: `swin_unet_full.pt`, `swin_unet_test_full.npz`,
# `swin_unet_full_report.json`.

# %%
import subprocess
import sys

# Pin the model implementations across local checks and the Kaggle runtime.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0", "timm==1.0.29"],
    check=True,
)

# %%
from zoo.experiment import run_full

run_full("swin_unet")
