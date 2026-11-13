# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin_unet_boundary_ema: five folds, out-of-fold maps for all 707 frames
#
# `zoo/swin_unet_boundary.py` again with the weights averaged over training (`EMA`, decay 0.998), same folds and seeds, so its
# out-of-fold maps differ from the first run's only by the averaging: does one averaged model do what averaging seeds did?
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_boundary_ema_oof_fold{k}.npz`,
# `swin_unet_boundary_ema_test_fold{k}.npz`, `swin_unet_boundary_ema_fold{k}.pt`, `swin_unet_boundary_ema_folds.json`,
# `swin_unet_boundary_ema_report.json`.

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

run("swin_unet_boundary_ema")
