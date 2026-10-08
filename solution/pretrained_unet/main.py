# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Pretrained U-Net comparison and skeleton recall ablation
# Same physical-image split, native-resolution tiles and training budget for all three runs.
# The third run starts from ImageNet weights, not from the winner's fitted checkpoint.
# Validation-only experiment; each run saves weights, probability maps, PQ sweep and error analysis.

# %%
import subprocess
import sys

# Pin the model implementations across local checks and the Kaggle runtime.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0", "timm==1.0.29"],
    check=True,
)

# %%
from pretrained_unet.experiment import run

run()
