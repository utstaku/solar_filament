# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Boundary loss ablation, followed by YOLO11s + crop U-Net
# Sequential validation-only experiments on the same physical-image split.
# No novel annotation weighting, candidate fusion, or competition submission.

# %%
import importlib.metadata
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "segmentation-models-pytorch==0.5.0", "ultralytics==8.3.200"],
    check=True,
)

# %%
from boundary_cascade import boundary, cascade
from boundary_cascade.common import save
from dataset.paths import output_dir

save(
    output_dir() / "versions.json",
    {
        p: importlib.metadata.version(p)
        for p in ("torch", "torchvision", "numpy", "ultralytics", "segmentation-models-pytorch", "scikit-learn")
    },
)
boundary.run()
cascade.run()
