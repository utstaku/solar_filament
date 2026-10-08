# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=false
# %% [markdown]
# # Plain U-Net: five-fold CV and test ensemble
# Each outer fold has an independent inner calibration model for the component filter.
# All postprocessing thresholds are fixed before this run. No external weights or data.
# %%
from unet_cv.experiment import run

run()
