# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, swin unet: five folds, out-of-fold maps for all 707 frames
#
# Swin-T (ImageNet) as the encoder of a U-Net, with two convolutional stages at full and half
# resolution for the skip connections Swin does not have (`models/swin_unet.py`).
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `swin_unet_oof_fold{k}.npz`,
# `swin_unet_test_fold{k}.npz`, `swin_unet_fold{k}.pt`, `swin_unet_folds.json`, `swin_unet_report.json`.

# %%
from zoo.experiment import run

run("swin_unet")
