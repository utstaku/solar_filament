# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=true
# %% [markdown]
# # Zoo, segformer: five folds, out-of-fold maps for all 707 frames
#
# SegFormer MiT-B2 (ImageNet) with its all-MLP decoder at 1/4 resolution: global self-attention
# over the tile, to follow long filaments without breaking them.
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `segformer_oof_fold{k}.npz`,
# `segformer_test_fold{k}.npz`, `segformer_fold{k}.pt`, `segformer_folds.json`, `segformer_report.json`.

# %%
from zoo.experiment import run

run("segformer")
