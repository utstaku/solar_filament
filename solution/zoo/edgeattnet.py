# kaggle: gpu=true accelerator=NvidiaTeslaT4
# %% [markdown]
# # Zoo, edgeattnet: five folds, out-of-fold maps for all 707 frames
#
# the plain U-Net with edge-guided self-attention at its bottleneck (`models/edgeattnet.py`),
# re-implemented from the EdgeAttNet paper (MAGFiLO, barbs).
# Everything else is shared by the zoo (`zoo/experiment.py`). Outputs: `edgeattnet_oof_fold{k}.npz`,
# `edgeattnet_test_fold{k}.npz`, `edgeattnet_fold{k}.pt`, `edgeattnet_folds.json`, `edgeattnet_report.json`.

# %%
from zoo.experiment import run

run("edgeattnet")
