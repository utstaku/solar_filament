# kaggle: kernel_sources={user}/filament-zoo-swin-unet-boundary,{user}/filament-zoo-swin-unet-boundary-full
# %% [markdown]
# # Zoo submission, swin_unet_boundary trained on all frames, post-processed
#
# The join and the component filter are fitted on the 707 out-of-fold maps of `zoo/swin_unet_boundary.py` and applied
# to the test maps of the model trained on all frames (`zoo/swin_unet_boundary_full.py`; `zoo/submission.py`).

# %%
from zoo.submission import write_submission

write_submission("swin_unet_boundary", fold="full")
