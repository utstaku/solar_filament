# kaggle: kernel_sources={user}/filament-zoo-swin-unet,{user}/filament-zoo-swin-unet-full
# %% [markdown]
# # Zoo submission, swin_unet trained on all frames, post-processed
#
# The join and the component filter are fitted on the 707 out-of-fold maps of `zoo/swin_unet.py` and applied
# to the test maps of the model trained on all frames (`zoo/swin_unet_full.py`; `zoo/submission.py`).

# %%
from zoo.submission import write_submission

write_submission("swin_unet", fold="full")
