# kaggle: kernel_sources={user}/filament-zoo-swin-unet
# %% [markdown]
# # Zoo submission, swin_unet: fold 0's model on the test frames, post-processed
#
# The join and the component filter are fitted on the 707 out-of-fold maps of `zoo/swin_unet.py` and applied to
# its fold-0 model's test maps (`zoo/submission.py`). A check that the out-of-fold ranking holds on the
# leaderboard before training the experiment on all frames.

# %%
from zoo.submission import write_submission

write_submission("swin_unet", fold=0)
