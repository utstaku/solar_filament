# kaggle: kernel_sources={user}/filament-zoo-boundary-bce
# %% [markdown]
# # Zoo submission, boundary_bce: fold 0's model on the test frames, post-processed
#
# The join and the component filter are fitted on the 707 out-of-fold maps of `zoo/boundary_bce.py` and applied to
# its fold-0 model's test maps (`zoo/submission.py`). A check that the out-of-fold ranking holds on the
# leaderboard before training the experiment on all frames.

# %%
from zoo.submission import write_submission

write_submission("boundary_bce", fold=0)
