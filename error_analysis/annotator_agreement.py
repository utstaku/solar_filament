# kaggle: gpu=false
# %% [markdown]
# # How well do the annotators agree with each other?
#
# Every frame with two or more annotation sets is scored both ways, one annotator's filaments as the
# "prediction" and another's as the truth, with the competition's Panoptic Quality. The result is the
# ceiling a model is measured against: it cannot agree with all annotators better than they agree with
# each other.
#
# Outputs: `human_ceiling.json`, `annotator_pairs.csv`.

# %%
import json
import time

import matplotlib.pyplot as plt

from dataset import IMAGE_SIZE
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from error_analysis.annotators import human_ceiling

T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


# %%
root = find_data_root()
annotations = load_annotations(next((root / "train").glob("*.json")))
files = sorted(annotations.file_to_entries)
multi = [f for f in files if len(annotations.file_to_entries[f]) >= 2]
log(f"{len(files)} frames, {len(multi)} with at least two annotators")

# %%
result = human_ceiling(annotations, multi, (IMAGE_SIZE, IMAGE_SIZE))
pairs = result.pop("pairs")
log(json.dumps(result, indent=2))
pairs.to_csv(output_dir() / "annotator_pairs.csv", index=False)
(output_dir() / "human_ceiling.json").write_text(json.dumps(result, indent=2))

# %% [markdown]
# ## Per-frame agreement
#
# Left: PQ of one annotator scored against another, per frame and direction. Right: per pair of
# annotators, how many of the truth filaments the other annotator also drew (recall) and how many of
# the other annotator's filaments were in the truth (precision).

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(pairs["pq"], bins=25, color="navy", alpha=0.75)
axes[0].axvline(
    result["overall"]["pq"], color="crimson", linestyle="--", label=f"overall {result['overall']['pq']:.3f}"
)
axes[0].set_xlabel("PQ, annotator vs annotator")
axes[0].set_ylabel("frame-direction pairs")
axes[0].legend()
recall = pairs["tp"] / (pairs["tp"] + pairs["fn"]).clip(lower=1)
precision = pairs["tp"] / (pairs["tp"] + pairs["fp"]).clip(lower=1)
axes[1].scatter(recall, precision, s=8, alpha=0.4, color="navy")
axes[1].set_xlabel("recall of the truth annotator's filaments")
axes[1].set_ylabel("precision of the other annotator's filaments")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(output_dir() / "annotator_agreement.png", dpi=110)
plt.show()
