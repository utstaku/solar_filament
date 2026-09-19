# %% [markdown]
# # Classical filament baseline (no learning)
#
# Flatten the limb darkening, threshold dark pixels, close small gaps and call every connected
# component a filament. It exists to exercise the whole loop (data mount → masks → RLE →
# `submission.csv`) in about two minutes of CPU, and to give the floor a model has to beat.
#
# Outputs: `submission.csv` and `baseline_report.json`.

# %%
import json
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_util

from solar_filament import IMAGE_SIZE, submission
from solar_filament.annotations import load_annotations
from solar_filament.classical import segment
from solar_filament.metric import PanopticQuality, instance_crops
from solar_filament.paths import find_data_root, output_dir
from solar_filament.preprocess import flatten

SEED = 0
N_TUNE_IMAGES = 60  # training images used for the parameter sweep
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def read_gray(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read {path}")
    return image


# %% [markdown]
# ## Data

# %%
root = find_data_root()
train_dir = root / "train" / "train_images"
test_dir = root / "test" / "test_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
train_files = sorted(p.name for p in train_dir.iterdir())
test_files = sorted(p.name for p in test_dir.iterdir())

gt_areas = np.concatenate([mask_util.area(rles) for rles in annotations.entry_to_rles.values()])
report = {
    "n_train_files": len(train_files),
    "n_test_files": len(test_files),
    "n_annotator_images": len(annotations.entry_to_rles),
    "n_annotations": annotations.n_annotations,
    "n_multi_polygon_annotations": annotations.n_multi_polygon,
    "gt_area_px_percentiles": {str(p): float(np.percentile(gt_areas, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
    "gt_foreground_fraction": round(float(gt_areas.sum()) / (len(annotations.entry_to_rles) * IMAGE_SIZE**2), 5),
}
log(json.dumps(report, indent=2))

# %% [markdown]
# ## Parameter sweep
#
# Each tuning image is flattened once and reused for every setting. Panoptic Quality is accumulated
# per annotator-image, exactly as the hosts score a submission.

# %%
random.seed(SEED)
tune_files = random.sample(train_files, min(N_TUNE_IMAGES, len(train_files)))
flats = {f: flatten(read_gray(train_dir / f)) for f in tune_files}
ground_truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in tune_files}
log(f"prepared {len(tune_files)} tuning images")

sweep = []
for ratio in (0.86, 0.90, 0.93):  # filaments are only ~10-15 % darker than their surroundings
    for close_px in (0, 7, 15):
        for min_area in (100, 300, 600):
            pq = PanopticQuality()
            for f in tune_files:
                labels, areas, _ = segment(flats[f], ratio, min_area, close_px)
                for gt_crops in ground_truth[f]:
                    pq.update(gt_crops, labels, areas)
            sweep.append({"ratio": ratio, "close_px": close_px, "min_area": min_area, **pq.summary()})
            log(sweep[-1])

best = max(sweep, key=lambda row: row["pq"])
params = {key: best[key] for key in ("ratio", "close_px", "min_area")}
report.update({"sweep": sweep, "best": best, "n_tune_images": len(tune_files)})
log("best:", best)

# %% [markdown]
# ## What it looks like
#
# First annotator's ground truth in green, prediction in red.

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 18))
for ax, f in zip(axes.ravel(), tune_files):
    canvas = cv2.cvtColor(read_gray(train_dir / f), cv2.COLOR_GRAY2RGB)
    for x, y, crop in ground_truth[f][0] if ground_truth[f] else []:
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x, y))
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    labels, _, _ = segment(flats[f], **params)
    contours, _ = cv2.findContours((labels > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)
    ax.imshow(canvas)
    ax.set_title(f)
    ax.axis("off")
plt.tight_layout()
plt.show()
del flats, ground_truth

# %% [markdown]
# ## Test predictions

# %%
rows = []
for f in test_files:
    labels, _, boxes = segment(flatten(read_gray(test_dir / f)), **params)
    rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
frame = submission.write(rows, output_dir() / "submission.csv")

report["n_test_predictions"] = len(frame)
report["n_test_images_with_predictions"] = int(frame["filament_id"].str.split("_").str[0].nunique())
report["elapsed_sec"] = round(time.time() - T0, 1)
(output_dir() / "baseline_report.json").write_text(json.dumps(report, indent=2))
log(f"wrote submission.csv with {len(frame)} rows")
