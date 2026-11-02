# kaggle: gpu=false
# %% [markdown]
# # Rotation check: which way does the solar surface move between consecutive days?
#
# CPU only, competition frames only. Moving a neighbouring day's filaments to today needs the image
# orientation: `cnn_baseline/heliographic.py` assumes solar north up and east on the left (position angle
# p = 0). The Sun rotates from east to west, so in such frames the disk-centre region moves **to the right**,
# by about R sin(13.4 deg x days) -- roughly 0.23 R per day.
#
# For every pair of frames taken 0.5-1.5 days apart, the central region of the second frame is phase-
# correlated with the first. The direction of the shift estimates p (0 = north up; 180 or a leftward shift
# would mean a mirrored or upside-down frame); its length, divided by the expected R sin(13.4 deg x days),
# should be about 1. Pairs are grouped by the sites of both frames, since sites could orient frames differently.
#
# Outputs: `rotation_pairs.csv`, `rotation_report.json`, `gallery_rotation.png` (stays within the team: real frames).

# %%
import json
import time
from datetime import UTC, datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.paths import find_data_root, output_dir
from preprocess.disk import disk_geometry, flatten
from preprocess.images import read_gray

MIN_DAYS, MAX_DAYS = 0.5, 1.5
CROP = 0.35  # half-size of the central square, in disk radii: well inside the limb
RATE = 13.38  # deg / day, synodic rotation at the equator (the centre region is near the equator)
MAX_PAIRS = 300
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:5.0f}s]", *args, flush=True)


def taken(name: str) -> datetime:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


# %% [markdown]
# ## Pairs of frames 0.5-1.5 days apart (training and test frames alike: no labels are used)

# %%
root = find_data_root()
dirs = {"train": root / "train" / "train_images", "test": root / "test" / "test_images"}
frames = pd.DataFrame(
    [
        {"name": p.name, "split": split, "time": taken(p.name), "site": p.name[14:16]}
        for split, d in dirs.items()
        for p in d.iterdir()
    ]
).sort_values("time", ignore_index=True)
pairs = []
for i, a in frames.iterrows():
    later = frames.iloc[i + 1 :]
    days = (later["time"] - a["time"]).dt.total_seconds() / 86400
    ok = later[(days >= MIN_DAYS) & (days <= MAX_DAYS)]
    if len(ok):
        b = ok.iloc[0]
        pairs.append(
            {
                "a": a["name"],
                "b": b["name"],
                "a_split": a["split"],
                "b_split": b["split"],
                "days": float((b["time"] - a["time"]).total_seconds() / 86400),
                "sites": "".join(sorted({a["site"], b["site"]})),
            }
        )
pairs = pd.DataFrame(pairs).head(MAX_PAIRS)
log(f"{len(frames)} frames, {len(pairs)} pairs {MIN_DAYS}-{MAX_DAYS} days apart")

# %% [markdown]
# ## Shift of the central region

# %%
cache = {}


def central(name: str, split: str, half: int | None = None):
    """(flattened central square as float32, disk radius); the square is centred on the disk centre."""
    if name not in cache:
        image = read_gray(dirs[split] / name)
        cx, cy, radius = disk_geometry(image)
        cache.clear()  # keep memory flat: consecutive pairs share at most one frame
        cache[name] = (flatten(image), cx, cy, radius)
    flat, cx, cy, radius = cache[name]
    half = half or int(CROP * radius)
    x0, y0 = round(cx) - half, round(cy) - half
    return flat[y0 : y0 + 2 * half, x0 : x0 + 2 * half].astype(np.float32), radius, half


rows, crops = [], {}
for p in pairs.itertuples(index=False):
    a, radius, half = central(p.a, p.a_split)
    b, radius_b, _ = central(p.b, p.b_split, half)
    if a.shape != b.shape or min(a.shape) < 2 * half:
        continue
    (dx, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    expected = radius * np.sin(np.radians(RATE * p.days))
    rows.append(
        {
            **p._asdict(),
            "dx": dx,
            "dy": dy,
            "response": response,
            "radius": radius,
            "radius_ratio": radius_b / radius,
            "p_est_deg": float(np.degrees(np.arctan2(-dy, dx))),  # 0 = moving right (west), 90 = moving up
            "speed_ratio": float(np.hypot(dx, dy) / expected),
        }
    )
    if len(crops) < 6 and response > 0.1:
        crops[(p.a, p.b)] = (a, b, dx, dy)
shifts = pd.DataFrame(rows)
shifts.to_csv(output_dir() / "rotation_pairs.csv", index=False)

# %% [markdown]
# ## Report
#
# Only pairs with a clear correlation peak count (response above the median): changing chromospheric texture
# and different seeing make the rest noisy.

# %%
good = shifts[shifts["response"] >= shifts["response"].median()]
report = {
    "pairs": len(shifts),
    "pairs_with_clear_peak": len(good),
    "share_moving_right": round(float((good["dx"] > 0).mean()), 3),
    "p_est_deg_median": round(float(good["p_est_deg"].median()), 2),
    "p_est_deg_iqr": [round(float(q), 2) for q in good["p_est_deg"].quantile([0.25, 0.75])],
    "speed_ratio_median": round(float(good["speed_ratio"].median()), 3),
    "by_sites": {
        s: {
            "pairs": len(g),
            "p_est_deg_median": round(float(g["p_est_deg"].median()), 2),
            "speed_ratio_median": round(float(g["speed_ratio"].median()), 3),
        }
        for s, g in good.groupby("sites")
    },
}
print(json.dumps(report, indent=2))
(output_dir() / "rotation_report.json").write_text(json.dumps(report, indent=2))

fig, axes = plt.subplots(len(crops), 2, figsize=(7, 3.4 * len(crops)), squeeze=False)
for (names, (a, b, dx, dy)), row in zip(crops.items(), axes):
    for ax, img, title in (
        (row[0], a, names[0][:16]),
        (row[1], b, f"{names[1][:16]}  shift ({dx:+.0f}, {dy:+.0f}) px"),
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    c = a.shape[0] / 2
    row[1].annotate("", xy=(c + dx, c + dy), xytext=(c, c), arrowprops={"color": "red", "width": 2})
fig.tight_layout()
fig.savefig(output_dir() / "gallery_rotation.png", dpi=90)
plt.show()
log("done")
