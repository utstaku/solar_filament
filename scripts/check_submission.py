"""Sanity-check a submission CSV against the competition's format before spending a submission."""

import re
import sys

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from solar_filament import IMAGE_SIZE, rle
from solar_filament.submission import COLUMNS

ID_PATTERN = re.compile(r"^\d{14}[A-Z][a-z]_\w+$")  # e.g. 20150125172714Mh_1
N_TEST_IMAGES = 180


def main(path: str) -> int:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUMNS:
        print(f"columns must be {COLUMNS}, got {list(df.columns)}")
        return 1
    errors = []
    if df["filament_id"].duplicated().any():
        errors.append(f"{int(df['filament_id'].duplicated().sum())} duplicated filament_id")
    bad_ids = df.loc[~df["filament_id"].str.match(ID_PATTERN), "filament_id"]
    if len(bad_ids):
        errors.append(f"{len(bad_ids)} malformed filament_id, e.g. {bad_ids.iloc[0]!r}")
    valid = df["segmentation_rle"].map(rle.is_valid)
    if not valid.all():
        errors.append(f"{int((~valid).sum())} invalid RLE strings, e.g. row {int(valid.idxmin())}")
        df = df[valid]

    areas = []
    overlap_images = 0
    image_ids = df["filament_id"].str.split("_", n=1).str[0]
    for _, group in df.groupby(image_ids):
        rles = [{"size": [IMAGE_SIZE, IMAGE_SIZE], "counts": c.encode("ascii")} for c in group["segmentation_rle"]]
        a = mask_util.area(rles)
        areas.extend(a.tolist())
        if len(rles) > 1 and int(mask_util.area(mask_util.merge(rles))) < int(a.sum()):
            overlap_images += 1

    n_images = image_ids.nunique()
    print(f"{path}: {len(df)} filaments over {n_images}/{N_TEST_IMAGES} test images")
    if areas:
        q = np.percentile(areas, [0, 50, 100]).astype(int)
        print(f"  filaments/image: {len(df) / max(n_images, 1):.1f} | area px min/median/max: {q[0]}/{q[1]}/{q[2]}")
    if 0 in areas:
        errors.append(f"{areas.count(0)} empty masks")
    if overlap_images:
        print(f"  warning: predicted filaments overlap each other in {overlap_images} images")
    if n_images < N_TEST_IMAGES:
        print(f"  warning: {N_TEST_IMAGES - n_images} test images have no prediction")
    for e in errors:
        print("  ERROR:", e)
    print("  OK" if not errors else f"  {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
