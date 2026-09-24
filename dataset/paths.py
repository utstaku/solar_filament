"""Where the competition data and the writable output directory live."""

import os
from pathlib import Path

DATASET_DIR_NAME = "MAGFiLO_1.0_Kaggle_2026"


def find_data_root(search_root: str = "/kaggle/input") -> Path:
    """Locate the dataset folder; Kaggle's mount point has changed between releases, so search for it.

    FILAMENT_DATA overrides the search: it points a local smoke test at a synthetic copy of the layout.
    """
    if os.environ.get("FILAMENT_DATA"):
        return Path(os.environ["FILAMENT_DATA"])
    hits = sorted(Path(search_root).glob(f"**/{DATASET_DIR_NAME}"))
    if not hits:
        raise FileNotFoundError(f"{DATASET_DIR_NAME} not found under {search_root}")
    return hits[0]


def output_dir() -> Path:
    """Kaggle's writable directory, else FILAMENT_OUTPUT, else the current directory."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    return Path(os.environ.get("FILAMENT_OUTPUT", "."))
