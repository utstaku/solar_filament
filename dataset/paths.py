"""Where the competition data, attached notebook outputs and the writable output directory live."""

import os
from pathlib import Path

DATASET_DIR_NAME = "MAGFiLO_1.0_Kaggle_2026"


def input_root() -> Path:
    """Kaggle mounts the competition data and any attached notebook outputs under /kaggle/input.

    FILAMENT_INPUT overrides it, so a notebook can be smoke-tested locally on a synthetic copy.
    """
    return Path(os.environ.get("FILAMENT_INPUT", "/kaggle/input"))


def find_data_root() -> Path:
    """Locate the dataset folder; Kaggle's mount point has changed between releases, so search for it.

    FILAMENT_DATA points straight at the folder and skips the search.
    """
    if os.environ.get("FILAMENT_DATA"):
        return Path(os.environ["FILAMENT_DATA"])
    hits = sorted(input_root().glob(f"**/{DATASET_DIR_NAME}"))
    if not hits:
        raise FileNotFoundError(f"{DATASET_DIR_NAME} not found under {input_root()}")
    return hits[0]


def find_input(pattern: str) -> Path:
    """First file under the input root matching a glob, e.g. a `unet.pt` saved by another notebook."""
    hits = sorted(input_root().glob(f"**/{pattern}"))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {input_root()}")
    return hits[0]


def output_dir() -> Path:
    """Kaggle's writable directory, else FILAMENT_OUTPUT, else the current directory."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working
    return Path(os.environ.get("FILAMENT_OUTPUT", "."))
