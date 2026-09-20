"""Where the competition data and the writable output directory live."""

from pathlib import Path

DATASET_DIR_NAME = "MAGFiLO_1.0_Kaggle_2026"


def find_data_root(search_root: str = "/kaggle/input") -> Path:
    """Locate the dataset folder; Kaggle's mount point has changed between releases, so search for it."""
    hits = sorted(Path(search_root).glob(f"**/{DATASET_DIR_NAME}"))
    if not hits:
        raise FileNotFoundError(f"{DATASET_DIR_NAME} not found under {search_root}")
    return hits[0]


def output_dir() -> Path:
    kaggle_working = Path("/kaggle/working")
    return kaggle_working if kaggle_working.is_dir() else Path.cwd()
