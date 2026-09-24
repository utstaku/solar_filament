"""The few facts the pipeline needs to know about this competition and this repository."""

from pathlib import Path

COMPETITION = "filament-segmentation-2026"
KERNEL_PREFIX = "filament"  # solution/unet_tiles.py runs as the notebook <your kaggle user>/filament-unet-tiles
N_TEST_IMAGES = 180

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solution"  # entry scripts and the packages they import
BUILD_DIR = ROOT / "build"
OUTPUT_DIR = ROOT / "outputs"

# Only these come back from a run. Weights and images stay on Kaggle.
OUTPUT_PATTERN = r".*\.(csv|json|txt)$"
