"""The few facts the pipeline needs to know about this competition and this repository."""

from pathlib import Path

COMPETITION = "filament-segmentation-2026"
KERNEL_PREFIX = "filament"  # solution/unet_tiles.py runs as the notebook <your kaggle user>/filament-unet-tiles
N_TEST_IMAGES = 180

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solution"  # entry scripts and the approach packages they import
MODELS_DIR = ROOT / "models"  # model architectures: a package of its own, shared by every approach
# import names that are packages outside solution/; everything else resolves under SOLUTION_DIR
PACKAGES = {"models": MODELS_DIR}
BUILD_DIR = ROOT / "build"
OUTPUT_DIR = ROOT / "outputs"

# Only these come back from a run. Weights and images stay on Kaggle.
OUTPUT_PATTERN = r".*\.(csv|json|txt)$"
