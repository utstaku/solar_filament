"""The few facts the pipeline needs to know about this competition and this repository."""

from pathlib import Path

COMPETITION = "filament-segmentation-2026"
KERNEL_PREFIX = "filament"  # solution/unet_tiles.py runs as the notebook <your kaggle user>/filament-unet-tiles

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "solution"  # one directory per approach, holding its entry scripts and modules
# where entry scripts may live -> prefix of the notebook name (solution/unet/train.py -> unet-train,
# error_analysis/cnn_errors.py -> error-analysis-cnn-errors)
ENTRY_DIRS = {SOLUTION_DIR: "", ROOT / "error_analysis": "error-analysis"}
# packages outside solution/ that notebooks may import; everything else resolves under SOLUTION_DIR
PACKAGES = {
    "dataset": ROOT / "dataset",  # data layout and annotations
    "scoring": ROOT / "scoring",  # submission format and metric
    "preprocess": ROOT / "preprocess",  # image preprocessing
    "postprocess": ROOT / "postprocess",  # probability maps -> joined, filtered instances
    "models": ROOT / "models",  # architectures
    "error_analysis": ROOT / "error_analysis",  # per-instance error tables, what-if bounds, galleries
}
BUILD_DIR = ROOT / "build"
OUTPUT_DIR = ROOT / "outputs"

# Only these come back from a run by default. Weights and frames stay on Kaggle; `filament run --figures`
# also fetches the PNG figures a notebook saved (galleries of error cases, plots).
OUTPUT_PATTERN = r".*\.(csv|json|txt)$"
FIGURES_PATTERN = r".*\.(csv|json|txt|png|jpg)$"
