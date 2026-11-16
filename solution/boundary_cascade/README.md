# Boundary loss, then detect/refine

`main.py` runs the following sequentially on Kaggle T4, with seed=0 and the
same 15% physical-image validation split as previous experiments:

1. Plain U-Net from scratch with boundary weights 0 (control), 2 and 4, each
   trained 16 epochs on 512px tiles, 16 tiles/batch, AdamW + OneCycle 1e-3.
   Boundary band is radius 2 around target>=0.5, with replicate padding.
   BCE retains soft targets and has weight `1 + w * band`; Dice is unchanged.
   Full-frame inference and solar-disk masking are identical across these runs.
2. COCO-pretrained YOLO11s detector: 1280px, batch=4, 40 epochs, last checkpoint.
   Detector uses the first sorted annotation entry per physical frame to avoid
   duplicated overlapping labels. Refiner sees all train annotators separately:
   ImageNet ResNet18 U-Net, 384px crops, 12 epochs, batch=8, LR=3e-4, BCE+Dice.
   Inputs are gray, CLAHE and a filled box seed, with training box jitter and
   identical crop geometry at inference. No hard-negative crop mining yet.
   Inference confidence=0.28, NMS IoU=0.5, mask threshold=128/255, min_area=150.
   Instances are greedily made disjoint in detector-confidence order.

The boundary models use fixed high=0.5/low=0.2/area=150 hysteresis. Both methods
report raw PQ and PQ after a 5-fold physical-frame OOF component classifier with
fixed cutoff=0.45. This cutoff was previously selected on this validation split,
so it is not an unbiased final estimate. Sparse/one-class folds have explicit
fallbacks. All experiments use final neural-network checkpoints; no checkpoint
selection on validation PQ in this initial run.

No annotation-consensus weighting, model fusion or other novelty is added.
No competition submission. Save weights, configs, loss histories, validation
maps/labels and small reports; only small reports return locally. Compare the
boundary models to the newly trained weight-0 control, not only old PQ=0.3767.
Cascade and semantic models have different inference/annotation treatment; their
comparison is between complete pipelines, not an architecture-only ablation.

Run: `uv run filament run solution/boundary_cascade/main.py`

YOLO11 API source: https://docs.ultralytics.com/models/yolo11/
Packages pinned in the entry: SMP 0.5.0, Ultralytics 8.3.200 (AGPL-3.0).
Training uses only competition training annotations, plus standard COCO/ImageNet
initialization. Other participants' trained filament weights are not used.
