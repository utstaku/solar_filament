# Pretrained U-Net experiment

Run on Kaggle (validation only, no competition submission):

```sh
uv run filament run solution/pretrained_unet/main.py
```

The notebook trains `efficientnet_b3.ra2_in1k` and `convnext_tiny.fb_in1k` encoders
with the same U-Net decoder, then selects the higher validation PQ and trains
that encoder again from ImageNet initialization with a skeleton recall weight of
0.2. All three runs use seed 0, a physical-image 15% validation split, 16 epochs,
512px tiles, four tiles per frame, and four tiles per optimizer step. The encoder
learning rate is 1e-4 and the decoder/head learning rate is 1e-3 (OneCycle).

Both raw and limb-corrected grayscale channels use `(x - 0.449) / 0.226`.
`timm` adapts the pretrained three-channel input weights to two channels.
Inference uses native-resolution overlapping 512px tiles with stride 384.
Postprocessing uses the main branch's connected components, not the separate
postprocessing branch's hysteresis. These results therefore should not be
compared directly to the previously downloaded PQ=0.3767 as an architecture-only
change.

The skeleton target is a radius-one dilated skeleton clipped to each annotator's
foreground, averaged across annotation sets, and transformed with the image.
It is a binary/soft-annotator adaptation of
[Skeleton Recall Loss](https://arxiv.org/abs/2404.03010), not a full reproduction
of the authors' nnU-Net configuration. Empty skeleton tiles contribute zero.
Only segmentation masks are used, not supplied spine metadata.

Outputs per run: `model.pt`, `config.json`, `report.json`, `val_probs.npz`,
`errors_gt.csv`, `errors_pred.csv`, `error_report.json`, and `breakdowns.txt`.
The top-level `comparison.json` updates after each completed experiment;
`experiment_config.json` records exact split filenames and package versions.
Model weights and probability maps stay on Kaggle; the CLI retrieves small
reports. Each epoch overwrites the checkpoint with model weights and history;
optimizer resume is not implemented.

Selection uses the final epoch's best PQ over the same threshold/area grid for
each model. Fixed postprocessing PQ (threshold=0.5, min_area=100) is also saved.
This is a single-split exploratory comparison; choosing models and thresholds
on this split gives an optimistic estimate. Skeleton recall is not guaranteed
to improve PQ or fragmentation.
