---
title: "Solar Filament Segmentation Challenge 2026: technical report"
author: "Team [TEAM NAME] — [authors]"
date: "15 November 2026"
geometry: "a4paper,margin=2cm"
fontsize: 10pt
mainfont: "STIX Two Text"
numbersections: true
header-includes:
  - \usepackage{newunicodechar}
  - \newunicodechar{→}{\ensuremath{\rightarrow}}
  - \newunicodechar{≈}{\ensuremath{\approx}}
  - \newunicodechar{≤}{\ensuremath{\leq}}
  - \newunicodechar{≥}{\ensuremath{\geq}}
  - \usepackage{booktabs}
  - \usepackage{framed}
  - \setlength{\parskip}{2pt}
  - \AtBeginDocument{\typeout{FIGSIZE textwidth=\the\textwidth textheight=\the\textheight}}
---

# Summary

The task is to outline every Hα filament in a 2048 × 2048 GONG full-disk image as its own mask, scored by
panoptic quality (PQ, a match needs IoU > 0.5) against each annotator separately. Filaments are only
10–15 % darker than the chromosphere around them, fade towards their ends and break into pieces. So the
score depends as much on **how pixels are grouped into filaments** as on the pixel map itself.

Our pipeline is a segmentation network followed by a physically motivated post-processing:
hysteresis thresholding, **joining pieces the way filaments physically break**, then a component filter.
Every choice was selected on out-of-fold (OOF) predictions for all 707 training frames. The final model is a
Swin U-Net with a boundary-weighted loss (§4.2); the post-processing and the analyses were developed on a
CNN U-Net (§4.1).

| | OOF PQ (707 frames) | public LB |
|---|---|---|
| annotator vs annotator (296 multi-annotator frames) | 0.34 | — |
| CNN U-Net + join-then-filter | 0.396 | 0.34 |
| **Swin U-Net (boundary) + TTA + hysteresis 0.6/0.45, cut 0.6** | **0.4357** | **0.37** |

The main lesson: the remaining errors sit where annotators trace filaments down to ~3 % absorption
contrast, the level of the background texture (§6). The network stops at the ~10 % core. That is a
limit of the signal and of label agreement, not of the post-processing.

# Data and metric: what the score rewards

**Data.** 707 training and 180 test frames (2011–2022, six GONG sites). 1,154 annotation sets cover
the 707 physical frames: up to three annotators outlined the same frame independently. The 8,199 filaments
are each a single polygon. The foreground is 0.36 % of all pixels, and the median filament area is 1,228 px.

**Metric.** PQ = Σ IoU over matches / (TP + ½FP + ½FN), accumulated per annotator-image. A near miss
(IoU ≤ 0.5) counts as both an FP and an FN.

**Ceiling.** Scoring one annotator's outlines against another's gives **PQ 0.34** (SQ 0.63, RQ 0.54; 1,196
directed pairs on 296 frames, `error_analysis/annotator_agreement.py`). Annotators disagree both on
*which* filaments exist and on their outlines.

**Break-even rule.** Per annotator-image, keeping a predicted component adds ½ to the denominator and
its IoU to the numerator only if it matches. So keeping it raises PQ exactly when
P(match) × IoU > PQ/2 ≈ 0.17–0.2. With IoU ≈ 0.65, a component is worth keeping at a ~30 % chance of
matching. We set every keep/drop cutoff with this rule, not by confidence alone
(`variants.expected_gain`, tested in `tests/test_variants.py`).

**Validation → leaderboard.** Our OOF scores are higher than the public LB by a stable factor. One
parameter, a constant sparsity r ≈ 0.73–0.77 of the test labels relative to training, predicted the public
score of the join-then-filter submission as 0.332–0.342 before it was submitted (observed 0.34;
`importance.public_pq`, `fit_r`).

# Validation design

- **Folds by physical frame:** the annotation sets of one frame never straddle folds. Five folds give OOF
  probability maps for all 707 frames. The component filter is itself fitted out of fold
  (`cnn_baseline/cv.py`, `oof_lab.py`).
- **Paired bootstrap over frames** for every comparison: the same resampled frames enter both
  variants, so frame difficulty cancels in the difference (`importance.paired_bootstrap`). A single 15 %
  split (106 frames) was not enough: fold PQ ranged from 0.31 to 0.35.
- **Caveat:** 36 % of test frames have a training frame within 24 h, and filaments last days. Neighbouring
  days are not independent, so our intervals are somewhat too narrow.
- **A pitfall we caught:** test maps averaged over five fold models are smoother than single-model OOF
  maps. A filter trained on the latter kept only about half as many filaments on test (3.9 vs 6.9 per
  frame). Fix: use single-model maps on both sides, and print the kept density on OOF and test as a check
  (`final_join.py`: 6.84 vs 6.21 per frame).

# Models

## CNN U-Net (development model)

A 4-level U-Net (32 base channels) trained from scratch on 512 px tiles (4 per frame per epoch, half centred
on a filament pixel, D4 rotations and flips), 16 epochs, AdamW with a one-cycle schedule, BCE + Dice loss.

- **Two input channels:** the raw frame and a limb-darkening-corrected frame. Limb darkening is
  *multiplicative*, so we divide by a smooth background (normalised convolution over the disk) rather than
  subtract it. A filament is then "0.88 × background" at the centre and at the limb alike.
- **Soft target:** the fraction of annotators who marked each pixel, which keeps their disagreement as
  a label instead of trusting one of them.
- **D4 test-time augmentation:** +0.002 [−0.001, +0.006] OOF, i.e. within noise (Fig. 1a).

## Swin U-Net with boundary loss (final model)

A U-Net decoder on timm's ImageNet Swin-T encoder (2 input channels as in §4.1, 8 × 8 windows on 512 px tiles),
with two small convolutional stages at full and half resolution for the 10–20 px wide filaments
(`models/swin_unet.py`). Loss: BCE + Dice on the soft target, with the BCE of a 3 px band around each filament's
border weighted ×5, since near misses lose their borders and faint ends, not their cores. Training as in §4.1
(16 epochs, encoder learning rate 1e-4); 37 min per fold on a T4, five folds for OOF maps, then one model on all
707 frames for the test. Inference: the mean of the 8 flips and quarter turns, +0.004 [+0.000, +0.007] OOF.
Post-processing: hysteresis 0.6/0.45, join and filter (§5), then each kept filament is cut to its pixels with
p ≥ 0.6, +0.006: the network's 0.3 contour is wider than the annotators' (median kept area 1,388 → 1,205 px).
OOF PQ 0.4223 → **0.4357** [+0.009, +0.018], public LB 0.37. In a ten-model zoo on the same folds and
post-processing, the Swin encoder (0.4216) and the boundary loss (U-Net 0.3983 → 0.4045) were the two gains;
HRNet-W18 0.397, SegFormer 0.382, Skeleton Recall loss 0.415 and weight averaging (EMA) did not help, and a second
seed (+0.004) was not worth doubling the model.

# Post-processing: from a probability map to filaments

1. **Hysteresis threshold** (A0: high 0.5, low 0.3, minimum area 300 px). A component of
   p ≥ low is kept whole if any pixel reaches high, so a filament's faint tails stay attached to its
   confident core.
2. **Joining by filament physics.** A filament breaks along its spine where its material thins, leaving
   pieces end to end with a gap along the axis. It also shows barbs: short side pieces at an angle. In
   pairs of nearby pieces, 46 % of same-filament pairs are end to end and **no** different-filament pair is;
   21 % of same-filament pairs meet at > 45° (barbs). Rule: join pieces that are end to end along their axes
   within 50 px, and any pieces within 10 px (`pairs.py`, `join_lab.py`).
3. **Component filter.** A gradient-boosted classifier on area, elongation, darkness in the corrected
   image, distance from the disk centre, and mean/max network probability predicts P(match). Its cutoff
   follows the break-even rule. Filtering *after* joining judges whole filaments rather than pieces.

![**(a)** OOF PQ change of post-processing variants against A0 on the 707 out-of-fold frames, with
paired-bootstrap 95 % intervals (variants on D4-TTA maps, A0 on plain maps). **(b)** Absorption contrast
$1 - I/I_\mathrm{bg}$ of annotated pixels in the corrected image: whole matched filaments, the part of a
near-missed filament that the prediction covers, the part it leaves out, and the median local background
spread (1σ).](figures/fig1_ablation_contrast.pdf){width=100%}

Join-then-filter gives **+0.010 [+0.006, +0.015]** OOF on TTA maps (+0.008 [+0.004, +0.012] with
single-model maps in the submitted pipeline). Two negative results are just as instructive. Isotropic
growing of masks and lower hysteresis thresholds did not help. A strict keep rule copied from a public
notebook cost **−0.041** on our maps (Fig. 1a): keep rules are tied to a model's calibration and do not
transfer between models.

**A cutoff for sparser test labels (negative).** The public scores fit test labels in which each filament
is present only with probability r ≈ 0.72–0.76 (§2). By the break-even rule, a kept component is then worth
less on test, so the filter should keep fewer. Choosing the cutoff by the expected public PQ at r instead of
by OOF PQ does move it, from 0.40 to 0.45–0.475 (6.8 → 6.1–6.4 filaments per frame). The expected public gain,
however, is only +0.0005 to +0.0009: the curve is flat between 0.40 and 0.50, far below the leaderboard's
±0.02 resolution (`sparse_cutoff_lab.py`). The submitted cutoff is therefore already near-optimal for the
test labels as well.

# Physics analyses: what the errors tell us about the data

**Where the misses are.** 75 % of the annotated pixels that a near miss leaves out have network
probability < 0.1 (`miss_lab.py`). The network did not see them at all, so no threshold can recover them.

**What annotators draw (Fig. 1b).** In the corrected image, matched filaments and the covered parts of
near misses have a median contrast of 0.10–0.11, i.e. the classical 10–15 % Hα absorption of a filament
core. The left-out parts have a median contrast of only **0.029**, equal to the local background spread
(SNR ≈ 1.0; `contrast_lab.py`). Annotators extend filaments into the faint ends and barbs, where the
absorption is comparable to the chromospheric texture. A pixel classifier cannot separate that from the
background without context; human annotators use continuity along the spine. Whole missed filaments sit
in between (median contrast 0.061, SNR 2.05).

**Limb.** Near the limb the line of sight crosses the chromosphere obliquely: limb darkening is strongest
and filaments are foreshortened in the radial direction. Without a filter, precision falls from 0.54 at
r < 0.5 R to 0.21 at 0.85–0.9 R. The component filter removes most of this excess (hallucinations at
0.85–0.9 R: 123 → 15), at the price of some near-limb recall (`notes/analysis/v1_limb.py`).

**Rotation and evolution.** Moving one day's annotations to a neighbouring day by solar rotation
(`heliographic.py`, `rotation_lab.py`) shows that GONG frames are oriented north-up, and that differential
and rigid rotation fit equally well over 0.5–3 days. Even within half a day, the rotated annotations reach
a median IoU of only 0.19 and cover 29 % of filament pixels. Filaments evolve, and annotators differ,
faster than rotation alone explains. Using neighbouring days at inference is not allowed by the rules
anyway; this result says it would also not have been a reliable prior.

# Quantitative distributions

On the 106-frame validation split (CNN U-Net + A0, `outputs/cnn-baseline-predict/error_report.json`):

| | count | | count |
|---|---|---|---|
| ground truth matched (TP) | 774 | predictions matched | 773 |
| near miss (IoU ≤ 0.5) | 202 | near miss | 202 |
| fragmented (one-to-many) | 74 | fragments | 171 |
| merged (many-to-one) | 33 | merges | 15 |
| missed | 304 | hallucinations | 176 |

PQ 0.378 = SQ 0.665 × RQ 0.568. The median IoU of matched filaments is 0.667, and the median predicted/true
area ratio is 1.07 (masks are not systematically too thin or too thick). On the 707 OOF frames, joining
lowered the fragment count while keeping merges rare: TP 4,674 → 4,782 and FP 3,244 → 3,130 for A0 →
join-then-filter (`final_join.py`). **[TO ADD: the same table for the final Swin pipeline.]**

# Limitations

- The faint ends (≈ 1σ) and faint whole filaments need a model that uses context along the spine.
  Candidates: transformer encoders (the gain of §4.2), targets that do not dilute faint parts (annotator
  disagreement lowers the soft target exactly there), and connectivity losses (clDice, Skeleton Recall;
  one untuned try lowered the score).
- Annotator conventions (barbs, where one filament ends) set a ceiling. PQ against one annotator is noisy:
  ±0.02 on ~100 frames, which is also the resolution of the public LB.
- Intervals ignore day-to-day correlation; a block bootstrap over neighbouring days would be more honest.

# Reproducibility and code

Everything runs as Kaggle notebooks generated from the scripts by one command
(`uv run filament run <script>`: tests → build notebook → push → wait → fetch outputs → format check).
Order: `cnn_baseline/main.py` (train, ~40 min T4) → `predict.py` → `cv.py` (5-fold OOF) →
`final_join.py` (CPU). The pipeline notebook `pipeline.ipynb` walks through one frame end to end.
83 unit tests on synthetic frames cover disk detection, the metric, RLE, post-processing, joining
and the break-even rule. Dependencies: `requirements.txt` (`uv export`).
**Final submission** (Swin): `zoo/swin_unet_boundary.py` (5-fold OOF, ~3.5 h T4) → `zoo/swin_unet_boundary_full.py`
(all frames, 45 min) → `zoo/tta.py` (8-view maps) → `final/export.py` (CPU: fits the post-processing, writes
`weights/`) → **`final/pipeline.py`** (`final_pipeline.ipynb`): loads `weights/` and writes `submission.csv`
from the test frames with nothing retrained, ~20 min on a T4.
