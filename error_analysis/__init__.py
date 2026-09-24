"""Where a submission loses Panoptic Quality, instance by instance.

PQ = sum of IoU over matches / (TP + 0.5 FP + 0.5 FN), a match being IoU > 0.5. Only three things
can go wrong, so the analysis classifies every ground-truth filament and every prediction by how it
relates to the other side (`instances`), aggregates the tables into what-if bounds and breakdowns by
size, contrast, position, site and annotator agreement (`report`), draws the worst cases (`gallery`),
and measures how far apart the human annotators are from each other (`annotators`), which is the
ceiling any model is scored against.

The tables are plain CSVs, so they come back from Kaggle and can be sliced locally without the images.
"""
