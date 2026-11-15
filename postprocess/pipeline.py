"""The post-processing of the best submission: candidates from the probability map, joined into whole
filaments, then filtered, with the filter and its cutoff learnt from out-of-fold frames.

1. **Candidates** (`A0`): hysteresis threshold 0.5 / 0.3 and components of at least 300 px.
2. **Join** (`joining.join_rule`): pieces that continue each other end to end along their axis join up to
   50 px apart, any two pieces up to 10 px (barbs). On 707 OOF frames this raised PQ by about +0.01.
3. **Filter** (`filtering.ComponentFilter`): gradient boosting on shape, darkness, limb distance and the
   network's confidence decides, per joined filament, whether it matched any annotator. Its cutoff is the
   one with the best PQ on out-of-fold filter scores.

Works on any model's maps: `fit` needs the maps of frames the model never saw (out of fold) with their
truth, and the fitted `Postprocess` is then applied to test maps of the same model, so both look alike.

    contexts = {f: frame_context(f, image, prob) for ...}
    post, oof = fit(contexts, truth, train_files)
    labels, areas, boxes = post(contexts[test_file])
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from error_analysis.instances import FrameContext
from postprocess.filtering import ComponentFilter, component_table, matched_by_any_annotator, out_of_fold_scores
from postprocess.instances import cut, instances, select
from postprocess.joining import join_pairs, join_rule, pair_table
from postprocess.tuning import best_cutoff
from scoring.metric import InstanceCrop

A0 = {"threshold": 0.5, "min_area": 300, "low": 0.3}
PAIR_RADIUS = 50  # pairs further apart are never joined (join_rule's longest reach)


def candidates(ctx: FrameContext, params: dict = A0, join: bool = True):
    """(labels, areas, boxes) before the filter: `instances` of `ctx.prob`, joined when `join`.

    Labels are uint8 (at most `MAX_INSTANCES` per frame) so the candidates of 707 frames fit in memory.
    """
    labels, areas, boxes = instances(ctx.prob, **params)
    if join:
        table = pair_table(ctx, labels, PAIR_RADIUS)
        if len(table):
            labels, areas, boxes = join_pairs(labels, table, join_rule(table))
    return labels.astype(np.uint8), areas, boxes


@dataclass
class Postprocess:
    """A fitted post-processing: `post(ctx)` -> (labels, areas, boxes) of the kept filaments."""

    filter: ComponentFilter
    params: dict = field(default_factory=lambda: dict(A0))
    join: bool = True
    mask_cut: float | None = None  # kept filaments reduced to their pixels with probability >= mask_cut

    def __call__(self, ctx: FrameContext):
        inst = candidates(ctx, self.params, self.join)
        if not len(inst[1]):
            return inst
        kept = select(*inst, self.filter.keep(component_table(ctx, inst[0], inst[2])))
        return cut(*kept, ctx.prob, self.mask_cut) if self.mask_cut is not None else kept


@dataclass
class OutOfFold:
    """How the fitted post-processing did on the frames it was fitted on, with out-of-fold filter scores."""

    terms: pd.DataFrame  # per-frame PQ terms (`tuning.frame_terms`)
    cutoff: float
    kept_per_frame: float


def fit(
    contexts: dict[str, FrameContext],
    truth: dict[str, list[list[InstanceCrop]]],
    files: list[str],
    params: dict = A0,
    join: bool = True,
) -> tuple[Postprocess, OutOfFold]:
    """Fit the filter on the candidates of `files` and choose its cutoff on their out-of-fold PQ.

    `contexts[f].prob` must come from a model that never saw frame f; `truth[f]` lists the ground truth per
    annotator. The returned filter is refitted on all of `files`; the `OutOfFold` scores are honest.
    """
    raw = {f: candidates(contexts[f], params, join) for f in files}
    table = pd.concat(
        [component_table(contexts[f], raw[f][0], raw[f][2]) for f in files if len(raw[f][1])], ignore_index=True
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    if not 0 < target.sum() < len(target):
        raise ValueError(
            f"the filter needs both matched and unmatched candidates; {target.sum()} of {len(target)} match"
        )
    scores = out_of_fold_scores(table, target)
    cutoff, terms = best_cutoff(scores, raw, table, truth, files)
    post = Postprocess(ComponentFilter(cutoff=cutoff).fit(table, target), dict(params), join)
    return post, OutOfFold(terms, cutoff, float((scores >= cutoff).sum()) / len(files))
