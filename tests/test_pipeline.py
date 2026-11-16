import cv2
import numpy as np
import pytest
from pycocotools import mask as mask_util

from error_analysis.instances import FrameContext
from postprocess import pipeline
from postprocess.tuning import pq_of
from preprocess.disk import REL_SCALE
from scoring.metric import instance_crops

SHAPE = (200, 300)


def frame(k: int):
    """(context, truth) of a frame: one filament the network split in two halves 20 px apart, which the annotator
    drew whole (each half alone has IoU 0.45), and round blobs no annotator drew."""
    rng = np.random.default_rng(k)
    flat = np.full(SHAPE, REL_SCALE, np.uint8)
    prob = np.zeros(SHAPE, np.uint8)
    truth = np.zeros(SHAPE, np.uint8)
    truth[100:106, 20:220] = 1
    flat[truth > 0] = int(REL_SCALE * 0.85)
    prob[100:106, 20:110] = prob[100:106, 130:220] = 230
    for _ in range(3):
        x, y = int(rng.integers(30, 270)), int(rng.choice([int(rng.integers(20, 70)), int(rng.integers(140, 180))]))
        cv2.circle(prob, (x, y), int(rng.integers(11, 15)), 200, -1)
    ctx = FrameContext(name=f"f{k}.jpeg", flat=flat, centre=(150.0, 100.0), radius=1000.0, prob=prob)
    return ctx, [instance_crops([mask_util.encode(np.asfortranarray(truth))])]


def test_join_then_filter_rebuilds_the_filament_and_drops_the_blobs():
    frames = {f"f{k}.jpeg": frame(k) for k in range(10)}
    contexts = {f: ctx for f, (ctx, _) in frames.items()}
    truth = {f: t for f, (_, t) in frames.items()}
    files = sorted(frames)

    with pytest.raises(ValueError, match="0 of"):  # without the join no half reaches IoU 0.5: nothing to learn
        pipeline.fit(contexts, truth, files, join=False)
    joined, joined_oof = pipeline.fit(contexts, truth, files, join=True)

    assert pq_of(joined_oof.terms) > 0.8
    assert joined_oof.kept_per_frame == 1

    labels, areas, boxes = joined(contexts["f0.jpeg"])
    assert len(areas) == 1 and areas[0] == 2 * 6 * 90  # both halves, no pixels added in the gap
    assert (boxes[0] == [20, 100, 200, 6]).all() and (labels > 0).sum() == areas[0]
