import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from cnn_baseline.variants import expected_gain, grow, join, out_of_fold_gain, score_frames, strict_keep
from postprocess.instances import instances, label_stats
from scoring.metric import PanopticQuality, instance_crops


def crops(*masks):
    return instance_crops([mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks])


def test_label_stats_matches_connected_components():
    prob = np.zeros((64, 64), np.uint8)
    prob[5:15, 10:40] = 200
    prob[30:34, 2:6] = 200
    labels, areas, boxes = instances(prob, 0.5, 1)
    got_areas, got_boxes = label_stats(labels, len(areas))
    assert got_areas.tolist() == areas.tolist() and got_boxes.tolist() == boxes.tolist()
    assert label_stats(np.zeros((8, 8), np.int32), 0)[1].shape == (0, 4)


def test_grow_follows_the_faint_tail_without_merging():
    prob = np.zeros((40, 100), np.uint8)
    prob[10:20, 10:30] = 250  # core of instance 1
    prob[10:20, 30:60] = 60  # faint tail (0.24)
    prob[10:20, 62:80] = 250  # instance 2, separated by a 2 px gap of prob 0
    labels, _, _ = instances(prob, 0.5, 10)
    grown, areas, _ = grow(labels, prob, low=0.2, steps=50)
    assert grown.max() == 2 and areas.tolist() == [200 + 300, 180]  # labels keep their numbers
    assert (grown[10:20, 30:60] == grown[15, 15]).all()  # the tail joined the core it touches
    assert (grown[10:20, 60:62] == 0).all()  # nothing grows where prob < low
    _, limited_areas, _ = grow(labels, prob, low=0.2, steps=5)
    assert limited_areas.sum() == 200 + 180 + 10 * 5  # five 1 px rings along the tail's 10 px front

    touching = np.zeros((20, 60), np.uint8)
    touching[5:15, 5:20] = 250
    touching[5:15, 40:55] = 250
    touching[5:15, 20:40] = 80  # a faint bridge
    labels, _, _ = instances(touching, 0.5, 10)
    grown, areas, _ = grow(labels, touching, low=0.2, steps=50)
    assert grown.max() == 2 and areas.sum() == 10 * 50  # the bridge is shared out, the instances stay two


def test_join_merges_only_close_instances_and_renumbers_by_size():
    prob = np.zeros((60, 120), np.uint8)
    prob[10:14, 10:40] = 250  # 120 px
    prob[10:14, 45:60] = 250  # 60 px, 5 px gap to the first
    prob[40:44, 80:90] = 250  # 40 px, far away
    labels, areas, _ = instances(prob, 0.5, 1)
    joined, joined_areas, boxes = join(labels, radius=6)
    assert joined_areas.tolist() == [180, 40]
    assert tuple(boxes[0]) == (10, 10, 50, 4)  # the merged instance spans both pieces
    assert (joined[10:14, 40:45] == 0).all()  # the gap itself is not filled
    assert join(labels, radius=3)[1].tolist() == sorted(areas.tolist(), reverse=True)  # gap too wide: nothing merges
    empty = np.zeros((10, 10), np.int32)
    assert join(empty, 5)[1].size == 0


def test_expected_gain_is_the_mean_matched_iou_per_annotator():
    shape = (64, 64)
    filament = np.zeros(shape, bool)
    filament[10:20, 10:50] = True
    prob = np.zeros(shape, np.uint8)
    prob[10:20, 10:42] = 250  # IoU 0.8 with the filament
    prob[40:50, 40:50] = 250  # matches nobody
    labels, areas, _ = instances(prob, 0.5, 1)
    other = np.zeros(shape, bool)
    other[55:60, 0:20] = True
    gain = expected_gain([crops(filament), crops(other)], labels, areas)  # only the first annotator drew it
    assert gain[0] == pytest.approx(0.8 / 2) and gain[1] == 0
    assert expected_gain([[]], labels, areas).tolist() == [0, 0]


@pytest.mark.parametrize("iou_sum_elsewhere", [5.0, 7.0, 9.0])  # PQ elsewhere 0.33, 0.47, 0.60
def test_keeping_a_component_raises_pq_exactly_when_its_gain_beats_half_the_pq(iou_sum_elsewhere):
    shape = (64, 64)
    filament = np.zeros(shape, bool)
    filament[10:20, 10:50] = True
    prob = np.zeros(shape, np.uint8)
    prob[10:20, 10:42] = 250
    labels, areas, _ = instances(prob, 0.5, 1)
    truth = [crops(filament), crops(), crops()]  # three annotators, one drew it: gain 0.8 / 3
    without, with_it = PanopticQuality(), PanopticQuality()
    for pq in (without, with_it):
        pq.tp, pq.fp, pq.fn, pq.iou_sum = 10, 4, 6, iou_sum_elsewhere  # the rest of the validation set
    for gt in truth:
        without.update(gt, np.zeros_like(labels), np.zeros(0))
        with_it.update(gt, labels, areas)
    gain = expected_gain(truth, labels, areas)[0]  # 0.8 / 3 = 0.27: keep below PQ 0.53, drop above
    assert (with_it.value > without.value) == (gain > without.value / 2)


def test_out_of_fold_gain_learns_from_other_frames():
    rng = np.random.default_rng(0)
    n = 400
    table = pd.DataFrame(
        {f: rng.normal(size=n) for f in ["log_area", "elongation", "darkness", "limb", "mean_prob", "max_prob"]}
    )
    table["frame"] = np.repeat([f"f{i}" for i in range(20)], 20)
    gain = np.clip(0.3 * table["mean_prob"].to_numpy(), 0, None)
    pred = out_of_fold_gain(table, gain)
    assert np.corrcoef(pred, gain)[0, 1] > 0.9


def test_score_frames_matches_the_metric():
    shape = (64, 64)
    filament = np.zeros(shape, bool)
    filament[10:20, 10:50] = True
    prob = np.zeros(shape, np.uint8)
    prob[10:20, 10:42] = 250
    prob[40:50, 40:50] = 250
    predictions = {"a": instances(prob, 0.5, 1)}
    terms = score_frames(predictions, {"a": [crops(filament)]}, ["a"])
    assert terms[["tp", "fp", "fn"]].iloc[0].tolist() == [1, 1, 0]
    assert terms["iou_sum"].iloc[0] == pytest.approx(0.8)


def test_strict_keep_needs_a_peak_a_high_mean_and_an_area():
    prob = np.zeros((64, 128), np.uint8)
    prob[5:15, 5:45] = 250  # peak 0.98 < 0.99: dropped
    prob[20:30, 5:45] = 240
    prob[24, 20] = 255  # peak 1.0, mean ~0.94: kept
    prob[40:50, 5:45] = 120
    prob[44, 20] = 255  # peak 1.0 but mean ~0.47: dropped
    prob[55:60, 100:110] = 255  # perfect but 50 px: dropped
    labels, areas, _ = strict_keep(prob)
    assert areas.tolist() == [400] and (labels[20:30, 5:45] == 1).all()
