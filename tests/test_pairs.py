import numpy as np
import pytest
from pycocotools import mask as mask_util

from cnn_baseline.pairs import (
    aligned_end_to_end,
    component_axes,
    fibril_direction,
    join_pairs,
    line_angle,
    pair_table,
    pairs,
    path_values,
    same_filament,
)
from error_analysis.instances import FrameContext
from preprocess.disk import REL_SCALE
from scoring.metric import instance_crops

SHAPE = (200, 300)


def context(labels: np.ndarray, flat=None, prob=None) -> FrameContext:
    flat = np.full(SHAPE, REL_SCALE, np.uint8) if flat is None else flat
    prob = (labels > 0).astype(np.uint8) * 230 if prob is None else prob
    return FrameContext(name="f.jpeg", flat=flat, centre=(150.0, 100.0), radius=1000.0, prob=prob)


def spine_and_barb() -> np.ndarray:
    """Label 1: a horizontal spine; 2: its continuation 20 px further along the axis; 3: a short vertical
    piece 6 px below the middle of the spine (a barb)."""
    labels = np.zeros(SHAPE, np.int32)
    labels[100:106, 20:140] = 1  # spine, x 20..139
    labels[100:106, 160:220] = 2  # continuation, gap x 140..159
    labels[112:140, 78:82] = 3  # barb below the spine's middle, gap y 106..111
    return labels


def test_component_axes():
    axes = component_axes(spine_and_barb(), 3)
    assert axes["angle"].iloc[0] == pytest.approx(0, abs=1e-6) or axes["angle"].iloc[0] == pytest.approx(np.pi)
    assert line_angle(axes["angle"].iloc[2], np.pi / 2) < 1e-6  # the barb is vertical
    assert axes["hi"].iloc[0] - axes["lo"].iloc[0] == pytest.approx(119)
    assert axes["elongation"].iloc[0] > 10


def test_line_angle_folds_to_a_quarter_turn():
    assert line_angle(0.1, np.pi - 0.1) == pytest.approx(0.2)
    assert line_angle(0, np.pi / 2) == pytest.approx(np.pi / 2)


def test_pairs_finds_nearest_pixels_within_the_radius():
    near = pairs(spine_and_barb(), radius=25).set_index(["a", "b"])
    assert set(near.index) == {(1, 2), (1, 3)}  # 2 and 3 are > 25 px apart
    assert near.loc[(1, 2), "dist"] == pytest.approx(21)  # x 139 -> x 160
    assert (near.loc[(1, 2), "ax"], near.loc[(1, 2), "bx"]) == (139, 160)
    assert near.loc[(1, 3), "dist"] == pytest.approx(7)  # y 105 -> y 112
    assert pairs(spine_and_barb(), radius=5).empty


def test_pair_table_tells_a_continuation_from_a_barb():
    labels = spine_and_barb()
    table = pair_table(context(labels), labels, radius=25).set_index(["a", "b"])
    cont, barb = table.loc[(1, 2)], table.loc[(1, 3)]
    assert cont["gap_large"] < 0.05 and cont["end_large"] < 0.02 and cont["angle_diff"] < 0.05
    assert barb["angle_diff"] > 1.5 and barb["end_large"] > 0.4 and barb["gap_large"] > 1.5
    assert aligned_end_to_end(table.reset_index()).tolist() == [True, False]
    assert cont["log_area_large"] == pytest.approx(np.log(720)) and cont["log_area_small"] == pytest.approx(np.log(360))


def test_path_darkness_and_confidence_of_the_gap():
    labels = spine_and_barb()
    flat = np.full(SHAPE, REL_SCALE, np.uint8)
    flat[100:106, 140:160] = int(REL_SCALE * 0.9)  # the gap of the continuation is 10 % darker
    prob = (labels > 0).astype(np.uint8) * 230
    prob[100:106, 140:160] = 51  # 0.2
    table = pair_table(context(labels, flat, prob), labels, radius=25).set_index(["a", "b"])
    assert table.loc[(1, 2), "path_dark"] == pytest.approx(0.1, abs=0.01)
    assert table.loc[(1, 2), "path_prob_mean"] == pytest.approx(0.2, abs=0.01)
    assert table.loc[(1, 3), "path_dark"] == pytest.approx(0, abs=0.01)
    assert path_values(flat, (139, 102), (160, 102)).size == 20


def test_fibril_direction_follows_the_stripes():
    yy, xx = np.mgrid[0:100, 0:100]
    horizontal = (128 + 40 * np.sin(yy / 2.0)).astype(np.uint8)  # stripes along x
    angle, coherence = fibril_direction(horizontal, 50, 50, np.zeros((100, 100), bool))
    assert line_angle(angle, 0) < 0.05 and coherence > 0.9
    diagonal = (128 + 40 * np.sin((xx - yy) / 2.0)).astype(np.uint8)  # stripes along y = x
    angle, _ = fibril_direction(diagonal, 50, 50, np.zeros((100, 100), bool))
    assert line_angle(angle, np.pi / 4) < 0.05
    assert fibril_direction(np.full((100, 100), 7, np.uint8), 50, 50, np.zeros((100, 100), bool)) == (0.0, 0.0)


def test_same_filament_labels():
    labels = spine_and_barb()
    table = pair_table(context(labels), labels, radius=25)
    whole = np.zeros(SHAPE, np.uint8)
    whole[98:142, 18:222] = 1  # one annotator drew all three pieces as one filament
    split = np.zeros(SHAPE, np.uint8)
    split[98:108, 18:150] = 1
    other = np.zeros(SHAPE, np.uint8)
    other[98:108, 155:222] = 1  # another drew the continuation as a separate filament
    areas = np.bincount(labels.ravel())[1:]

    def crops(*masks):
        return instance_crops([mask_util.encode(np.asfortranarray(m)) for m in masks])

    same, different = same_filament([crops(split, other)], labels, areas, table)
    assert same.tolist() == [False, False] and different.tolist() == [True, False]  # barb drawn by nobody here
    same, different = same_filament([crops(split, other), crops(whole)], labels, areas, table)
    assert same.tolist() == [True, True] and not different.any()  # one annotator joining them is enough


def test_join_pairs_merges_only_the_chosen_pairs():
    labels = spine_and_barb()
    table = pair_table(context(labels), labels, radius=25)
    joined, areas, _ = join_pairs(labels, table, np.array([True, False]))
    assert areas.tolist() == [720 + 360, 112]
    assert joined.max() == 2 and (joined[100:106, 140:160] == 0).all()
