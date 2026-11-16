import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util
from sklearn.metrics import roc_auc_score

from cnn_baseline.importance import drop_column, fit_r, paired_bootstrap, permutation_importance, public_pq
from postprocess.filtering import out_of_fold_scores
from postprocess.instances import instances
from postprocess.tuning import best_cutoff, frame_terms, pq_of
from scoring.metric import PanopticQuality, instance_crops


def crops(*masks):
    return instance_crops([mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks])


@pytest.fixture()
def two_frames():
    """Frame a: a filament found well plus a hallucination; frame b: two annotators, one filament found."""
    prob_a = np.zeros((64, 64), np.uint8)
    prob_a[10:20, 5:45] = 230  # matches the filament of frame a
    prob_a[40:50, 40:60] = 230  # nobody drew this
    prob_b = np.zeros((64, 64), np.uint8)
    prob_b[30:36, 10:50] = 230
    gt_a = np.zeros((64, 64), bool)
    gt_a[10:20, 5:40] = True
    gt_b, other = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    gt_b[30:36, 12:50] = True
    other[50:60, 0:20] = True  # drawn by the second annotator only, never predicted
    raw = {"a": instances(prob_a, 0.5, 10), "b": instances(prob_b, 0.5, 10)}
    truth = {"a": [crops(gt_a)], "b": [crops(gt_b), crops(gt_b, other)]}
    # one row per component, frame after frame, in label order (largest first: a has 400 then 200 px)
    table = pd.DataFrame({"frame": ["a", "a", "b"], "component": [1, 2, 1]})
    return raw, truth, table


def test_frame_terms_add_up_to_the_metric(two_frames):
    raw, truth, table = two_frames
    terms = frame_terms(raw, table, np.ones(3, bool), truth, ["a", "b"])
    assert terms["frame"].tolist() == ["a", "b"]
    assert terms[["tp", "fp", "fn"]].to_numpy().tolist() == [[1, 1, 0], [2, 0, 1]]
    pq = PanopticQuality()
    for f in ["a", "b"]:
        labels, areas, _ = raw[f]
        for gt in truth[f]:
            pq.update(gt, labels, areas)
    assert pq_of(terms) == pytest.approx(pq.value)

    dropped = frame_terms(raw, table, np.array([True, False, True]), truth, ["a", "b"])
    assert dropped[["tp", "fp", "fn"]].to_numpy().tolist() == [[1, 0, 0], [2, 0, 1]]
    assert pq_of(dropped) > pq_of(terms)
    assert pq_of(frame_terms(raw, table, np.zeros(3, bool), truth, ["a", "b"])) == 0.0


def test_pq_of_pools_the_terms_before_dividing():
    terms = pd.DataFrame({"frame": ["x", "y"], "tp": [1, 0], "fp": [0, 9], "fn": [0, 0], "iou_sum": [1.0, 0.0]})
    assert pq_of(terms) == pytest.approx(1 / (1 + 4.5))  # not the mean of per-frame PQs, (1 + 0) / 2


def test_best_cutoff_drops_the_hallucination(two_frames):
    raw, truth, table = two_frames
    cutoff, terms = best_cutoff(np.array([0.9, 0.3, 0.8]), raw, table, truth, ["a", "b"])
    assert cutoff == 0.35  # the lowest cutoff above the hallucination's score
    assert terms["fp"].sum() == 0


# ---------------------------------------------------------------------------------------------------
# The paired bootstrap
# ---------------------------------------------------------------------------------------------------


@pytest.fixture()
def heterogeneous_frames():
    """40 frames of very different size and difficulty; b turns one missed filament per frame into a match."""
    rng = np.random.default_rng(1)
    tp = rng.integers(1, 200, 40)
    a = pd.DataFrame(
        {
            "frame": [f"f{i:02d}" for i in range(40)],
            "tp": tp,
            "fp": (tp * rng.uniform(0.1, 3, 40)).astype(int),
            "fn": (tp * rng.uniform(0.1, 2, 40)).astype(int) + 1,
            "iou_sum": tp * rng.uniform(0.5, 0.9, 40),
        }
    )
    b = a.assign(tp=a["tp"] + 1, fn=a["fn"] - 1, iou_sum=a["iou_sum"] + 0.9)
    return a, b


def test_bootstrap_of_identical_variants_is_exactly_zero(heterogeneous_frames):
    a, _ = heterogeneous_frames
    assert paired_bootstrap(a, a.copy()) == (0.0, 0.0, 0.0)


def test_bootstrap_estimate_is_the_pooled_difference(heterogeneous_frames):
    a, b = heterogeneous_frames
    delta, lo, hi = paired_bootstrap(a, b)
    assert delta == pytest.approx(pq_of(b) - pq_of(a))
    assert lo < delta < hi


def test_bootstrap_resamples_the_same_frames_for_both_variants(heterogeneous_frames):
    """b is better on every frame, so every paired resample favours b. Drawing frames for a and b
    independently mixes the large frame-to-frame spread into the difference and the interval crosses 0."""
    a, b = heterogeneous_frames
    _, lo, _ = paired_bootstrap(a, b)
    assert lo > 0


def test_bootstrap_is_reproducible_and_checks_the_frames(heterogeneous_frames):
    a, b = heterogeneous_frames
    assert paired_bootstrap(a, b, seed=3) == paired_bootstrap(a, b, seed=3)
    with pytest.raises(ValueError):
        paired_bootstrap(a, b.iloc[::-1])
    with pytest.raises(ValueError):
        paired_bootstrap(a, b.iloc[1:])


# ---------------------------------------------------------------------------------------------------
# Permutation importance and drop-column
# ---------------------------------------------------------------------------------------------------

FEATURES = ["signal", "noise", "constant"]


@pytest.fixture(scope="module")
def components():
    """30 frames x 20 components; only `signal` says whether a component matched a filament."""
    rng = np.random.default_rng(0)
    n = 600
    target = rng.random(n) < 0.4
    table = pd.DataFrame(
        {
            "frame": np.repeat([f"f{i:02d}" for i in range(30)], 20),
            "signal": 1.5 * target + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "constant": np.zeros(n),
        }
    )
    return table, target


def auc_of(target):
    return lambda scores: roc_auc_score(target, scores)


def test_permutation_importance_finds_the_signal_not_the_noise(components):
    table, target = components
    drops = permutation_importance(table, target, "signal", auc_of(target), features=FEATURES)
    assert drops.shape == (5,)
    assert drops.mean() > 0.2
    assert abs(permutation_importance(table, target, "noise", auc_of(target), features=FEATURES).mean()) < 0.03


def test_permutation_of_an_unused_feature_changes_nothing(components):
    """Shuffling a constant leaves every score as it was -- only true if the unshuffled baseline is the
    out-of-fold scores of the very same fold models."""
    table, target = components
    drops = permutation_importance(table, target, "constant", auc_of(target), features=FEATURES)
    assert np.all(drops == 0)


def test_permutation_importance_is_reproducible_and_leaves_the_table_alone(components):
    table, target = components
    before = table.copy()

    def run(seed):
        return permutation_importance(table, target, "signal", auc_of(target), seed=seed, features=FEATURES)

    first = run(7)
    pd.testing.assert_frame_equal(table, before)
    assert np.array_equal(first, run(7)) and not np.array_equal(first, run(8))


def test_drop_column_refits_without_the_feature(components):
    table, target = components
    full = roc_auc_score(target, out_of_fold_scores(table, target, features=FEATURES))
    assert full > 0.8
    assert drop_column(table, target, "signal", auc_of(target), features=FEATURES) < 0.6
    assert drop_column(table, target, "noise", auc_of(target), features=FEATURES) == pytest.approx(full, abs=0.03)
    with pytest.raises(ValueError):
        drop_column(table, target, "limb", auc_of(target), features=FEATURES)


def test_public_pq_reduces_to_pq_and_fit_r_inverts_it():
    terms = pd.DataFrame({"frame": ["x"], "tp": [774], "fp": [564], "fn": [613], "iou_sum": [514.56]})
    assert public_pq(terms, 1.0) == pytest.approx(pq_of(terms))
    assert public_pq(terms, 0.77) < pq_of(terms)  # sparser test labels: lower score
    r = fit_r(terms, 0.33)
    assert 0.7 < r < 0.85 and public_pq(terms, r) == pytest.approx(0.33, abs=1e-6)
    assert fit_r(terms, 0.5) == 1.0  # above validation: nothing to explain
