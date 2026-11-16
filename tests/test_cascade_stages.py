import numpy as np

from boundary_cascade.crops import exclusive_labels
from error_analysis.cascade_stages import diagnose


def test_stages_and_postprocess():
    shape = (40, 40)
    gt = (0, 0, np.ones((20, 20), bool))
    empty = exclusive_labels([], shape)[0]
    assert diagnose(gt, [], empty)["stage"] == "no_proposal_crop_overlap"

    def run(p):
        cs = [(0.9, 0, 0, p)]
        return diagnose(gt, cs, exclusive_labels(cs, shape)[0])["stage"]

    assert run(np.full((20, 20), 255, np.uint8)) == "matched"
    assert run(np.zeros((20, 20), np.uint8)) == "refiner_empty"
    assert run(np.full((5, 5), 255, np.uint8)) == "insufficient_proposal_crop_coverage"
    p = np.zeros((20, 20), np.uint8)
    p[:5] = 255
    assert run(p) == "refiner_iou_at_most_half"
    tiny = (0, 0, np.ones((10, 10), bool))
    cs = [(0.9, 0, 0, np.full((10, 10), 255, np.uint8))]
    assert diagnose(tiny, cs, empty)["stage"] == "minimum_area"
    good = [(0.9, 0, 0, np.full((20, 20), 255, np.uint8))]
    assert diagnose(gt, good, empty)["stage"] == "overlap_resolution"
    final = exclusive_labels(good, shape)[0]
    assert diagnose(gt, good, final, empty)["lost_to_component_filter"]
