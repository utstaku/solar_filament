from unet_cv.experiment import folds


def test_outer_and_calibration_exclusion():
    files = [f"{i:04d}.jpeg" for i in range(100)]
    splits = list(folds(files))
    assert splits == list(folds(list(reversed(files))))
    seen = []
    for split in splits:
        train, val = set(split["train"]), set(split["validation"])
        inner, calibration = set(split["inner_train"]), set(split["calibration"])
        assert train.isdisjoint(val)
        assert train | val == set(files)
        assert inner.isdisjoint(calibration)
        assert inner | calibration == train
        assert (len(train), len(val), len(inner), len(calibration)) == (80, 20, 64, 16)
        seen.extend(val)
    assert sorted(seen) == files
