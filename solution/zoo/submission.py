"""A submission from one zoo experiment without more training: one fold model's test maps, post-processed.

The post-processing (`postprocess.pipeline`: join, then the component filter) is fitted on the experiment's
707 out-of-fold maps and applied to the test maps of fold `fold`'s model. Both are single-model maps made
the same way (512 px tiles, Hann blending, no TTA), so the filter sees on test what it learnt from; averaging
the five fold models would give smoother maps on which the filter keeps far fewer filaments.

The model has seen four fifths of the training frames, so this is a check of whether the out-of-fold
ranking holds on the leaderboard, not the strongest submission of the experiment.

Outputs: `submission.csv`, `<name>_submission_report.json` (OOF PQ, cutoff, filaments kept per validation
frame and per test frame of each fold's maps: they should be close).
"""

import json
import time

import numpy as np

from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess import pipeline
from postprocess.filtering import frame_context
from postprocess.tuning import pq_of
from preprocess.images import read_gray
from scoring import submission
from scoring.metric import instance_crops
from zoo.experiment import N_FOLDS


def load_maps(pattern: str) -> dict[str, np.ndarray]:
    maps = {}
    for k in range(N_FOLDS):
        saved = np.load(find_input(pattern.format(k=k)))
        maps |= {f: saved[f] for f in saved.files}
    return maps


def write_submission(name: str, fold: int = 0) -> dict:
    t0 = time.time()

    def log(*args):
        print(f"[{time.time() - t0:5.0f}s]", *args, flush=True)

    root = find_data_root()
    train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
    annotations = load_annotations(next((root / "train").glob("*.json")))
    oof = load_maps(f"{name}_oof_fold{{k}}.npz")
    files = sorted(oof)
    truth = {f: [instance_crops(rles) for rles in annotations.rles_for_file(f)] for f in files}
    contexts = {f: frame_context(f, read_gray(train_dir / f), oof[f]) for f in files}
    post, scores = pipeline.fit(contexts, truth, files)
    log(f"{name}: OOF PQ {pq_of(scores.terms):.4f}, cutoff {scores.cutoff}, {len(files)} frames")
    del contexts, oof

    test_files = sorted(p.name for p in test_dir.iterdir())
    images = {f: read_gray(test_dir / f) for f in test_files}
    kept, rows = {}, []
    for k in range(N_FOLDS):
        saved = np.load(find_input(f"{name}_test_fold{k}.npz"))
        assert set(saved.files) == set(test_files), f"fold {k}'s test maps do not cover every test frame"
        count = 0
        for f in test_files:
            labels, _, boxes = post(frame_context(f, images[f], saved[f]))
            count += len(boxes)
            if k == fold:
                rows += submission.rows_from_label_map(f.rsplit(".", 1)[0], labels, boxes)
        kept[k] = round(count / len(test_files), 2)
        log(f"fold {k}'s test maps: {kept[k]} filaments kept per frame")
    submission.write(rows, output_dir() / "submission.csv")

    report = {
        "name": name,
        "fold": fold,
        "oof_pq": round(pq_of(scores.terms), 4),
        "cutoff": scores.cutoff,
        "kept_per_val_frame": round(scores.kept_per_frame, 2),
        "kept_per_test_frame": kept,
        "rows": len(rows),
    }
    (output_dir() / f"{name}_submission_report.json").write_text(json.dumps(report, indent=2))
    log(report)
    return report
