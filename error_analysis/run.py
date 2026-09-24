"""The whole analysis of one model's validation predictions, as one call.

Every training notebook ends with it, so each run yields its error tables, report and galleries next
to submission.csv; `cnn_errors.py` calls the same function on saved weights.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.annotations import Annotations
from error_analysis.gallery import save_galleries, worst_cases
from error_analysis.instances import FrameContext, analyze_frame, tables
from error_analysis.report import breakdowns, summary, what_if
from preprocess.disk import disk_geometry, flatten
from preprocess.images import read_gray
from scoring.metric import instance_crops

Instances = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
"""(prob, **params) -> (label map, areas, boxes), e.g. cnn_baseline.postprocess.instances."""


def run(
    annotations: Annotations,
    image_dir: Path,
    files: list[str],
    probs: dict[str, np.ndarray],
    instances: Instances,
    params: dict,
    out_dir: Path,
    galleries: bool = True,
    log=print,
) -> dict:
    """Classify every filament and prediction of `files`, write the tables, report, breakdowns and
    galleries into `out_dir`, and return the summary.

    `probs` holds the model's uint8 probability map per frame; `instances(prob, **params)` turns one
    into predictions the way the submission does.
    """
    out_dir = Path(out_dir)
    images, gt_rows, pred_rows = {}, [], []
    for f in files:
        images[f] = image = read_gray(Path(image_dir) / f)
        cx, cy, radius = disk_geometry(image)
        ctx = FrameContext(name=f, flat=flatten(image), centre=(cx, cy), radius=radius, prob=probs[f])
        labels, areas, boxes = instances(probs[f], **params)
        entries = annotations.file_to_entries.get(f, [])
        g, p = analyze_frame(ctx, [(e, annotations.entry_to_rles.get(e, [])) for e in entries], labels, areas, boxes)
        gt_rows += g
        pred_rows += p
    gt, pred = tables(gt_rows, pred_rows)
    gt.to_csv(out_dir / "errors_gt.csv", index=False)
    pred.to_csv(out_dir / "errors_pred.csv", index=False)
    log(f"error analysis: {len(gt)} ground-truth rows, {len(pred)} prediction rows")

    report = summary(gt, pred)
    report["post_processing"] = params
    (out_dir / "error_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "what_if"}, indent=1))
    print("\n== what if ==")
    print(what_if(gt, pred).to_string(index=False))

    pd.set_option("display.width", 160)
    lines = []
    for name, table in breakdowns(gt, pred).items():
        lines += [f"== {name} ==", table.to_string(), ""]
    (out_dir / "breakdowns.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    if galleries:

        def context(frame, annotator):
            labels, _, _ = instances(probs[frame], **params)
            return images[frame], instance_crops(annotations.entry_to_rles.get(annotator, [])), labels

        paths = save_galleries(worst_cases(gt, pred, k=8), context, out_dir)
        log("galleries:", [p.name for p in paths])
    return report
