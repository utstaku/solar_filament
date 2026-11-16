# kaggle: gpu=false kernel_sources={user}/filament-pretrained-unet
# %% [markdown]
# # Re-evaluate saved validation maps with the baseline's chosen postprocessing
# high=0.5, low=0.2, area=150, component filter cutoff=0.45.
# Component classifier is refitted per model with physical-frame GroupKFold OOF predictions.
# No neural-network training, test prediction, or submission.

# %%
import json

import numpy as np
import pandas as pd

from cnn_baseline.data import split_files
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from postprocess.filtering import component_table, frame_context, matched_by_any_annotator, out_of_fold_scores
from postprocess.instances import instances, select
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops

root = find_data_root()
image_dir = root / "train" / "train_images"
annotations = load_annotations(next((root / "train").glob("*.json")))
source = find_input("experiment_config.json").parent
config = json.loads((source / "experiment_config.json").read_text())
files = config["val_files"]
_, expected = split_files([p.name for p in image_dir.iterdir()], 0.15, 0)
assert files == expected, "Validation split differs from original baseline"
truth = {f: [instance_crops(rs) for rs in annotations.rles_for_file(f)] for f in files}
images = {f: read_gray(image_dir / f) for f in files}
params = {"threshold": 0.5, "low": 0.2, "min_area": 150}
cutoff = 0.45
results = []


def score(predictions):
    metric = PanopticQuality()
    for f in files:
        labels, areas, _ = predictions[f]
        for gt in truth[f]:
            metric.update(gt, labels, areas)
    return {**metric.summary(), "pq_exact": metric.value}


for name in ("efficientnet_b3", "convnext_tiny", "efficientnet_b3_skeleton"):
    print("Processing", name, flush=True)
    with np.load(source / name / "val_probs.npz") as archive:
        assert set(archive.files) == set(files)
        probs = {f: archive[f] for f in files}
    raw = {f: instances(probs[f], **params) for f in files}
    table = pd.concat(
        [component_table(frame_context(f, images[f], probs[f]), raw[f][0], raw[f][2]) for f in files], ignore_index=True
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], raw[f][0], raw[f][1]) for f in files])
    oof = out_of_fold_scores(table, target)
    keep = oof >= cutoff
    filtered = {f: select(*raw[f], keep[(table["frame"] == f).to_numpy()]) for f in files}
    row = {
        "name": name,
        "params": params,
        "filter_cutoff": cutoff,
        "hysteresis_only": score(raw),
        "with_oof_filter": score(filtered),
        "components_before": len(keep),
        "components_after": int(keep.sum()),
    }
    results.append(row)
    table["target"] = target
    table["oof_score"] = oof
    table.to_csv(output_dir() / f"{name}_components.csv", index=False)
    (output_dir() / "reprocess_report.json").write_text(
        json.dumps(
            {
                "baseline_reference_pq": 0.3767,
                "val_files": files,
                "results": results,
                "caveat": "Saved maps are disk-masked; old baseline predicted full frames without this mask. "
                "OOF classifier predictions use held-out frames, but old cutoff was selected on this validation set.",
            },
            indent=2,
        )
    )
    print(json.dumps(row), flush=True)
    del probs, raw, filtered
