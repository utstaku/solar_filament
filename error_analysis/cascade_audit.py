# kaggle: gpu=true accelerator=NvidiaTeslaT4 kernel_sources={user}/filament-boundary-cascade
# %% [markdown]
# # Cascade FN attribution from saved models, no retraining
# Reproduce confidence .28, then diagnose .05 as a separate recall counterfactual.
# GT crop coverage is only a geometric upper bound, not a learned detector recall metric.

# %%
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.3.200", "segmentation-models-pytorch==0.5.0"],
    check=True,
)

# %%
import json

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO, settings

from boundary_cascade.common import data, save, score
from boundary_cascade.crops import crop_input, exclusive_labels
from cnn_baseline.training import pick_device
from dataset.paths import find_input, output_dir
from error_analysis.cascade_stages import best_box_iou, diagnose
from models.crop_unet import crop_unet
from preprocess.images import read_gray
from pretrained_unet.postprocess import select
from scoring.metric import instance_crops

settings.update({"wandb": False, "mlflow": False, "comet": False, "tensorboard": False})
device = pick_device()
source = find_input("yolo_crop/config.json").parent
config = json.loads((source / "config.json").read_text())
directory, annotations, _, expected = data()
files = config["val_files"]
assert files == expected
truth = {f: [instance_crops(rs) for rs in annotations.rles_for_file(f)] for f in files}
refiner = crop_unet(pretrained=False).to(device).eval()
refiner.load_state_dict(torch.load(source / "model.pt", map_location=device, weights_only=True))
detector = YOLO(str(source / "detector/weights/last.pt"))
old_components = pd.read_csv(source / "components.csv")
old_report = json.loads((source / "report.json").read_text())
results, rows = [], []

for confidence in (0.28, 0.05):
    predictions = {}
    different_pixels = 0
    baseline = np.load(source / "val_labels.npz") if confidence == 0.28 else None
    with torch.inference_mode():
        for frame_index, name in enumerate(files):
            image = read_gray(directory / name)
            result = detector.predict(
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                imgsz=1280,
                conf=confidence,
                iou=0.5,
                max_det=40,
                device=0,
                verbose=False,
            )[0]
            candidates = []
            if result.boxes is not None:
                for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    x0, y0, x1, y1 = box
                    inputs, (left, top, side) = crop_input(image, [x0, y0, x1 - x0, y1 - y0])
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        logits = refiner(torch.from_numpy(inputs).unsqueeze(0).to(device))
                    prob = logits.float().sigmoid()[0, 0].cpu().numpy()
                    prob = np.rint(cv2.resize(prob, (side, side)) * 255).astype(np.uint8)
                    candidates.append((float(conf), left, top, prob))
            final, _ = exclusive_labels(candidates, image.shape)
            predictions[name] = final
            filtered = None
            if baseline is not None:
                identical = np.array_equal(final[0], baseline[name])
                different_pixels += int(np.count_nonzero(final[0] != baseline[name]))
                # OOF decisions belong to old component IDs; reuse only with exact label identity.
                if identical:
                    frame_table = old_components[old_components.frame == name].sort_values("component")
                    assert len(frame_table) == len(final[1])
                    filtered = select(*final, frame_table.oof_score.to_numpy() >= 0.45)
            for annotator, gts in zip(annotations.file_to_entries[name], truth[name]):
                for index, gt in enumerate(gts):
                    rows.append(
                        {
                            "confidence": confidence,
                            "frame": name,
                            "annotator": annotator,
                            "gt_index": index,
                            "best_detector_box_iou": best_box_iou(
                                gt, result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                            ),
                            **diagnose(gt, candidates, final, filtered),
                        }
                    )
            if (frame_index + 1) % 10 == 0:
                print(confidence, frame_index + 1, len(files), flush=True)
    if baseline is not None:
        baseline.close()
    table = pd.DataFrame([r for r in rows if r["confidence"] == confidence])
    metrics = score(predictions, truth)
    assert int((table.stage != "matched").sum()) == metrics["fn"]
    summary = {
        "confidence": confidence,
        "metrics": metrics,
        "gt_stages": table.stage.value_counts().to_dict(),
        "saved_map_different_pixels": different_pixels if confidence == 0.28 else None,
    }
    if "lost_to_component_filter" in table:
        summary["lost_to_component_filter"] = int(table.lost_to_component_filter.fillna(False).sum())
    results.append(summary)
    pd.DataFrame(rows).to_csv(output_dir() / "cascade_gt_stages.csv", index=False)
    save(
        output_dir() / "cascade_audit_report.json",
        {
            "reference": old_report,
            "results": results,
            "interpretation": "Mutually exclusive operational FN categories, not causal proof. "
            "Coverage means GT-mask fraction inside a candidate crop. Low-confidence run is a separate "
            "pipeline counterfactual before component filtering; never reuse old filter IDs for it.",
        },
    )
    print(summary, flush=True)
    del predictions
