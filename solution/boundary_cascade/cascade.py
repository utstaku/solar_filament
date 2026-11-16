"""Standard YOLO11s detector followed by an individual-mask crop U-Net."""

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader

from boundary_cascade.common import data, evaluate, save, seed, train_model
from boundary_cascade.crops import Crops, crop_input, exclusive_labels
from cnn_baseline.data import worker_init
from cnn_baseline.training import pick_device
from dataset.paths import output_dir
from models.crop_unet import crop_unet
from preprocess.images import read_gray


def detector_dataset(directory, annotations, train, val, dest):
    """Use one deterministic annotation set per physical image for detector boxes.

    Avoid duplicate overlapping boxes from independent annotation sets. The refiner
    still samples all annotation sets as separate individual-mask training records.
    """
    for split, files in (("train", train), ("val", val)):
        imgs, labs = dest / "images" / split, dest / "labels" / split
        imgs.mkdir(parents=True, exist_ok=True)
        labs.mkdir(parents=True, exist_ok=True)
        for name in files:
            link = imgs / name
            if not link.exists():
                link.symlink_to((directory / name).resolve())
            entry = min(annotations.file_to_entries[name])
            lines = []
            for rle in annotations.entry_to_rles.get(entry, []):
                x, y, w, h = mask_util.toBbox(rle)
                ih, iw = rle["size"]
                lines.append(f"0 {(x + w / 2) / iw} {(y + h / 2) / ih} {w / iw} {h / ih}")
            (labs / (Path(name).stem + ".txt")).write_text("\n".join(lines))
    path = dest / "data.yaml"
    path.write_text(
        f"path: {json.dumps(str(dest.resolve()))}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n"
    )
    return path


def run():
    from ultralytics import YOLO, settings

    settings.update({"wandb": False, "mlflow": False, "comet": False, "tensorboard": False})
    device = pick_device()
    seed()
    directory, annotations, train, val = data()
    dest = output_dir() / "yolo_crop"
    dest.mkdir(parents=True, exist_ok=True)
    save(
        dest / "config.json",
        {
            "train_files": train,
            "val_files": val,
            "seed": 0,
            "detector": "yolo11s.pt",
            "detector_epochs": 40,
            "detector_imgsz": 1280,
            "detector_batch": 4,
            "detector_targets": "first sorted annotation entry per physical image",
            "refiner": "ImageNet ResNet18 U-Net",
            "crop_size": 384,
            "refiner_epochs": 12,
            "detector_conf": 0.28,
            "nms_iou": 0.5,
            "mask_threshold": 128 / 255,
            "min_area": 150,
            "note": "last detector epoch, fixed thresholds; no external filament weights or novelty additions",
        },
    )
    yaml = detector_dataset(directory, annotations, train, val, Path("/tmp/filament-yolo-data"))
    detector = YOLO("yolo11s.pt")
    detector.train(
        data=str(yaml),
        epochs=40,
        imgsz=1280,
        batch=4,
        device=0,
        workers=2,
        project=str(dest),
        name="detector",
        exist_ok=True,
        seed=0,
        deterministic=True,
        patience=0,
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        cache=False,
        plots=False,
        save=True,
    )
    weights = Path(detector.trainer.save_dir) / "weights" / "last.pt"
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    seed()
    refiner = crop_unet()
    dataset = Crops(directory, annotations, train)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(0),
    )
    train_model(refiner, loader, device, 12, dest, lr=3e-4)
    refiner.eval()
    detector = YOLO(str(weights))
    predictions, probs = {}, {}
    with torch.inference_mode():
        for index, name in enumerate(val):
            image = read_gray(directory / name)
            result = detector.predict(
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                imgsz=1280,
                conf=0.28,
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
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = refiner(torch.from_numpy(inputs).unsqueeze(0).to(device))
                    p = logits.float().sigmoid()[0, 0].cpu().numpy()
                    p = np.rint(cv2.resize(p, (side, side)) * 255).astype(np.uint8)
                    candidates.append((float(conf), left, top, p))
            predictions[name], probs[name] = exclusive_labels(candidates, image.shape)
            if (index + 1) % 10 == 0:
                print("Cascade validation", index + 1, len(val), flush=True)
    np.savez_compressed(dest / "val_labels.npz", **{f: p[0] for f, p in predictions.items()})
    np.savez_compressed(dest / "val_probs.npz", **probs)
    report = evaluate("yolo_crop", predictions, probs, directory, annotations, val, dest)
    save(output_dir() / "cascade_report.json", report)
    return report
