"""Run two encoder baselines, then retrain the winner with skeleton recall."""

import gc
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import error_analysis.run
from cnn_baseline.data import FilamentFrames, channels, collate_tiles, split_files, worker_init
from cnn_baseline.postprocess import instances, on_disk
from cnn_baseline.training import pick_device
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from models.pretrained_unet import ENCODERS, PretrainedUNet
from preprocess.images import read_gray
from pretrained_unet.data import SkeletonFrames, collate_skeleton
from pretrained_unet.training import fit, predict_frame
from scoring.metric import PanopticQuality, instance_crops


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def evaluate(probs, truth):
    sweep = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        for min_area in (50, 100, 200, 400):
            pq = PanopticQuality()
            for name, prob in probs.items():
                labels, areas, _ = instances(prob, threshold, min_area)
                for annotation in truth[name]:
                    pq.update(annotation, labels, areas)
            sweep.append({"threshold": threshold, "min_area": min_area, **pq.summary(), "pq_exact": pq.value})
    return sweep


def run(epochs=16, quick=False, pretrained=True, workers=2):
    device = pick_device()
    print(f"device={device}, torch={torch.__version__}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(), flush=True)
    root, output = find_data_root(), output_dir()
    output.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(next((root / "train").glob("*.json")))
    image_dir = root / "train" / "train_images"
    files = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if quick:
        files, epochs = files[:12], 1
    train_files, val_files = split_files(files, 0.15, 0)
    config = {
        "seed": 0,
        "epochs": epochs,
        "tile": 512,
        "tiles_per_frame": 4,
        "frames_per_batch": 1,
        "encoder_lr": 1e-4,
        "decoder_lr": 1e-3,
        "inference_stride": 384,
        "normalization": {"mean": 0.449, "std": 0.226},
        "pretrained": pretrained,
        "train_files": train_files,
        "val_files": val_files,
        "quick": quick,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "timm", "segmentation-models-pytorch", "scikit-image", "numpy")
        },
    }
    save_json(output / "experiment_config.json", config)
    truth = {name: [instance_crops(rs) for rs in annotations.rles_for_file(name)] for name in val_files}
    results = []
    candidates = [("efficientnet_b3", 0.0), ("convnext_tiny", 0.0)]
    for index in range(3):
        if index == 2:
            winner = max(results, key=lambda r: r["best"]["pq_exact"])
            candidates.append((winner["encoder"], 0.2))
            print(f"Selected {winner['encoder']} for skeleton recall ablation", flush=True)
        encoder, skeleton_weight = candidates[index]
        name = encoder + ("_skeleton" if skeleton_weight else "")
        dest = output / name
        dest.mkdir(exist_ok=True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.reset_peak_memory_stats()
        local_config = {
            **config,
            "encoder": encoder,
            "encoder_id": ENCODERS[encoder],
            "skeleton_weight": skeleton_weight,
            "skeleton_tube_radius": 1,
        }
        save_json(dest / "config.json", local_config)
        cls = SkeletonFrames if skeleton_weight else FilamentFrames
        dataset = cls(train_files, image_dir, annotations, tile=512, tiles_per_frame=4)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_skeleton if skeleton_weight else collate_tiles,
            worker_init_fn=worker_init,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(0),
        )

        def log(*args, run_name=name):
            print(run_name, *args, flush=True)

        log("start", local_config["encoder_id"])
        started = time.time()
        model = PretrainedUNet(encoder, pretrained=pretrained)
        history = fit(model, loader, device, epochs, skeleton_weight, dest, local_config, log)
        train_seconds = time.time() - started
        started = time.time()
        probs = {}
        for j, file in enumerate(val_files):
            image = read_gray(image_dir / file)
            prob = predict_frame(model, torch.from_numpy(channels(image)), device)
            probs[file] = on_disk(prob, image)
            if (j + 1) % 10 == 0:
                log("validation", j + 1, "/", len(val_files))
        inference_seconds = time.time() - started
        # Retain validation maps on Kaggle for post-processing audits, not downloaded by the CLI.
        np.savez_compressed(dest / "val_probs.npz", **probs)
        sweep = evaluate(probs, truth)
        best = max(sweep, key=lambda r: r["pq_exact"])
        fixed = next(r for r in sweep if r["threshold"] == 0.5 and r["min_area"] == 100)
        params = {k: best[k] for k in ("threshold", "min_area")}
        report = {
            "name": name,
            "encoder": encoder,
            "skeleton_weight": skeleton_weight,
            "best": best,
            "fixed": fixed,
            "sweep": sweep,
            "history": history,
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        }
        save_json(dest / "report.json", report)
        analysis = error_analysis.run.run(
            annotations, image_dir, val_files, probs, instances, params, dest, galleries=False, log=log
        )
        report["error_counts"] = analysis["counts"]
        save_json(dest / "report.json", report)
        results.append(report)
        save_json(output / "comparison.json", results)
        log("completed", best)
        del model, loader, dataset, probs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(
        "COMPARISON",
        json.dumps([{k: r[k] for k in ("name", "best", "fixed", "error_counts")} for r in results]),
        flush=True,
    )
    return results
