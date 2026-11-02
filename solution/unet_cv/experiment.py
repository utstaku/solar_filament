"""Outer U-Net CV with independently calibrated component filters; fixed hyperparameters."""

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, split_files, worker_init
from cnn_baseline.training import fit, pick_device, predict
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from models.unet import UNet
from postprocess.filtering import ComponentFilter, component_table, frame_context, matched_by_any_annotator
from postprocess.instances import instances, select
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops
from scoring.submission import rows_from_label_map, write

PARAMS = {"threshold": 0.5, "low": 0.2, "min_area": 150}
CUTOFF = 0.45


def folds(files, seed=0):
    files = sorted(files)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate physical frames")
    for fold, (train_idx, val_idx) in enumerate(KFold(5, shuffle=True, random_state=seed).split(files)):
        train = [files[i] for i in train_idx]
        validation = [files[i] for i in val_idx]
        inner_train, calibration = split_files(train, 0.2, seed + 100 + fold)
        assert not set(validation) & (set(inner_train) | set(calibration))
        assert not set(inner_train) & set(calibration)
        yield {
            "fold": fold,
            "train": train,
            "validation": validation,
            "inner_train": inner_train,
            "calibration": calibration,
        }


def run():
    out = output_dir()
    root = find_data_root()
    train_dir, test_dir = root / "train/train_images", root / "test/test_images"
    annotations = load_annotations(next((root / "train").glob("*.json")))
    files = sorted(p.name for p in train_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    test_files = sorted(p.name for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    splits = list(folds(files))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    device = pick_device()
    start = time.time()

    def log(*args):
        print(f"[{time.time() - start:.0f}s]", *args, flush=True)

    def train(names, seed, path):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader = DataLoader(
            FilamentFrames(names, train_dir, annotations, tile=512, tiles_per_frame=4),
            batch_size=4,
            shuffle=True,
            num_workers=3,
            collate_fn=collate_tiles,
            worker_init_fn=worker_init,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )
        model = UNet(in_channels=2, base=32, depth=4)
        history = fit(model, loader, epochs=16, lr=1e-3, device=device, log=log)
        torch.save(model.state_dict(), path)
        path.with_suffix(".json").write_text(json.dumps({"seed": seed, "loss": history}))
        del loader
        gc.collect()
        return model

    def components(model, names, directory):
        for name in names:
            image = read_gray(directory / name)
            prob = next(predict(model, [torch.from_numpy(channels(image))], device))
            raw = instances(prob, **PARAMS)
            table = component_table(frame_context(name, image, prob), raw[0], raw[2])
            yield name, prob, raw, table

    def truth(name):
        return [instance_crops(r) for r in annotations.rles_for_file(name)]

    def fit_filter(tables, targets):
        table, target = pd.concat(tables, ignore_index=True), np.concatenate(targets)
        if len(np.unique(target)) != 2:
            raise RuntimeError("Calibration has only one class; cannot fit component filter")
        return ComponentFilter(CUTOFF).fit(table, target)

    totals = {"raw": PanopticQuality(), "filtered": PanopticQuality()}
    report = {
        "params": PARAMS,
        "cutoff": CUTOFF,
        "epochs": 16,
        "folds": [],
        "evaluation": "Outer 5-fold physical-frame CV. Inner holdout calibrates filters without outer labels.",
        "limitations": [
            "Fixed parameters were previously selected on part of this dataset.",
            "Outer scores assess single models; test inference averages five models.",
            "Random physical-frame splits do not establish temporal generalization.",
        ],
    }
    all_tables, all_targets = [], []
    for split in splits:
        k = split["fold"]
        log("fold", k, "calibration model")
        inner = train(split["inner_train"], 1000 + k, out / f"calibration_{k}.pt")
        tables, targets = [], []
        for name, _, raw, table in components(inner, split["calibration"], train_dir):
            tables.append(table)
            targets.append(matched_by_any_annotator(truth(name), raw[0], raw[1]))
        filt = fit_filter(tables, targets)
        del inner, tables, targets
        gc.collect()
        torch.cuda.empty_cache()
        log("fold", k, "outer model")
        model = train(split["train"], k, out / f"unet_fold_{k}.pt")
        scores = {"raw": PanopticQuality(), "filtered": PanopticQuality()}
        prob_dir = out / "oof_probs"
        prob_dir.mkdir(exist_ok=True)
        for name, prob, raw, table in components(model, split["validation"], train_dir):
            np.save(prob_dir / (name + ".npy"), prob)
            target = matched_by_any_annotator(truth(name), raw[0], raw[1])
            filtered = select(*raw, filt.keep(table))
            all_tables.append(table.assign(fold=k))
            all_targets.append(target)
            for kind, result in [("raw", raw), ("filtered", filtered)]:
                for gt in truth(name):
                    scores[kind].update(gt, result[0], result[1])
                    totals[kind].update(gt, result[0], result[1])
        report["folds"].append(
            dict(
                fold=k,
                train_frames=len(split["train"]),
                val_frames=len(split["validation"]),
                **{kind: value.summary() for kind, value in scores.items()},
            )
        )
        report["pooled"] = {kind: value.summary() for kind, value in totals.items()}
        (out / "cv_report.json").write_text(json.dumps(report, indent=2))
        log(report["folds"][-1])
        # Disk-backed uint16 sums avoid retaining all full-resolution test maps in RAM.
        sums = out / "test_sums"
        sums.mkdir(exist_ok=True)
        for name, prob, _, _ in components(model, test_files, test_dir):
            path = sums / (name + ".npy")
            total = np.load(path) if k else np.zeros(prob.shape, dtype=np.uint16)
            total += prob.astype(np.uint16)
            np.save(path, total)
        del model, filt
        gc.collect()
        torch.cuda.empty_cache()

    final_filter = fit_filter(all_tables, all_targets)
    table = pd.concat(all_tables, ignore_index=True)
    table["target"] = np.concatenate(all_targets)
    table.to_csv(out / "oof_components.csv", index=False)
    import pickle

    with (out / "component_filter.pkl").open("wb") as stream:
        pickle.dump(final_filter, stream)
    rows = []
    for name in test_files:
        prob = np.rint(np.load(out / "test_sums" / (name + ".npy")) / 5).astype(np.uint8)
        image = read_gray(test_dir / name)
        raw = instances(prob, **PARAMS)
        table = component_table(frame_context(name, image, prob), raw[0], raw[2])
        labels, _, boxes = select(*raw, final_filter.keep(table))
        rows.extend(rows_from_label_map(Path(name).stem, labels, boxes))
    frame = write(rows, out / "submission.csv")
    report.update(test_frames=len(test_files), submission_rows=len(frame), elapsed_sec=time.time() - start)
    (out / "cv_report.json").write_text(json.dumps(report, indent=2))
    log("complete", report)
