"""Shared split, training and fixed postprocessing for the two experiments."""

import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from cnn_baseline.data import split_files
from cnn_baseline.training import bce_dice_loss
from dataset.annotations import load_annotations
from dataset.paths import find_data_root
from postprocess.filtering import ComponentFilter, component_table, frame_context, matched_by_any_annotator
from postprocess.instances import select
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data():
    root = find_data_root()
    directory = root / "train" / "train_images"
    annotations = load_annotations(next((root / "train").glob("*.json")))
    train, val = split_files(sorted(p.name for p in directory.iterdir() if p.is_file()), 0.15, 0)
    return directory, annotations, train, val


def train_model(model, loader, device, epochs, dest, loss_fn=bce_dice_loss, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(epochs):
        start, total = time.time(), 0.0
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        row = {"epoch": epoch + 1, "loss": total / len(loader), "seconds": time.time() - start}
        history.append(row)
        torch.save(model.state_dict(), dest / "model.pt")
        save(dest / "history.json", history)
        print(dest.name, row, flush=True)
    return history


def score(predictions, truth):
    pq = PanopticQuality()
    for f, (labels, areas, _) in predictions.items():
        for annotation in truth[f]:
            pq.update(annotation, labels, areas)
    return {**pq.summary(), "pq_exact": pq.value}


def evaluate(name, predictions, probs, directory, annotations, files, dest):
    """Same fixed 0.45 OOF component filter; group by physical frame, matching the earlier filter protocol."""
    truth = {f: [instance_crops(rs) for rs in annotations.rles_for_file(f)] for f in files}
    table = pd.concat(
        [
            component_table(frame_context(f, read_gray(directory / f), probs[f]), predictions[f][0], predictions[f][2])
            for f in files
        ],
        ignore_index=True,
    )
    target = np.concatenate([matched_by_any_annotator(truth[f], predictions[f][0], predictions[f][1]) for f in files])
    groups = table["frame"].to_numpy()
    oof = np.zeros(len(table))
    n_splits = min(5, len(np.unique(groups)))
    splits = GroupKFold(n_splits=n_splits).split(table, target, groups) if n_splits >= 2 else []
    for train_idx, test_idx in splits:
        y = target[train_idx]
        if len(np.unique(y)) < 2:
            oof[test_idx] = float(y.mean()) if len(y) else 0.0
        else:
            clf = ComponentFilter().fit(table.iloc[train_idx], y)
            oof[test_idx] = clf.scores(table.iloc[test_idx])
    keep = oof >= 0.45
    filtered = {f: select(*predictions[f], keep[groups == f]) for f in files}
    table["target"], table["oof_score"] = target, oof
    table.to_csv(dest / "components.csv", index=False)
    report = {
        "name": name,
        "raw": score(predictions, truth),
        "filtered": score(filtered, truth) if n_splits >= 2 else None,
        "cutoff": 0.45,
        "n_val": len(files),
        "filter_folds": n_splits,
        "filter_available": n_splits >= 2,
    }
    save(dest / "report.json", report)
    print(report, flush=True)
    return report
