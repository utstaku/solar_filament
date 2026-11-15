"""Five-fold training of one network / loss / label setting: out-of-fold maps for all 707 training frames.

Every experiment of the zoo goes through this one function, so they differ only in what `EXPERIMENTS`
says: the folds (`fold_files`, seed 0, as `cnn_baseline/cv.py`), tiles, augmentation, epochs, schedule
and inference are shared. Inference runs on 512 px tiles with Hann-window blending for every network,
the size the transformers were built for, so all maps are made the same way.

Outputs, with the experiment's name as prefix so several can be attached to one comparison notebook:
`<name>_oof_fold{k}.npz` (uint8 probability * 255 of fold k's frames), `<name>_test_fold{k}.npz` (fold k's
model on the test frames), `<name>_fold{k}.pt`, `<name>_folds.json`, `<name>_report.json` (losses, timings,
peak GPU memory and a quick PQ: A0's instances without the component filter).
"""

import json
import os
import random
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, fold_files, worker_init
from cnn_baseline.training import (
    bce_dice_loss,
    boundary_bce_dice_loss,
    fit,
    pick_device,
    skeleton_recall_bce_dice_loss,
)
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, output_dir
from postprocess.instances import instances
from postprocess.pipeline import A0
from preprocess.images import read_gray
from scoring.metric import PanopticQuality, instance_crops

QUICK = bool(os.environ.get("FILAMENT_QUICK"))  # a few frames, one epoch, small tiles and no downloads
SEED = 0
N_FOLDS = 5
TILE = 256 if QUICK else 512
STRIDE = TILE * 3 // 4
TILES_PER_FRAME = 4
FRAMES_PER_BATCH = 2 if QUICK else 4  # -> 16 tiles per step, as the baseline
EPOCHS = 1 if QUICK else 16
LR = 1e-3
PRETRAINED_LR = 1e-4  # ImageNet encoders: a tenth of the decoder's peak learning rate
WORKERS = 0 if QUICK else 3
BUDGET_SEC = 11 * 3600  # Kaggle stops a session at 12 h; no fold starts that would not finish before this

# name -> (network, loss, label); the first three differ from the baseline in one thing each
EXPERIMENTS = {
    "baseline": ("unet", "bce_dice", "mean"),
    "boundary_bce": ("unet", "boundary", "mean"),
    "annotator_sampling": ("unet", "bce_dice", "sample"),
    "edgeattnet": ("edgeattnet", "bce_dice", "mean"),
    "hrnet": ("hrnet_w18", "bce_dice", "mean"),
    "segformer": ("segformer", "bce_dice", "mean"),
    "swin_unet": ("swin_unet", "bce_dice", "mean"),
    "swin_unet_boundary": ("swin_unet", "boundary", "mean"),  # the two that helped, together
    "swin_unet_skeleton": ("swin_unet", "skeleton", "mean"),  # against filaments cut by faint stretches
    "swin_unet_skeleton_majority": ("swin_unet", "skeleton_majority", "mean"),  # the same, on agreed lines
    "swin_unet_boundary_seed1": ("swin_unet", "boundary", "mean"),  # a second seed, for a seed ensemble
    "swin_unet_boundary_ema": ("swin_unet", "boundary", "mean"),  # one model like the mean of several
}
# training seeds start here instead of SEED (the folds stay those of SEED, so out-of-fold maps line up)
TRAIN_SEEDS = {"swin_unet_boundary_seed1": 100}
# weights averaged over training (`fit`'s `ema` decay; 0.998 ~ the last 500 steps, 3.5 of 16 epochs)
EMA = {"swin_unet_boundary_ema": 0.998}
LOSSES = {
    "bce_dice": bce_dice_loss,
    "boundary": boundary_bce_dice_loss,
    "skeleton": skeleton_recall_bce_dice_loss,
    "skeleton_majority": partial(skeleton_recall_bce_dice_loss, weight=0.2, level=0.5),
}


def build(network: str) -> torch.nn.Module:
    """The network, with ImageNet weights where it has a pretrained encoder (none in quick runs)."""
    pretrained = not QUICK
    if network == "unet":
        from models.unet import UNet

        return UNet(in_channels=2, base=8 if QUICK else 32, depth=4)
    if network == "edgeattnet":
        from models.edgeattnet import EdgeAttNet

        return EdgeAttNet(in_channels=2, base=8 if QUICK else 32, depth=4)
    if network == "hrnet_w18":
        from models.pretrained_unet import PretrainedUNet

        return PretrainedUNet("hrnet_w18", pretrained=pretrained)
    if network == "segformer":
        from models.segformer import SegFormer

        return SegFormer(pretrained=pretrained)
    if network == "swin_unet":
        from models.swin_unet import SwinUNet

        return SwinUNet(pretrained=pretrained, tile=TILE)
    raise ValueError(f"unknown network {network!r}")


def starts(length: int, tile: int, stride: int) -> list[int]:
    return [0] if length <= tile else sorted({*range(0, length - tile + 1, stride), length - tile})


@torch.no_grad()
def predict_tiled(model, frame: torch.Tensor, device, tile: int = TILE, stride: int = STRIDE, batch: int = 8):
    """uint8 (H, W) probability * 255 of a (2, H, W) frame from overlapping tiles, blended with a Hann window."""
    model.to(device).eval()
    _, height, width = frame.shape
    window = torch.from_numpy(np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 1e-3).astype(np.float32))
    total = torch.zeros(height, width)
    weight = torch.zeros(height, width)
    corners = [(top, left) for top in starts(height, tile, stride) for left in starts(width, tile, stride)]
    for i in range(0, len(corners), batch):
        chunk = corners[i : i + batch]
        x = torch.stack([frame[:, t : t + tile, l : l + tile] for t, l in chunk]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(x).float())[:, 0].cpu()
        for (t, l), p in zip(chunk, prob):
            total[t : t + tile, l : l + tile] += p * window
            weight[t : t + tile, l : l + tile] += window
    return (total / weight).mul(255).round().to(torch.uint8).numpy()


# (quarter turns, mirrored) of the 8 views of the square; the first four are identity, flips and half turn
VIEWS = [(0, False), (0, True), (2, True), (2, False), (1, False), (1, True), (3, False), (3, True)]


def predict_views(model, frame: torch.Tensor, device, views: int = 8) -> np.ndarray:
    """uint8 (H, W) mean of `predict_tiled` over the first `views` flips and quarter turns, each turned back."""
    total = torch.zeros(frame.shape[1:])
    for turns, flip in VIEWS[:views]:
        view = torch.rot90(frame, turns, dims=(-2, -1))
        view = view.flip(-1) if flip else view
        prob = torch.from_numpy(predict_tiled(model, view.contiguous(), device)).float()
        prob = prob.flip(-1) if flip else prob
        total += torch.rot90(prob, -turns, dims=(-2, -1))
    return total.div(views).round().to(torch.uint8).numpy()


def train(name: str, train_files: list[str], train_dir, annotations, device, seed: int, log=print):
    """The experiment's network trained on `train_files` with its loss and label: (model, epoch losses)."""
    network, loss, label = EXPERIMENTS[name]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    loader = DataLoader(
        FilamentFrames(train_files, train_dir, annotations, tile=TILE, tiles_per_frame=TILES_PER_FRAME, label=label),
        batch_size=FRAMES_PER_BATCH,
        shuffle=True,
        num_workers=WORKERS,
        collate_fn=collate_tiles,
        worker_init_fn=worker_init,
        pin_memory=device.type == "cuda",
        persistent_workers=WORKERS > 0,
        drop_last=True,
    )
    model = build(network)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    history = fit(
        model,
        loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        log=log,
        loss=LOSSES[loss],
        pretrained_lr=PRETRAINED_LR,
        ema=EMA.get(name),
    )
    return model, history


def run(name: str, folds_to_run: list[int] | None = None) -> dict:
    network, loss, label = EXPERIMENTS[name]
    t0 = time.time()

    def log(*args):
        print(f"[{time.time() - t0:5.0f}s]", *args, flush=True)

    device = pick_device()
    out = output_dir()
    root = find_data_root()
    train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
    annotations = load_annotations(next((root / "train").glob("*.json")))
    files = sorted(p.name for p in train_dir.iterdir())
    test_files = sorted(p.name for p in test_dir.iterdir())
    if QUICK:
        files, test_files = files[:20], test_files[:3]
    folds = fold_files(files, N_FOLDS, SEED)
    (out / f"{name}_folds.json").write_text(json.dumps({"seed": SEED, "folds": folds}, indent=1))
    folds_to_run = list(range(N_FOLDS)) if folds_to_run is None else folds_to_run
    log(f"{name}: {network}, loss {loss}, label {label}; folds {folds_to_run} of {[len(f) for f in folds]}; {device}")

    def frame(directory, f):
        return torch.from_numpy(channels(read_gray(directory / f)))

    report = {
        "name": name,
        "network": network,
        "loss": loss,
        "label": label,
        "quick": QUICK,
        "seed": SEED,
        "train_seed": TRAIN_SEEDS.get(name, SEED),
        "ema": EMA.get(name),
        "tile": TILE,
        "stride": STRIDE,
        "epochs": EPOCHS,
        "lr": LR,
        "pretrained_lr": PRETRAINED_LR,
        "quick_pq_params": A0,
        "folds": [],
    }
    for k in folds_to_run:
        longest = max((f["train_sec"] + f["predict_sec"] for f in report["folds"]), default=0)
        if time.time() - t0 + longest > BUDGET_SEC:
            log(f"stopping before fold {k}: it would not finish within the session")
            report["stopped_before_fold"] = k
            break
        start = time.time()
        held_out = folds[k]
        train_files = sorted(set(files) - set(held_out))
        log(f"fold {k}: {len(train_files)} frames")
        model, history = train(name, train_files, train_dir, annotations, device, TRAIN_SEEDS.get(name, SEED) + k, log)
        torch.save(model.state_dict(), out / f"{name}_fold{k}.pt")
        trained = time.time()

        oof = {f: predict_tiled(model, frame(train_dir, f), device) for f in held_out}
        np.savez_compressed(out / f"{name}_oof_fold{k}.npz", **oof)
        pq = PanopticQuality()
        for f, prob in oof.items():
            labels, areas, _ = instances(prob, **A0)
            for rles in annotations.rles_for_file(f):
                pq.update(instance_crops(rles), labels, areas)
        del oof
        test = {f: predict_tiled(model, frame(test_dir, f), device) for f in test_files}
        np.savez_compressed(out / f"{name}_test_fold{k}.npz", **test)
        del test, model
        report["folds"].append(
            {
                "fold": k,
                "train_frames": len(train_files),
                "held_out_frames": len(held_out),
                "loss": [round(v, 4) for v in history],
                "quick_pq": {**pq.summary(), "iou_sum": round(pq.iou_sum, 4)},
                "train_sec": round(trained - start, 1),
                "predict_sec": round(time.time() - trained, 1),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"fold {k} done:", report["folds"][-1]["quick_pq"])
        (out / f"{name}_report.json").write_text(json.dumps(report, indent=2))

    pooled = PanopticQuality()
    for fold in report["folds"]:
        s = fold["quick_pq"]
        pooled.tp, pooled.fp, pooled.fn = pooled.tp + s["tp"], pooled.fp + s["fp"], pooled.fn + s["fn"]
        pooled.iou_sum += s["iou_sum"]
    report["pooled_quick_pq"] = pooled.summary()
    report["elapsed_sec"] = round(time.time() - t0, 1)
    (out / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    log("pooled quick PQ (A0 instances, no filter):", report["pooled_quick_pq"])
    return report


def run_full(name: str) -> dict:
    """The experiment trained once on all training frames, for the submission: `<name>_full.pt`,
    `<name>_test_full.npz` (its test maps, single model, made as the out-of-fold maps are), and
    `<name>_full_report.json`. Same schedule and epochs as each fold, so 25 % more steps."""
    t0 = time.time()

    def log(*args):
        print(f"[{time.time() - t0:5.0f}s]", *args, flush=True)

    device = pick_device()
    out = output_dir()
    root = find_data_root()
    train_dir, test_dir = root / "train" / "train_images", root / "test" / "test_images"
    annotations = load_annotations(next((root / "train").glob("*.json")))
    files = sorted(p.name for p in train_dir.iterdir())
    test_files = sorted(p.name for p in test_dir.iterdir())
    if QUICK:
        files, test_files = files[:20], test_files[:3]
    log(f"{name}: {EXPERIMENTS[name]} on all {len(files)} frames; {device}")
    model, history = train(name, files, train_dir, annotations, device, TRAIN_SEEDS.get(name, SEED) + N_FOLDS, log)
    torch.save(model.state_dict(), out / f"{name}_full.pt")
    trained = time.time()
    test = {f: predict_tiled(model, torch.from_numpy(channels(read_gray(test_dir / f))), device) for f in test_files}
    np.savez_compressed(out / f"{name}_test_full.npz", **test)
    report = {
        "name": name,
        "experiment": EXPERIMENTS[name],
        "quick": QUICK,
        "train_frames": len(files),
        "test_frames": len(test_files),
        "loss": [round(v, 4) for v in history],
        "train_sec": round(trained - t0, 1),
        "predict_sec": round(time.time() - trained, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
    }
    (out / f"{name}_full_report.json").write_text(json.dumps(report, indent=2))
    log(report)
    return report
