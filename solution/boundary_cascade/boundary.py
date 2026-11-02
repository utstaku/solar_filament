"""Controlled boundary-BCE experiment: same plain U-Net, weights 0, 2 and 4."""

import gc
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from boundary_cascade.common import data, evaluate, save, seed, train_model
from cnn_baseline.data import FilamentFrames, channels, collate_tiles, worker_init
from cnn_baseline.training import bce_dice_loss, pick_device, predict
from dataset.paths import output_dir
from models.unet import UNet
from postprocess.instances import instances, on_disk
from preprocess.images import read_gray


def boundary_loss(logits, target, weight=2.0):
    """BCE+Dice plus BCE weight on a radius-2 band of the soft target's 0.5 contour.

    Boundary detection alone uses a hard mask; BCE/Dice retain the original soft target.
    weight=0 is exactly the old loss. Weight map is 1+weight*band (not renormalized).
    """
    hard = (target >= 0.5).float().unsqueeze(1)
    padded = F.pad(hard, (2, 2, 2, 2), mode="replicate")
    dilated = F.max_pool2d(padded, 5, stride=1)
    eroded = -F.max_pool2d(-padded, 5, stride=1)
    band = (dilated - eroded).squeeze(1)
    extra = F.binary_cross_entropy_with_logits(logits.squeeze(1).float(), target.float(), reduction="none")
    return bce_dice_loss(logits, target) + weight * (extra * band).mean()


def run():
    device = pick_device()
    directory, annotations, train, val = data()
    reports = []
    for weight in (0, 2, 4):
        seed()
        name = f"boundary_w{weight}"
        dest = output_dir() / name
        dest.mkdir(parents=True, exist_ok=True)
        save(
            dest / "config.json",
            {
                "weight": weight,
                "band_radius": 2,
                "train_files": train,
                "val_files": val,
                "epochs": 16,
                "seed": 0,
                "tile": 512,
                "batch_tiles": 16,
                "threshold": 0.5,
                "low": 0.2,
                "min_area": 150,
                "inference": "full-frame, disk-masked",
            },
        )
        dataset = FilamentFrames(train, directory, annotations, tile=512, tiles_per_frame=4)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            worker_init_fn=worker_init,
            collate_fn=collate_tiles,
            generator=torch.Generator().manual_seed(0),
            drop_last=True,
        )
        model = UNet(in_channels=2, base=32, depth=4)
        train_model(model, loader, device, 16, dest, partial(boundary_loss, weight=weight))
        frames = (torch.from_numpy(channels(read_gray(directory / f))) for f in val)
        probs = {f: on_disk(p, read_gray(directory / f)) for f, p in zip(val, predict(model, frames, device))}
        np.savez_compressed(dest / "val_probs.npz", **probs)
        predictions = {f: instances(probs[f], threshold=0.5, low=0.2, min_area=150) for f in val}
        reports.append(evaluate(name, predictions, probs, directory, annotations, val, dest))
        save(output_dir() / "boundary_comparison.json", reports)
        del model, loader, predictions, probs
        gc.collect()
        torch.cuda.empty_cache()
    return reports
