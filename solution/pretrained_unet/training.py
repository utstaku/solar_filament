"""Controlled training and overlap-tile inference for encoder comparison."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from cnn_baseline.training import bce_dice_loss


def skeleton_recall_loss(logits, skeleton):
    """Soft recall on averaged annotator tubes; empty targets contribute zero.

    Binary, soft-annotator adaptation of Kirchhoff et al. (ECCV 2024).
    https://github.com/MIC-DKFZ/Skeleton-Recall
    """
    prob = logits.squeeze(1).float().sigmoid()
    skeleton = skeleton.float()
    mass = skeleton.sum(dim=(1, 2))
    recall = (prob * skeleton).sum(dim=(1, 2)) / mass.clamp_min(1e-6)
    valid = mass > 0
    return ((1 - recall) * valid).sum() / valid.sum().clamp_min(1)


def fit(model, loader, device, epochs, skeleton_weight, out_dir, config, log=print):
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.net.encoder.parameters(), "lr": 1e-4},
            {"params": model.net.decoder.parameters(), "lr": 1e-3},
            {"params": model.net.segmentation_head.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-4, 1e-3, 1e-3], total_steps=epochs * len(loader)
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(epochs):
        model.train()
        start, total = time.time(), 0.0
        for batch in loader:
            x, y = (t.to(device) for t in batch[:2])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = bce_dice_loss(logits, y)
                if skeleton_weight:
                    loss = loss + skeleton_weight * skeleton_recall_loss(logits, batch[2].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += loss.item()
        row = {"epoch": epoch + 1, "loss": total / len(loader), "seconds": time.time() - start}
        history.append(row)
        log(row)
        torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, out_dir / "model.pt")
    return history


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    return sorted({*range(0, length - tile + 1, stride), length - tile})


@torch.inference_mode()
def predict_frame(model, frame, device, tile=512, stride=384):
    """Native pixel spacing; positive Hann weights also cover frame boundaries."""
    model.eval()
    _, height, width = frame.shape
    frame = F.pad(frame, (0, max(0, tile - width), 0, max(0, tile - height)), mode="replicate")
    ph, pw = frame.shape[-2:]
    total, weight = np.zeros((ph, pw), np.float32), np.zeros((ph, pw), np.float32)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 1e-3).astype(np.float32)
    for top in starts(ph, tile, stride):
        for left in starts(pw, tile, stride):
            x = frame[:, top : top + tile, left : left + tile].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            p = logits.float().sigmoid()[0, 0].cpu().numpy()
            total[top : top + tile, left : left + tile] += p * window
            weight[top : top + tile, left : left + tile] += window
    return np.rint(total[:height, :width] / weight[:height, :width] * 255).astype(np.uint8)
