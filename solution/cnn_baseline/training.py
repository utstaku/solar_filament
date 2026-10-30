"""Training loop and full-frame inference for a binary segmentation network."""

import os
import time
from collections.abc import Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def pick_device() -> torch.device:
    """CUDA if there is one; locally fall back to Apple's MPS or the CPU.

    On Kaggle a missing GPU is an error, not a fallback: an account without phone verification (or
    without GPU quota) gets a CPU machine for a GPU notebook without any warning, and this U-Net then
    needs ~3.5 h per epoch instead of minutes.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError(
            "Kaggle gave this notebook no GPU. Check that the account is phone-verified and has GPU quota "
            "left (Settings on kaggle.com); stopping instead of training on the CPU."
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE for per-pixel accuracy plus soft Dice for the 0.4 % of pixels that are filament.

    Both accept soft targets, so annotator disagreement flows into the loss as fractional labels.
    """
    logits, target = logits.squeeze(1).float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2))
    dice = 1 - (2 * intersection + 1) / (prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


def fit(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    log=print,
) -> list[float]:
    """AdamW + one-cycle schedule, mixed precision on CUDA. Returns the mean loss of each epoch."""
    device = device or pick_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, total, steps = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = bce_dice_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total, steps = total + loss.item(), steps + 1
        history.append(total / max(steps, 1))
        log(
            f"epoch {epoch}/{epochs}  loss {history[-1]:.4f}  lr {schedule.get_last_lr()[0]:.2e}  {time.time() - start:.0f}s"
        )
    return history


@torch.no_grad()
def predict(
    model: nn.Module, frames: Iterable[torch.Tensor], device: torch.device | None = None
) -> Iterator[np.ndarray]:
    """Yield a uint8 (H, W) probability map (probability * 255) for each (2, H, W) frame."""
    device = device or pick_device()
    model.to(device).eval()
    use_amp = device.type == "cuda"
    for frame in frames:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(frame.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits.float()).squeeze().mul(255).round().to(torch.uint8)
        yield prob.cpu().numpy()


D4 = [(k, flip) for flip in (False, True) for k in range(4)]  # (quarter turns, mirrored); identity first


@torch.no_grad()
def predict_tta(
    model: nn.Module, frames: Iterable[torch.Tensor], device: torch.device | None = None, views: int = 8
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (plain, averaged) uint8 maps per (2, H, W) frame: the identity view alone, as `predict` gives it,
    and the mean probability over the first `views` rotations / mirror images, each turned back first.

    A filament has no preferred orientation in the image, so the network should answer the same for every
    view; averaging removes the part of its output that depends on the view.
    """
    device = device or pick_device()
    model.to(device).eval()
    use_amp = device.type == "cuda"
    for frame in frames:
        plain, total = None, None
        for k, flip in D4[:views]:
            x = torch.rot90(frame.flip(-1) if flip else frame, k, dims=(-2, -1))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x.unsqueeze(0).to(device))
            prob = torch.rot90(torch.sigmoid(logits.float()).squeeze(), -k, dims=(-2, -1))
            prob = prob.flip(-1) if flip else prob
            plain = prob if plain is None else plain
            total = prob.clone() if total is None else total + prob
        yield (
            plain.mul(255).round().to(torch.uint8).cpu().numpy(),
            total.div(views).mul(255).round().to(torch.uint8).cpu().numpy(),
        )
