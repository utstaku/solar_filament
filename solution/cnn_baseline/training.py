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


def boundary_band(target: torch.Tensor, width: int) -> torch.Tensor:
    """bool (B, H, W): pixels within `width` px of a place where the (soft) target changes value.

    Morphological gradient by max pooling; on the mean of several annotators it covers each one's border.
    """
    t = target.unsqueeze(1).float()
    k = 2 * width + 1
    dilated = F.max_pool2d(t, k, stride=1, padding=width)
    eroded = -F.max_pool2d(-t, k, stride=1, padding=width)
    return (dilated - eroded).squeeze(1) > 0


def boundary_bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: float = 4.0, width: int = 3
) -> torch.Tensor:
    """`bce_dice_loss` with the BCE of pixels near a filament's border counted `1 + weight` times.

    Near misses (IoU just under 0.5) lose their filament's border and faint ends, not its core; weighting
    the border band moves the gradient there. The BCE is a weighted mean, so its scale stays comparable.
    """
    logits, target = logits.squeeze(1).float(), target.float()
    w = 1 + weight * boundary_band(target, width).float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum()
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2))
    dice = 1 - (2 * intersection + 1) / (prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


def soft_skeleton(image: torch.Tensor, iterations: int = 30) -> torch.Tensor:
    """(B, H, W) skeleton of a (soft) mask by iterated min / max pooling, as clDice's soft-skeletonisation.

    On a binary mask it is close to the morphological skeleton; on the mean of several annotators each
    skeleton pixel keeps the fraction that drew it. `iterations` must reach half the widest filament (a 60 px wide blob needs 30).
    """

    def erode(t):
        return torch.minimum(-F.max_pool2d(-t, (3, 1), 1, (1, 0)), -F.max_pool2d(-t, (1, 3), 1, (0, 1)))

    def opened(t):
        return F.max_pool2d(erode(t), 3, 1, 1)

    t = image.unsqueeze(1).float()
    skeleton = F.relu(t - opened(t))
    for _ in range(iterations):
        t = erode(t)
        delta = F.relu(t - opened(t))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.squeeze(1)


def skeleton_recall_bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: float = 1.0, level: float | None = None
) -> torch.Tensor:
    """`bce_dice_loss` plus `weight` x (1 - recall of the target's skeleton), as Skeleton Recall (Kirchhoff
    et al., 2024).

    A filament whose faint middle falls below 0.5 comes out in pieces or as a fragment; BCE and Dice barely
    notice the few pixels of the gap, but they are a large share of the filament's centre line. The skeleton
    is thickened to 3 px ("tubed") so a prediction a pixel off still covers it; tiles without filament add 0.

    With `level`, the skeleton is that of the pixels at least that share of the annotators drew (0.5: a
    majority) instead of the soft target's: recall rewards p = 1 on every skeleton pixel, so a skeleton of
    filaments one annotator in three drew pulls the map up where they disagree (`zoo/swin_unet_skeleton`).
    """
    with torch.no_grad():
        drawn = target >= level if level is not None else target > 0
        source = drawn.float() if level is not None else target
        skeleton = F.max_pool2d(soft_skeleton(source).unsqueeze(1), 3, 1, 1).squeeze(1).clamp(max=1)
        skeleton = skeleton * drawn  # the tube stays inside the drawn filament
    prob = torch.sigmoid(logits.squeeze(1).float())
    length = skeleton.sum(dim=(1, 2))
    recall = (prob * skeleton).sum(dim=(1, 2)) / length.clamp(min=1)
    has_filament = (length > 0).float()
    missed = ((1 - recall) * has_filament).sum() / has_filament.sum().clamp(min=1)
    return bce_dice_loss(logits, target) + weight * missed


def parameter_groups(model: nn.Module, lr: float, pretrained_lr: float | None) -> list[dict]:
    """One group at `lr`, or -- for a model with `pretrained_parameters()` and a `pretrained_lr` -- the
    pretrained part at `pretrained_lr` and the rest (decoder, head) at `lr`."""
    if pretrained_lr is None or not hasattr(model, "pretrained_parameters"):
        return [{"params": list(model.parameters()), "lr": lr}]
    pretrained = list(model.pretrained_parameters())
    ids = {id(p) for p in pretrained}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": pretrained, "lr": pretrained_lr}, {"params": rest, "lr": lr}]


def fit(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    log=print,
    loss=bce_dice_loss,
    pretrained_lr: float | None = None,
) -> list[float]:
    """AdamW + one-cycle schedule, mixed precision on CUDA. Returns the mean loss of each epoch.

    `loss(logits, target)` defaults to BCE + Dice; `pretrained_lr` sets a smaller peak learning rate for the
    pretrained encoder of a model that has one (see `parameter_groups`).
    """
    device = device or pick_device()
    model.to(device)
    groups = parameter_groups(model, lr, pretrained_lr)
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, total, steps = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                value = loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(value).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total, steps = total + value.item(), steps + 1
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
