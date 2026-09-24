"""Frames as network input, soft filament targets, and tile sampling for training."""

from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from dataset.annotations import Annotations
from preprocess.disk import OFF_DISK, flatten
from preprocess.images import read_gray

OFF_DISK_VALUE = OFF_DISK / 255.0  # what the flattened channel holds outside the disk


def channels(image: np.ndarray) -> np.ndarray:
    """float32 (2, H, W): the raw frame and its limb-darkening-corrected version, both scaled to [0, 1]."""
    return np.stack([image, flatten(image)]).astype(np.float32) / 255.0


def soft_target(annotations: Annotations, file_name: str) -> np.ndarray:
    """float32 (H, W) in [0, 1]: the fraction of annotators who marked each pixel as filament.

    Up to three annotators labelled the same frame independently; averaging them keeps the
    disagreement as a soft label instead of arbitrarily trusting one of them.
    """
    per_annotator = annotations.rles_for_file(file_name)
    if not per_annotator:
        raise KeyError(f"no annotations for {file_name}")
    masks = [mask_util.decode(rles).any(axis=2) if rles else None for rles in per_annotator]
    shape = next(m.shape for m in masks if m is not None)
    stacked = np.stack([m if m is not None else np.zeros(shape, bool) for m in masks])
    return stacked.mean(axis=0, dtype=np.float32)


class FilamentFrames(Dataset):
    """One item = one physical frame.

    Training mode returns `tiles_per_frame` random tiles from it: (K, 2, tile, tile) inputs and
    (K, tile, tile) targets. Half of the tiles are centred on a filament pixel, because filaments
    cover 0.4 % of the disk and purely random tiles would mostly be empty. Evaluation mode returns
    the whole frame, (2, H, W) and (H, W).
    """

    def __init__(
        self,
        files: list[str],
        image_dir: Path,
        annotations: Annotations,
        tile: int = 512,
        tiles_per_frame: int = 4,
        train: bool = True,
    ):
        self.files, self.image_dir, self.annotations = files, Path(image_dir), annotations
        self.tile, self.tiles_per_frame, self.train = tile, tiles_per_frame, train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        x = channels(read_gray(self.image_dir / self.files[index]))
        y = soft_target(self.annotations, self.files[index])
        if not self.train:
            return torch.from_numpy(x), torch.from_numpy(y)
        # torch seeds each DataLoader worker differently every epoch, so tiles differ across epochs
        rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,))))
        tiles = [self.sample_tile(x, y, rng) for _ in range(self.tiles_per_frame)]
        return torch.from_numpy(np.stack([t[0] for t in tiles])), torch.from_numpy(np.stack([t[1] for t in tiles]))

    def sample_tile(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        positives = np.flatnonzero(y.ravel() > 0.5)
        if len(positives) and rng.random() < 0.5:
            centre = positives[rng.integers(len(positives))]
        else:
            on_disk = np.flatnonzero(x[1].ravel() < OFF_DISK_VALUE)
            centre = on_disk[rng.integers(len(on_disk))] if len(on_disk) else rng.integers(y.size)
        height, width = y.shape
        row, col = divmod(int(centre), width)
        top = int(np.clip(row - self.tile // 2, 0, height - self.tile))
        left = int(np.clip(col - self.tile // 2, 0, width - self.tile))
        tile_x = x[:, top : top + self.tile, left : left + self.tile]
        tile_y = y[top : top + self.tile, left : left + self.tile]
        # only masks are scored, so every rotation and flip is a valid augmentation
        k = int(rng.integers(4))
        tile_x, tile_y = np.rot90(tile_x, k, axes=(1, 2)), np.rot90(tile_y, k)
        if rng.random() < 0.5:
            tile_x, tile_y = tile_x[:, :, ::-1], tile_y[:, ::-1]
        return np.ascontiguousarray(tile_x), np.ascontiguousarray(tile_y)


def worker_init(_: int) -> None:
    """OpenCV's own thread pool and forked DataLoader workers do not mix; one thread per worker is plenty."""
    cv2.setNumThreads(0)


def collate_tiles(batch):
    """Merge the per-frame tile stacks into one batch: (B*K, 2, tile, tile), (B*K, tile, tile)."""
    xs, ys = zip(*batch)
    return torch.cat(xs), torch.cat(ys)
