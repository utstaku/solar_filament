"""Aligned tiles with per-annotator, tubed skeleton targets (no spine metadata)."""

import numpy as np
import torch
from pycocotools import mask as mask_util
from skimage.morphology import binary_dilation, disk, skeletonize

from cnn_baseline.data import FilamentFrames, channels, soft_target
from preprocess.images import read_gray


def tubed_skeleton(mask: np.ndarray) -> np.ndarray:
    """Radius-one tube clipped to the annotated foreground."""
    mask = np.asarray(mask, dtype=bool)
    return binary_dilation(skeletonize(mask), footprint=disk(1)) & mask


def skeleton_target(annotations, name: str, shape: tuple[int, int]) -> np.ndarray:
    sets = annotations.rles_for_file(name)
    if not sets:
        raise ValueError(f"No annotation sets for {name}")
    target = np.zeros(shape, dtype=np.float32)
    for rles in sets:
        mask = mask_util.decode(rles).any(axis=2) if rles else np.zeros(shape, bool)
        target += tubed_skeleton(mask)
    return target / len(sets)


class SkeletonFrames(FilamentFrames):
    def __getitem__(self, index):
        name = self.files[index]
        x = channels(read_gray(self.image_dir / name))
        y = soft_target(self.annotations, name)
        s = skeleton_target(self.annotations, name, y.shape)
        # Use the existing sampler: adding a channel ensures identical spatial transforms.
        xs = np.concatenate([x, s[None]], axis=0)
        rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,))))
        tiles = [self.sample_tile(xs, y, rng) for _ in range(self.tiles_per_frame)]
        return (
            torch.from_numpy(np.stack([t[0][:2] for t in tiles])),
            torch.from_numpy(np.stack([t[1] for t in tiles])),
            torch.from_numpy(np.stack([t[0][2] for t in tiles])),
        )


def collate_skeleton(batch):
    return tuple(torch.cat(items) for items in zip(*batch))
