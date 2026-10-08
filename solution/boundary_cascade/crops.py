"""Box-conditioned crops with individual masks, shared train/inference geometry."""

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from preprocess.images import read_gray


def crop_input(image, box, size=384):
    h, w = image.shape
    x, y, bw, bh = box
    side = min(max(96, int(np.ceil(max(bw, bh) * 1.5))), h, w)
    left = int(np.clip(round(x + bw / 2 - side / 2), 0, w - side))
    top = int(np.clip(round(y + bh / 2 - side / 2), 0, h - side))
    gray = cv2.resize(image[top : top + side, left : left + side], (size, size))
    seed = np.zeros((side, side), np.uint8)
    x0, x1 = np.clip([int(np.floor(x - left)), int(np.ceil(x + bw - left))], 0, side)
    y0, y1 = np.clip([int(np.floor(y - top)), int(np.ceil(y + bh - top))], 0, side)
    seed[y0:y1, x0:x1] = 255
    seed = cv2.resize(seed, (size, size), interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    inputs = np.stack([gray, clahe, seed]).astype(np.float32) / 255
    # Match ImageNet preprocessing used by the crop refiner, including the seed channel.
    inputs = inputs - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    inputs /= np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    return inputs, (left, top, side)


class Crops(Dataset):
    """One individual target per record; do not merge different annotators' instances."""

    def __init__(self, directory, annotations, files, size=384):
        self.directory, self.size = directory, size
        self.records = [(f, r) for f in files for rs in annotations.rles_for_file(f) for r in rs]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name, rle = self.records[index]
        image = read_gray(self.directory / name)
        mask = mask_util.decode(rle)
        x, y, w, h = mask_util.toBbox(rle)
        rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,))))
        # Box jitter at training; target remains the same instance even near neighbours.
        box = [
            x + rng.uniform(-0.1, 0.1) * w,
            y + rng.uniform(-0.1, 0.1) * h,
            w * rng.uniform(0.9, 1.1),
            h * rng.uniform(0.9, 1.1),
        ]
        inputs, (left, top, side) = crop_input(image, box, self.size)
        target = cv2.resize(
            mask[top : top + side, left : left + side], (self.size, self.size), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
        k = int(rng.integers(4))
        inputs, target = np.rot90(inputs, k, axes=(1, 2)), np.rot90(target, k)
        if rng.random() < 0.5:
            inputs, target = inputs[:, :, ::-1], target[:, ::-1]
        return torch.from_numpy(inputs.copy()), torch.from_numpy(target.copy())


def exclusive_labels(candidates, shape, min_area=150):
    """Candidates are (confidence, x, y, probability crop); masks are greedily disjoint."""
    labels, probabilities, areas, boxes = np.zeros(shape, np.int32), np.zeros(shape, np.uint8), [], []
    for _, x, y, prob in sorted(candidates, key=lambda c: -c[0]):
        h, w = prob.shape
        view = labels[y : y + h, x : x + w]
        foreground = (prob >= 128) & (view == 0)
        area = int(foreground.sum())
        if area < min_area:
            continue
        view[foreground] = len(areas) + 1
        probabilities[y : y + h, x : x + w][foreground] = prob[foreground]
        rows, cols = np.where(foreground)
        areas.append(area)
        boxes.append([x + int(cols.min()), y + int(rows.min()), int(np.ptp(cols)) + 1, int(np.ptp(rows)) + 1])
    return (labels, np.array(areas, dtype=np.int64), np.array(boxes, dtype=np.int32).reshape(-1, 4)), probabilities
