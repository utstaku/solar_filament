"""Which kept components belong to the same filament: features of nearby pairs, and joining by a decision.

Joining kept components within a fixed distance helped (`variants.join`), but distance alone cannot tell
how two pieces relate. Physically a filament breaks in two ways:

- **along its spine**, where the material thins out: the pieces line up end to end, and the gap runs
  along their common axis, often still a little darker than the surroundings;
- **at a barb**, a short side extension from the spine down to the chromosphere: a small piece next to
  the *side* of the spine, at an angle to it.

Two separate filaments, in contrast, lie side by side or end to end in different filament channels. The
features below describe a pair in those terms: distance, the angles between the axes and the gap, whether
the gap leaves the larger piece at its end, how dark and how confident the gap is, and the direction of
the chromospheric fibrils around the gap (fibrils in a filament channel run along the channel).
"""

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from cnn_baseline.postprocess import select
from cnn_baseline.variants import label_stats, merge
from error_analysis.instances import FrameContext, overlaps
from preprocess.disk import REL_SCALE
from scoring.metric import InstanceCrop

PAIR_FEATURES = [
    "dist",
    "angle_diff",
    "gap_large",
    "gap_small",
    "end_large",
    "end_small",
    "path_dark",
    "path_prob_mean",
    "path_prob_min",
    "fibril_angle",
    "fibril_coherence",
    "log_area_large",
    "log_area_small",
    "elong_large",
    "elong_small",
    "limb",
]
ORIENTED = 2.0  # elongation above which a piece has a meaningful axis
FIBRIL_HALF = 20  # half-width (px) of the window around the gap in which the fibril direction is measured


def line_angle(a, b):
    """Angle between two undirected lines given by their angles (radians), in [0, pi / 2]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % np.pi
    return np.minimum(d, np.pi - d)


def component_axes(labels: np.ndarray, n: int) -> pd.DataFrame:
    """Per instance 1..n: centroid, principal-axis direction (unit vector, angle in [0, pi)), elongation and
    the extent along the axis (lo..hi, relative to the centroid)."""
    _, boxes = label_stats(labels, n)
    rows = []
    for k, (x, y, w, h) in enumerate(boxes, start=1):
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == k)
        xs, ys = (xs + x).astype(float), (ys + y).astype(float)
        cx, cy = xs.mean(), ys.mean()
        if len(xs) > 2:
            values, vectors = np.linalg.eigh(np.cov(np.stack([xs, ys])))
            ux, uy = vectors[:, 1]
            elongation = float(np.sqrt(max(values[1], 1e-9) / max(values[0], 1e-9)))
        else:
            ux, uy, elongation = 1.0, 0.0, 1.0
        proj = (xs - cx) * ux + (ys - cy) * uy
        rows.append(
            {
                "cx": cx,
                "cy": cy,
                "ux": ux,
                "uy": uy,
                "angle": float(np.arctan2(uy, ux) % np.pi),
                "elongation": elongation,
                "lo": proj.min(),
                "hi": proj.max(),
                "area": len(xs),
            }
        )
    return pd.DataFrame(rows, columns=["cx", "cy", "ux", "uy", "angle", "elongation", "lo", "hi", "area"])


def pairs(labels: np.ndarray, radius: int) -> pd.DataFrame:
    """Every pair a < b of instances within `radius` px: their distance and nearest pixels (ax, ay), (bx, by)."""
    n = int(labels.max())
    _, boxes = label_stats(labels, n)
    rows = []
    for a, (x, y, w, h) in enumerate(boxes, start=1):
        x0, y0 = max(x - radius, 0), max(y - radius, 0)
        x1, y1 = min(x + w + radius, labels.shape[1]), min(y + h + radius, labels.shape[0])
        window = labels[y0:y1, x0:x1]
        dist, (iy, ix) = ndimage.distance_transform_edt(window != a, return_indices=True)
        for b in np.unique(window[(dist <= radius) & (window > a)]):
            d = np.where(window == b, dist, np.inf)
            by, bx = np.unravel_index(int(np.argmin(d)), d.shape)
            rows.append(
                {
                    "a": a,
                    "b": int(b),
                    "dist": float(d[by, bx]),
                    "ax": int(ix[by, bx]) + x0,
                    "ay": int(iy[by, bx]) + y0,
                    "bx": int(bx) + x0,
                    "by": int(by) + y0,
                }
            )
    return pd.DataFrame(rows, columns=["a", "b", "dist", "ax", "ay", "bx", "by"])


def path_values(image: np.ndarray, p: tuple[int, int], q: tuple[int, int]) -> np.ndarray:
    """Values of `image` on the straight path from pixel p to pixel q (x, y), endpoints excluded when possible."""
    m = max(int(np.hypot(q[0] - p[0], q[1] - p[1])) - 1, 1)
    t = np.linspace(0, 1, m + 2)[1:-1]
    xs = np.rint(p[0] + t * (q[0] - p[0])).astype(int)
    ys = np.rint(p[1] + t * (q[1] - p[1])).astype(int)
    return image[ys, xs]


def fibril_direction(image: np.ndarray, x: int, y: int, exclude: np.ndarray, half: int = FIBRIL_HALF):
    """(angle of the dominant line direction in [0, pi), coherence in [0, 1]) around (x, y).

    Structure tensor of the image gradients: the dominant gradient direction is across the fibrils, so the
    fibrils run perpendicular to it. Pixels where `exclude` is set (the components themselves, whose edges
    would otherwise dominate) do not count. Coherence 0 = no preferred direction, 1 = perfectly parallel lines.
    """
    x0, y0 = max(x - half, 0), max(y - half, 0)
    x1, y1 = min(x + half + 1, image.shape[1]), min(y + half + 1, image.shape[0])
    window = image[y0:y1, x0:x1].astype(np.float32)
    gx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
    weight = (~exclude[y0:y1, x0:x1]).astype(np.float32)
    jxx, jyy, jxy = (float((weight * g).sum()) for g in (gx * gx, gy * gy, gx * gy))
    if jxx + jyy <= 0:
        return 0.0, 0.0
    gradient = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    coherence = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy)
    return float((gradient + np.pi / 2) % np.pi), float(coherence)


def end_position(axes: pd.Series, x: float, y: float) -> float:
    """Where (x, y) projects along a piece's axis: 0 at either end, 0.5 in the middle."""
    span = axes["hi"] - axes["lo"]
    if span <= 0:
        return 0.5
    t = ((x - axes["cx"]) * axes["ux"] + (y - axes["cy"]) * axes["uy"] - axes["lo"]) / span
    t = float(np.clip(t, 0, 1))
    return min(t, 1 - t)


def pair_table(ctx: FrameContext, labels: np.ndarray, radius: int = 50) -> pd.DataFrame:
    """One row per pair of instances within `radius` px: PAIR_FEATURES plus frame, a, b."""
    near = pairs(labels, radius)
    if not len(near):
        return pd.DataFrame(columns=["frame", "a", "b", *PAIR_FEATURES])
    axes = component_axes(labels, int(labels.max()))
    exclude = cv2.dilate((labels > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = []
    for p in near.itertuples(index=False):
        A, B = axes.iloc[p.a - 1], axes.iloc[p.b - 1]
        a_large = A["area"] >= B["area"]
        L, S = (A, B) if a_large else (B, A)
        pl, ps = ((p.ax, p.ay), (p.bx, p.by)) if a_large else ((p.bx, p.by), (p.ax, p.ay))
        gap = float(np.arctan2(p.by - p.ay, p.bx - p.ax) % np.pi)
        mid_x, mid_y = (p.ax + p.bx) // 2, (p.ay + p.by) // 2
        flat = path_values(ctx.flat, pl, ps).astype(float)
        prob = path_values(ctx.prob, pl, ps).astype(float) / 255
        fibril, coherence = fibril_direction(ctx.flat, mid_x, mid_y, exclude)
        rows.append(
            {
                "frame": ctx.name,
                "a": p.a,
                "b": p.b,
                "dist": p.dist,
                "angle_diff": float(line_angle(L["angle"], S["angle"])),
                "gap_large": float(line_angle(gap, L["angle"])),
                "gap_small": float(line_angle(gap, S["angle"])),
                "end_large": end_position(L, *pl),
                "end_small": end_position(S, *ps),
                "path_dark": float(1 - flat.mean() / REL_SCALE),
                "path_prob_mean": float(prob.mean()),
                "path_prob_min": float(prob.min()),
                "fibril_angle": float(line_angle(fibril, gap)),
                "fibril_coherence": coherence,
                "log_area_large": float(np.log(L["area"])),
                "log_area_small": float(np.log(S["area"])),
                "elong_large": L["elongation"],
                "elong_small": S["elongation"],
                "limb": float(np.hypot(mid_x - ctx.centre[0], mid_y - ctx.centre[1]) / ctx.radius)
                if ctx.radius
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["frame", "a", "b", *PAIR_FEATURES])


def same_filament(
    gt_crops_per_annotator: list[list[InstanceCrop]],
    labels: np.ndarray,
    areas: np.ndarray,
    table: pd.DataFrame,
    share: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """(same, different) per pair: for some annotator, `share` or more of both pieces lies in one filament
    (same), or in two different filaments of theirs (different, and never same)."""
    same = np.zeros(len(table), bool)
    different = np.zeros(len(table), bool)
    if not len(table):
        return same, different
    a, b = table["a"].to_numpy(int) - 1, table["b"].to_numpy(int) - 1
    for crops in gt_crops_per_annotator:
        if not crops:
            continue
        intersection, _ = overlaps(crops, labels, areas)
        inside = intersection / np.maximum(areas, 1)[None, :] >= share  # (n_gt, n_pred)
        same |= (inside[:, a] & inside[:, b]).any(axis=0)
        different |= inside[:, a].any(axis=0) & inside[:, b].any(axis=0)
    return same, different & ~same


def join_pairs(labels: np.ndarray, table: pd.DataFrame, decide: np.ndarray):
    """Merge the pairs of `table` where `decide` is True (transitively); renumber largest first."""
    chosen = table.loc[np.asarray(decide, bool), ["a", "b"]].to_numpy(int)
    return merge(labels, [tuple(link) for link in chosen])


def aligned_end_to_end(table: pd.DataFrame, max_angle: float = np.radians(30), max_end: float = 0.15) -> np.ndarray:
    """Pairs that continue each other along the spine: the gap leaves the larger piece at its end along its
    axis, and also follows the smaller piece's axis when that piece is elongated enough to have one."""
    large = (table["elong_large"] >= ORIENTED) & (table["gap_large"] <= max_angle) & (table["end_large"] <= max_end)
    small = (table["elong_small"] < ORIENTED) | (table["gap_small"] <= max_angle)
    return (large & small).to_numpy()


def join_rule(table: pd.DataFrame, near: int = 10, far: int = 50) -> np.ndarray:
    """The rule of `join_lab.py`: pieces that continue each other end to end along the axis join up to `far`
    px apart, any two pieces up to `near` px (barbs, blobs)."""
    return ((table["dist"] <= near) | ((table["dist"] <= far) & aligned_end_to_end(table))).to_numpy()


def rescue(ctx: FrameContext, labels: np.ndarray, keep: np.ndarray, radius: int = 50):
    """Kept instances joined by `join_rule`, with each other and with dropped instances that continue them.

    `labels` holds every candidate 1..K before the filter and `keep` (bool[K]) the filter's decision. A dropped
    piece comes back only as part of a group that contains a kept one; groups of dropped pieces stay dropped.
    """
    keep = np.asarray(keep, bool)
    assert int(labels.max()) == len(keep), "one decision per candidate"
    table = pair_table(ctx, labels, radius)
    if len(table):
        a, b = table["a"].to_numpy(int) - 1, table["b"].to_numpy(int) - 1
        joined, areas, boxes = join_pairs(labels, table, join_rule(table) & (keep[a] | keep[b]))
    else:
        joined, areas, boxes = merge(labels, [])
    has_kept = np.zeros(len(areas) + 1, bool)
    has_kept[np.unique(joined[np.r_[False, keep][labels]])] = True
    has_kept[0] = False
    return select(joined, areas, boxes, has_kept[1:])
