"""Image pixels <-> heliographic coordinates, and moving a filament mask by the Sun's differential rotation.

Used to bring the annotated filaments of a neighbouring day to the time of the frame being processed.

Conventions (all angles in degrees):
- Image: x to the right, y down; disk centre (cx, cy) and radius R in pixels (`preprocess.disk.disk_geometry`).
- Solar north is up and solar east is on the left, unless rotated by the position angle `p` (solar north
  measured counter-clockwise from image up). Whether GONG frames have p = 0 is checked by `rotation_check.py`.
- Observer frame, in units of R: X = (x - cx) / R towards west, Y = -(y - cy) / R towards north, and
  Z = sqrt(1 - X^2 - Y^2) towards the observer (orthographic projection: the Sun is far away).
- Heliographic latitude `lat` (north positive) and longitude `lon` from the central meridian (west
  positive). `b0` is the latitude of the disk centre: the solar equator tilts towards us by up to 7.25 deg.
"""

from datetime import UTC, datetime

import numpy as np

SIDEREAL = (14.366, -2.297, -1.624)  # A, B, C (deg / day) of magnetic features, Snodgrass 1983 [to verify]
EARTH_ORBIT = 0.9856  # deg / day: the Earth's mean motion, subtracted to get the rate seen from Earth


def _turn(X, Y, degrees):
    """(X, Y) turned counter-clockwise by `degrees`."""
    a = np.radians(degrees)
    return X * np.cos(a) - Y * np.sin(a), X * np.sin(a) + Y * np.cos(a)


def b0_angle(when: datetime) -> float:
    """Heliographic latitude of the disk centre at `when` (timezone-aware, e.g. UTC; degrees, within +-7.25).

    Low-precision recipe (good to ~0.1 deg): with d = days since 2000-01-01 12:00 UT,
    - the Sun's mean anomaly g = 357.529 + 0.98560028 d and mean longitude q = 280.459 + 0.98564736 d;
    - its ecliptic longitude L = q + 1.915 sin g + 0.020 sin 2g;
    - the longitude of the ascending node of the solar equator Omega = 73.6667 + 1.3958333 (JD - 2396758) / 36525,
      with the Julian date JD = 2451545 + d;
    - B0 = asin(sin(L - Omega) * sin(7.25)).
    """
    d = (when - datetime(2000, 1, 1, 12, tzinfo=UTC)).total_seconds() / 86400
    g = np.radians(357.529 + 0.98560028 * d)
    q = 280.459 + 0.98564736 * d
    ecliptic = q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    node = 73.6667 + 1.3958333 * (2451545 + d - 2396758) / 36525
    return float(np.degrees(np.arcsin(np.sin(np.radians(ecliptic - node)) * np.sin(np.radians(7.25)))))


def pixel_to_heliographic(x, y, centre, radius, b0, p=0.0):
    """(lat, lon, on_disk) for image pixels (arrays or scalars). Off-disk pixels get on_disk False (lat, lon NaN).

    Steps: X, Y in units of R (see the module conventions); undo the position angle by rotating (X, Y) by
    -p; Z = sqrt(1 - X^2 - Y^2); then tilt by b0 about the X axis:
    sin(lat) = Y cos(b0) + Z sin(b0),  lon = atan2(X, Z cos(b0) - Y sin(b0)).
    """
    X = (np.asarray(x, float) - centre[0]) / radius
    Y = -(np.asarray(y, float) - centre[1]) / radius
    X, Y = _turn(X, Y, -p)
    r2 = X**2 + Y**2
    on_disk = r2 <= 1
    Z = np.sqrt(np.clip(1 - r2, 0, None))
    b = np.radians(b0)
    lat = np.degrees(np.arcsin(np.clip(Y * np.cos(b) + Z * np.sin(b), -1, 1)))
    lon = np.degrees(np.arctan2(X, Z * np.cos(b) - Y * np.sin(b)))
    return np.where(on_disk, lat, np.nan), np.where(on_disk, lon, np.nan), on_disk


def heliographic_to_pixel(lat, lon, centre, radius, b0, p=0.0):
    """(x, y, visible): the inverse of `pixel_to_heliographic`. visible is False on the far side (Z <= 0).

    X = cos(lat) sin(lon); then with Zs = cos(lat) cos(lon) and sin(lat):
    Y = sin(lat) cos(b0) - Zs sin(b0),  Z = sin(lat) sin(b0) + Zs cos(b0); rotate (X, Y) by +p; back to pixels.
    """
    lat, lon, b = np.radians(lat), np.radians(lon), np.radians(b0)
    X = np.cos(lat) * np.sin(lon)
    towards = np.cos(lat) * np.cos(lon)
    Y = np.sin(lat) * np.cos(b) - towards * np.sin(b)
    Z = np.sin(lat) * np.sin(b) + towards * np.cos(b)
    X, Y = _turn(X, Y, p)
    return centre[0] + X * radius, centre[1] - Y * radius, Z > 0


def synodic_rate(lat) -> np.ndarray:
    """Rotation rate seen from Earth (deg / day) at latitude `lat`: A + B sin^2 + C sin^4 - EARTH_ORBIT."""
    s2 = np.sin(np.radians(lat)) ** 2
    a, b, c = SIDEREAL
    return a + b * s2 + c * s2**2 - EARTH_ORBIT


def rotate_mask(
    mask: np.ndarray,
    centre,
    radius,
    b0_from: float,
    b0_to: float,
    days: float,
    p: float = 0.0,
    source_disk=None,
    rate: float | None = None,
):
    """`mask` (bool, H x W) of a frame, moved to a frame taken `days` later (negative: earlier).

    Work backwards from the target so the result has no holes: for each target pixel on the disk, find its
    (lat, lon) at the target time (b0_to), subtract synodic_rate(lat) * days from lon, project back with
    b0_from, and take the source mask's value there (nearest pixel). Target pixels whose source point is on
    the far side, or off the source disk, are False.

    `centre`, `radius` describe the target frame's disk; `source_disk` = (centre, radius) the source frame's
    when it differs (frames from different sites). `rate` (deg / day) replaces differential rotation by a
    rigid one, for comparison.
    """
    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width]
    lat, lon, on_disk = pixel_to_heliographic(xs, ys, centre, radius, b0_to, p)
    shift = (synodic_rate(lat) if rate is None else rate) * days
    source_centre, source_radius = source_disk or (centre, radius)
    sx, sy, visible = heliographic_to_pixel(lat, lon - shift, source_centre, source_radius, b0_from, p)
    sx = np.rint(np.nan_to_num(sx, nan=-1)).astype(int)
    sy = np.rint(np.nan_to_num(sy, nan=-1)).astype(int)
    ok = on_disk & visible & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
    moved = np.zeros(mask.shape, bool)
    moved[ok] = mask[sy[ok], sx[ok]]
    return moved
