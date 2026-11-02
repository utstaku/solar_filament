from datetime import UTC, datetime

import numpy as np
import pytest

from cnn_baseline.heliographic import (
    b0_angle,
    heliographic_to_pixel,
    pixel_to_heliographic,
    rotate_mask,
    synodic_rate,
)

CENTRE, RADIUS = (1024.0, 1024.0), 900.0


@pytest.mark.parametrize(
    "when, expected",
    [
        (datetime(2014, 6, 6, tzinfo=UTC), 0.0),  # the solar equator is edge-on twice a year
        (datetime(2014, 12, 8, tzinfo=UTC), 0.0),
        (datetime(2014, 9, 8, tzinfo=UTC), 7.25),  # north pole tipped towards us
        (datetime(2014, 3, 6, tzinfo=UTC), -7.25),
    ],
)
def test_b0_follows_the_year(when, expected):
    assert b0_angle(when) == pytest.approx(expected, abs=0.3)


def test_disk_centre_is_at_latitude_b0_on_the_central_meridian():
    lat, lon, on_disk = pixel_to_heliographic(CENTRE[0], CENTRE[1], CENTRE, RADIUS, b0=5.0)
    assert on_disk and lat == pytest.approx(5.0) and lon == pytest.approx(0.0, abs=1e-9)


def test_limb_and_off_disk():
    lat, lon, on_disk = pixel_to_heliographic(CENTRE[0] + RADIUS * 0.999999, CENTRE[1], CENTRE, RADIUS, b0=0.0)
    assert on_disk and lat == pytest.approx(0, abs=1e-3) and lon == pytest.approx(90, abs=0.2)  # west limb
    _, _, on_disk = pixel_to_heliographic(CENTRE[0] + RADIUS * 1.01, CENTRE[1], CENTRE, RADIUS, b0=0.0)
    assert not on_disk
    lat, _, _ = pixel_to_heliographic(CENTRE[0], CENTRE[1] - RADIUS * 0.5, CENTRE, RADIUS, b0=0.0)
    assert lat == pytest.approx(30.0)  # half way up the disk: north, sin(lat) = 0.5


@pytest.mark.parametrize("b0, p", [(0.0, 0.0), (6.5, 0.0), (-4.0, 12.0)])
def test_pixel_heliographic_round_trip(b0, p):
    rng = np.random.default_rng(0)
    angle, r = rng.uniform(0, 2 * np.pi, 500), RADIUS * np.sqrt(rng.uniform(0, 0.95, 500))
    x, y = CENTRE[0] + r * np.cos(angle), CENTRE[1] + r * np.sin(angle)
    lat, lon, on_disk = pixel_to_heliographic(x, y, CENTRE, RADIUS, b0, p)
    assert on_disk.all()
    x2, y2, visible = heliographic_to_pixel(lat, lon, CENTRE, RADIUS, b0, p)
    assert visible.all() and np.allclose(x2, x) and np.allclose(y2, y)


def test_position_angle_turns_the_image():
    # with p = 90, solar north points to the image's left: a point left of centre is in the north
    lat, lon, _ = pixel_to_heliographic(CENTRE[0] - RADIUS * 0.5, CENTRE[1], CENTRE, RADIUS, b0=0.0, p=90.0)
    assert lat == pytest.approx(30.0) and lon == pytest.approx(0.0, abs=1e-9)


def test_far_side_is_not_visible():
    _, _, visible = heliographic_to_pixel(0.0, 120.0, CENTRE, RADIUS, b0=0.0)
    assert not visible


def test_synodic_rate():
    assert synodic_rate(0.0) == pytest.approx(14.366 - 0.9856)  # ~13.4 deg / day at the equator
    rates = synodic_rate(np.array([0.0, 30.0, 60.0]))
    assert rates[0] > rates[1] > rates[2] > 9  # slower towards the poles


def test_rotate_mask_moves_a_spot_west_by_the_rotation():
    shape = (2048, 2048)
    mask = np.zeros(shape, bool)
    mask[1019:1030, 1019:1030] = True  # an 11 px spot at the disk centre, on the equator (b0 = 0)
    same = rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=0.0)
    assert (same == mask).all()
    moved = rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=1.0)
    ys, xs = np.nonzero(moved)
    expected_x = CENTRE[0] + RADIUS * np.sin(np.radians(synodic_rate(0.0)))  # ~ +208 px to the west (right)
    assert xs.mean() == pytest.approx(expected_x, abs=2) and ys.mean() == pytest.approx(CENTRE[1], abs=1)
    farther = rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=3.0)  # ~40 deg west: cos 40 = 0.77
    assert farther.sum() < 0.9 * mask.sum()  # foreshortened: narrower in x away from the centre
    back = rotate_mask(moved, CENTRE, RADIUS, 0.0, 0.0, days=-1.0)
    assert np.nonzero(back)[1].mean() == pytest.approx(CENTRE[0], abs=2)


def test_rotate_mask_loses_what_rotates_behind_the_limb():
    shape = (2048, 2048)
    mask = np.zeros(shape, bool)
    x = int(CENTRE[0] + RADIUS * np.sin(np.radians(80)))  # 80 deg west, on the equator
    mask[1019:1030, x - 2 : x + 3] = True
    assert not rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=2.0).any()  # 80 + 27 deg: on the far side
    assert rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=-2.0).any()  # earlier it was further east: visible


def test_rotate_mask_between_disks_of_different_frames_and_rigid_rotation():
    shape = (2048, 2048)
    mask = np.zeros(shape, bool)
    mask[1019:1030, 1019:1030] = True
    source = ((CENTRE[0] - 10, CENTRE[1] + 4), RADIUS)  # the source frame's disk sits 10 px left, 4 px lower
    moved = rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=0.0, source_disk=source)
    ys, xs = np.nonzero(moved)
    assert xs.mean() == pytest.approx(1024 + 10, abs=1) and ys.mean() == pytest.approx(1024 - 4, abs=1)
    frozen = rotate_mask(mask, CENTRE, RADIUS, 0.0, 0.0, days=1.0, rate=0.0)
    assert (frozen == mask).all()
    high = np.zeros(shape, bool)
    high[1024 - 640 : 1024 - 630, 1019:1030] = True  # ~45 deg north, where the Sun turns more slowly
    differential = np.nonzero(rotate_mask(high, CENTRE, RADIUS, 0.0, 0.0, days=1.0))[1].mean()
    rigid = np.nonzero(rotate_mask(high, CENTRE, RADIUS, 0.0, 0.0, days=1.0, rate=float(synodic_rate(0.0))))[1].mean()
    assert differential < rigid - 10  # differential rotation moves high latitudes less
