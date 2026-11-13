"""The occlusion mask's brush and snapshots (pure numpy)."""

import numpy as np
import pytest

from text3d_studio.occlusion import decode, encode, paint_segment


def blank(width=100, height=60):
    return np.zeros((height, width), np.uint8)


def test_a_dab_covers_a_disc_of_the_radius():
    mask = blank()
    paint_segment(mask, (50, 30), (50, 30), radius=10, feather=0.0)
    assert mask[30, 50] == 255
    assert mask[30, 50 + 8] == 255
    assert mask[30, 50 + 12] == 0
    assert mask[0, 0] == 0


def test_a_segment_covers_the_capsule_between_its_ends():
    mask = blank()
    paint_segment(mask, (20, 30), (80, 30), radius=5, feather=0.0)
    assert (mask[30, 20:80] == 255).all()
    assert mask[30 + 8, 50] == 0


def test_feather_fades_the_edge_through_intermediate_values():
    hard, soft = blank(), blank()
    paint_segment(hard, (50, 30), (50, 30), radius=20, feather=0.0)
    paint_segment(soft, (50, 30), (50, 30), radius=20, feather=1.0)
    row = soft[30, 50:75]
    assert row[0] >= 250 and row[-1] == 0
    assert np.all(np.diff(row.astype(int)) <= 0)  # falls off monotonically
    assert ((row > 0) & (row < 255)).sum() > 10
    assert ((hard > 0) & (hard < 255)).sum() < ((soft > 0) & (soft < 255)).sum()


def test_painting_keeps_the_larger_value_and_erasing_the_smaller():
    mask = blank()
    paint_segment(mask, (50, 30), (50, 30), radius=10, feather=0.0)
    paint_segment(mask, (50, 30), (50, 30), radius=10, feather=1.0)  # a softer dab over it takes nothing away
    assert mask[30, 50 + 8] == 255
    paint_segment(mask, (50, 30), (50, 30), radius=4, feather=0.0, erase=True)
    assert mask[30, 50] == 0
    assert mask[30, 50 + 8] == 255


def test_the_changed_rect_bounds_the_stroke_and_is_clipped_to_the_mask():
    mask = blank()
    x0, y0, x1, y1 = paint_segment(mask, (-5, 10), (20, 10), radius=4, feather=0.5)
    assert (x0, y0) == (0, 6) and x1 <= 26 and y1 <= 16
    changed = np.argwhere(mask > 0)
    assert changed[:, 0].min() >= y0 and changed[:, 0].max() < y1
    assert changed[:, 1].min() >= x0 and changed[:, 1].max() < x1


def test_a_stroke_off_the_mask_changes_nothing():
    mask = blank()
    assert paint_segment(mask, (-50, -50), (-30, -40), radius=5, feather=0.5) is None
    assert not mask.any()


@pytest.mark.parametrize("radius", [0.0, -3.0, 0.2])
def test_a_tiny_or_invalid_radius_still_paints_about_one_pixel(radius):
    mask = blank()
    paint_segment(mask, (10.5, 10.5), (10.5, 10.5), radius=radius, feather=0.5)
    assert 1 <= (mask > 0).sum() <= 9


def test_snapshots_round_trip_and_none_means_no_mask():
    mask = blank()
    paint_segment(mask, (50, 30), (70, 40), radius=12, feather=0.5)
    snapshot = encode(mask)
    assert np.array_equal(decode(snapshot), mask)
    assert encode(None) is None and decode(None) is None
    assert len(snapshot[1]) < mask.nbytes / 4  # compressed


def test_a_decoded_mask_is_a_writable_copy():
    mask = blank()
    restored = decode(encode(mask))
    restored[0, 0] = 9
    assert mask[0, 0] == 0
