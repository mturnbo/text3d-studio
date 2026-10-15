import numpy as np
import pytest

from text3d_studio.model import TextParams
from text3d_studio.text_raster import MAX_TEXTURE_WIDTH, extract_walls, rasterize_text


def decode_direction(rgba):
    return rgba[..., :2].astype(np.float64) / 255 * 2 - 1


def test_rasterizes_text_into_mask_with_coverage(qapp):
    raster = rasterize_text(TextParams(content="Hi"))
    assert raster is not None
    assert raster.rgba.dtype == np.uint8 and raster.rgba.shape[2] == 4
    assert raster.rgba[..., 3].max() == 255
    assert raster.size[0] > 0 and raster.size[1] > 0


def test_ink_bottom_sits_on_local_ground_and_is_centered(qapp):
    raster = rasterize_text(TextParams(content="HHH"))
    alpha = raster.rgba[..., 3]
    height, width = alpha.shape
    ink_rows = np.flatnonzero(alpha.max(axis=1) > 5)
    bottom_local = raster.origin[1] + (1 - (ink_rows[-1] + 1) / height) * raster.size[1]
    assert bottom_local == pytest.approx(0.0, abs=2 / height * raster.size[1])
    assert raster.origin[0] == pytest.approx(-raster.size[0] / 2)


def test_edge_normals_point_outward(qapp):
    raster = rasterize_text(TextParams(content="I", bold=True))
    alpha = raster.rgba[..., 3]
    direction = decode_direction(raster.rgba)
    row = alpha.shape[0] // 2
    cols = np.flatnonzero(alpha[row] > 127)
    left_edge, right_edge = cols[0], cols[-1]
    assert direction[row, left_edge, 0] < -0.8
    assert direction[row, right_edge, 0] > 0.8


def test_bevel_factor_is_high_at_edge_and_zero_deep_inside(qapp):
    raster = rasterize_text(TextParams(content="O", bevel_width=4))
    alpha, bevel = raster.rgba[..., 3], raster.rgba[..., 2]
    row = alpha.shape[0] // 2
    cols = np.flatnonzero(alpha[row] > 127)
    gaps = np.flatnonzero(np.diff(cols) > 1)
    left_stroke = cols[: gaps[0] + 1]
    assert bevel[row, left_stroke[0]] > 150
    assert bevel[row, left_stroke[len(left_stroke) // 2]] < 60


@pytest.mark.parametrize("content", ["", "   ", "\t"])
def test_blank_text_returns_none(qapp, content):
    assert rasterize_text(TextParams(content=content)) is None


def test_very_long_text_is_clamped_to_max_texture_width(qapp):
    raster = rasterize_text(TextParams(content="W" * 200))
    assert raster.rgba.shape[1] <= MAX_TEXTURE_WIDTH


def test_unknown_font_family_falls_back_instead_of_failing(qapp):
    assert rasterize_text(TextParams(content="Ok", font_family="No Such Font 12345")) is not None


def test_walls_trace_a_closed_square_outline():
    alpha = np.zeros((20, 20), dtype=np.float32)
    alpha[5:15, 5:15] = 1.0
    walls = extract_walls(alpha, origin=(0.0, 0.0), size=(20.0, 20.0), step=1).reshape(-1, 6, 3)
    assert len(walls) == 4  # one merged quad per side of the square
    xs, ys = walls[..., 0], walls[..., 1]
    assert xs.min() == pytest.approx(5) and xs.max() == pytest.approx(15)
    assert ys.min() == pytest.approx(5) and ys.max() == pytest.approx(15)
    assert set(np.unique(walls[..., 2])) == {0.0, 1.0}


def test_walls_include_inner_hole_outline():
    alpha = np.zeros((30, 30), dtype=np.float32)
    alpha[5:25, 5:25] = 1.0
    alpha[12:18, 12:18] = 0.0
    walls = extract_walls(alpha, origin=(0.0, 0.0), size=(30.0, 30.0), step=1).reshape(-1, 6, 3)
    assert len(walls) == 8


def test_walls_handle_glyph_touching_mask_border():
    alpha = np.ones((4, 4), dtype=np.float32)
    walls = extract_walls(alpha, origin=(0.0, 0.0), size=(4.0, 4.0), step=1).reshape(-1, 6, 3)
    assert len(walls) == 4
