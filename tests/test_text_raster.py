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


def test_covers_follows_the_ink_and_its_holes(qapp):
    raster = rasterize_text(TextParams(content="O"))
    pad = -raster.origin[1]
    top = raster.size[1] - 2 * pad  # the ink spans y = 0 to here and is centered on x = 0
    ring_x = raster.origin[0] + pad + 0.04  # just inside the O's left edge
    assert raster.covers(ring_x, top / 2)
    assert not raster.covers(0.0, top / 2)  # the O's counter
    assert not raster.covers(0.0, top + pad / 2)  # padding above the ink


@pytest.mark.parametrize("point", [(-10.0, 0.3), (10.0, 0.3), (0.0, -5.0), (0.0, 5.0)])
def test_covers_is_false_off_the_raster(qapp, point):
    assert not rasterize_text(TextParams(content="H")).covers(*point)


# ----- Multi-line text ----------------------------------------------------------------------------------


def ink_bands(raster):
    """The (first, last) row of each horizontal band of ink, top to bottom."""
    rows = raster.rgba[..., 3].max(axis=1) > 5
    edges = np.diff(np.concatenate([[0], rows.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def ink_columns(raster, band):
    """The (first, last) column with ink within one band's rows."""
    cols = np.flatnonzero(raster.rgba[band[0] : band[1] + 1, :, 3].max(axis=0) > 127)
    return cols[0], cols[-1]


def assert_walls_closed(raster):
    """Every vertex of the traced outline joins an even number of unit wall edges, so no wall is left open."""
    from collections import Counter

    from text3d_studio.text_raster import WALL_STEP

    height, width = raster.rgba.shape[:2]
    quads = raster.walls.reshape(-1, 6, 3)
    assert len(quads) > 0
    p0, p1 = quads[:, 0, :2], quads[:, 1, :2]
    to_grid = lambda p: np.rint(np.stack([
        (p[:, 0] - raster.origin[0]) / raster.size[0] * width,
        (1 - (p[:, 1] - raster.origin[1]) / raster.size[1]) * height,
    ], axis=1) / WALL_STEP).astype(int)
    degree = Counter()
    for (x0, y0), (x1, y1) in zip(to_grid(p0), to_grid(p1)):
        steps = max(abs(x1 - x0), abs(y1 - y0))
        dx, dy = np.sign(x1 - x0), np.sign(y1 - y0)
        for i in range(steps):
            degree[(x0 + i * dx, y0 + i * dy)] += 1
            degree[(x0 + (i + 1) * dx, y0 + (i + 1) * dy)] += 1
    assert all(count % 2 == 0 for count in degree.values())


@pytest.mark.parametrize("lines", [2, 3])
def test_each_line_is_its_own_band_of_ink(qapp, lines):
    single = rasterize_text(TextParams(content="HI"))
    raster = rasterize_text(TextParams(content="\n".join(["HI"] * lines)))
    assert len(ink_bands(raster)) == lines
    assert raster.rgba.shape[0] > single.rgba.shape[0] * (lines - 0.5)
    assert raster.size[1] > single.size[1] * (lines - 0.5)
    assert_walls_closed(raster)


def test_lines_are_one_line_spacing_apart(qapp):
    raster = rasterize_text(TextParams(content="HI\nHI"))
    first, second = ink_bands(raster)
    em_rows = raster.rgba.shape[0] / raster.size[1]  # pixels per em
    assert (second[0] - first[0]) / em_rows == pytest.approx(1.0, abs=0.02)  # Helvetica's line spacing is 1 em


def test_blank_line_keeps_its_height(qapp):
    tight = ink_bands(rasterize_text(TextParams(content="HI\nHI")))
    spaced_raster = rasterize_text(TextParams(content="HI\n\nHI"))
    spaced = ink_bands(spaced_raster)
    em_rows = spaced_raster.rgba.shape[0] / spaced_raster.size[1]
    assert len(spaced) == 2
    assert ((spaced[1][0] - spaced[0][0]) - (tight[1][0] - tight[0][0])) / em_rows == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("content", ["HI\n", "HI\n\n", "\nHI", "\n\nHI\n"])
def test_leading_and_trailing_newlines_change_nothing(qapp, content):
    plain = rasterize_text(TextParams(content="HI"))
    raster = rasterize_text(TextParams(content=content))
    assert np.array_equal(raster.rgba, plain.rgba)
    assert raster.origin == plain.origin and raster.size == plain.size


@pytest.mark.parametrize("content", ["\n", "\n\n", " \n \t\n"])
def test_text_of_only_blank_lines_returns_none(qapp, content):
    assert rasterize_text(TextParams(content=content)) is None


def test_ink_bottom_of_the_last_line_sits_on_local_ground(qapp):
    raster = rasterize_text(TextParams(content="HI\nHH"))
    last = ink_bands(raster)[-1]
    height = raster.rgba.shape[0]
    bottom_local = raster.origin[1] + (1 - (last[1] + 1) / height) * raster.size[1]
    assert bottom_local == pytest.approx(0.0, abs=2 / height * raster.size[1])


def test_many_lines_are_clamped_to_the_texture_limit(qapp):
    raster = rasterize_text(TextParams(content="\n".join(["HI"] * 40)))
    assert max(raster.rgba.shape[:2]) <= MAX_TEXTURE_WIDTH
    assert len(ink_bands(raster)) == 40


def test_covers_hits_the_second_line(qapp):
    raster = rasterize_text(TextParams(content="I\nOOOO"))
    first, second = ink_bands(raster)
    row = (second[0] + second[1]) // 2
    col = int(np.flatnonzero(raster.rgba[row, :, 3] == 255)[0]) + 2
    height, width = raster.rgba.shape[:2]
    x = raster.origin[0] + (col + 0.5) / width * raster.size[0]
    y = raster.origin[1] + (1 - (row + 0.5) / height) * raster.size[1]
    assert raster.covers(x, y)
    gap_row = (first[1] + second[0]) // 2
    assert not raster.covers(x, raster.origin[1] + (1 - (gap_row + 0.5) / height) * raster.size[1])


# ----- Alignment ----------------------------------------------------------------------------------------

ALIGN_CONTENT = "HHHHHH\nHH\nHHHH"  # flat-sided letters, so ink extents are exact


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_lines_line_up_as_aligned(qapp, align):
    raster = rasterize_text(TextParams(content=ALIGN_CONTENT, align=align))
    extents = [ink_columns(raster, band) for band in ink_bands(raster)]
    assert len(extents) == 3
    lefts, rights = [e[0] for e in extents], [e[1] for e in extents]
    centers = [(e[0] + e[1]) / 2 for e in extents]
    lined_up = {"left": lefts, "center": centers, "right": rights}[align]
    assert max(lined_up) - min(lined_up) <= 3
    assert max(rights) - min(lefts) - max(e[1] - e[0] for e in extents) <= 3  # the widest line sets the span


def test_alignment_changes_nothing_for_a_single_line(qapp):
    rasters = [rasterize_text(TextParams(content="HELLO", align=a)) for a in ("left", "center", "right")]
    assert all(np.array_equal(r.rgba, rasters[0].rgba) for r in rasters)


def test_left_and_right_alignment_mirror_the_short_lines(qapp):
    left = rasterize_text(TextParams(content=ALIGN_CONTENT, align="left"))
    right = rasterize_text(TextParams(content=ALIGN_CONTENT, align="right"))
    width = left.rgba.shape[1]
    short_left = ink_columns(left, ink_bands(left)[1])
    short_right = ink_columns(right, ink_bands(right)[1])
    assert short_left[0] < width / 3 and short_right[1] > width * 2 / 3
    assert left.rgba.shape == right.rgba.shape


def test_unknown_alignment_raises(qapp):
    with pytest.raises(ValueError, match="alignment"):
        rasterize_text(TextParams(content="HI", align="justify"))


# ----- Letter spacing and line height -------------------------------------------------------------------


def test_letter_spacing_widens_the_mask_by_the_gaps_between_letters(qapp):
    widths = {s: rasterize_text(TextParams(content="HHHH", letter_spacing=s)).size[0] for s in (-0.1, 0.0, 0.2, 0.5)}
    assert widths[-0.1] < widths[0.0] < widths[0.2] < widths[0.5]
    assert widths[0.2] - widths[0.0] == pytest.approx(3 * 0.2, abs=0.02)  # three gaps, no trailing space
    assert widths[0.0] - widths[-0.1] == pytest.approx(3 * 0.1, abs=0.02)


def test_letter_spacing_keeps_the_height(qapp):
    plain = rasterize_text(TextParams(content="HI"))
    spaced = rasterize_text(TextParams(content="HI", letter_spacing=0.5))
    assert spaced.rgba.shape[0] == plain.rgba.shape[0]


def test_letter_spacing_applies_to_every_line_and_keeps_them_aligned(qapp):
    raster = rasterize_text(TextParams(content="HHHH\nHH", letter_spacing=0.3, align="left"))
    extents = [ink_columns(raster, band) for band in ink_bands(raster)]
    assert abs(extents[0][0] - extents[1][0]) <= 3


@pytest.mark.parametrize("spacing", [-0.1, -0.3])
def test_negative_letter_spacing_overlaps_letters_with_closed_walls(qapp, spacing):
    raster = rasterize_text(TextParams(content="HOHO", letter_spacing=spacing))
    assert raster.rgba[..., 3].max() == 255
    assert_walls_closed(raster)


@pytest.mark.parametrize("line_height", [1.0, 1.5, 2.0])
def test_line_height_sets_the_distance_between_lines(qapp, line_height):
    raster = rasterize_text(TextParams(content="HI\nHI", line_height=line_height))
    first, second = ink_bands(raster)
    em_rows = raster.rgba.shape[0] / raster.size[1]
    assert (second[0] - first[0]) / em_rows == pytest.approx(line_height, abs=0.02)


def test_tight_line_height_shrinks_the_mask_and_keeps_walls_closed(qapp):
    normal = rasterize_text(TextParams(content="HI\nHI\nHI"))
    tight = rasterize_text(TextParams(content="HI\nHI\nHI", line_height=0.7))
    assert tight.size[1] == pytest.approx(normal.size[1] - 2 * 0.3, abs=0.02)
    assert_walls_closed(tight)


def test_line_height_changes_nothing_for_a_single_line(qapp):
    plain = rasterize_text(TextParams(content="HI"))
    assert np.array_equal(rasterize_text(TextParams(content="HI", line_height=2.0)).rgba, plain.rgba)
