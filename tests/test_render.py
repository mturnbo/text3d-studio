"""End-to-end GPU tests: render real frames offscreen and inspect the pixels."""

import time

import numpy as np
import pytest

from text3d_studio.model import Document
from text3d_studio.render.compositor import Compositor
from text3d_studio.render.exporter import render_document
from text3d_studio.text_raster import rasterize_text

BG_COLOR = (200, 180, 150)


@pytest.fixture
def compositor(gl_context):
    comp = Compositor()
    yield comp
    comp.delete()


def make_document(width=640, height=400, **scene_overrides) -> Document:
    doc = Document()
    background = np.zeros((height, width, 4), np.uint8)
    background[..., :3] = BG_COLOR
    background[..., 3] = 255
    doc.background = background
    for section, values in scene_overrides.items():
        for name, value in values.items():
            setattr(getattr(doc.scene, section), name, value)
    return doc


def render(compositor, doc):
    compositor.set_background(doc.background)
    compositor.set_glyph(rasterize_text(doc.scene.text))
    return render_document(compositor, doc)


def changed_mask(pixels):
    return np.any(pixels[..., :3] != BG_COLOR, axis=-1)


def test_export_matches_background_native_resolution(compositor):
    pixels = render(compositor, make_document(width=731, height=411))
    assert pixels.shape == (411, 731, 4)
    assert pixels.dtype == np.uint8


def test_text_is_drawn_over_background(compositor):
    pixels = render(compositor, make_document(shadow={"enabled": False}))
    assert changed_mask(pixels).mean() > 0.02
    assert np.all(pixels[0, 0, :3] == BG_COLOR)  # corners stay untouched


def test_empty_text_leaves_background_untouched(compositor):
    pixels = render(compositor, make_document(text={"content": ""}))
    assert not changed_mask(pixels).any()


def test_extrusion_depth_adds_visible_side_walls(compositor):
    flat = render(compositor, make_document(transform={"depth": 0.0, "rot_y": -40}, shadow={"enabled": False}))
    deep = render(compositor, make_document(transform={"depth": 0.8, "rot_y": -40}, shadow={"enabled": False}))
    assert changed_mask(deep).sum() > changed_mask(flat).sum() * 1.1


def test_shadow_darkens_ground_beyond_the_text(compositor):
    light = {"azimuth": -110, "elevation": 30}
    without = render(compositor, make_document(light=light, shadow={"enabled": False}))
    with_shadow = render(compositor, make_document(light=light, shadow={"softness": 0.1, "opacity": 0.8}))
    darker = without[..., :3].astype(int).sum(-1) - with_shadow[..., :3].astype(int).sum(-1)
    assert (darker > 30).sum() > 500


def test_grazing_light_still_casts_shadow_via_side_walls(compositor):
    # Light nearly parallel to the text face: layer slices alone would cast almost nothing.
    doc = make_document(transform={"rot_y": 0}, light={"azimuth": 90, "elevation": 20}, shadow={"softness": 0.05})
    no_shadow = make_document(transform={"rot_y": 0}, light={"azimuth": 90, "elevation": 20}, shadow={"enabled": False})
    darker = render(compositor, no_shadow)[..., :3].astype(int).sum(-1) - render(compositor, doc)[..., :3].astype(int).sum(-1)
    assert (darker > 30).sum() > 500


def test_lighting_azimuth_changes_text_shading(compositor):
    lit_front = render(compositor, make_document(light={"azimuth": 0, "elevation": 20}, shadow={"enabled": False}))
    lit_back = render(compositor, make_document(light={"azimuth": 180, "elevation": 20}, shadow={"enabled": False}))
    assert lit_front[..., :3].astype(int).sum() > lit_back[..., :3].astype(int).sum()


def test_render_without_background_uses_default_canvas(compositor):
    doc = Document()
    compositor.set_background(None)
    compositor.set_glyph(rasterize_text(doc.scene.text))
    assert render_document(compositor, doc).shape == (1000, 1600, 4)


def test_oversized_background_is_rejected(compositor):
    too_big = compositor.max_texture_size + 1
    with pytest.raises(ValueError):
        compositor.set_background(np.zeros((1, too_big, 4), np.uint8))


def test_interactive_frame_rate_exceeds_30_fps(compositor, gl_context):
    from OpenGL import GL

    from text3d_studio.render.gl_utils import RenderTarget

    doc = make_document(width=1600, height=1000)
    compositor.set_background(doc.background)
    compositor.set_glyph(rasterize_text(doc.scene.text))
    target = RenderTarget(1600, 1000, samples=4)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        frames = 30
        start = time.perf_counter()
        for i in range(frames):
            doc.scene.transform.rot_y = -40 + i  # simulate a slider drag
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        fps = frames / (time.perf_counter() - start)
    finally:
        target.delete()
    assert fps > 30, f"only {fps:.1f} FPS"


def test_layer_count_auto_increases_when_extrusion_spans_many_pixels():
    from text3d_studio.render.text3d_renderer import MAX_LAYERS, Text3DRenderer

    doc = make_document()
    doc.scene.transform.layers = 8
    doc.scene.transform.depth = 0.5
    assert Text3DRenderer.effective_layers(doc.scene, 4000) > 8
    doc.scene.transform.z = -1000
    assert Text3DRenderer.effective_layers(doc.scene, 400) == 8  # far away: user count suffices
    doc.scene.transform.z = -0.01
    assert Text3DRenderer.effective_layers(doc.scene, 4000) == MAX_LAYERS
    doc.scene.transform.depth = 0.0
    assert Text3DRenderer.effective_layers(doc.scene, 4000) == 1


# ----- Grid overlay (preview only) ----------------------------------------------------------------


def render_preview(compositor, doc, with_grid):
    """Composite plus (optionally) the grid overlay into an offscreen target, like the canvas does."""
    from OpenGL import GL

    from text3d_studio.render.gl_utils import RenderTarget
    from text3d_studio.render.grid_renderer import GridRenderer

    height, width = doc.background.shape[:2]
    raster = rasterize_text(doc.scene.text)
    compositor.set_background(doc.background)
    compositor.set_glyph(raster)
    target = RenderTarget(width, height, 0)
    grid = GridRenderer(compositor.quad)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, width, height))
        if with_grid:
            grid.render(doc.scene, raster, target.fbo, (0, 0, width, height))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    finally:
        grid.delete()
        target.delete()
    return np.frombuffer(data, np.uint8).reshape(height, width, 4)[::-1].copy()


def grid_document(grid=None, **overrides):
    grid = {"visible": True, "opacity": 1.0, **(grid or {})}
    return make_document(grid=grid, shadow={"enabled": False}, **overrides)


def test_grid_draws_lines_on_the_ground_only(compositor):
    doc = grid_document()
    plain, gridded = render_preview(compositor, doc, False), render_preview(compositor, doc, True)
    changed = np.any(plain[..., :3] != gridded[..., :3], axis=-1)
    height = changed.shape[0]
    assert changed[height // 2 + 20 :].mean() > 0.02  # ground below the horizon gets lines
    # Nothing above the horizon (mid-image at pitch 0) except the vanishing-point rings centred on it.
    assert not changed[: height // 2 - 10].any()


def test_text_hides_the_grid_behind_it(compositor):
    doc = grid_document()
    plain, gridded = render_preview(compositor, doc, False), render_preview(compositor, doc, True)
    text = changed_mask(plain)
    differs = np.any(plain[..., :3] != gridded[..., :3], axis=-1)
    assert (differs & text).sum() < 0.01 * text.sum()  # only antialiased silhouette edges may blend


@pytest.mark.parametrize("pitch", [0.0, 6.0, -4.0])
def test_horizon_line_lands_where_the_ground_vanishes(compositor, pitch):
    from text3d_studio import math3d

    doc = grid_document(ground={"pitch": pitch}, text={"content": ""})
    plain, gridded = render_preview(compositor, doc, False), render_preview(compositor, doc, True)
    height = plain.shape[0]
    expected_row = height / 2 - math3d.focal_length_px(doc.scene.camera.fov, height) * np.tan(np.radians(pitch))
    column = 15  # left edge, away from the vanishing-point rings
    rows = np.flatnonzero(np.any(plain[:, column, :3] != gridded[:, column, :3], axis=-1))
    first_marked = rows.min()
    assert abs(first_marked - expected_row) <= 2


def test_export_is_identical_with_grid_shown_or_hidden(compositor):
    hidden = render(compositor, make_document(grid={"visible": False}))
    shown = render(compositor, make_document(grid={"visible": True, "opacity": 1.0}))
    np.testing.assert_array_equal(hidden, shown)


def test_footprint_hugs_the_ink_and_handles_no_text():
    from text3d_studio.model import TextParams
    from text3d_studio.render.grid_renderer import footprint

    raster = rasterize_text(TextParams(content="HHH"))
    x_min, x_max, z_min, z_max = footprint(raster, size=2.0, depth=0.5)
    ink_cols = np.flatnonzero(raster.rgba[..., 3].max(axis=0) > 127)
    px_to_world = raster.size[0] * 2.0 / raster.rgba.shape[1]
    assert x_max - x_min == pytest.approx((ink_cols[-1] - ink_cols[0] + 1) * px_to_world, rel=0.02)
    assert (z_min, z_max) == (-1.0, 0.0)
    assert footprint(None, 2.0, 0.5) == (0.0, 0.0, 0.0, 0.0)


def test_preview_with_grid_still_exceeds_30_fps(compositor):
    from OpenGL import GL

    from text3d_studio.render.gl_utils import RenderTarget
    from text3d_studio.render.grid_renderer import GridRenderer

    doc = grid_document(ground={"pitch": 8.0})
    doc.background = make_document(width=1600, height=1000).background
    raster = rasterize_text(doc.scene.text)
    compositor.set_background(doc.background)
    compositor.set_glyph(raster)
    target, grid = RenderTarget(1600, 1000, samples=4), GridRenderer(compositor.quad)
    try:
        GL.glFinish()
        start = time.perf_counter()
        for i in range(30):
            doc.scene.transform.rot_y = -40 + i
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
            grid.render(doc.scene, raster, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        fps = 30 / (time.perf_counter() - start)
    finally:
        grid.delete()
        target.delete()
    assert fps > 30, f"only {fps:.1f} FPS"


def test_grid_offset_slides_lines_and_wraps_every_cell(compositor):
    base = render_preview(compositor, grid_document(text={"content": ""}), True)
    half = render_preview(compositor, grid_document(text={"content": ""}, grid={"offset_x": 0.5}), True)
    whole = render_preview(compositor, grid_document(text={"content": ""}, grid={"offset_x": 1.0, "offset_z": -2.0}), True)
    assert np.abs(base.astype(int) - half).max() > 50  # lines visibly moved
    assert np.abs(base.astype(int) - whole).max() <= 1  # a whole-cell shift lands on the same lines (float rounding aside)


# ----- Physically based materials -----------------------------------------------------------------

RED = (0.8, 0.08, 0.08)


def material_document(background_rgb=BG_COLOR, **material):
    doc = make_document(shadow={"enabled": False}, material=material, light={"azimuth": 10, "elevation": 20})
    doc.background[..., :3] = background_rgb
    return doc


def text_pixels(compositor, doc, erode=2):
    """Colors of pixels well inside the text. The background is a flat color, so anything else is text;
    eroding the mask drops antialiased edges, which blend in the background's color."""
    pixels = render(compositor, doc)[..., :3].astype(float)
    covered = np.any(pixels != doc.background[0, 0, :3], axis=-1)
    interior = covered.copy()
    for axis in (0, 1):
        for shift in range(1, erode + 1):
            interior &= np.roll(covered, shift, axis) & np.roll(covered, -shift, axis)
    return pixels[interior]


def highlight_saturation(colors, count=5):
    """Saturation of the brightest few pixels (the light's highlight): ~0 when white, high when tinted."""
    brightest = colors[np.argsort(colors.sum(-1))[-count:]]
    return float(np.mean((brightest.max(-1) - brightest.min(-1)) / np.maximum(brightest.max(-1), 1)))


def test_metal_tints_highlights_while_plastic_keeps_them_white(compositor):
    def saturation(metallic):
        doc = material_document(metallic=metallic, face_color=RED, side_color=RED, roughness=0.2)
        doc.scene.light.reflect_photo = False  # isolate the light's own highlight
        return highlight_saturation(text_pixels(compositor, doc))

    assert saturation(0.0) < 0.1
    assert saturation(1.0) > 0.3


def test_rougher_surfaces_have_dimmer_highlights(compositor):
    shared = dict(metallic=0.0, face_color=(0.5, 0.5, 0.5), side_color=(0.5, 0.5, 0.5))
    smooth = text_pixels(compositor, material_document(roughness=0.08, **shared)).sum(-1)
    rough = text_pixels(compositor, material_document(roughness=0.9, **shared)).sum(-1)
    assert smooth.max() > rough.max() + 40


@pytest.mark.parametrize("reflect", [True, False])
def test_reflections_follow_the_photo_only_when_enabled(compositor, reflect):
    def on(background):
        doc = material_document(background_rgb=background, metallic=1.0, roughness=0.1,
                                face_color=(0.9, 0.9, 0.9), side_color=(0.9, 0.9, 0.9))
        doc.scene.light.reflect_photo = reflect
        return text_pixels(compositor, doc).mean(axis=0)

    red_scene, blue_scene = on((200, 40, 40)), on((40, 40, 200))
    red_minus_blue = (red_scene[0] - red_scene[2]) - (blue_scene[0] - blue_scene[2])
    if reflect:
        assert red_minus_blue > 40  # a mirror-like metal takes on the scene's color
    else:
        assert abs(red_minus_blue) < 1


def test_reflection_strength_scales_metal_brightness(compositor):
    def brightness(strength):
        doc = material_document(metallic=1.0, roughness=0.3)
        doc.scene.light.reflection_strength = strength
        return text_pixels(compositor, doc).mean()

    assert brightness(0.0) < brightness(1.0) < brightness(2.0)


def test_every_material_preset_renders_distinctly(compositor):
    from itertools import combinations

    from text3d_studio.materials import KINDS, PRESETS

    means = {kind: text_pixels(compositor, material_document(**PRESETS[kind])).mean(axis=0) for kind in KINDS}
    for a, b in combinations(KINDS, 2):
        assert np.abs(means[a] - means[b]).max() > 8, (a, b)


@pytest.mark.parametrize("kind", ["metal", "plastic", "glass"])
def test_neutral_materials_stay_neutral_at_grazing_angles(compositor, kind):
    # On a neutral gray photo with white light, a near-neutral material can't produce saturated pixels.
    from text3d_studio.materials import PRESETS

    neutral = (0.85, 0.85, 0.85)
    doc = material_document(background_rgb=(128, 128, 128), **{**PRESETS[kind], "face_color": neutral, "side_color": neutral})
    doc.scene.light.color = (1.0, 1.0, 1.0)
    doc.scene.light.ambient_color = (1.0, 1.0, 1.0)
    doc.scene.transform.rot_y = -60  # plenty of grazing side walls
    pixels = text_pixels(compositor, doc)
    saturation = (pixels.max(-1) - pixels.min(-1)) / np.maximum(pixels.max(-1), 1)
    assert saturation.max() < 0.2



@pytest.mark.parametrize("kind", ["glass", "metal"])
def test_preview_framebuffer_stays_opaque(compositor, kind):
    # Qt composites the canvas as premultiplied color: any alpha below 1 lets bright channels overflow and
    # wrap (e.g. a 245 blue becoming 9). The text's coverage and the grid's blending must not touch alpha.
    from text3d_studio.materials import PRESETS

    doc = grid_document(material={**PRESETS[kind], "kind": kind})
    doc.background = make_document().background
    pixels = render_preview(compositor, doc, with_grid=True)
    assert pixels[..., 3].min() == 255


# ----- Depth pre-pass -----------------------------------------------------------------------------


def compositor_with_prepass(monkeypatch, enabled):
    import text3d_studio.render.text3d_renderer as text3d_renderer

    monkeypatch.setattr(text3d_renderer, "DEPTH_PREPASS", enabled)
    return Compositor()


@pytest.mark.parametrize("kind", ["metal", "plastic", "glass"])
def test_depth_prepass_matches_single_pass_rendering(gl_context, monkeypatch, kind):
    from text3d_studio.materials import PRESETS

    doc = make_document(material={**PRESETS[kind], "kind": kind}, transform={"depth": 0.8, "rot_y": -45})
    single, prepass = compositor_with_prepass(monkeypatch, False), compositor_with_prepass(monkeypatch, True)
    try:
        np.testing.assert_array_equal(render(prepass, doc), render(single, doc))
    finally:
        single.delete()
        prepass.delete()


@pytest.mark.parametrize("kind", ["metal", "plastic", "stone", "wood", "glass"])
def test_deep_text_stays_above_30_fps_for_every_material(compositor, kind):
    from OpenGL import GL

    from text3d_studio.materials import PRESETS
    from text3d_studio.render.gl_utils import RenderTarget

    doc = make_document(width=1600, height=1000, material={**PRESETS[kind], "kind": kind},
                        transform={"size": 6.0, "depth": 1.2, "z": -10.0})
    compositor.set_background(doc.background)
    compositor.set_glyph(rasterize_text(doc.scene.text))
    target = RenderTarget(1600, 1000, samples=4)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        start = time.perf_counter()
        for i in range(20):
            doc.scene.transform.rot_y = -40 + i
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        fps = 20 / (time.perf_counter() - start)
    finally:
        target.delete()
    assert fps > 30, f"{kind}: only {fps:.1f} FPS"
