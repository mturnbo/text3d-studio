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
        owner = doc.scene.objects[0] if section in ("text", "transform", "material") else doc.scene
        for name, value in values.items():
            setattr(getattr(owner, section), name, value)
    return doc


def upload_glyphs(compositor, doc):
    """Rasterizes and uploads every object's text, and frees glyphs of objects no longer in the scene."""
    compositor.retain_glyphs({obj.id for obj in doc.scene.objects})
    for obj in doc.scene.objects:
        compositor.set_glyph(obj.id, rasterize_text(obj.text))


def render(compositor, doc):
    compositor.set_background(doc.background)
    upload_glyphs(compositor, doc)
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
    upload_glyphs(compositor, doc)
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
    upload_glyphs(compositor, doc)
    target = RenderTarget(1600, 1000, samples=4)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        frames = 30
        start = time.perf_counter()
        for i in range(frames):
            doc.scene.objects[0].transform.rot_y = -40 + i  # simulate a slider drag
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        fps = frames / (time.perf_counter() - start)
    finally:
        target.delete()
    assert fps > 30, f"only {fps:.1f} FPS"


def test_layer_count_auto_increases_when_extrusion_spans_many_pixels():
    from text3d_studio.render.text3d_renderer import MAX_LAYERS, Text3DRenderer

    doc = make_document()
    doc.scene.objects[0].transform.layers = 8
    doc.scene.objects[0].transform.depth = 0.5
    assert Text3DRenderer.effective_layers(doc.scene, doc.scene.objects[0].transform, 4000) > 8
    doc.scene.objects[0].transform.z = -1000
    assert Text3DRenderer.effective_layers(doc.scene, doc.scene.objects[0].transform, 400) == 8  # far away: user count suffices
    doc.scene.objects[0].transform.z = -0.01
    assert Text3DRenderer.effective_layers(doc.scene, doc.scene.objects[0].transform, 4000) == MAX_LAYERS
    doc.scene.objects[0].transform.depth = 0.0
    assert Text3DRenderer.effective_layers(doc.scene, doc.scene.objects[0].transform, 4000) == 1


# ----- Grid overlay (preview only) ----------------------------------------------------------------


def render_preview(compositor, doc, with_grid, selected=0):
    """Composite plus (optionally) the grid overlay for the selected object into an offscreen target, like the
    canvas does."""
    from OpenGL import GL

    from text3d_studio.render.gl_utils import RenderTarget
    from text3d_studio.render.grid_renderer import GridRenderer

    height, width = doc.background.shape[:2]
    compositor.set_background(doc.background)
    upload_glyphs(compositor, doc)
    obj = doc.scene.objects[selected]
    target = RenderTarget(width, height, 0)
    grid = GridRenderer(compositor.quad)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, width, height))
        if with_grid:
            grid.render(doc.scene, obj, compositor.glyph_raster(obj.id), target.fbo, (0, 0, width, height))
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
    raster = rasterize_text(doc.scene.objects[0].text)
    compositor.set_background(doc.background)
    compositor.set_glyph(doc.scene.objects[0].id, raster)
    target, grid = RenderTarget(1600, 1000, samples=4), GridRenderer(compositor.quad)
    try:
        GL.glFinish()
        start = time.perf_counter()
        for i in range(30):
            doc.scene.objects[0].transform.rot_y = -40 + i
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
            grid.render(doc.scene, doc.scene.objects[0], raster, target.fbo, (0, 0, 1600, 1000))
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


def text_image(compositor, doc, erode=2):
    """The rendered colors and a mask of pixels well inside the text. The background is a flat color, so
    anything else is text; eroding the mask drops antialiased edges, which blend in the background."""
    pixels = render(compositor, doc)[..., :3].astype(float)
    covered = np.any(pixels != doc.background[0, 0, :3], axis=-1)
    interior = covered.copy()
    for axis in (0, 1):
        for shift in range(1, erode + 1):
            interior &= np.roll(covered, shift, axis) & np.roll(covered, -shift, axis)
    return pixels, interior


def text_pixels(compositor, doc, erode=2):
    pixels, interior = text_image(compositor, doc, erode)
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
    doc.scene.objects[0].transform.rot_y = -60  # plenty of grazing side walls
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


@pytest.mark.parametrize("kind", ["metal", "plastic", "stone", "wood", "glass"])
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
    upload_glyphs(compositor, doc)
    target = RenderTarget(1600, 1000, samples=4)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        start = time.perf_counter()
        for i in range(20):
            doc.scene.objects[0].transform.rot_y = -40 + i
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        fps = 20 / (time.perf_counter() - start)
    finally:
        target.delete()
    assert fps > 30, f"{kind}: only {fps:.1f} FPS"



# ----- Procedural stone and wood ------------------------------------------------------------------


def textured_document(kind, **material):
    from text3d_studio.materials import PRESETS

    doc = material_document(**{**PRESETS[kind], "kind": kind, **material})
    doc.scene.objects[0].transform.rot_y = 0.0  # face the camera so the face fills most of the text
    doc.scene.objects[0].transform.size = 4.0
    doc.scene.light.reflect_photo = False
    return doc


def directional_detail(compositor, doc):
    """Mean luminance change between vertical and between horizontal neighbours inside the text."""
    pixels, interior = text_image(compositor, doc)
    luminance = pixels.sum(-1) / 3.0
    vertical = np.abs(np.diff(luminance, axis=0))[interior[1:] & interior[:-1]].mean()
    horizontal = np.abs(np.diff(luminance, axis=1))[interior[:, 1:] & interior[:, :-1]].mean()
    return vertical, horizontal


def test_stone_has_far_more_fine_detail_than_plain_material(compositor):
    stone = sum(directional_detail(compositor, textured_document("stone")))
    plain = sum(directional_detail(compositor, textured_document("stone", texture_strength=0.0, bump_strength=0.0)))
    assert stone > 2 * plain  # plain text still has bevel and edge shading, so 2x is a clear margin


def test_stone_veining_strength_darkens_more_of_the_surface(compositor):
    def dark_fraction(strength):
        doc = textured_document("stone", texture_strength=strength, bump_strength=0.0)
        luminance = text_pixels(compositor, doc).sum(-1)
        return np.mean(luminance < 0.75 * np.median(luminance))  # veins and dark speckle

    assert dark_fraction(1.0) > dark_fraction(0.3) + 0.06


def test_stone_bump_adds_detail(compositor):
    flat = sum(directional_detail(compositor, textured_document("stone", bump_strength=0.0)))
    bumpy = sum(directional_detail(compositor, textured_document("stone", bump_strength=1.0)))
    assert bumpy > 1.2 * flat


@pytest.mark.parametrize("angle, across", [(0.0, "vertical"), (90.0, "horizontal")])
def test_wood_rings_vary_across_the_grain(compositor, angle, across):
    vertical, horizontal = directional_detail(compositor, textured_document("wood", grain_angle=angle, bump_strength=0.0))
    if across == "vertical":
        assert vertical > 1.5 * horizontal  # horizontal grain: rings stack up the letters
    else:
        assert horizontal > 1.5 * vertical


def test_tighter_ring_spacing_means_more_rings(compositor):
    def ring_detail(spacing):
        return directional_detail(compositor, textured_document("wood", texture_scale=spacing, bump_strength=0.0))[0]

    assert ring_detail(0.05) > 1.5 * ring_detail(0.25)


def test_varnish_adds_a_sharp_highlight(compositor):
    def highlight(varnish):
        doc = textured_document("wood", varnish=varnish)
        # Light from the camera's direction puts the coat's mirror-sharp highlight on the flat face.
        doc.scene.light.azimuth, doc.scene.light.elevation = 0.0, 12.0
        return np.sort(text_pixels(compositor, doc).sum(-1))[-30:].mean()

    assert highlight(1.0) > highlight(0.0) + 150


# ----- Glass --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind, expected", [
    ("plastic", (0.0, 0.0, 0.0)),
    ("glass", (0.65 * 0.8, 0.65 * 0.9, 0.65 * 1.0)),
])
def test_shadow_color_lightens_and_tints_only_for_transmissive_materials(kind, expected):
    from text3d_studio.render.shadow_renderer import shadow_color

    doc = make_document(material={"kind": kind, "face_color": (0.8, 0.9, 1.0)}, shadow={"color": (0.0, 0.0, 0.0)})
    assert shadow_color(doc.scene.shadow, doc.scene.objects[0].material) == pytest.approx(expected)


def test_glass_shadow_keeps_a_custom_shadow_color_in_proportion():
    from text3d_studio.render.shadow_renderer import shadow_color

    doc = make_document(material={"kind": "glass", "face_color": (1.0, 1.0, 1.0)}, shadow={"color": (0.2, 0.4, 0.6)})
    assert shadow_color(doc.scene.shadow, doc.scene.objects[0].material) == pytest.approx((0.72, 0.79, 0.86))


def glass_document(kind="glass", background=None, **material):
    from text3d_studio.materials import PRESETS

    doc = material_document(**{**PRESETS[kind], "kind": kind, **material})
    doc.scene.objects[0].transform.rot_y = 0.0
    doc.scene.objects[0].transform.size = 4.0
    doc.scene.light.reflect_photo = False  # opaque materials then ignore the photo entirely
    if background is not None:
        doc.background = background
    return doc


def glass_interior(compositor):
    """Pixels well inside the text. Found from opaque text on a flat background: glass on a patterned
    background can't be told apart from it by color, and every material covers the same pixels."""
    _, interior = text_image(compositor, glass_document("plastic"), erode=3)
    return interior


def stripes(period=16):
    background = np.full((400, 640, 4), 255, np.uint8)
    background[:, (np.arange(640) // (period // 2)) % 2 == 1, :3] = 0
    return background


def luminance(pixels):
    return pixels[..., :3].astype(float).mean(-1)


def test_glass_shows_the_background_through_the_text_but_plastic_hides_it(compositor):
    interior = glass_interior(compositor)
    glass = luminance(render(compositor, glass_document(background=stripes())))[interior]
    plastic = luminance(render(compositor, glass_document("plastic", background=stripes())))[interior]
    assert glass.std() > 60
    assert glass.std() > 2 * plastic.std()  # plastic varies only with its own shading


def test_glass_bends_the_background_rather_than_just_tinting_it(compositor):
    # Pure tinting keeps every stripe exactly where it was; refraction moves bright stripes over dark ones.
    interior = glass_interior(compositor)
    background = stripes()
    glass = luminance(render(compositor, glass_document(background=background)))[interior]
    behind = luminance(np.flipud(background))[interior]  # exported rows are flipped relative to the photo
    assert np.mean((glass > 150) & (behind < 50)) > 0.05  # bright stripes seen over dark ones...
    assert np.mean((glass < 80) & (behind > 200)) > 0.05  # ...and dark over bright


def test_changing_the_photo_changes_glass_but_not_opaque_materials(compositor):
    interior = glass_interior(compositor)

    def text_colors(kind, rgb):
        doc = glass_document(kind)
        doc.background[..., :3] = rgb
        return render(compositor, doc)[..., :3].astype(float)[interior]

    for kind in ("plastic", "stone"):
        changed = np.any(text_colors(kind, (200, 180, 150)) != text_colors(kind, (60, 90, 230)), axis=-1)
        assert changed.mean() < 1e-3  # only stray diagonal edge pixels, which the erosion keeps, blend the photo
    glass_shift = text_colors("glass", (200, 180, 150)) - text_colors("glass", (60, 90, 230))
    assert glass_shift[:, 0].mean() > 60 and glass_shift[:, 2].mean() < 0


def test_frost_blurs_the_background_seen_through_glass(compositor):
    interior = glass_interior(compositor)

    def contrast(roughness):
        doc = glass_document(background=stripes(), roughness=roughness)
        return luminance(render(compositor, doc))[interior].std()

    assert contrast(0.6) < 0.4 * contrast(0.05)


def test_absorption_deepens_the_glass_tint(compositor):
    interior = glass_interior(compositor)
    tint = (0.6, 0.8, 1.0)

    def mean_color(absorption):
        doc = glass_document(background_rgb=(255, 255, 255), face_color=tint, side_color=tint, absorption=absorption)
        doc.background[..., :3] = 255
        return render(compositor, doc)[..., :3].astype(float)[interior].mean(axis=0)

    clear, light, deep = mean_color(0.0), mean_color(1.0), mean_color(4.0)
    assert clear[0] > 230 and abs(clear[0] - clear[2]) < 10  # no absorption: the white photo shows untinted
    assert deep[0] < light[0] - 20 < clear[0] - 20  # red is absorbed more with every step
    assert deep[2] > 230  # the tint passes blue untouched


def test_glass_casts_a_lighter_shadow_than_plastic(compositor):
    light = {"azimuth": -110, "elevation": 30}
    _, text_area = text_image(compositor, make_document(light=light, shadow={"enabled": False}), erode=0)
    ground = ~text_area

    def darkening(kind):
        from text3d_studio.materials import PRESETS

        material = {**PRESETS[kind], "kind": kind}
        lit = render(compositor, make_document(light=light, material=material, shadow={"enabled": False}))
        shaded = render(compositor, make_document(light=light, material=material, shadow={"softness": 0.1, "opacity": 0.8}))
        return (luminance(lit) - luminance(shaded))[ground].sum()

    assert 0 < darkening("glass") < 0.5 * darkening("plastic")


# ----- Text layer ---------------------------------------------------------------------------------


def read_text_layer(compositor):
    """The resolved text layer as an (H, W, 4) float array, top row first."""
    from OpenGL import GL

    layer = compositor.text_layer
    layer.color.bind(0)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    return np.frombuffer(data, np.uint8).reshape(layer.height, layer.width, 4)[::-1].astype(float)


@pytest.mark.parametrize("kind", ["metal", "glass"])
def test_text_layer_is_premultiplied_with_coverage_alpha(compositor, kind):
    from text3d_studio.materials import PRESETS

    render(compositor, make_document(material={**PRESETS[kind], "kind": kind}, shadow={"enabled": False}))
    layer = read_text_layer(compositor)
    alpha = layer[..., 3]
    assert alpha.max() == 255 and alpha.min() == 0  # solid interior, empty surroundings
    edges = (alpha > 0) & (alpha < 255)
    assert edges.sum() > 200  # antialiased outline
    assert np.all(layer[..., :3] <= alpha[..., None] + 1)  # premultiplied: color never exceeds coverage
    assert not layer[alpha == 0, :3].any()


def test_export_is_the_text_layer_composited_over_the_background(compositor):
    doc = make_document(shadow={"enabled": False})
    exported = render(compositor, doc)[..., :3].astype(float)
    layer = read_text_layer(compositor)
    background = doc.background[..., :3].astype(float)
    expected = layer[..., :3] + (1 - layer[..., 3:] / 255) * background
    assert np.abs(exported - expected).max() <= 1.0


def test_preview_without_multisampling_still_antialiases_text_edges(compositor):
    doc = grid_document()
    doc.background = make_document().background
    render_preview(compositor, doc, with_grid=False)  # into a target with no multisampling
    assert compositor.text_layer.samples >= 4
    alpha = read_text_layer(compositor)[..., 3]
    assert ((alpha > 0) & (alpha < 255)).sum() > 200


# ----- Multiple text objects ----------------------------------------------------------------------


def text_object(object_id, content="HI", material="plastic", **transform):
    from text3d_studio.materials import PRESETS
    from text3d_studio.model import MaterialParams, TextObject, TextParams, TransformParams

    return TextObject(
        id=object_id,
        text=TextParams(content=content),
        transform=TransformParams(**{"rot_y": 0.0, "size": 2.0, **transform}),
        material=MaterialParams(**{**PRESETS[material], "kind": material}),
    )


def scene_of(*objects, **overrides):
    doc = make_document(**{"shadow": {"enabled": False}, **overrides})
    doc.scene.objects = list(objects)
    return doc


def eroded(mask, erode=2):
    interior = mask.copy()
    for axis in (0, 1):
        for shift in range(1, erode + 1):
            interior &= np.roll(mask, shift, axis) & np.roll(mask, -shift, axis)
    return interior


@pytest.mark.parametrize("kind", ["plastic", "glass"])
def test_every_object_is_drawn(compositor, kind):
    left, right = text_object(0, x=-3.0), text_object(1, material=kind, x=3.0)
    alone_left = changed_mask(render(compositor, scene_of(left)))
    alone_right = changed_mask(render(compositor, scene_of(right)))
    both = changed_mask(render(compositor, scene_of(left, right)))
    assert alone_left.sum() > 2000 and alone_right.sum() > 2000
    assert not (alone_left & alone_right).any()  # side by side
    assert np.mean(both != (alone_left | alone_right)) < 1e-3


def test_a_nearer_object_hides_a_farther_one_whatever_the_order(compositor):
    far = text_object(0, content="WW", z=-16.0, size=3.0)
    near = text_object(1, content="O", material="metal", x=0.6, z=-9.0)
    near_alone = render(compositor, scene_of(near))
    far_alone = changed_mask(render(compositor, scene_of(far)))
    near_interior = eroded(changed_mask(near_alone))
    overlap = near_interior & far_alone
    assert overlap.sum() > 500
    for doc in (scene_of(far, near), scene_of(near, far)):
        both = render(compositor, doc)
        np.testing.assert_array_equal(both[near_interior], near_alone[near_interior])
        assert changed_mask(both)[far_alone & ~changed_mask(near_alone)].all()  # the rest of the far one shows


def test_every_object_casts_a_shadow(compositor):
    light = {"azimuth": -110, "elevation": 30}
    shadow = {"softness": 0.1, "opacity": 0.8}
    left, right = text_object(0, x=-4.0), text_object(1, material="metal", x=4.0)

    def shaded(*objects):
        lit = render(compositor, scene_of(*objects, light=light))
        dark = render(compositor, scene_of(*objects, light=light, shadow=shadow))
        return luminance(lit) - luminance(dark) > 10, changed_mask(lit)

    shadow_left, _ = shaded(left)
    shadow_right, _ = shaded(right)
    shadow_both, text_both = shaded(left, right)
    for alone in (shadow_left, shadow_right):
        alone = alone & ~text_both
        assert alone.sum() > 500
        assert (shadow_both & alone).sum() >= 0.98 * alone.sum()


def test_each_object_tints_its_own_shadow(compositor):
    light = {"azimuth": -110, "elevation": 30}
    shadow = {"softness": 0.1, "opacity": 0.8, "color": (0.0, 0.0, 0.0)}
    glass, solid = text_object(0, material="glass", x=-4.0), text_object(1, x=4.0)
    glass.material.face_color = (0.2, 0.4, 1.0)

    def darkening(*objects):
        lit = render(compositor, scene_of(*objects, light=light))
        dark = render(compositor, scene_of(*objects, light=light, shadow=shadow))
        return lit[..., :3].astype(float) - dark[..., :3].astype(float), changed_mask(lit)

    glass_only, _ = darkening(glass)
    solid_only, _ = darkening(solid)
    both, text = darkening(glass, solid)
    glass_shadow = (glass_only.sum(-1) > 30) & ~text
    solid_shadow = (solid_only.sum(-1) > 30) & ~text & ~glass_shadow
    assert glass_shadow.sum() > 300 and solid_shadow.sum() > 300
    glass_rgb, solid_rgb = both[glass_shadow].mean(0), both[solid_shadow].mean(0)
    assert glass_rgb[0] > glass_rgb[2] + 10  # blue light passes through the tinted glass, so it darkens red most
    assert abs(solid_rgb[0] - solid_rgb[2]) < 0.25 * solid_rgb.mean() + 5  # a black shadow darkens evenly
    np.testing.assert_allclose(both[glass_shadow], glass_only[glass_shadow], atol=2)  # unchanged by the other


def test_deleting_an_object_removes_it_from_the_render(compositor):
    keep, drop = text_object(0, x=-3.0), text_object(1, x=3.0)
    doc = scene_of(keep, drop)
    render(compositor, doc)
    assert set(compositor.glyphs) == {0, 1}
    doc.scene.objects.remove(drop)
    after = render(compositor, doc)  # frees the dropped object's glyph, like the canvas
    assert set(compositor.glyphs) == {0}
    np.testing.assert_array_equal(after, render(compositor, scene_of(keep)))


def test_an_object_without_ink_draws_nothing_and_hides_nothing(compositor):
    shown, empty = text_object(0), text_object(1, content="  ", z=-4.0, size=6.0)
    with_empty = render(compositor, scene_of(shown, empty))
    assert set(compositor.glyphs) == {0}
    np.testing.assert_array_equal(with_empty, render(compositor, scene_of(shown)))


def test_set_glyph_replaces_and_removes_an_objects_glyph(compositor):
    from text3d_studio.model import TextParams

    first, second = rasterize_text(TextParams(content="A")), rasterize_text(TextParams(content="BCD"))
    compositor.set_glyph(5, first)
    glyph = compositor.glyphs[5]
    compositor.set_glyph(5, second)
    assert compositor.glyphs[5] is glyph and compositor.glyph_raster(5) is second  # re-uploaded in place
    compositor.set_glyph(5, None)
    assert 5 not in compositor.glyphs and compositor.glyph_raster(5) is None
    compositor.set_glyph(6, None)  # removing what isn't there is fine
    assert compositor.glyphs == {}


def test_grid_lanes_follow_the_selected_object(compositor):
    doc = scene_of(text_object(0, x=-5.0), text_object(1, x=5.0, rot_y=35.0),
                   grid={"visible": True, "opacity": 1.0, "color": (0.0, 0.0, 1.0), "accent_color": (1.0, 0.0, 0.0)},
                   ground={"pitch": 10.0})
    plain = render_preview(compositor, doc, False).astype(int)

    def lane_columns(selected):
        moved = render_preview(compositor, doc, True, selected).astype(int) - plain
        accent = (moved[..., 0] > 20) & (moved[..., 2] < 0)  # redder: the accent, not the blue grid lines
        assert accent.sum() > 300
        return np.nonzero(accent)[1]

    left, right = lane_columns(0), lane_columns(1)
    width = plain.shape[1]
    assert np.median(left) < width / 2 < np.median(right)


def test_two_objects_stay_above_30_fps(compositor):
    from OpenGL import GL

    from text3d_studio.render.gl_utils import RenderTarget

    doc = scene_of(text_object(0, material="stone", x=-3.0, depth=0.8), text_object(1, material="glass", x=3.0),
                   width=1600, height=1000, shadow={"enabled": True})
    compositor.set_background(doc.background)
    upload_glyphs(compositor, doc)
    target = RenderTarget(1600, 1000, samples=4)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        start = time.perf_counter()
        for i in range(20):
            doc.scene.objects[0].transform.rot_y = -40 + i
            compositor.render(doc.scene, target.fbo, (0, 0, 1600, 1000))
        GL.glFinish()
        fps = 20 / (time.perf_counter() - start)
    finally:
        target.delete()
    assert fps > 30, f"only {fps:.1f} FPS"
