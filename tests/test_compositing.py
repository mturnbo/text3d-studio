"""Compositing passes on the real GPU: occlusion masks, wall and contact shadows, the ground reflection, bloom and
photo matching. Tests check rendered properties against control renders, never reference images."""

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


def make_document(width=640, height=400, **sections) -> Document:
    doc = Document()
    background = np.zeros((height, width, 4), np.uint8)
    background[..., :3] = BG_COLOR
    background[..., 3] = 255
    doc.background = background
    for section, values in sections.items():
        owner = doc.scene.objects[0] if section in ("text", "transform", "material") else doc.scene
        for name, value in values.items():
            setattr(getattr(owner, section), name, value)
    return doc


def render(compositor, doc, **options):
    compositor.set_background(doc.background)
    compositor.set_mask(doc.mask)
    compositor.retain_glyphs({obj.id for obj in doc.scene.objects})
    for obj in doc.scene.objects:
        compositor.set_glyph(obj.id, rasterize_text(obj.text))
    return render_document(compositor, doc, **options)


def changed(pixels, reference):
    return np.any(pixels[..., :3] != reference[..., :3], axis=-1)


def luminance(pixels):
    return pixels[..., :3].astype(np.float64) @ (0.2126, 0.7152, 0.0722)


def eroded(mask, erode=2):
    interior = mask.copy()
    for axis in (0, 1):
        for shift in range(1, erode + 1):
            interior &= np.roll(mask, shift, axis) & np.roll(mask, -shift, axis)
    return interior


def dilated(mask, grow):
    grown = mask.copy()
    for axis in (0, 1):
        for shift in range(1, grow + 1):
            grown |= np.roll(mask, shift, axis) | np.roll(mask, -shift, axis)
    return grown


NO_SHADOW = {"enabled": False}


def text_mask(compositor, **sections):
    """Where the text draws, from a render without shadow."""
    sections = {"shadow": NO_SHADOW, **sections}
    return changed(render(compositor, make_document(**sections)), np.array(BG_COLOR)[None, None])


# ----- Occlusion masks ----------------------------------------------------------------------------


def left_half_mask(doc):
    mask = np.zeros(doc.background.shape[:2], np.uint8)
    mask[:, : mask.shape[1] // 2] = 255
    return mask


def test_masked_pixels_show_the_photo_exactly_and_the_rest_is_unchanged(compositor):
    plain = make_document(shadow={"softness": 0.1})
    masked = make_document(shadow={"softness": 0.1})
    masked.mask = left_half_mask(masked)
    before, after = render(compositor, plain), render(compositor, masked)
    left = masked.mask == 255
    assert changed(before, after).any()
    assert np.all(after[left][:, :3] == BG_COLOR)  # text and shadow alike are covered
    assert changed(before, np.array(BG_COLOR)[None, None])[left].any()  # there was text or shadow there
    assert np.array_equal(after[:, after.shape[1] // 2 + 2:], before[:, before.shape[1] // 2 + 2:])


def test_an_empty_or_absent_mask_changes_nothing(compositor):
    doc = make_document()
    reference = render(compositor, doc)
    doc.mask = np.zeros(doc.background.shape[:2], np.uint8)
    assert np.array_equal(render(compositor, doc), reference)


def test_feathered_mask_values_blend_the_photo_over_the_text(compositor):
    inside = eroded(text_mask(compositor), 3)
    doc = make_document(shadow=NO_SHADOW)
    text = render(compositor, doc)
    doc.mask = np.full(doc.background.shape[:2], 128, np.uint8)
    half = render(compositor, doc).astype(int)
    expected = (text.astype(int) + np.array([*BG_COLOR, 255])) / 2
    assert np.abs(half[inside][:, :3] - expected[inside][:, :3]).max() <= 2


def test_a_mask_of_another_size_than_the_photo_is_ignored(compositor):
    doc = make_document()
    reference = render(compositor, doc)
    doc.mask = np.full((10, 10), 255, np.uint8)
    assert np.array_equal(render(compositor, doc), reference)


def test_an_oversized_mask_is_refused_and_leaves_no_mask(compositor):
    doc = make_document()
    reference = render(compositor, doc)
    compositor.set_mask(left_half_mask(doc))
    with pytest.raises(ValueError):
        compositor.set_mask(np.zeros((1, compositor.max_texture_size + 1), np.uint8))
    assert not compositor.has_mask
    compositor.set_background(doc.background)
    assert np.array_equal(render_document(compositor, doc), reference)


def test_transparent_export_leaves_masked_text_and_shadow_out(compositor):
    plain = make_document(light={"azimuth": -110, "elevation": 30})
    masked = make_document(light={"azimuth": -110, "elevation": 30})
    masked.mask = left_half_mask(masked)
    before = render(compositor, plain, transparent=True)
    after = render(compositor, masked, transparent=True)
    left = masked.mask == 255
    assert before[left][:, 3].any()
    assert not after[left][:, 3].any()
    right = slice(after.shape[1] // 2 + 2, None)
    assert np.array_equal(after[:, right], before[:, right])


def render_preview(compositor, doc, mask_overlay=False):
    """The composite (plus the Mask tool's overlay) in a single-sampled target, like the canvas draws it."""
    from OpenGL import GL

    from text3d_studio.render.gl_utils import RenderTarget

    height, width = doc.background.shape[:2]
    render(compositor, doc)  # uploads everything
    target = RenderTarget(width, height, 0)
    try:
        compositor.render(doc.scene, target.fbo, (0, 0, width, height))
        if mask_overlay:
            compositor.render_mask_overlay(target.fbo, (0, 0, width, height), (1.0, 0.0, 0.0, 0.5))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    finally:
        target.delete()
    return np.frombuffer(data, np.uint8).reshape(height, width, 4)[::-1].copy()


def test_the_mask_overlay_tints_only_masked_pixels_and_stays_out_of_the_export(compositor):
    doc = make_document()
    doc.mask = left_half_mask(doc)
    export = render(compositor, doc)
    plain, tinted = render_preview(compositor, doc), render_preview(compositor, doc, mask_overlay=True)
    differs = changed(plain, tinted)
    assert differs[doc.mask == 255].all()
    assert not differs[doc.mask == 0].any()
    assert tinted[..., 3].min() == 255
    assert np.array_equal(render(compositor, doc), export)


# ----- Shadows on the wall ------------------------------------------------------------------------

WALL_GROUND = {"pitch": 12.0, "height": 1.5}
WALL = {"angle": 0.0, "offset": 10.0, "enabled": True}
FRONT_LIGHT = {"azimuth": 25.0, "elevation": 35.0}  # on the camera's side of the wall
SHADOW = {"enabled": True, "opacity": 0.8, "softness": 0.05, "falloff": 0.0}


def text_object(object_id=0, content="HI", **transform):
    from text3d_studio.materials import PRESETS
    from text3d_studio.model import MaterialParams, TextObject, TextParams, TransformParams

    return TextObject(
        id=object_id,
        text=TextParams(content=content),
        transform=TransformParams(**{"rot_y": 0.0, "size": 2.0, **transform}),
        material=MaterialParams(**{**PRESETS["plastic"], "kind": "plastic"}),
    )


def wall_text(**transform):
    return text_object(mount="wall", **{"wall_x": 0.0, "wall_y": 1.0, "size": 1.6, "depth": 0.3, **transform})


def scene_of(*objects, **sections):
    doc = make_document(**{"ground": WALL_GROUND, "wall": WALL, **sections})
    doc.scene.objects = list(objects)
    return doc


def surfaces(doc):
    """Per pixel (top row first): True where the camera ray shows the floor in front of the wall, False for the wall."""
    from text3d_studio import ground, math3d

    scene = doc.scene
    width, height = doc.native_size
    focal = math3d.focal_length_px(scene.camera.fov, height)
    ys, xs = np.mgrid[0:height, 0:width] + 0.5
    rays = np.stack([(xs - width / 2) / focal, -(ys - height / 2) / focal, -np.ones_like(xs)], axis=-1)
    floor, wall = ground.ground_plane(scene.ground), ground.wall_plane(scene.ground, scene.wall)
    toward = rays @ floor[:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = rays * (-floor[3] / toward)[..., None]
    return (toward < 0) & (hits @ wall[:3] + wall[3] >= 0)


def image_point(doc, point):
    from text3d_studio import math3d

    u, v = math3d.project_to_image(point, doc.scene.camera.fov, doc.native_size)
    return int(v), int(u)


def floor_shadow_of(doc, point):
    """Where a point's shadow lands on the (infinite) floor, as a pixel (row, column)."""
    from text3d_studio import ground
    from text3d_studio.lighting import LightingEngine

    light = LightingEngine().compute(doc.scene.light, doc.scene.ground).light_dir
    plane = ground.ground_plane(doc.scene.ground)
    point = np.asarray(point, float)
    t = (plane[:3] @ point + plane[3]) / (plane[:3] @ light)
    return image_point(doc, point - t * light)


def darkening(compositor, doc):
    """How much darker each pixel is than with the shadow off, and where the text is."""
    import dataclasses

    unshadowed = make_document()
    unshadowed.scene = dataclasses.replace(doc.scene, shadow=dataclasses.replace(doc.scene.shadow, enabled=False))
    lit = render(compositor, unshadowed)
    return luminance(lit) - luminance(render(compositor, doc)), changed(lit, np.array(BG_COLOR)[None, None])


def test_wall_text_darkens_the_wall_around_it_but_not_the_floor_behind_the_wall_line(compositor):
    from text3d_studio import ground

    doc = scene_of(wall_text(), light=FRONT_LIGHT, shadow=SHADOW)
    darker, text = darkening(compositor, doc)
    on_floor = surfaces(doc)
    around = dilated(text, 12) & ~dilated(text, 1)
    assert (darker[around & ~on_floor] > 8).sum() > 100  # the wall around the text
    # The floor shadow would land behind the wall, where the wall shows: it must not appear there.
    scene = doc.scene
    middle = (ground.wall_to_world(scene.ground, scene.wall, 0.0, 1.0 + 0.4 * 1.6)
              + 0.3 * 1.6 * ground.wall_normal(scene.ground, scene.wall))
    row, column = floor_shadow_of(doc, middle)
    assert 0 <= row < 400 and 0 <= column < 640 and not on_floor[row, column]
    legacy, _ = darkening(compositor, scene_of(wall_text(), light=FRONT_LIGHT, shadow=SHADOW,
                                               wall={**WALL, "enabled": False}))
    window = (slice(row - 2, row + 3), slice(column - 2, column + 3))
    assert legacy[window].mean() > 20  # without the wall, the floor shadow showed there
    assert darker[window].mean() < 2


def test_floor_text_near_the_wall_shadows_the_floor_as_before_and_climbs_the_wall(compositor):
    objects = (text_object(z=-8.5, size=2.0),)
    light = {"azimuth": 10.0, "elevation": 25.0}  # the shadow runs back toward the wall and reaches it
    with_wall = render(compositor, scene_of(*objects, light=light, shadow=SHADOW))
    legacy = render(compositor, scene_of(*objects, light=light, shadow=SHADOW, wall={**WALL, "enabled": False}))
    doc = scene_of(*objects, light=light, shadow=SHADOW)
    darker, text = darkening(compositor, doc)
    on_floor = surfaces(doc)
    boundary = dilated(on_floor, 2) & dilated(~on_floor, 2)
    floor = on_floor & ~boundary
    diff = np.abs(with_wall[floor].astype(int) - legacy[floor])
    assert diff.max() <= 1  # the floor's shadow is the same, to rounding
    assert (darker[~on_floor & ~text & ~boundary] > 8).sum() > 100  # and it carries on up the wall


def test_a_light_behind_the_wall_casts_nothing_on_it(compositor):
    behind = {"azimuth": 180.0, "elevation": 35.0}
    doc = scene_of(wall_text(), light=behind, shadow=SHADOW)
    darker, text = darkening(compositor, doc)
    assert not (darker[~surfaces(doc) & ~dilated(text, 1)] > 1).any()


def test_without_a_wall_in_the_photo_the_wall_changes_nothing(compositor):
    objects = (text_object(z=-8.5),)
    first = render(compositor, scene_of(*objects, light=FRONT_LIGHT, shadow=SHADOW, wall={**WALL, "enabled": False}))
    moved = render(compositor, scene_of(*objects, light=FRONT_LIGHT, shadow=SHADOW,
                                        wall={"angle": 30.0, "offset": 9.0, "enabled": False}))
    assert np.array_equal(first, moved)


def test_the_wall_shadow_is_in_the_transparent_export(compositor):
    doc = scene_of(wall_text(), light=FRONT_LIGHT, shadow=SHADOW)
    darker, text = darkening(compositor, doc)
    pixels = render(compositor, doc, transparent=True)
    shadowed = (darker > 15) & ~dilated(text, 1) & ~surfaces(doc)
    assert shadowed.sum() > 50
    assert (pixels[shadowed][:, 3] > 20).all()


# ----- Contact shadow -----------------------------------------------------------------------------

CONTACT_GROUND = {"pitch": 15.0, "height": 1.5}


def contact_document(**shadow):
    shadow = {"enabled": True, "opacity": 0.0, "contact_strength": 1.0, "contact_size": 0.15, **shadow}
    return make_document(ground=CONTACT_GROUND, transform={"z": -8.0, "rot_y": -20.0}, shadow=shadow)


def test_zero_contact_strength_is_identical_to_before(compositor):
    reference = render(compositor, make_document(ground=CONTACT_GROUND))
    unused = render(compositor, make_document(ground=CONTACT_GROUND, shadow={"contact_strength": 0.0, "contact_size": 0.9}))
    assert np.array_equal(reference, unused)
    off = render(compositor, contact_document(enabled=False))
    assert np.array_equal(off, render(compositor, contact_document(enabled=False, contact_strength=0.0)))


def test_the_contact_shadow_darkens_tightly_around_the_base(compositor):
    darker, text = darkening(compositor, contact_document())
    shadowed = (darker > 6) & ~text
    assert shadowed.sum() > 150
    rows = np.flatnonzero(text.any(axis=1))
    top, base = rows.min(), rows.max()
    shadow_rows = np.nonzero(shadowed)[0]
    assert (shadow_rows > base - 0.3 * (base - top)).mean() > 0.95  # by the base, not up the text
    assert shadow_rows.max() <= base + 12  # and only a few pixels out
    # Darkest right at the base, falling off below it.
    columns = text[base]
    assert darker[base + 2, columns].mean() > darker[base + 8, columns].mean()


def test_a_larger_contact_size_spreads_further(compositor):
    small, text = darkening(compositor, contact_document(contact_size=0.05))
    large, _ = darkening(compositor, contact_document(contact_size=0.5))
    assert ((large > 6) & ~text).sum() > ((small > 6) & ~text).sum()


def test_wall_text_gets_no_contact_shadow(compositor):
    doc = scene_of(wall_text(), shadow={"enabled": True, "opacity": 0.0, "contact_strength": 1.0}, wall={**WALL, "enabled": False})
    darker, _ = darkening(compositor, doc)
    assert not (darker > 1).any()


def test_the_contact_shadow_is_in_the_transparent_export(compositor):
    doc = contact_document()
    darker, text = darkening(compositor, doc)
    pixels = render(compositor, doc, transparent=True)
    shadowed = (darker > 15) & ~dilated(text, 1)
    assert shadowed.sum() > 20
    assert (pixels[shadowed][:, 3] > 20).all()
    assert not pixels[~dilated((darker > 0) | text, 3)][:, 3].any()  # nothing where nothing darkened


# ----- Ground reflection --------------------------------------------------------------------------

REFLECTION_GROUND = {"pitch": 8.0, "height": 1.5}


def reflection_document(**effects):
    effects = {"reflection_strength": 1.0, "reflection_roughness": 0.0, "reflection_fade": 3.0, **effects}
    return make_document(ground=REFLECTION_GROUND, transform={"z": -9.0, "rot_y": -10.0}, shadow=NO_SHADOW,
                         effects=effects)


def reflection_change(compositor, **effects):
    """How much each pixel changed from the reflection-free render, and where the text is."""
    plain = render(compositor, reflection_document(reflection_strength=0.0))
    shown = render(compositor, reflection_document(**effects))
    text = changed(plain, np.array(BG_COLOR)[None, None])
    return np.abs(shown[..., :3].astype(int) - plain[..., :3]).sum(-1), text


def test_zero_reflection_strength_is_identical_to_before(compositor):
    reference = render(compositor, make_document(ground=REFLECTION_GROUND))
    unused = render(compositor, make_document(ground=REFLECTION_GROUND, effects={
        "reflection_strength": 0.0, "reflection_roughness": 1.0, "reflection_fade": 0.5}))
    assert np.array_equal(reference, unused)


def test_the_reflection_appears_below_the_text_and_fades_with_depth(compositor):
    change, text = reflection_change(compositor)
    rows = np.flatnonzero(text.any(axis=1))
    top, base = rows.min(), rows.max()
    reflected = (change > 6) & ~text
    reflected_rows = np.nonzero(reflected)[0]
    assert reflected.sum() > 500
    assert np.median(reflected_rows) > base  # below the text, on the floor
    assert not (change[:top] > 0).any()
    height = base - top
    near = change[base + 1:base + height // 3].mean()
    far = change[base + 2 * height // 3:base + height].mean()
    assert near > 2 * far


def test_a_short_fade_shortens_the_reflection(compositor):
    long, text = reflection_change(compositor, reflection_fade=6.0)
    short, _ = reflection_change(compositor, reflection_fade=0.5)
    assert ((short > 6) & ~text).sum() < 0.6 * ((long > 6) & ~text).sum()


def test_roughness_blurs_the_reflection(compositor):
    sharp, text = reflection_change(compositor)
    rough, _ = reflection_change(compositor, reflection_roughness=1.0)
    below = ~dilated(text, 2)
    below[: np.flatnonzero(text.any(axis=1)).max()] = False

    def detail(change):
        return np.abs(np.diff(change.astype(float), axis=1))[below[:, 1:]].mean()

    assert detail(rough) < 0.75 * detail(sharp)


def test_wall_text_is_not_reflected_in_the_floor(compositor):
    plain = scene_of(wall_text(), wall={**WALL, "enabled": False}, shadow=NO_SHADOW)
    reflected = scene_of(wall_text(), wall={**WALL, "enabled": False}, shadow=NO_SHADOW,
                         effects={"reflection_strength": 1.0})
    assert np.array_equal(render(compositor, plain), render(compositor, reflected))


def test_the_reflection_is_in_the_transparent_export_under_the_text(compositor):
    change, text = reflection_change(compositor)
    pixels = render(compositor, reflection_document(), transparent=True)
    reflected = (change > 30) & ~dilated(text, 2)
    assert reflected.sum() > 100
    assert (pixels[reflected][:, 3] > 0).all()
    assert (pixels[eroded(text, 3)][:, 3] == 255).mean() > 0.999  # the text stays opaque over it
    assert not pixels[~dilated((change > 0) | text, 2)][:, 3].any()
