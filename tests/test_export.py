"""Export variants: transparent PNGs and scaled exports, rendered on the real GPU."""

import numpy as np
import pytest

from text3d_studio.materials import PRESETS
from text3d_studio.model import Document
from text3d_studio.render.compositor import Compositor
from text3d_studio.render.exporter import compose_transparent, render_document
from text3d_studio.render.shadow_renderer import shadow_color
from text3d_studio.text_raster import rasterize_text

BG_COLOR = (200, 180, 150)
SHADOW_LIGHT = {"azimuth": -110, "elevation": 30}


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
    compositor.retain_glyphs({obj.id for obj in doc.scene.objects})
    for obj in doc.scene.objects:
        compositor.set_glyph(obj.id, rasterize_text(obj.text))
    return render_document(compositor, doc, **options)


def eroded(mask, erode=3):
    interior = mask.copy()
    for axis in (0, 1):
        for shift in range(1, erode + 1):
            interior &= np.roll(mask, shift, axis) & np.roll(mask, -shift, axis)
    return interior


def text_only_mask(compositor, doc):
    """Where the text is, from an export without shadow."""
    no_shadow = make_document(**{"light": SHADOW_LIGHT, "shadow": {"enabled": False}})
    no_shadow.scene.objects = doc.scene.objects
    return np.any(render(compositor, no_shadow)[..., :3] != BG_COLOR, axis=-1)


# ----- Transparent export -------------------------------------------------------------------------


def test_transparent_export_is_empty_away_from_the_text_and_shadow(compositor):
    doc = make_document(light=SHADOW_LIGHT, shadow={"softness": 0.1, "opacity": 0.8})
    normal = render(compositor, doc)
    transparent = render(compositor, doc, transparent=True)
    assert transparent.shape == normal.shape
    untouched = np.all(normal[..., :3] == BG_COLOR, axis=-1)
    assert untouched.mean() > 0.5
    assert transparent[untouched, 3].max() <= 2  # the shadow's faintest tail rounds away over the photo
    far = eroded(untouched, erode=10)
    assert far.mean() > 0.5
    assert not transparent[far].any()  # alpha 0, and no stray color under it
    assert (transparent[..., 3] > 0).sum() >= (~untouched).sum()


@pytest.mark.parametrize("kind", ["metal", "plastic", "stone"])
def test_transparent_text_interior_is_opaque_with_the_normal_colors(compositor, kind):
    doc = make_document(material={**PRESETS[kind], "kind": kind}, shadow={"enabled": False})
    normal = render(compositor, doc)
    transparent = render(compositor, doc, transparent=True)
    interior = eroded(np.any(normal[..., :3] != BG_COLOR, axis=-1))
    assert interior.sum() > 1000
    opaque = interior & (transparent[..., 3] == 255)
    assert opaque.sum() >= 0.999 * interior.sum()  # the odd interior pixel is only partly covered in both
    assert np.abs(transparent[opaque, :3].astype(int) - normal[opaque, :3].astype(int)).max() <= 1
    alpha = transparent[..., 3:] / 255
    over = transparent[..., :3] * alpha + np.array(BG_COLOR) * (1 - alpha)
    assert np.abs(over - normal[..., :3]).max() <= 2  # edges too, once put back over the photo


def test_transparent_shadow_has_the_shadow_color_at_partial_alpha(compositor):
    shadow = {"softness": 0.1, "opacity": 0.6, "color": (0.30, 0.10, 0.50)}
    doc = make_document(light=SHADOW_LIGHT, shadow=shadow)
    transparent = render(compositor, doc, transparent=True)
    shadow_only = (transparent[..., 3] > 0) & ~text_only_mask(compositor, doc)
    alpha = transparent[shadow_only, 3]
    assert shadow_only.sum() > 500
    assert alpha.max() <= round(0.6 * 255) + 1  # alpha is opacity × strength, never above the opacity
    assert np.all(alpha < 255)
    solid = shadow_only & (transparent[..., 3] > 60)
    assert solid.sum() > 200
    expected = np.array(shadow["color"]) * 255
    assert np.abs(transparent[solid, :3] - expected).max() <= 4


def test_transparent_shadow_over_the_photo_reproduces_the_normal_export(compositor):
    doc = make_document(light=SHADOW_LIGHT, shadow={"softness": 0.2, "opacity": 0.7})
    normal = render(compositor, doc).astype(float)
    transparent = render(compositor, doc, transparent=True).astype(float)
    alpha = transparent[..., 3:] / 255
    over = transparent[..., :3] * alpha + np.array(BG_COLOR) * (1 - alpha)
    shadow_only = (transparent[..., 3] > 0) & ~text_only_mask(compositor, doc)
    # The photo's own color shows through a multiplied shadow, so alpha-over matches it only roughly.
    assert np.abs(over[shadow_only] - normal[shadow_only, :3]).mean() < 12


def test_glass_shadow_keeps_its_tint_in_the_transparent_export(compositor):
    glass = {**PRESETS["glass"], "kind": "glass", "face_color": (0.2, 0.9, 0.3)}
    doc = make_document(light=SHADOW_LIGHT, shadow={"softness": 0.1, "opacity": 0.8}, material=glass)
    transparent = render(compositor, doc, transparent=True)
    solid = (transparent[..., 3] > 40) & ~text_only_mask(compositor, doc)
    expected = np.array(shadow_color(doc.scene.shadow, doc.scene.objects[0].material)) * 255
    assert solid.sum() > 200
    assert np.abs(np.median(transparent[solid, :3], axis=0) - expected).max() <= 4


def test_glass_still_refracts_the_photo_in_a_transparent_export(compositor):
    doc = make_document(material={**PRESETS["glass"], "kind": "glass"}, shadow={"enabled": False})
    normal = render(compositor, doc)
    transparent = render(compositor, doc, transparent=True)
    interior = eroded(np.any(normal[..., :3] != BG_COLOR, axis=-1))
    opaque = interior & (transparent[..., 3] == 255)
    assert opaque.sum() >= 0.999 * interior.sum()
    assert np.abs(transparent[opaque, :3].astype(int) - normal[opaque, :3].astype(int)).max() <= 1


def test_transparent_export_without_text_is_fully_transparent(compositor):
    transparent = render(compositor, make_document(text={"content": ""}), transparent=True)
    assert not transparent.any()


def test_transparent_export_with_shadow_off_is_only_the_text(compositor):
    doc = make_document(light=SHADOW_LIGHT, shadow={"enabled": False})
    transparent = render(compositor, doc, transparent=True)
    assert np.array_equal(transparent[..., 3] > 0, text_only_mask(compositor, doc))


def test_compose_transparent_unpremultiplies_text_edges():
    text = np.array([[[100, 50, 25, 128]]], np.uint8)  # half-covered edge pixel, premultiplied
    pixels = compose_transparent(text, np.ones((1, 1, 3)), np.ones((1, 1, 1)))
    assert pixels[0, 0, 3] == 128
    assert np.abs(pixels[0, 0, :3].astype(int) - np.array([199, 100, 50])).max() <= 1


def test_compose_transparent_puts_text_over_its_shadow():
    text = np.array([[[0, 0, 0, 0], [255, 0, 0, 255]]], np.uint8)
    strength = np.full((1, 2, 1), 0.5)  # alpha 0.5
    factor = 1 - 0.5 * (1 - np.array([0.2, 0.4, 0.6]))  # shadow color (0.2, 0.4, 0.6)
    pixels = compose_transparent(text, np.broadcast_to(factor, (1, 2, 3)), strength)
    assert pixels[0, 0, 3] == 128
    assert np.abs(pixels[0, 0, :3].astype(int) - np.array([51, 102, 153])).max() <= 1
    assert list(pixels[0, 1]) == [255, 0, 0, 255]  # opaque text hides the shadow
