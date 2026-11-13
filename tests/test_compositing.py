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
