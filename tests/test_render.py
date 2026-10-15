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

