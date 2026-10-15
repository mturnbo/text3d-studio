import numpy as np
import pytest

from text3d_studio import math3d
from text3d_studio.lighting import MIN_SHADOW_ELEVATION, LightingEngine
from text3d_studio.model import LightParams, TransformParams


def test_light_direction_is_unit_and_scaled_colors():
    light = LightParams(intensity=2.0, color=(1, 0.5, 0.25), ambient_color=(1, 1, 1), ambient_intensity=0.3)
    state = LightingEngine().compute(light, TransformParams())
    assert np.linalg.norm(state.light_dir) == pytest.approx(1.0)
    np.testing.assert_allclose(state.light_color, [2.0, 1.0, 0.5])
    np.testing.assert_allclose(state.ambient, [0.3, 0.3, 0.3])


def test_light_follows_ground_tilt_but_not_text_yaw():
    light = LightParams(azimuth=40, elevation=30)
    engine = LightingEngine()
    flat = engine.compute(light, TransformParams(rot_x=0, rot_y=0))
    yawed = engine.compute(light, TransformParams(rot_x=0, rot_y=90))
    tilted = engine.compute(light, TransformParams(rot_x=25, rot_y=0))
    np.testing.assert_allclose(flat.light_dir, yawed.light_dir)
    expected = math3d.rotation_x(25)[:3, :3] @ flat.light_dir
    np.testing.assert_allclose(tilted.light_dir, expected, atol=1e-12)


def test_ground_plane_passes_through_text_origin():
    t = TransformParams(x=2, y=-3, z=-15, rot_x=10, rot_z=-5)
    plane = LightingEngine().compute(LightParams(), t).ground_plane
    assert plane[:3] @ np.array([t.x, t.y, t.z]) + plane[3] == pytest.approx(0.0, abs=1e-12)


def test_grazing_light_shadow_stays_finite():
    state = LightingEngine().compute(LightParams(elevation=0.0), TransformParams())
    point = state.shadow_matrix @ np.array([0.0, 1.0, -12.0, 1.0])
    assert abs(point[3]) > 1e-3
    assert np.all(np.isfinite(point[:3] / point[3]))
    assert MIN_SHADOW_ELEVATION > 0
