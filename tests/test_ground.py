import numpy as np
import pytest

from text3d_studio import ground, math3d
from text3d_studio.model import GroundParams, TransformParams


def legacy_model_matrix(x, y, z, rot_x, rot_y, rot_z, size):
    """v0.5.0 placement, where the ground was attached to the text."""
    return (
        math3d.translation(x, y, z)
        @ math3d.ground_frame(rot_x, rot_z)
        @ math3d.rotation_y(rot_y)
        @ math3d.scaling(size)
    )


def legacy_to_ground_model(x, y, z, rot_x, rot_y, rot_z):
    """Converts a v0.5.0 transform (ground attached to the text) into ground model B params."""
    g = GroundParams(pitch=rot_x, roll=rot_z, yaw=0.0, height=1.0)
    up = ground.ground_up(g)
    g.height = -float(up @ np.array([x, y, z]))
    gx, gz = ground.world_to_ground(g, (x, y, z))
    return g, TransformParams(x=gx, z=gz, rot_y=rot_y)


@pytest.mark.parametrize(
    "legacy",
    [(0, -1.5, -12, 0, -20, 0), (2, -3, -15, 10, 45, -5), (-4, -0.5, -30, 25, 170, 8), (1, -6, -9, -5, 0, 2)],
)
def test_model_matrix_matches_legacy_transform(legacy):
    x, y, z, rot_x, rot_y, rot_z = legacy
    g, t = legacy_to_ground_model(*legacy)
    t.size = 2.5
    expected = legacy_model_matrix(x, y, z, rot_x, rot_y, rot_z, 2.5)
    np.testing.assert_allclose(ground.text_model_matrix(g, t), expected, atol=1e-9)


def test_default_scene_matches_v050_defaults():
    g, t = GroundParams(), TransformParams()
    expected = legacy_model_matrix(0, -1.5, -12, 0, -20, 0, t.size)
    np.testing.assert_allclose(ground.text_model_matrix(g, t), expected)


@pytest.mark.parametrize("pitch, roll, yaw", [(0, 0, 0), (25, -10, 40), (-15, 30, 170)])
def test_camera_sits_height_above_the_ground(pitch, roll, yaw):
    g = GroundParams(pitch=pitch, roll=roll, yaw=yaw, height=2.75)
    plane = ground.ground_plane(g)
    assert plane[3] == pytest.approx(2.75)  # signed distance of the camera (origin) from the plane
    np.testing.assert_allclose(np.linalg.norm(plane[:3]), 1.0)


def test_ground_coordinates_round_trip():
    g = GroundParams(pitch=20, roll=5, yaw=35, height=1.8)
    point = ground.ground_to_world(g, 3.2, -14.0)
    assert ground.ground_plane(g)[:3] @ point + ground.ground_plane(g)[3] == pytest.approx(0, abs=1e-12)
    assert ground.world_to_ground(g, point) == pytest.approx((3.2, -14.0))


def test_text_stands_on_the_ground_for_any_yaw():
    g = GroundParams(pitch=12, roll=-4, yaw=60, height=2.0)
    for rot_y in (0, 90, -135):
        m = ground.text_model_matrix(g, TransformParams(x=1, z=-10, rot_y=rot_y))
        base = m @ np.array([0.7, 0.0, -0.3, 1.0])  # any point at the text's local y = 0
        assert ground.ground_plane(g)[:3] @ base[:3] + ground.ground_plane(g)[3] == pytest.approx(0, abs=1e-9)


def test_grid_yaw_rotates_text_placement_but_not_the_light():
    still, turned = GroundParams(yaw=0), GroundParams(yaw=90)
    np.testing.assert_allclose(ground.light_frame(still), ground.light_frame(turned))
    np.testing.assert_allclose(ground.ground_to_world(turned, 0, -5), ground.ground_to_world(still, -5, 0), atol=1e-12)


def test_text_distance_is_positive_depth():
    assert ground.text_distance(GroundParams(), TransformParams(z=-12)) == pytest.approx(12)
    assert ground.text_distance(GroundParams(), TransformParams(z=5)) == pytest.approx(1e-3)


def anchor_direction(g, t):
    p = ground.text_position(g, t)
    return p / np.linalg.norm(p)


@pytest.mark.parametrize(
    "change", [{"pitch": 9.0}, {"roll": -6.0}, {"height": 3.0}, {"yaw": 40.0}, {"pitch": -3.0, "yaw": -25.0, "roll": 4.0}]
)
def test_reanchored_text_stays_on_the_same_camera_ray(change):
    old = GroundParams(pitch=2.0, yaw=5.0)
    t = TransformParams(x=1.5, z=-14.0, rot_y=-20.0)
    new = GroundParams(**{**vars(old), **change})
    x, z, rot_y = ground.reanchor_text(old, new, t)
    moved = TransformParams(x=x, z=z, rot_y=rot_y)
    np.testing.assert_allclose(anchor_direction(new, moved), anchor_direction(old, t), atol=1e-9)


def test_reanchoring_after_yaw_keeps_world_position_and_facing():
    old, new = GroundParams(yaw=0.0), GroundParams(yaw=30.0)
    t = TransformParams(x=2.0, z=-10.0, rot_y=-20.0)
    x, z, rot_y = ground.reanchor_text(old, new, t)
    moved = TransformParams(x=x, z=z, rot_y=rot_y)
    np.testing.assert_allclose(ground.text_model_matrix(new, moved), ground.text_model_matrix(old, t), atol=1e-9)


def test_reanchor_fails_when_the_anchor_would_be_above_the_new_horizon():
    old = GroundParams(pitch=0.0)
    t = TransformParams(z=-60.0)  # far away, just below the horizon
    assert ground.reanchor_text(old, GroundParams(pitch=-10.0), t) is None
