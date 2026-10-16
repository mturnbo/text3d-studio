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
    x, z, rot_y, size = ground.reanchor_text(old, new, t)
    moved = TransformParams(x=x, z=z, rot_y=rot_y, size=size)
    np.testing.assert_allclose(anchor_direction(new, moved), anchor_direction(old, t), atol=1e-9)
    # Apparent size ~ size / distance along the ray: unchanged.
    old_distance = np.linalg.norm(ground.text_position(old, t))
    new_distance = np.linalg.norm(ground.text_position(new, moved))
    assert size / new_distance == pytest.approx(t.size / old_distance)


def test_reanchoring_after_yaw_keeps_world_position_and_facing():
    old, new = GroundParams(yaw=0.0), GroundParams(yaw=30.0)
    t = TransformParams(x=2.0, z=-10.0, rot_y=-20.0)
    x, z, rot_y, size = ground.reanchor_text(old, new, t)
    moved = TransformParams(x=x, z=z, rot_y=rot_y, size=size)
    assert size == pytest.approx(t.size)  # yaw alone doesn't move the anchor, so nothing scales
    np.testing.assert_allclose(ground.text_model_matrix(new, moved), ground.text_model_matrix(old, t), atol=1e-9)


def test_reanchor_fails_when_the_anchor_would_be_above_the_new_horizon():
    old = GroundParams(pitch=0.0)
    t = TransformParams(z=-60.0)  # far away, just below the horizon
    assert ground.reanchor_text(old, GroundParams(pitch=-10.0), t) is None


def test_reanchoring_there_and_back_restores_the_placement():
    a, b = GroundParams(pitch=0.0), GroundParams(pitch=10.0)
    t = TransformParams(x=1.5, z=-14.0, rot_y=-20.0, size=3.0)
    x, z, rot_y, size = ground.reanchor_text(a, b, t)
    back = ground.reanchor_text(b, a, TransformParams(x=x, z=z, rot_y=rot_y, size=size))
    np.testing.assert_allclose(back, (1.5, -14.0, -20.0, 3.0), atol=1e-9)


IMAGE = (1200, 800)


def same_grid_phase(a, b, cell):
    """Offsets a and b put grid lines in the same places (they differ by a whole number of cells)."""
    turns = (a - b) / cell
    return abs(turns - round(turns)) < 1e-6


def project_ground_rectangle(g, fov, corner=(1.2, -9.0), size=(2.0, 3.0)):
    """Pixel corners of a size[0] x size[1] rectangle lying on ground `g`, edge 0-1 along the grid's x."""
    x0, z0 = corner
    w, d = size
    corners = [(x0, z0), (x0 + w, z0), (x0 + w, z0 - d), (x0, z0 - d)]
    return [math3d.project_to_image(ground.ground_to_world(g, x, z), fov, IMAGE) for x, z in corners]


@pytest.mark.parametrize(
    "pose", [dict(pitch=12, yaw=30, roll=-4, height=1.7), dict(pitch=3, yaw=-50, roll=8, height=4.0), dict(pitch=25, yaw=0, roll=0, height=1.2)]
)
def test_rectangle_fit_recovers_the_ground(pose):
    g = GroundParams(**pose)
    fit = ground.fit_ground_to_rectangle(project_ground_rectangle(g, 45.0), (2.0, 3.0), 45.0, IMAGE, cell_size=0.5)
    np.testing.assert_allclose((fit.pitch, fit.yaw, fit.roll, fit.height), (g.pitch, g.yaw, g.roll, g.height), atol=1e-6)
    assert fit.fov == 45.0
    assert same_grid_phase(fit.offset_x, 1.2, 0.5) and same_grid_phase(fit.offset_z, -9.0, 0.5)


@pytest.mark.parametrize("fov", [30.0, 55.0, 80.0])
def test_rectangle_fit_can_estimate_field_of_view(fov):
    g = GroundParams(pitch=15, yaw=35, roll=3, height=1.6)
    quad = project_ground_rectangle(g, fov)
    fit = ground.fit_ground_to_rectangle(quad, (2.0, 3.0), 40.0, IMAGE, cell_size=1.0, estimate_fov=True)
    assert fit.fov == pytest.approx(fov, abs=1e-6)
    assert fit.pitch == pytest.approx(15, abs=1e-6) and fit.height == pytest.approx(1.6, abs=1e-6)


def test_fitted_grid_lines_run_through_the_rectangle_corner():
    g = GroundParams(pitch=10, yaw=20, roll=0, height=2.0)
    fit = ground.fit_ground_to_rectangle(project_ground_rectangle(g, 40.0), (2.0, 3.0), 40.0, IMAGE, cell_size=0.75)
    fitted = GroundParams(pitch=fit.pitch, yaw=fit.yaw, roll=fit.roll, height=fit.height)
    corner = ground.ground_to_world(g, 1.2, -9.0)
    x, z = ground.world_to_ground(fitted, corner)
    for value, offset in ((x, fit.offset_x), (z, fit.offset_z)):
        assert ((value - offset) / 0.75) == pytest.approx(round((value - offset) / 0.75), abs=1e-6)


def test_crossed_corners_cannot_be_fitted():
    a, b, c, d = project_ground_rectangle(GroundParams(pitch=10, yaw=20), 40.0)
    with pytest.raises(ValueError):
        ground.fit_ground_to_rectangle([a, c, b, d], (2.0, 3.0), 40.0, IMAGE, 1.0)


def test_fit_beyond_the_supported_roll_is_rejected():
    quad = project_ground_rectangle(GroundParams(pitch=10, yaw=20, roll=70), 40.0)
    with pytest.raises(ValueError, match="tilted"):
        ground.fit_ground_to_rectangle(quad, (2.0, 3.0), 40.0, IMAGE, 1.0)


def test_fov_estimate_fails_for_a_square_on_rectangle():
    quad = project_ground_rectangle(GroundParams(pitch=10, yaw=0, roll=0), 40.0)  # front edge stays horizontal
    with pytest.raises(ValueError):
        ground.fit_ground_to_rectangle(quad, (2.0, 3.0), 40.0, IMAGE, 1.0, estimate_fov=True)
