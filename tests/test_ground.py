import numpy as np
import pytest

from text3d_studio import ground, math3d
from text3d_studio.model import GroundParams, TransformParams, WallParams


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


@pytest.mark.parametrize("pitch, yaw, rot_y", [(0.0, 0.0, -20.0), (25.0, 40.0, 70.0), (-5.0, -15.0, 0.0)])
def test_face_hit_finds_the_local_point_a_ray_sees(pitch, yaw, rot_y):
    g = GroundParams(pitch=pitch, yaw=yaw, height=1.8)
    t = TransformParams(x=1.0, z=-9.0, rot_y=rot_y, size=2.5)
    point = ground.text_model_matrix(g, t) @ np.array([0.3, 0.4, 0.0, 1.0])  # on the front face, in em units
    distance, x, y = ground.text_face_hit(g, t, point[:3] / np.linalg.norm(point[:3]))
    assert (x, y) == pytest.approx((0.3, 0.4), abs=1e-9)
    assert distance == pytest.approx(np.linalg.norm(point[:3]))


def test_face_hit_misses_rays_parallel_to_or_away_from_the_face():
    g, t = GroundParams(), TransformParams(rot_y=0.0)  # the face looks straight at the camera
    assert ground.text_face_hit(g, t, np.array([1.0, 0.0, 0.0])) is None  # parallel to the face
    assert ground.text_face_hit(g, t, np.array([0.0, 0.0, 1.0])) is None  # pointing away from it


# ----- The wall ---------------------------------------------------------------------------------

POSES = [GroundParams(), GroundParams(pitch=14, roll=-6, yaw=25, height=2.2), GroundParams(pitch=-8, roll=12, yaw=-140)]
WALLS = [WallParams(), WallParams(angle=35, offset=9), WallParams(angle=-120, offset=4.5)]


def plane_distance(plane, point):
    return float(plane[:3] @ np.asarray(point)[:3] + plane[3])


def wall_text(**fields):
    return TransformParams(mount="wall", **{"wall_x": 1.5, "wall_y": 2.0, "depth": 0.4, "size": 1.5, **fields})


@pytest.mark.parametrize("g", POSES)
@pytest.mark.parametrize("wall", WALLS)
def test_wall_stands_upright_on_the_ground_facing_the_camera(g, wall):
    np.testing.assert_allclose(ground.wall_normal(g, wall) @ ground.ground_up(g), 0.0, atol=1e-12)
    for along in (-5.0, 0.0, 7.0):
        base = ground.wall_to_world(g, wall, along, 0.0)
        assert plane_distance(ground.ground_plane(g), base) == pytest.approx(0, abs=1e-9)
        assert plane_distance(ground.wall_plane(g, wall), base) == pytest.approx(0, abs=1e-9)
    # The camera is on the wall's front side, `offset` from it (the wall is vertical, the camera above its origin).
    assert plane_distance(ground.wall_plane(g, wall), (0, 0, 0)) == pytest.approx(wall.offset)


@pytest.mark.parametrize("g", POSES)
@pytest.mark.parametrize("wall", WALLS)
def test_wall_coordinates_round_trip(g, wall):
    point = ground.wall_to_world(g, wall, -3.25, 4.5)
    assert ground.world_to_wall(g, wall, point) == pytest.approx((-3.25, 4.5))
    np.testing.assert_allclose(ground.wall_to_world(g, wall, 0, 4.5) - ground.wall_to_world(g, wall, 0, 0),
                               4.5 * ground.ground_up(g), atol=1e-12)


@pytest.mark.parametrize("g", POSES)
@pytest.mark.parametrize("wall", WALLS)
def test_wall_text_lies_flat_on_the_wall_facing_out(g, wall):
    t = wall_text()
    m = ground.text_model_matrix(g, t, wall)
    plane = ground.wall_plane(g, wall)
    for corner in ((-0.8, 0.0), (0.9, 0.0), (0.4, 0.7)):
        back = m @ np.array([*corner, -t.depth, 1.0])  # the back cap, in em units
        face = m @ np.array([*corner, 0.0, 1.0])
        assert plane_distance(plane, back) == pytest.approx(0, abs=1e-9)
        assert plane_distance(plane, face) == pytest.approx(t.depth * t.size)  # in front of the wall, not inside it
    np.testing.assert_allclose(m[:3, 2] / t.size, ground.wall_normal(g, wall), atol=1e-12)
    np.testing.assert_allclose(m[:3, 1] / t.size, ground.ground_up(g), atol=1e-12)  # upright


@pytest.mark.parametrize("g", POSES)
def test_wall_text_at_height_zero_sits_on_the_floor_line(g):
    wall, t = WallParams(angle=20, offset=8), wall_text(wall_y=0.0)
    m = ground.text_model_matrix(g, t, wall)
    bottom_back = m @ np.array([0.5, 0.0, -t.depth, 1.0])
    assert plane_distance(ground.ground_plane(g), bottom_back) == pytest.approx(0, abs=1e-9)
    assert plane_distance(ground.wall_plane(g, wall), bottom_back) == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(ground.text_position(g, t, wall), ground.wall_to_world(g, wall, t.wall_x, 0.0))


def test_wall_text_reads_left_to_right_from_the_camera():
    g, wall = GroundParams(), WallParams(angle=30, offset=10)
    m = ground.text_model_matrix(g, wall_text(), wall)
    left = math3d.project_to_image((m @ np.array([-1.0, 0.3, 0.0, 1.0]))[:3], 40, (800, 500))
    right = math3d.project_to_image((m @ np.array([1.0, 0.3, 0.0, 1.0]))[:3], 40, (800, 500))
    assert left[0] < right[0]


def test_wall_mount_ignores_floor_fields_and_floor_mount_ignores_wall_fields():
    g, wall = GroundParams(pitch=5), WallParams(angle=10, offset=6)
    a, b = wall_text(x=1.0, z=-3.0, rot_y=40.0), wall_text(x=-7.0, z=-30.0, rot_y=-80.0)
    np.testing.assert_allclose(ground.text_model_matrix(g, a, wall), ground.text_model_matrix(g, b, wall))
    np.testing.assert_allclose(
        ground.text_model_matrix(g, TransformParams(x=1.0, z=-9.0), wall),
        ground.text_model_matrix(g, TransformParams(x=1.0, z=-9.0, wall_x=5.0, wall_y=3.0)),
    )


@pytest.mark.parametrize("a, b", [((-2.0, -10.0), (3.0, -12.0)), ((4.0, -3.0), (4.0, -20.0)), ((-6.0, 2.0), (6.0, 2.0))])
def test_wall_through_two_ground_points_runs_its_base_through_both(a, b):
    g = GroundParams(pitch=10, yaw=15)
    for first, second in ((a, b), (b, a)):
        wall = ground.wall_through(first, second)
        assert wall.offset >= 0 and -180 < wall.angle <= 180
        for x, z in (a, b):
            point = ground.ground_to_world(g, x, z)
            assert plane_distance(ground.wall_plane(g, wall), point) == pytest.approx(0, abs=1e-9)
            assert ground.world_to_wall(g, wall, point)[1] == pytest.approx(0, abs=1e-9)
        assert plane_distance(ground.wall_plane(g, wall), (0, 0, 0)) >= 0  # the camera is on the front side


def test_wall_through_is_the_same_wall_whichever_way_it_is_dragged():
    forward, backward = ground.wall_through((-2, -10), (3, -12)), ground.wall_through((3, -12), (-2, -10))
    assert (forward.angle, forward.offset) == pytest.approx((backward.angle, backward.offset))


def test_wall_through_one_point_is_no_wall():
    assert ground.wall_through((1.0, -5.0), (1.0, -5.0)) is None


@pytest.mark.parametrize(
    "change", [{"pitch": 9.0}, {"roll": -6.0}, {"height": 3.0}, {"yaw": 40.0}, {"pitch": -3.0, "yaw": -25.0, "roll": 4.0}]
)
def test_reanchored_wall_keeps_its_base_line_on_screen(change):
    old = GroundParams(pitch=4.0, yaw=5.0)
    new = GroundParams(**{**vars(old), **change})
    wall = WallParams(angle=25, offset=10)
    moved = ground.reanchor_wall(old, new, wall)
    for along in (-4.0, 0.0, 6.0):
        ray = ground.wall_to_world(old, wall, along, 0.0)
        hit = math3d.intersect_ray_plane((0, 0, 0), ray / np.linalg.norm(ray), ground.ground_plane(new))
        assert plane_distance(ground.wall_plane(new, moved), hit) == pytest.approx(0, abs=1e-8)


def test_reanchoring_the_wall_on_an_unchanged_ground_keeps_it():
    g, wall = GroundParams(pitch=7, yaw=-30), WallParams(angle=-50, offset=7)
    moved = ground.reanchor_wall(g, g, wall)
    assert (moved.angle, moved.offset) == pytest.approx((wall.angle, wall.offset))


def test_reanchored_wall_text_stays_on_its_ray_at_the_same_apparent_size():
    old, new = GroundParams(pitch=3.0), GroundParams(pitch=12.0, roll=4.0, height=2.0)
    wall = WallParams(angle=15, offset=12)
    new_wall = ground.reanchor_wall(old, new, wall)
    t = wall_text()
    wall_x, wall_y, size = ground.reanchor_wall_text(old, wall, new, new_wall, t)
    moved = wall_text(wall_x=wall_x, wall_y=wall_y, size=size)
    before, after = ground.text_position(old, t, wall), ground.text_position(new, moved, new_wall)
    np.testing.assert_allclose(after / np.linalg.norm(after), before / np.linalg.norm(before), atol=1e-9)
    assert size / np.linalg.norm(after) == pytest.approx(t.size / np.linalg.norm(before))


def test_reanchoring_wall_text_fails_when_its_ray_misses_the_wall():
    g, wall = GroundParams(), WallParams(angle=0, offset=10)
    behind = WallParams(angle=180, offset=10)  # a wall behind the camera: the ray to the text never meets it
    assert ground.reanchor_wall_text(g, wall, g, behind, wall_text(wall_x=0.0, wall_y=1.0)) is None


def test_duplicated_wall_text_moves_down_the_wall_or_up_off_the_floor():
    high = wall_text(wall_y=4.0, size=2.0)
    lower = ground.placement_in_front(GroundParams(), high)
    assert lower.wall_y < high.wall_y - high.size * ground.INK_HEIGHT_EM and lower.size == high.size
    low = wall_text(wall_y=0.5, size=2.0)
    assert ground.placement_in_front(GroundParams(), low).wall_y > low.wall_y + low.size * ground.INK_HEIGHT_EM


def screen_direction(point):
    return np.asarray(point) / np.linalg.norm(point)


@pytest.mark.parametrize("g", POSES[:2])
def test_mounting_floor_text_behind_the_wall_keeps_it_on_its_ray_and_apparent_size(g):
    wall = WallParams(angle=10, offset=8)
    t = TransformParams(x=0.5, z=-20.0, size=2.0)  # beyond the wall: its ray meets the wall above the floor
    wall_x, wall_y, size = ground.mount_on_wall(g, wall, t)
    moved = wall_text(wall_x=wall_x, wall_y=wall_y, size=size)
    before, after = ground.text_position(g, t), ground.text_position(g, moved, wall)
    assert wall_y > 0
    np.testing.assert_allclose(screen_direction(after), screen_direction(before), atol=1e-9)
    assert size / np.linalg.norm(after) == pytest.approx(t.size / np.linalg.norm(before))


def test_mounting_floor_text_in_front_of_the_wall_puts_it_on_the_base_line():
    g, wall = GroundParams(pitch=10), WallParams(offset=20)
    wall_x, wall_y, size = ground.mount_on_wall(g, wall, TransformParams(x=1.0, z=-8.0))
    assert wall_y == 0.0
    assert ground.world_to_wall(g, wall, ground.wall_to_world(g, wall, wall_x, 0.0)) == pytest.approx((wall_x, 0.0))


def test_mounting_on_a_wall_the_ray_misses_drops_the_text_straight_onto_it():
    g, wall = GroundParams(), WallParams(angle=180, offset=5)  # behind the camera
    t = TransformParams(x=2.0, z=-10.0)
    wall_x, wall_y, _size = ground.mount_on_wall(g, wall, t)
    np.testing.assert_allclose(ground.wall_to_world(g, wall, wall_x, wall_y)[[0, 1]], ground.text_position(g, t)[[0, 1]])


@pytest.mark.parametrize("g", POSES[:2])
def test_mounting_wall_text_on_the_floor_keeps_it_on_its_ray_facing_the_walls_way(g):
    wall = WallParams(angle=25, offset=12)
    t = wall_text(wall_y=0.5)
    x, z, rot_y, size = ground.mount_on_floor(g, wall, t)
    moved = TransformParams(x=x, z=z, rot_y=rot_y, size=size)
    before, after = ground.text_position(g, t, wall), ground.text_position(g, moved)
    np.testing.assert_allclose(screen_direction(after), screen_direction(before), atol=1e-9)
    assert size / np.linalg.norm(after) == pytest.approx(t.size / np.linalg.norm(before))
    np.testing.assert_allclose(ground.text_model_matrix(g, moved)[:3, 2] / size, ground.wall_normal(g, wall), atol=1e-12)


def test_wall_text_above_the_horizon_drops_to_the_walls_foot_on_the_floor():
    g, wall = GroundParams(), WallParams(offset=10)
    t = wall_text(wall_x=2.0, wall_y=5.0)  # well above the camera: its ray never meets the floor
    x, z, _rot_y, _size = ground.mount_on_floor(g, wall, t)
    np.testing.assert_allclose(ground.ground_to_world(g, x, z), ground.wall_to_world(g, wall, 2.0, 0.0), atol=1e-12)


def test_floor_to_wall_and_back_returns_to_the_same_spot_on_screen():
    g, wall = GroundParams(pitch=6), WallParams(angle=-15, offset=9)
    t = TransformParams(x=-1.0, z=-25.0, rot_y=-15.0, size=2.5)
    wall_x, wall_y, size = ground.mount_on_wall(g, wall, t)
    x, z, rot_y, back_size = ground.mount_on_floor(g, wall, wall_text(wall_x=wall_x, wall_y=wall_y, size=size))
    assert (x, z, rot_y, back_size) == pytest.approx((t.x, t.z, t.rot_y, t.size))
