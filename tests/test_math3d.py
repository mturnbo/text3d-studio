import numpy as np
import pytest

from text3d_studio import math3d


def test_perspective_maps_near_and_far_planes_to_ndc_bounds():
    proj = math3d.perspective(60, 1.5, 0.5, 100)
    near = proj @ np.array([0, 0, -0.5, 1])
    far = proj @ np.array([0, 0, -100, 1])
    assert near[2] / near[3] == pytest.approx(-1.0)
    assert far[2] / far[3] == pytest.approx(1.0)


def test_perspective_respects_vertical_fov():
    proj = math3d.perspective(90, 1.0, 0.1, 10)
    top_edge = proj @ np.array([0, 2, -2, 1])  # 45 degrees up
    assert top_edge[1] / top_edge[3] == pytest.approx(1.0)


def test_direction_from_angles_is_unit_and_straight_up_at_90_elevation():
    for az, el in [(0, 0), (35, 40), (-170, 5), (90, 89)]:
        assert np.linalg.norm(math3d.direction_from_angles(az, el)) == pytest.approx(1.0)
    np.testing.assert_allclose(math3d.direction_from_angles(123, 90), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(math3d.direction_from_angles(0, 0), [0, 0, 1], atol=1e-12)


def test_ground_yaw_keeps_ground_normal_fixed():
    a = math3d.ground_rotation(20, 0, 10)
    b = math3d.ground_rotation(20, 75, 10)
    up = np.array([0, 1, 0, 0])
    np.testing.assert_allclose(a @ up, b @ up, atol=1e-12)


def test_planar_shadow_matrix_projects_points_onto_plane_along_light():
    plane = math3d.plane_from_point_normal(np.array([0, -1.0, 0]), np.array([0, 1.0, 0]))
    light = math3d.direction_from_angles(30, 45)
    shadow = math3d.planar_shadow_matrix(plane, light)
    point = np.array([1.0, 2.0, -3.0, 1.0])
    projected = shadow @ point
    projected = projected[:3] / projected[3]
    assert plane[:3] @ projected + plane[3] == pytest.approx(0.0, abs=1e-12)
    offset = point[:3] - projected
    np.testing.assert_allclose(offset / np.linalg.norm(offset), light, atol=1e-12)


def test_planar_shadow_matrix_leaves_points_on_plane_in_place():
    plane = math3d.plane_from_point_normal(np.array([0, 0, 0.0]), np.array([0, 1.0, 0]))
    shadow = math3d.planar_shadow_matrix(plane, math3d.direction_from_angles(-60, 20))
    on_plane = np.array([4.0, 0.0, 7.0, 1.0])
    result = shadow @ on_plane
    np.testing.assert_allclose(result[:3] / result[3], on_plane[:3], atol=1e-12)


def test_to_gl_is_column_major_float32():
    m = math3d.translation(1, 2, 3)
    gl = math3d.to_gl(m)
    assert gl.dtype == np.float32 and gl.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(gl.ravel()[12:15], [1, 2, 3])


# ----- Image-space perspective helpers ----------------------------------------------------------

IMAGE = (1200, 800)


def test_focal_and_fov_are_inverse():
    for fov in (10.0, 40.0, 90.0, 120.0):
        assert math3d.fov_from_focal(math3d.focal_length_px(fov, 800), 800) == pytest.approx(fov)


def test_focal_matches_ninety_degree_fov():
    assert math3d.focal_length_px(90, 800) == pytest.approx(400)


def test_point_on_axis_projects_to_image_center():
    np.testing.assert_allclose(math3d.project_to_image((0, 0, -5), 40, IMAGE), [600, 400])


def test_projection_agrees_with_gl_perspective_matrix():
    point = np.array([1.5, -0.7, -9.0])
    pixel = math3d.project_to_image(point, 40, IMAGE)
    clip = math3d.perspective(40, IMAGE[0] / IMAGE[1], 0.1, 100) @ np.append(point, 1.0)
    ndc = clip[:2] / clip[3]
    expected = [(ndc[0] + 1) / 2 * IMAGE[0], (1 - ndc[1]) / 2 * IMAGE[1]]
    np.testing.assert_allclose(pixel, expected, atol=1e-9)


def test_image_y_grows_downward():
    above = math3d.project_to_image((0, 1, -5), 40, IMAGE)
    below = math3d.project_to_image((0, -1, -5), 40, IMAGE)
    assert above[1] < 400 < below[1]


@pytest.mark.parametrize("z", [0.0, 3.0])
def test_points_at_or_behind_camera_do_not_project(z):
    assert math3d.project_to_image((1, 1, z), 40, IMAGE) is None


def test_image_ray_inverts_projection():
    point = np.array([2.0, -1.5, -11.0])
    pixel = math3d.project_to_image(point, 55, IMAGE)
    ray = math3d.image_ray(pixel, 55, IMAGE)
    np.testing.assert_allclose(ray, point / np.linalg.norm(point), atol=1e-12)


def test_center_ray_looks_down_negative_z():
    np.testing.assert_allclose(math3d.image_ray((600, 400), 40, IMAGE), [0, 0, -1])


def test_ray_hits_ground_plane_below_camera():
    ground = math3d.plane_from_point_normal(np.array([0, -2.0, 0]), np.array([0, 1.0, 0]))
    hit = math3d.intersect_ray_plane((0, 0, 0), math3d.image_ray((600, 700), 40, IMAGE), ground)
    assert hit[1] == pytest.approx(-2.0)
    assert hit[2] < 0


def test_ray_parallel_to_plane_misses():
    ground = math3d.plane_from_point_normal(np.array([0, -2.0, 0]), np.array([0, 1.0, 0]))
    assert math3d.intersect_ray_plane((0, 0, 0), (0, 0, -1), ground) is None


def test_ray_pointing_away_from_plane_misses():
    ground = math3d.plane_from_point_normal(np.array([0, -2.0, 0]), np.array([0, 1.0, 0]))
    above_horizon = math3d.image_ray((600, 100), 40, IMAGE)
    assert math3d.intersect_ray_plane((0, 0, 0), above_horizon, ground) is None


UNIT_SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_homography_maps_square_onto_perspective_quad():
    quad = [(412.5, 610.0), (905.0, 640.0), (780.0, 380.0), (450.0, 372.0)]
    h = math3d.homography_from_points(UNIT_SQUARE, quad)
    np.testing.assert_allclose(math3d.apply_homography(h, UNIT_SQUARE), quad, atol=1e-6)


def test_homography_recovers_a_known_matrix_from_pixel_scale_points():
    known = np.array([[1.2, 0.1, 300.0], [-0.05, 0.9, 150.0], [0.0004, 0.0002, 1.0]])
    source = np.array([(0, 0), (1600, 0), (1600, 1000), (0, 1000)], dtype=float)
    h = math3d.homography_from_points(source, math3d.apply_homography(known, source))
    np.testing.assert_allclose(h / h[2, 2], known, rtol=1e-6, atol=1e-9)


def test_identity_correspondence_gives_identity():
    h = math3d.homography_from_points(UNIT_SQUARE, UNIT_SQUARE)
    np.testing.assert_allclose(h / h[2, 2], np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "target",
    [
        [(0, 0), (1, 1), (2, 2), (0, 5)],  # three collinear
        [(3, 3), (3, 3), (3, 3), (3, 3)],  # all the same point
    ],
)
def test_degenerate_targets_are_rejected(target):
    with pytest.raises(ValueError):
        math3d.homography_from_points(UNIT_SQUARE, target)


@pytest.mark.parametrize("count", [3, 5])
def test_point_count_other_than_four_is_rejected(count):
    points = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 3)][:count]
    with pytest.raises(ValueError):
        math3d.homography_from_points(points, points)


def ground_square(fov, pitch=20.0, yaw=35.0, roll=5.0, origin=(0.5, -2.0, -10.0), side=3.0):
    """Pixel corners of a square lying on a tilted ground plane, in order around the square."""
    frame = math3d.ground_frame(pitch, roll)[:3, :3] @ math3d.rotation_y(yaw)[:3, :3]
    corners = [np.array(origin) + frame @ np.array([a, 0.0, b]) for a, b in ((0, 0), (side, 0), (side, -side), (0, -side))]
    return [math3d.project_to_image(c, fov, IMAGE) for c in corners]


@pytest.mark.parametrize("fov", [30.0, 50.0, 75.0])
def test_focal_estimated_from_a_perspective_square(fov):
    first, second = math3d.vanishing_points(ground_square(fov))
    focal = math3d.focal_from_vanishing_points(first, second, IMAGE)
    assert focal == pytest.approx(math3d.focal_length_px(fov, IMAGE[1]), rel=1e-6)


def test_vanishing_points_lie_on_both_edge_lines():
    quad = ground_square(50)
    first, _ = math3d.vanishing_points(quad)
    vp = first[:2] / first[2]
    for a, b in ((quad[0], quad[1]), (quad[3], quad[2])):
        cross = (b[0] - a[0]) * (vp[1] - a[1]) - (b[1] - a[1]) * (vp[0] - a[0])
        assert abs(cross) < 1e-6 * np.linalg.norm(b - a) * np.linalg.norm(vp - a)


def test_square_facing_the_camera_squarely_cannot_estimate_focal():
    # With no yaw or roll the front and back edges stay horizontal in the image: that vanishing point is at infinity.
    first, second = math3d.vanishing_points(ground_square(50, yaw=0.0, roll=0.0))
    with pytest.raises(ValueError):
        math3d.focal_from_vanishing_points(first, second, IMAGE)


def test_vanishing_points_on_the_same_side_are_rejected():
    first, second = np.array([900.0, 300.0, 1.0]), np.array([1000.0, 350.0, 1.0])
    with pytest.raises(ValueError):
        math3d.focal_from_vanishing_points(first, second, IMAGE)


def test_ground_rotation_applies_yaw_before_tilt_and_roll():
    expected = math3d.rotation_z(7) @ math3d.rotation_x(15) @ math3d.rotation_y(-20)
    np.testing.assert_allclose(math3d.ground_rotation(15, -20, 7), expected, atol=1e-12)


@pytest.mark.parametrize(
    "angles", [(0.0, 0.0, 0.0), (20.0, 35.0, 5.0), (-40.0, 170.0, -25.0), (75.0, -90.0, 60.0), (10.0, -179.0, 0.5)]
)
def test_ground_angles_round_trip(angles):
    recovered = math3d.ground_angles_from_rotation(math3d.ground_rotation(*angles))
    np.testing.assert_allclose(recovered, angles, atol=1e-9)


def test_ground_angles_accept_4x4_matrices():
    m = math3d.ground_rotation(12, 34, -5)
    assert m.shape == (4, 4)
    np.testing.assert_allclose(math3d.ground_angles_from_rotation(m), (12, 34, -5), atol=1e-9)


@pytest.mark.parametrize(
    "fov, pitch, yaw, roll",
    [(50.0, 20.0, 35.0, 5.0), (30.0, 8.0, -60.0, -3.0), (75.0, 45.0, 10.0, 12.0), (40.0, 15.0, 0.0, 0.0)],
)
def test_ground_pose_recovered_from_projected_rectangle(fov, pitch, yaw, roll):
    origin = np.array([0.5, -2.0, -10.0])
    quad = ground_square(fov, pitch, yaw, roll, origin=origin, side=3.0)
    rotation, corner = math3d.ground_pose_from_rectangle(quad, (3.0, 3.0), fov, IMAGE)
    np.testing.assert_allclose(rotation, math3d.ground_rotation(pitch, yaw, roll)[:3, :3], atol=1e-7)
    np.testing.assert_allclose(corner, origin, atol=1e-6)
    np.testing.assert_allclose(math3d.ground_angles_from_rotation(rotation), (pitch, yaw, roll), atol=1e-5)


def test_non_square_rectangle_recovers_scale_from_known_size():
    frame = math3d.ground_rotation(18, 25, 0)[:3, :3]
    origin = np.array([-1.0, -1.5, -8.0])
    corners = [origin + frame @ np.array([a, 0, -b]) for a, b in ((0, 0), (4, 0), (4, 1.5), (0, 1.5))]
    quad = [math3d.project_to_image(c, 45, IMAGE) for c in corners]
    rotation, corner = math3d.ground_pose_from_rectangle(quad, (4.0, 1.5), 45, IMAGE)
    np.testing.assert_allclose(corner, origin, atol=1e-6)
    np.testing.assert_allclose(rotation, frame, atol=1e-7)


def test_reversed_corner_order_keeps_up_toward_camera():
    quad = ground_square(50)
    rotation, _ = math3d.ground_pose_from_rectangle(quad[::-1], (3.0, 3.0), 50, IMAGE)
    expected_up = math3d.ground_rotation(20, 35, 5)[:3, 1]
    np.testing.assert_allclose(rotation[:, 1], expected_up, atol=1e-7)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_crossed_corners_are_rejected():
    a, b, c, d = ground_square(50)
    with pytest.raises(ValueError, match="convex"):
        math3d.ground_pose_from_rectangle([a, c, b, d], (3.0, 3.0), 50, IMAGE)



def test_collinear_corners_are_rejected():
    with pytest.raises(ValueError):
        math3d.ground_pose_from_rectangle([(100, 500), (300, 500), (500, 500), (300, 300)], (1.0, 1.0), 40, IMAGE)


def test_parallel_lines_converge_on_their_vanishing_point():
    direction = np.array([0.4, -0.1, -1.0])
    vp = math3d.vanishing_point(direction, 50, IMAGE)
    for start in ([1.0, -2.0, -5.0], [-3.0, -2.0, -8.0]):
        far = math3d.project_to_image(np.array(start) + 1e9 * direction, 50, IMAGE)
        np.testing.assert_allclose(far, vp, atol=1e-3)


def test_vanishing_point_ignores_line_direction_sign():
    d = np.array([0.3, 0.0, 1.0])
    np.testing.assert_allclose(math3d.vanishing_point(d, 40, IMAGE), math3d.vanishing_point(-d, 40, IMAGE))


def test_direction_parallel_to_image_plane_has_no_vanishing_point():
    assert math3d.vanishing_point((1.0, 0.5, 0.0), 40, IMAGE) is None
