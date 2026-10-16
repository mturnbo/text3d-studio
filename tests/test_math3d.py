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


def test_model_matrix_yaw_keeps_ground_normal_fixed():
    a = math3d.model_matrix(0, 0, 0, 20, 0, 10, 1)
    b = math3d.model_matrix(0, 0, 0, 20, 75, 10, 1)
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
