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
