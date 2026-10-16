"""Row-major 4x4 matrix helpers (numpy). Use to_gl() when uploading as a uniform."""

import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translation(x: float, y: float, z: float) -> np.ndarray:
    m = identity()
    m[:3, 3] = (x, y, z)
    return m


def scaling(s: float) -> np.ndarray:
    m = identity()
    m[0, 0] = m[1, 1] = m[2, 2] = s
    return m


def rotation_x(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    m = identity()
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rotation_y(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    m = identity()
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rotation_z(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    m = identity()
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def ground_frame(rot_x: float, rot_z: float) -> np.ndarray:
    """Orientation of the ground plane the text stands on (tilt + roll, no yaw)."""
    return rotation_z(rot_z) @ rotation_x(rot_x)


def model_matrix(x, y, z, rot_x, rot_y, rot_z, size) -> np.ndarray:
    # Yaw is applied first, around the local up axis, so it spins the text on its ground plane.
    return translation(x, y, z) @ ground_frame(rot_x, rot_z) @ rotation_y(rot_y) @ scaling(size)


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y_degrees) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def direction_from_angles(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    """Unit vector pointing toward the light in a Y-up frame. Azimuth 0 = toward +Z (the camera)."""
    az, el = np.radians(azimuth_degrees), np.radians(elevation_degrees)
    return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def plane_from_point_normal(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    return np.array([n[0], n[1], n[2], -float(np.dot(n, point))])


def planar_shadow_matrix(plane: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
    """Flattens geometry onto `plane` along a directional light (light_dir points toward the light)."""
    light = np.array([light_dir[0], light_dir[1], light_dir[2], 0.0])
    return float(plane @ light) * identity() - np.outer(light, plane)


def to_gl(m: np.ndarray) -> np.ndarray:
    """Column-major float32 copy for glUniformMatrix4fv(..., GL_FALSE, ...)."""
    return np.ascontiguousarray(m.T, dtype=np.float32)


# ----- Image-space perspective helpers ----------------------------------------------------------
# The camera sits at the origin looking down -Z (view = identity). Image coordinates are pixels with
# the origin at the top-left and y pointing down; the principal point is the image center.


def focal_length_px(fov_y_degrees: float, image_height: float) -> float:
    return (image_height / 2.0) / np.tan(np.radians(fov_y_degrees) / 2.0)


def fov_from_focal(focal_px: float, image_height: float) -> float:
    return float(np.degrees(2.0 * np.arctan((image_height / 2.0) / focal_px)))


def project_to_image(point, fov_y_degrees: float, image_size: tuple[float, float]) -> np.ndarray | None:
    """Pixel position of a camera-space point, or None if it is not in front of the camera."""
    x, y, z = point
    if z >= -1e-9:
        return None
    width, height = image_size
    f = focal_length_px(fov_y_degrees, height)
    return np.array([width / 2.0 + f * x / -z, height / 2.0 - f * y / -z])


def image_ray(pixel, fov_y_degrees: float, image_size: tuple[float, float]) -> np.ndarray:
    """Unit direction from the camera through a pixel (the inverse of project_to_image)."""
    width, height = image_size
    f = focal_length_px(fov_y_degrees, height)
    direction = np.array([(pixel[0] - width / 2.0) / f, -(pixel[1] - height / 2.0) / f, -1.0])
    return direction / np.linalg.norm(direction)


def intersect_ray_plane(origin, direction, plane: np.ndarray) -> np.ndarray | None:
    """Where a ray hits a plane (a, b, c, d), or None if it is parallel or the hit is behind the origin."""
    origin, direction = np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
    denominator = float(plane[:3] @ direction)
    if abs(denominator) < 1e-9:
        return None
    t = -(float(plane[:3] @ origin) + plane[3]) / denominator
    if t <= 0:
        return None
    return origin + t * direction
