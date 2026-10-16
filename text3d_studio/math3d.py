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


def apply_homography(h: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mapped = np.hstack([points, np.ones((len(points), 1))]) @ h.T
    return mapped[:, :2] / mapped[:, 2:3]


def homography_from_points(source, target) -> np.ndarray:
    """3x3 matrix mapping four source points onto four target points (direct linear transform).

    Raises ValueError unless there are exactly four pairs with no three points of either set collinear.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("Need exactly four matching point pairs")
    if _has_collinear_triple(source) or _has_collinear_triple(target):
        raise ValueError("Three of the points lie on one line")

    # Hartley normalization keeps the SVD well conditioned for pixel-sized coordinates.
    source_norm, target_norm = _normalizing_transform(source), _normalizing_transform(target)
    s = apply_homography(source_norm, source)
    t = apply_homography(target_norm, target)
    rows = []
    for (x, y), (u, v) in zip(s, t):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    h = np.linalg.inv(target_norm) @ vt[-1].reshape(3, 3) @ source_norm
    return h / np.linalg.norm(h)


def _normalizing_transform(points: np.ndarray) -> np.ndarray:
    """Moves the centroid to the origin and scales the mean distance from it to sqrt(2)."""
    centroid = points.mean(axis=0)
    mean_distance = np.linalg.norm(points - centroid, axis=1).mean()
    s = np.sqrt(2.0) / mean_distance
    return np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1.0]])


def _has_collinear_triple(points: np.ndarray) -> bool:
    extent = np.ptp(points, axis=0).max()
    if extent == 0:
        return True
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                a, b = points[j] - points[i], points[k] - points[i]
                if abs(a[0] * b[1] - a[1] * b[0]) < 1e-6 * extent**2:
                    return True
    return False


def vanishing_points(quad) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous vanishing points of a quad's two pairs of opposite edges (w ~ 0 when parallel).

    The quad's corners go around the shape: edges 0-1 / 3-2 meet at the first point, 1-2 / 0-3 at the second.
    """
    p = [np.append(np.asarray(corner, dtype=np.float64), 1.0) for corner in quad]
    first = np.cross(np.cross(p[0], p[1]), np.cross(p[3], p[2]))
    second = np.cross(np.cross(p[1], p[2]), np.cross(p[0], p[3]))
    return first, second


def focal_from_vanishing_points(first, second, image_size: tuple[float, float]) -> float:
    """Focal length in pixels implied by the vanishing points of two perpendicular directions.

    Raises ValueError when a vanishing point is at infinity or the points imply no real focal length.
    """
    principal = np.array(image_size, dtype=np.float64) / 2.0
    offsets = []
    for point in (first, second):
        if abs(point[2]) < 1e-9 * np.linalg.norm(point[:2]):
            raise ValueError("Opposite edges are parallel in the image; the field of view can't be estimated")
        offsets.append(point[:2] / point[2] - principal)
    focal_squared = -float(offsets[0] @ offsets[1])
    if focal_squared <= 0:
        raise ValueError("These corners don't match a rectangle seen through this camera")
    return float(np.sqrt(focal_squared))


def ground_rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Orientation of a ground frame: yaw spins it in its own plane, then pitch and roll tilt it."""
    return ground_frame(pitch, roll) @ rotation_y(yaw)


def ground_angles_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees such that ground_rotation(pitch, yaw, roll) reproduces `rotation`."""
    r = np.asarray(rotation, dtype=np.float64)[:3, :3]
    # The ground's up axis is unaffected by yaw: up = Rz(roll) Rx(pitch) e_y.
    pitch = np.degrees(np.arcsin(np.clip(r[2, 1], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(-r[0, 1], r[1, 1]))
    yaw_only = (rotation_x(pitch)[:3, :3].T @ rotation_z(roll)[:3, :3].T) @ r
    yaw = np.degrees(np.arctan2(yaw_only[0, 2], yaw_only[0, 0]))
    return float(pitch), float(yaw), float(roll)


def ground_pose_from_rectangle(quad, size, fov_y_degrees: float, image_size) -> tuple[np.ndarray, np.ndarray]:
    """Pose of the ground plane from a rectangle on it, seen in the image as `quad`.

    `quad` holds pixel corners in order around the rectangle; `size` is (length of edge 0-1, length of
    edge 0-3) in world units. Returns (rotation, corner): rotation's columns are the ground's x, up and
    z axes in camera space (x runs along edge 0-1, up points toward the camera), and corner is the
    camera-space position of corner 0. Raises ValueError for shapes no rectangle could project to.
    """
    quad = np.asarray(quad, dtype=np.float64)
    if not _is_convex(quad):
        raise ValueError("The corners must form a convex shape, in order around it")
    width, height = image_size
    f = focal_length_px(fov_y_degrees, height)
    normalized = np.column_stack([(quad[:, 0] - width / 2.0) / f, -(quad[:, 1] - height / 2.0) / f])
    edge, depth = size
    plane_corners = [(0, 0), (edge, 0), (edge, depth), (0, depth)]

    # Ground point (a, b) sits at camera position a*e1 + b*e2 + corner, and its image is (x, y, -z).
    m = np.diag([1.0, 1.0, -1.0]) @ homography_from_points(plane_corners, normalized)
    m /= (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1])) / 2.0
    if m[2, 2] > 0:
        m = -m  # corner 0 must be in front of the camera

    x_axis = m[:, 0] / np.linalg.norm(m[:, 0])
    up = np.cross(x_axis, m[:, 1])
    up /= np.linalg.norm(up)
    if up @ -m[:, 2] < 0:
        up = -up  # the camera looks at the ground from its upper side
    rotation = np.column_stack([x_axis, up, np.cross(x_axis, up)])
    u, _, vt = np.linalg.svd(rotation)  # snap to the nearest exact rotation
    return u @ vt, m[:, 2].copy()


def _is_convex(quad: np.ndarray) -> bool:
    crosses = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(x > 0 for x in crosses) or all(x < 0 for x in crosses)
