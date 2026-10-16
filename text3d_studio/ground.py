"""Ground model: where the ground plane is, and where text placed on it ends up in camera space.

The ground is posed relative to the camera (GroundParams). Text stands on it at ground coordinates
(transform.x, transform.z), spun by transform.rot_y around the ground's up axis. Light directions
are measured in the ground's tilt frame and ignore the grid's yaw, so rotating the grid never moves the sun.
"""

from dataclasses import dataclass

import numpy as np

from . import math3d
from .model import GroundParams, TransformParams

# Supported ranges, shared by the sliders, the canvas gestures and the rectangle fit.
MAX_PITCH = 89.0
MAX_ROLL = 60.0
MIN_FOV, MAX_FOV = 10.0, 100.0


@dataclass
class RectangleFit:
    pitch: float
    yaw: float
    roll: float
    height: float
    fov: float
    offset_x: float  # grid offsets that run grid lines along the rectangle's edges
    offset_z: float


def ground_rotation(ground: GroundParams) -> np.ndarray:
    return math3d.ground_rotation(ground.pitch, ground.yaw, ground.roll)


def ground_up(ground: GroundParams) -> np.ndarray:
    return ground_rotation(ground)[:3, 1]


def ground_origin(ground: GroundParams) -> np.ndarray:
    """The point on the ground directly below the camera."""
    return -ground.height * ground_up(ground)


def ground_plane(ground: GroundParams) -> np.ndarray:
    return math3d.plane_from_point_normal(ground_origin(ground), ground_up(ground))


def light_frame(ground: GroundParams) -> np.ndarray:
    return math3d.ground_frame(ground.pitch, ground.roll)[:3, :3]


def ground_to_world(ground: GroundParams, x: float, z: float) -> np.ndarray:
    return ground_origin(ground) + ground_rotation(ground)[:3, :3] @ np.array([x, 0.0, z])


def world_to_ground(ground: GroundParams, point) -> tuple[float, float]:
    """Ground (x, z) of a camera-space point, measured along the ground's axes."""
    local = ground_rotation(ground)[:3, :3].T @ (np.asarray(point, dtype=np.float64) - ground_origin(ground))
    return float(local[0]), float(local[2])


def text_position(ground: GroundParams, transform: TransformParams) -> np.ndarray:
    return ground_to_world(ground, transform.x, transform.z)


def text_model_matrix(ground: GroundParams, transform: TransformParams) -> np.ndarray:
    position = text_position(ground, transform)
    return (
        math3d.translation(*position)
        @ ground_rotation(ground)
        @ math3d.rotation_y(transform.rot_y)
        @ math3d.scaling(transform.size)
    )


def text_distance(ground: GroundParams, transform: TransformParams) -> float:
    """Depth of the text's anchor along the view axis (always positive for visible text)."""
    return max(-float(text_position(ground, transform)[2]), 1e-3)


def reanchor_text(
    old: GroundParams, new: GroundParams, transform: TransformParams
) -> tuple[float, float, float, float] | None:
    """Placement (x, z, rot_y, size) on `new` that keeps the text where, and as big as, it appears on screen.

    The anchor stays on the same camera ray, now meeting the new ground. The size scales with the
    anchor's change in distance along that ray, so the text keeps its apparent size. The facing is kept
    too: a change in grid yaw is offset by the opposite change in text yaw. Returns None when the ray no
    longer meets the ground (the anchor would sit above the new horizon).
    """
    anchor = text_position(old, transform)
    distance = np.linalg.norm(anchor)
    hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), anchor / distance, ground_plane(new))
    if hit is None:
        return None
    x, z = world_to_ground(new, hit)
    size = transform.size * float(np.linalg.norm(hit)) / distance
    return x, z, transform.rot_y + old.yaw - new.yaw, size


def fit_ground_to_rectangle(
    quad, size, fov: float, image_size, cell_size: float, estimate_fov: bool = False
) -> RectangleFit:
    """Ground pose that makes `quad` (pixel corners, in order around the shape) the image of a rectangle.

    `size` is the rectangle's (edge 0-1, edge 0-3) length in world units; edge 0-1 becomes the grid's
    x direction. With `estimate_fov` the field of view comes from the two vanishing points instead of
    `fov`. Raises ValueError for shapes no rectangle could project to, or poses outside the supported range.
    """
    if estimate_fov:
        first, second = math3d.vanishing_points(quad)
        fov = math3d.fov_from_focal(math3d.focal_from_vanishing_points(first, second, image_size), image_size[1])
        if not MIN_FOV <= fov <= MAX_FOV:
            raise ValueError(f"The estimated field of view ({fov:.0f}°) is outside {MIN_FOV:.0f}–{MAX_FOV:.0f}°")
    rotation, corner = math3d.ground_pose_from_rectangle(quad, size, fov, image_size)
    pitch, yaw, roll = math3d.ground_angles_from_rotation(rotation)
    if abs(pitch) > MAX_PITCH or abs(roll) > MAX_ROLL:
        raise ValueError("That shape implies a ground tilted further than the editor supports")
    ground = GroundParams(pitch=pitch, roll=roll, yaw=yaw, height=float(rotation[:, 1] @ -corner))
    corner_x, corner_z = world_to_ground(ground, corner)
    return RectangleFit(
        pitch=pitch, yaw=yaw, roll=roll, height=ground.height, fov=fov,
        offset_x=corner_x % cell_size, offset_z=corner_z % cell_size,
    )
