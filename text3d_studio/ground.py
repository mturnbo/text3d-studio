"""Ground model: where the ground plane is, and where text placed on it ends up in camera space.

The ground is posed relative to the camera (GroundParams). Text stands on it at ground coordinates
(transform.x, transform.z), spun by transform.rot_y around the ground's up axis. Light directions
are measured in the ground's tilt frame and ignore the grid's yaw, so rotating the grid never moves the sun.
"""

import numpy as np

from . import math3d
from .model import GroundParams, TransformParams


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
