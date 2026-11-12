"""Ground model: where the ground plane is, and where text placed on it ends up in camera space.

The ground is posed relative to the camera (GroundParams). Text stands on it at ground coordinates
(transform.x, transform.z), spun by transform.rot_y around the ground's up axis. Light directions
are measured in the ground's tilt frame and ignore the grid's yaw, so rotating the grid never moves the sun.
"""

from dataclasses import dataclass, replace

import numpy as np

from . import math3d
from .model import GroundParams, TransformParams, WallParams

# Supported ranges, shared by the sliders, the canvas gestures and the rectangle fit.
MAX_PITCH = 89.0
MAX_ROLL = 60.0
MIN_FOV, MAX_FOV = 10.0, 100.0
INK_HEIGHT_EM = 0.75  # about a capital's height: how tall placement_in_front assumes the text stands
MOUNTS = ("floor", "wall")


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


def text_position(ground: GroundParams, transform: TransformParams, wall: WallParams | None = None) -> np.ndarray:
    """The text's anchor (the middle of its ink's bottom edge): on the ground, or on the wall for wall mounts."""
    if transform.mount == "wall":
        return wall_to_world(ground, wall, transform.wall_x, transform.wall_y)
    return ground_to_world(ground, transform.x, transform.z)


def text_model_matrix(ground: GroundParams, transform: TransformParams, wall: WallParams | None = None) -> np.ndarray:
    if transform.mount == "wall":
        return wall_text_model_matrix(ground, wall, transform)
    position = text_position(ground, transform)
    return (
        math3d.translation(*position)
        @ ground_rotation(ground)
        @ math3d.rotation_y(transform.rot_y)
        @ math3d.scaling(transform.size)
    )


def text_face_hit(
    ground: GroundParams, transform: TransformParams, ray, wall: WallParams | None = None
) -> tuple[float, float, float] | None:
    """Where a camera ray meets the plane of the text's front face: (distance along the ray, local x, local y),
    local coordinates in em units like the glyph raster's. None when the ray runs parallel to or away from it."""
    model = text_model_matrix(ground, transform, wall)
    plane = math3d.plane_from_point_normal(model[:3, 3], model[:3, 2])
    hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), ray, plane)
    if hit is None:
        return None
    local = np.linalg.inv(model) @ np.append(hit, 1.0)
    return float(np.linalg.norm(hit)), float(local[0]), float(local[1])


def text_distance(ground: GroundParams, transform: TransformParams, wall: WallParams | None = None) -> float:
    """Depth of the text's anchor along the view axis (always positive for visible text)."""
    return max(-float(text_position(ground, transform, wall)[2]), 1e-3)


# ----- The wall ---------------------------------------------------------------------------------
# A vertical plane standing on the ground. Its frame is the ground's, spun by wall.angle like a text's rot_y:
# x runs along the base line, y is the ground's up and z is the wall's normal, facing the camera's side.
# The base line passes wall.offset from the point below the camera. Wall coordinates are
# (distance along the base line from the point nearest the camera, height above the floor).


def wall_rotation(ground: GroundParams, wall: WallParams) -> np.ndarray:
    return ground_rotation(ground) @ math3d.rotation_y(wall.angle)


def wall_normal(ground: GroundParams, wall: WallParams) -> np.ndarray:
    return wall_rotation(ground, wall)[:3, 2]


def wall_to_world(ground: GroundParams, wall: WallParams, along: float, height: float) -> np.ndarray:
    return ground_origin(ground) + wall_rotation(ground, wall)[:3, :3] @ np.array([along, height, -wall.offset])


def world_to_wall(ground: GroundParams, wall: WallParams, point) -> tuple[float, float]:
    """Wall coordinates (along, height) of a camera-space point, projected straight onto the wall."""
    local = wall_rotation(ground, wall)[:3, :3].T @ (np.asarray(point, dtype=np.float64) - ground_origin(ground))
    return float(local[0]), float(local[1])


def wall_plane(ground: GroundParams, wall: WallParams) -> np.ndarray:
    return math3d.plane_from_point_normal(wall_to_world(ground, wall, 0.0, 0.0), wall_normal(ground, wall))


def wall_text_model_matrix(ground: GroundParams, wall: WallParams, transform: TransformParams) -> np.ndarray:
    """Text lying flat on the wall: its back on the wall, its face toward the camera's side, reading along it."""
    rotation = wall_rotation(ground, wall)
    back = wall_to_world(ground, wall, transform.wall_x, transform.wall_y)
    position = back + rotation[:3, 2] * transform.depth * transform.size
    return math3d.translation(*position) @ rotation @ math3d.scaling(transform.size)


def mount_on_wall(ground: GroundParams, wall: WallParams, transform: TransformParams) -> tuple[float, float, float]:
    """(wall_x, wall_y, size) that put floor text on the wall where, and as big as, it appears on screen.

    The anchor moves along its camera ray onto the wall, or straight onto it when the ray misses. It never goes
    below the floor: text standing in front of the wall lands on the wall's base line instead.
    """
    anchor = text_position(ground, transform)
    distance = float(np.linalg.norm(anchor))
    plane = wall_plane(ground, wall)
    hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), anchor / distance, plane)
    if hit is None:
        hit = anchor - (plane[:3] @ anchor + plane[3]) * plane[:3]
    wall_x, wall_y = world_to_wall(ground, wall, hit)
    wall_y = max(wall_y, 0.0)
    point = wall_to_world(ground, wall, wall_x, wall_y)
    return wall_x, wall_y, transform.size * float(np.linalg.norm(point)) / distance


def mount_on_floor(
    ground: GroundParams, wall: WallParams, transform: TransformParams
) -> tuple[float, float, float, float]:
    """(x, z, rot_y, size) that stand wall text on the floor where, and as big as, it appears on screen, facing the
    way the wall does. When its ray misses the floor (it shows above the horizon), it drops to the wall's foot."""
    anchor = text_position(ground, transform, wall)
    distance = float(np.linalg.norm(anchor))
    hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), anchor / distance, ground_plane(ground))
    if hit is None:
        hit = wall_to_world(ground, wall, transform.wall_x, 0.0)
    x, z = world_to_ground(ground, hit)
    return x, z, wall.angle, transform.size * float(np.linalg.norm(hit)) / distance


def wall_through(a, b) -> WallParams | None:
    """The wall whose base line passes through ground points `a` and `b` (x, z), or None when they coincide.

    The wall faces the point below the camera, and its x axis then reads left to right from the camera's side.
    """
    dx, dz = b[0] - a[0], b[1] - a[1]
    if np.hypot(dx, dz) < 1e-9:
        return None
    angle = np.degrees(np.arctan2(-dz, dx))  # the base line runs along (cos angle, -sin angle)
    normal = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
    offset = -float(normal @ np.asarray(a, dtype=np.float64))
    if offset < 0:
        angle, offset = angle + 180.0, -offset
    return WallParams(angle=float(180.0 - (180.0 - angle) % 360.0), offset=offset)  # angle in (-180, 180]


def reanchor_wall(old: GroundParams, new: GroundParams, wall: WallParams) -> WallParams | None:
    """The wall on `new` whose base line shows where `wall`'s base line showed on `old`.

    The camera and the old base line span a plane; the new base line is where that plane meets the new ground.
    None when that plane runs parallel to the new ground (the base line would be the horizon).
    """
    first = wall_to_world(old, wall, 0.0, 0.0)
    second = wall_to_world(old, wall, 1.0, 0.0)
    normal = ground_rotation(new)[:3, :3].T @ np.cross(first, second)  # in the new ground's frame
    # Ground points (x, 0, z) on that plane: normal . ((x, 0, z) - camera) = 0, with the camera at (0, height, 0).
    line = np.array([normal[0], normal[2]])
    length = float(np.linalg.norm(line))
    if length < 1e-9 * float(np.linalg.norm(normal)):
        return None
    distance = float(normal[1]) * new.height / length  # signed distance of the line along line / length
    foot = line / length * distance
    return wall_through(foot, foot + np.array([-line[1], line[0]]))


def reanchor_wall_text(
    old: GroundParams, old_wall: WallParams, new: GroundParams, new_wall: WallParams, transform: TransformParams
) -> tuple[float, float, float] | None:
    """Wall placement (wall_x, wall_y, size) on `new_wall` that keeps wall text where, and as big as, it appears.

    Like reanchor_text: the anchor stays on its camera ray and the size scales with its distance. None when the ray
    misses the new wall."""
    anchor = text_position(old, transform, old_wall)
    distance = np.linalg.norm(anchor)
    hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), anchor / distance, wall_plane(new, new_wall))
    if hit is None:
        return None
    wall_x, wall_y = world_to_wall(new, new_wall, hit)
    return wall_x, wall_y, transform.size * float(np.linalg.norm(hit)) / distance


def placement_in_front(ground: GroundParams, transform: TransformParams) -> TransformParams:
    """A copy of `transform` moved toward the camera, and shrunk with its distance, so text standing there shows
    below the original instead of over it.

    Placement and size scale by the same factor f around the point below the camera, in the ground's frame.
    The camera's ray to the original's base then passes height (1 - f) * camera height above the new spot,
    which clears the new text's top (f * size * INK_HEIGHT_EM) when f < height / (size * INK_HEIGHT_EM + height).
    Wall text stays the same size and moves down the wall by a line, or up where that would sink it into the floor.
    """
    if transform.mount == "wall":
        step = transform.size * INK_HEIGHT_EM * 1.25
        below = transform.wall_y - step
        return replace(transform, wall_y=below if below >= 0 else transform.wall_y + step)
    f = 0.9 * ground.height / (transform.size * INK_HEIGHT_EM + ground.height)
    return replace(transform, x=transform.x * f, z=transform.z * f, size=transform.size * f)


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


def wall_grid_lines(ground: GroundParams, wall: WallParams, spacing: float, half_width: float, height: float):
    """Camera-space segments of a grid on the wall: verticals every `spacing` within `half_width` of the point
    nearest the camera, and horizontals up to `height`. The base line (height 0) isn't included."""
    count = int(half_width / spacing)
    lines = []
    for i in range(-count, count + 1):
        along = i * spacing
        lines.append((wall_to_world(ground, wall, along, 0.0), wall_to_world(ground, wall, along, height)))
    for j in range(1, int(height / spacing) + 1):
        lines.append((wall_to_world(ground, wall, -half_width, j * spacing), wall_to_world(ground, wall, half_width, j * spacing)))
    return lines
