"""LightingEngine: turns user-facing light/shadow params into world-space shader inputs."""

from dataclasses import dataclass

import numpy as np

from . import ground as ground_model
from . import math3d
from .model import GroundParams, LightParams

MIN_SHADOW_ELEVATION = 3.0  # below this a planar shadow stretches toward infinity


def srgb_to_linear(color) -> np.ndarray:
    """Colors are picked in sRGB; lighting math happens in linear light."""
    c = np.asarray(color, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@dataclass
class LightingState:
    light_dir: np.ndarray  # world space, unit, pointing toward the light
    light_color: np.ndarray  # linear color * intensity
    ambient: np.ndarray  # linear ambient color * intensity
    ground_plane: np.ndarray  # (a, b, c, d) world plane the shadow falls on
    shadow_matrix: np.ndarray  # 4x4 planar projection onto the ground plane


def solve_sun(ground: GroundParams, fov: float, image_size, top, base, tip) -> tuple[float, float]:
    """(azimuth, elevation) of the sun from a vertical object and its shadow in the photo.

    `top` and `base` are the pixels of the object's ends, `tip` the pixel of its shadow's tip, all on the floor
    except the top. The azimuth points from the shadow's tip back past the base; the elevation comes from the
    object's height, found where the top's camera ray comes closest to the vertical line through the base, over
    the shadow's length. Raises ValueError when the clicks can't describe a shadow.
    """
    plane = ground_model.ground_plane(ground)
    base_point = math3d.intersect_ray_plane((0.0, 0.0, 0.0), math3d.image_ray(base, fov, image_size), plane)
    if base_point is None:
        raise ValueError("The base must be on the floor, below the horizon")
    tip_point = math3d.intersect_ray_plane((0.0, 0.0, 0.0), math3d.image_ray(tip, fov, image_size), plane)
    if tip_point is None:
        raise ValueError("The shadow's tip must be on the floor, below the horizon")
    shadow = tip_point - base_point
    length = float(np.linalg.norm(shadow))
    if length < 1e-6 * float(np.linalg.norm(base_point)):
        raise ValueError("The shadow has no length: click its tip away from the base")
    up = ground_model.ground_up(ground)
    height = math3d.closest_line_parameter(base_point, up, (0.0, 0.0, 0.0), math3d.image_ray(top, fov, image_size))
    if height is None:
        raise ValueError("The object's height can't be measured looking straight down on it")
    if height <= 0:
        raise ValueError("The top must be above the base")
    toward_light = ground_model.light_frame(ground).T @ (-shadow / length)
    azimuth = float(np.degrees(np.arctan2(toward_light[0], toward_light[2])))
    return azimuth, float(np.degrees(np.arctan2(height, length)))


class LightingEngine:
    def compute(self, light: LightParams, ground: GroundParams) -> LightingState:
        frame = ground_model.light_frame(ground)
        light_dir = frame @ math3d.direction_from_angles(light.azimuth, light.elevation)
        plane = ground_model.ground_plane(ground)

        shadow_dir = frame @ math3d.direction_from_angles(
            light.azimuth, max(light.elevation, MIN_SHADOW_ELEVATION)
        )
        return LightingState(
            light_dir=light_dir,
            light_color=srgb_to_linear(light.color) * light.intensity,
            ambient=srgb_to_linear(light.ambient_color) * light.ambient_intensity,
            ground_plane=plane,
            shadow_matrix=math3d.planar_shadow_matrix(plane, shadow_dir),
        )
