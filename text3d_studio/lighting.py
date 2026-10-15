"""LightingEngine: turns user-facing light/shadow params into world-space shader inputs."""

from dataclasses import dataclass

import numpy as np

from . import math3d
from .model import LightParams, TransformParams

MIN_SHADOW_ELEVATION = 3.0  # below this a planar shadow stretches toward infinity


@dataclass
class LightingState:
    light_dir: np.ndarray  # world space, unit, pointing toward the light
    light_color: np.ndarray  # color * intensity
    ambient: np.ndarray  # ambient color * intensity
    ground_plane: np.ndarray  # (a, b, c, d) world plane the shadow falls on
    shadow_matrix: np.ndarray  # 4x4 planar projection onto the ground plane


class LightingEngine:
    def compute(self, light: LightParams, transform: TransformParams) -> LightingState:
        frame = math3d.ground_frame(transform.rot_x, transform.rot_z)[:3, :3]
        local_dir = math3d.direction_from_angles(light.azimuth, light.elevation)
        light_dir = frame @ local_dir

        ground_normal = frame @ np.array([0.0, 1.0, 0.0])
        ground_point = np.array([transform.x, transform.y, transform.z])
        plane = math3d.plane_from_point_normal(ground_point, ground_normal)

        shadow_dir = frame @ math3d.direction_from_angles(
            light.azimuth, max(light.elevation, MIN_SHADOW_ELEVATION)
        )
        return LightingState(
            light_dir=light_dir,
            light_color=np.array(light.color) * light.intensity,
            ambient=np.array(light.ambient_color) * light.ambient_intensity,
            ground_plane=plane,
            shadow_matrix=math3d.planar_shadow_matrix(plane, shadow_dir),
        )
