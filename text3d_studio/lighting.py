"""LightingEngine: turns user-facing light/shadow params into world-space shader inputs."""

from dataclasses import dataclass

import numpy as np

from . import ground as ground_model
from . import math3d
from .model import GroundParams, LightParams

MIN_SHADOW_ELEVATION = 3.0  # below this a planar shadow stretches toward infinity


@dataclass
class LightingState:
    light_dir: np.ndarray  # world space, unit, pointing toward the light
    light_color: np.ndarray  # color * intensity
    ambient: np.ndarray  # ambient color * intensity
    ground_plane: np.ndarray  # (a, b, c, d) world plane the shadow falls on
    shadow_matrix: np.ndarray  # 4x4 planar projection onto the ground plane


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
            light_color=np.array(light.color) * light.intensity,
            ambient=np.array(light.ambient_color) * light.ambient_intensity,
            ground_plane=plane,
            shadow_matrix=math3d.planar_shadow_matrix(plane, shadow_dir),
        )
