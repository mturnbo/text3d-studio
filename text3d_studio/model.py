"""Pure-data scene model. No Qt or OpenGL imports live here."""

from dataclasses import dataclass, field

import numpy as np

Color = tuple[float, float, float]


@dataclass
class TextParams:
    content: str = "HELLO"
    font_family: str = "Helvetica"
    bold: bool = True
    italic: bool = False
    bevel_width: float = 6.0  # in raster pixels; controls how wide the beveled rim is


@dataclass
class TransformParams:
    x: float = 0.0
    y: float = -1.5
    z: float = -12.0
    rot_x: float = 0.0  # ground tilt (pitch)
    rot_y: float = -20.0  # yaw: spins the text on its ground plane
    rot_z: float = 0.0  # ground roll
    size: float = 3.0  # world height of one em
    depth: float = 0.35  # extrusion depth as a fraction of size
    layers: int = 96  # number of duplicated layers in the stack


@dataclass
class MaterialParams:
    face_color: Color = (0.92, 0.78, 0.45)
    side_color: Color = (0.70, 0.52, 0.25)
    side_darkening: float = 0.45  # how much darker the back of the stack gets
    bevel_strength: float = 1.6
    specular: float = 0.35
    shininess: float = 32.0


@dataclass
class LightParams:
    azimuth: float = 35.0  # degrees around the ground normal; 0 = from the camera
    elevation: float = 40.0  # degrees above the ground plane
    intensity: float = 1.0
    color: Color = (1.0, 0.97, 0.90)
    ambient_color: Color = (0.55, 0.62, 0.75)
    ambient_intensity: float = 0.35


@dataclass
class ShadowParams:
    enabled: bool = True
    opacity: float = 0.6
    softness: float = 0.35
    color: Color = (0.08, 0.10, 0.14)


@dataclass
class CameraParams:
    fov: float = 40.0  # vertical field of view in degrees


@dataclass
class Scene:
    text: TextParams = field(default_factory=TextParams)
    transform: TransformParams = field(default_factory=TransformParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    light: LightParams = field(default_factory=LightParams)
    shadow: ShadowParams = field(default_factory=ShadowParams)
    camera: CameraParams = field(default_factory=CameraParams)


@dataclass
class Document:
    """The editable document: a background photo plus the 3D text scene."""

    scene: Scene = field(default_factory=Scene)
    background: np.ndarray | None = None  # (H, W, 4) uint8 RGBA, top row first
    background_path: str | None = None

    @property
    def native_size(self) -> tuple[int, int]:
        if self.background is None:
            return (1600, 1000)
        height, width = self.background.shape[:2]
        return (width, height)
