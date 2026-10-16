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
    """Where the text stands on the ground (see ground.py); x and z run along the ground's axes."""

    x: float = 0.0
    z: float = -12.0  # negative is away from the camera
    rot_y: float = -20.0  # yaw: spins the text on the ground
    size: float = 3.0  # world height of one em
    depth: float = 0.35  # extrusion depth as a fraction of size
    layers: int = 96  # number of duplicated layers in the stack


@dataclass
class GroundParams:
    """Pose of the ground plane relative to the camera. The text stands on this plane."""

    pitch: float = 0.0  # tilt toward the camera; positive looks down onto the ground
    roll: float = 0.0  # tilt around the view axis
    yaw: float = 0.0  # spins the ground's grid axes (and the text placed on them) within the plane
    height: float = 1.5  # camera height above the ground, in world units


@dataclass
class GridParams:
    """Editor overlay drawn on the ground plane. Never part of an export."""

    visible: bool = False
    cell_size: float = 1.0  # world units between grid lines
    offset_x: float = 0.0  # slides the grid lines (not the ground) to line them up with floor seams
    offset_z: float = 0.0
    opacity: float = 0.6
    color: Color = (0.85, 0.90, 1.0)
    accent_color: Color = (1.0, 0.70, 0.20)  # the text's footprint and the lanes it slides along
    # Fit to Rectangle: the real size of the rectangle being traced sets the ground's scale (camera height).
    fit_width: float = 1.0  # along its first edge
    fit_depth: float = 1.0  # along its second edge
    fit_estimate_fov: bool = False  # take the field of view from the rectangle's vanishing points


@dataclass
class MaterialParams:
    face_color: Color = (0.92, 0.78, 0.45)
    side_color: Color = (0.70, 0.52, 0.25)
    side_darkening: float = 0.45  # how much darker the back of the stack gets
    bevel_strength: float = 1.6
    metallic: float = 1.0  # 0 = dielectric (plastic, stone, wood), 1 = metal; colors the highlight
    roughness: float = 0.35  # 0 = mirror-polished, 1 = fully matte


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
    softness: float = 0.2
    falloff: float = 0.8  # how quickly the shadow fades for parts of the text far from the ground
    color: Color = (0.08, 0.10, 0.14)


@dataclass
class CameraParams:
    fov: float = 40.0  # vertical field of view in degrees


@dataclass
class Scene:
    text: TextParams = field(default_factory=TextParams)
    transform: TransformParams = field(default_factory=TransformParams)
    ground: GroundParams = field(default_factory=GroundParams)
    grid: GridParams = field(default_factory=GridParams)
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
