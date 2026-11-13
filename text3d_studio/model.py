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
    align: str = "center"  # "left", "center" or "right": how the lines line up, relative to the widest one
    letter_spacing: float = 0.0  # extra space after each letter, as a fraction of the em; negative tightens
    line_height: float = 1.0  # distance between baselines, as a multiple of the font's line spacing
    arc: float = 0.0  # degrees the widest line bends through (-180..180); positive bends the middle up


@dataclass
class TransformParams:
    """Where the text stands on the ground (see ground.py); x and z run along the ground's axes."""

    x: float = 0.0
    z: float = -12.0  # negative is away from the camera
    rot_y: float = -20.0  # yaw: spins the text on the ground
    size: float = 3.0  # world height of one em
    depth: float = 0.35  # extrusion depth as a fraction of size
    layers: int = 96  # number of duplicated layers in the stack
    mount: str = "floor"  # "floor": stands on the ground at (x, z); "wall": lies flat on the wall at (wall_x, wall_y)
    wall_x: float = 0.0  # distance along the wall's base line
    wall_y: float = 0.0  # height of the ink's bottom above the floor


@dataclass
class GroundParams:
    """Pose of the ground plane relative to the camera. The text stands on this plane."""

    pitch: float = 0.0  # tilt toward the camera; positive looks down onto the ground
    roll: float = 0.0  # tilt around the view axis
    yaw: float = 0.0  # spins the ground's grid axes (and the text placed on them) within the plane
    height: float = 1.5  # camera height above the ground, in world units


@dataclass
class WallParams:
    """A vertical wall standing on the ground, set by its base line in ground coordinates (see ground.py)."""

    angle: float = 0.0  # facing of the wall, like transform.rot_y: 0 runs the base line along the ground's x axis
    offset: float = 20.0  # distance of the base line from the point below the camera
    enabled: bool = False  # the photo shows this wall: it catches shadows, and floor shadows stop at its base


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
    kind: str = "metal"  # one of materials.KINDS; picking a kind loads its preset (see materials.py)
    face_color: Color = (0.92, 0.78, 0.45)
    side_color: Color = (0.70, 0.52, 0.25)
    side_darkening: float = 0.45  # how much darker the back of the stack gets
    bevel_strength: float = 1.6
    metallic: float = 1.0  # 0 = dielectric (plastic, stone, wood), 1 = metal; colors the highlight
    roughness: float = 0.35  # 0 = mirror-polished, 1 = fully matte
    # Procedural textures (stone, wood). Patterns live in the text's own space, in em units.
    secondary_color: Color = (0.70, 0.52, 0.25)  # stone veins, wood's dark rings
    texture_scale: float = 1.0  # feature size (stone grain, wood ring spacing)
    texture_strength: float = 0.0  # how strongly the pattern shows
    bump_strength: float = 0.0  # surface relief from the pattern
    grain_angle: float = 0.0  # wood grain direction on the face, degrees (0 = horizontal)
    varnish: float = 0.0  # glossy clear coat over the material
    absorption: float = 0.0  # glass: how quickly the tint deepens with the thickness light passes through
    side_kind: str = "same"  # "same", or a kind whose preset look the side walls and back cap take (materials.SIDE_KINDS)
    # A band of a second material along the face's edge, following the glyph's contour.
    outline_width: float = 0.0  # in bevel widths: 1 covers the whole beveled rim, 0 = no outline
    outline_color: Color = (0.08, 0.08, 0.08)
    outline_metallic: float = 0.0
    # Unlit glow (neon), added on top of the shading; the side walls and back cap glow dimmer.
    emission_color: Color = (1.0, 0.25, 0.75)
    emission_strength: float = 0.0


@dataclass
class LightParams:
    azimuth: float = 35.0  # degrees around the ground normal; 0 = from the camera
    elevation: float = 40.0  # degrees above the ground plane
    intensity: float = 1.0
    color: Color = (1.0, 0.97, 0.90)
    ambient_color: Color = (0.55, 0.62, 0.75)
    ambient_intensity: float = 0.35
    reflect_photo: bool = True  # glossy surfaces reflect a blurred copy of the background photo
    reflection_strength: float = 1.0


@dataclass
class ShadowParams:
    enabled: bool = True
    opacity: float = 0.6
    softness: float = 0.2
    falloff: float = 0.8  # how quickly the shadow fades for parts of the text far from the ground
    color: Color = (0.08, 0.10, 0.14)
    contact_strength: float = 0.0  # a tight, dark shadow where floor text meets the ground; 0 = off
    contact_size: float = 0.15  # how far up the text (in em) the contact shadow reaches, and so how far it spreads


@dataclass
class CameraParams:
    fov: float = 40.0  # vertical field of view in degrees


@dataclass
class MaskParams:
    """The Mask tool's brush. The occlusion mask itself is per photo and lives on the Document."""

    brush_size: float = 0.05  # diameter, as a fraction of the photo's height
    feather: float = 0.5  # how much of the brush's radius fades out softly


@dataclass
class EffectsParams:
    """Finishing passes over the composite. All of them are off (or neutral) by default."""

    reflection_strength: float = 0.0  # a mirrored copy of the floor text on the ground; 0 = off
    reflection_roughness: float = 0.3  # blurs the reflection
    reflection_fade: float = 3.0  # world units below the ground over which the reflection fades out
    bloom_strength: float = 0.0  # bright parts of the text glow; 0 = off
    bloom_threshold: float = 0.8  # brightness (0-1) above which the text glows
    bloom_radius: float = 0.3  # how far the glow spreads, relative to the image height
    grain_amount: float = 0.0  # film grain on the text; 0 = off
    grain_size: float = 1.0  # size of a grain, in thousandths of the image height


@dataclass
class TextObject:
    """One piece of 3D text: what it says, where it stands and what it's made of."""

    id: int = 0  # stable and unique within a scene; the renderer keys its glyph rasters by it
    text: TextParams = field(default_factory=TextParams)
    transform: TransformParams = field(default_factory=TransformParams)
    material: MaterialParams = field(default_factory=MaterialParams)


@dataclass
class Scene:
    """The text objects plus the sections they share. There is always at least one object."""

    objects: list[TextObject] = field(default_factory=lambda: [TextObject()])
    ground: GroundParams = field(default_factory=GroundParams)
    grid: GridParams = field(default_factory=GridParams)
    light: LightParams = field(default_factory=LightParams)
    shadow: ShadowParams = field(default_factory=ShadowParams)
    camera: CameraParams = field(default_factory=CameraParams)
    wall: WallParams = field(default_factory=WallParams)
    mask: MaskParams = field(default_factory=MaskParams)
    effects: EffectsParams = field(default_factory=EffectsParams)


@dataclass
class Document:
    """The editable document: a background photo plus the 3D text scene."""

    scene: Scene = field(default_factory=Scene)
    background: np.ndarray | None = None  # (H, W, 4) uint8 RGBA, top row first
    background_path: str | None = None
    # Occlusion mask: (H, W) uint8 at the photo's size, 255 where the photo is in front of the text. None: no mask.
    mask: np.ndarray | None = None

    @property
    def native_size(self) -> tuple[int, int]:
        if self.background is None:
            return (1600, 1000)
        height, width = self.background.shape[:2]
        return (width, height)


# Ranges of numeric fields, shared by the panel's sliders and project loading (which clamps to them).
FIELD_RANGES = {
    ShadowParams: {"contact_strength": (0.0, 1.0), "contact_size": (0.02, 1.0)},
    MaskParams: {"brush_size": (0.005, 0.5), "feather": (0.0, 1.0)},
    EffectsParams: {
        "reflection_strength": (0.0, 1.0),
        "reflection_roughness": (0.0, 1.0),
        "reflection_fade": (0.1, 30.0),
        "bloom_strength": (0.0, 2.0),
        "bloom_threshold": (0.0, 1.0),
        "bloom_radius": (0.0, 1.0),
        "grain_amount": (0.0, 1.0),
        "grain_size": (0.5, 5.0),
    },
}
