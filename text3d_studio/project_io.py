"""Project files (.t3d): the scene plus a reference to (or an embedded copy of) the background photo, as JSON."""

import base64
import dataclasses
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6.QtGui import QImage, QImageReader

from .image_io import SUPPORTED_EXTENSIONS, load_image, qimage_to_rgba, rgba_to_qimage
from .materials import sanitized_material_params
from .model import (
    CameraParams,
    Document,
    EffectsParams,
    GridParams,
    GroundParams,
    FIELD_RANGES,
    LightParams,
    MaskParams,
    MaterialParams,
    Scene,
    ShadowParams,
    TextObject,
    TextParams,
    TransformParams,
    WallParams,
)
from .text_raster import sanitized_text_params

PROJECT_EXTENSION = ".t3d"
PROJECT_FILTER = f"Text3D Studio project (*{PROJECT_EXTENSION})"
FORMAT_NAME = "text3d-studio-project"
# Bump when the schema changes incompatibly, and migrate older files in load_project before reading them.
FORMAT_VERSION = 1

# Colour fields are tuples in the model but lists in JSON; these are turned back into tuples on load.
COLOR_FIELDS = {
    GridParams: ("color", "accent_color"),
    MaterialParams: ("face_color", "side_color", "secondary_color", "outline_color", "emission_color"),
    LightParams: ("color", "ambient_color"),
    ShadowParams: ("color",),
}


@dataclass
class LoadedProject:
    document: Document
    warnings: list[str] = field(default_factory=list)
    missing_photo: str | None = None  # the photo the project refers to, when it couldn't be found or read
    photo_embedded: bool = False


def save_project(path: str, document: Document, embed_photo: bool = False) -> None:
    """Writes the project atomically: a failure part-way leaves any existing file at `path` untouched.

    A background without a path (it came embedded in a project) is always embedded, since there is nothing to link to.
    """
    project = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "scene": dataclasses.asdict(document.scene),
        "photo": _photo_entry(path, document, embed_photo),
        "mask": None if document.mask is None else {"embedded_png": _encode_mask_png(document.mask)},
    }
    directory = os.path.dirname(os.path.abspath(path))
    handle, temp_path = tempfile.mkstemp(prefix=".", suffix=PROJECT_EXTENSION + ".tmp", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(project, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        os.unlink(temp_path)
        raise


def _photo_entry(path: str, document: Document, embed_photo: bool) -> dict | None:
    if document.background is None:
        return None
    if embed_photo or document.background_path is None:
        return {"embedded_png": _encode_png(document.background)}
    photo = os.path.abspath(document.background_path)
    project_dir = os.path.dirname(os.path.abspath(path))
    return {"relative_path": os.path.relpath(photo, project_dir), "absolute_path": photo}


def _encode_png(pixels) -> str:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not rgba_to_qimage(pixels).save(buffer, "PNG"):
        raise OSError("Could not encode the photo as PNG")
    return base64.b64encode(bytes(data)).decode("ascii")


def load_project(path: str) -> LoadedProject:
    """Reads a project. Raises ValueError for a file that isn't a readable project of a supported version.

    Missing fields take their defaults and unknown ones are ignored. A missing photo only adds a warning.
    """
    try:
        with open(path, encoding="utf-8") as file:
            project = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{os.path.basename(path)} is not a valid project file (it may be damaged)") from exc
    if not isinstance(project, dict) or project.get("format") != FORMAT_NAME:
        raise ValueError(f"{os.path.basename(path)} is not a Text3D Studio project")
    version = project.get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"{os.path.basename(path)} has no valid project version")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{os.path.basename(path)} was saved by a newer Text3D Studio (project version {version}); "
            f"this version reads up to {FORMAT_VERSION}"
        )
    try:
        scene = _scene_from_dict(project.get("scene") or {})
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"{os.path.basename(path)} has a damaged scene: {exc}") from exc
    loaded = LoadedProject(Document(scene=scene))
    _load_photo(path, project.get("photo"), loaded)
    _load_mask(project.get("mask"), loaded)
    return loaded


def _scene_from_dict(data: dict) -> Scene:
    objects = [_object_from_dict(item) for item in data.get("objects") or []]
    if len({obj.id for obj in objects}) != len(objects):
        for index, obj in enumerate(objects):
            obj.id = index  # duplicate or missing ids would mix up the objects' cached glyphs
    scene = Scene(
        ground=_params_from_dict(GroundParams, data.get("ground")),
        grid=_params_from_dict(GridParams, data.get("grid")),
        light=_params_from_dict(LightParams, data.get("light")),
        shadow=_params_from_dict(ShadowParams, data.get("shadow")),
        camera=_params_from_dict(CameraParams, data.get("camera")),
        wall=_params_from_dict(WallParams, data.get("wall")),
        mask=_params_from_dict(MaskParams, data.get("mask")),
        effects=_params_from_dict(EffectsParams, data.get("effects")),
    )
    if objects:
        scene.objects = objects
    if not isinstance(scene.wall.enabled, bool) or "enabled" not in (data.get("wall") or {}):
        # Projects from before walls caught shadows: the photo shows a wall if some text was mounted on it.
        scene.wall.enabled = any(obj.transform.mount == "wall" for obj in scene.objects)
    return scene


def _object_from_dict(data: dict) -> TextObject:
    return TextObject(
        id=data.get("id", 0),
        text=_params_from_dict(TextParams, data.get("text")),
        transform=_params_from_dict(TransformParams, data.get("transform")),
        material=_params_from_dict(MaterialParams, data.get("material")),
    )


def _params_from_dict(cls, data: dict | None):
    """One flat parameter section: known fields only, colours back to tuples, anything missing left at its default.

    A number that isn't a finite float (NaN, Infinity, or an int too large to convert), on its own or in a colour,
    takes the field's default, and text layout values are clamped to their ranges (material shading options too),
    so a crafted file can't reach the panel's controls or the renderer with them.
    """
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls)}
    values = {name: value for name, value in data.items() if name in names and not _is_unusable_number(value)}
    for name in COLOR_FIELDS.get(cls, ()):
        if name in values:
            if isinstance(values[name], list) and any(_is_unusable_number(c) for c in values[name]):
                del values[name]
            else:
                values[name] = tuple(values[name])
    params = cls(**values)
    for name, (low, high) in FIELD_RANGES.get(cls, {}).items():
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            setattr(params, name, getattr(cls(), name))
        else:
            setattr(params, name, min(max(float(value), low), high))
    if cls is TextParams:
        return sanitized_text_params(params)
    if cls is MaterialParams:
        return sanitized_material_params(params)
    return params


def _is_unusable_number(value) -> bool:
    """A number that isn't a finite float: NaN, Infinity, or an int too large to convert (JSON allows all three)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return not math.isfinite(float(value))
    except OverflowError:
        return True


def _load_photo(path: str, photo: dict | None, loaded: LoadedProject) -> None:
    if not photo:
        return
    if "embedded_png" in photo:
        image = QImage()
        try:
            data = base64.b64decode(photo["embedded_png"], validate=True)
        except ValueError:
            data = b""
        if not image.loadFromData(data, "PNG"):
            loaded.warnings.append("The photo embedded in this project could not be read")
            return
        loaded.document.background = qimage_to_rgba(image)
        loaded.photo_embedded = True
        return
    project_dir = os.path.dirname(os.path.abspath(path))
    candidates = []  # (as stored, resolved), each checked before any filesystem call
    relative = photo.get("relative_path")
    if relative:
        if _is_rooted(relative):
            candidates.append((relative, None))
        else:
            candidates.append((relative, os.path.normpath(os.path.join(project_dir, relative))))
    absolute = photo.get("absolute_path")
    if absolute:
        candidates.append((absolute, os.path.normpath(absolute) if is_allowed_photo_path(absolute) else None))
    reasons = []
    for stored, candidate in candidates:
        # A shared project can name any path: never open network shares or non-image files it points at.
        if candidate is None or not is_allowed_photo_path(candidate):
            reasons.append(f"not a local path: {stored}")
            continue
        if not candidate.lower().endswith(SUPPORTED_EXTENSIONS):
            reasons.append(f"not a supported image type: {stored}")
            continue
        if os.path.isfile(candidate):
            try:
                loaded.document.background = load_image(candidate)
            except ValueError:
                continue
            loaded.document.background_path = candidate
            return
    loaded.missing_photo = absolute or relative or "(unknown)"
    detail = f" ({'; '.join(reasons)})" if reasons else ""
    loaded.warnings.append(f"Photo not found: {loaded.missing_photo}{detail}")


def _encode_mask_png(mask) -> str:
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    height, width = mask.shape
    image = QImage(mask.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise OSError("Could not encode the occlusion mask as PNG")
    return base64.b64encode(bytes(data)).decode("ascii")


def _load_mask(entry, loaded: LoadedProject) -> None:
    """The occlusion mask, a grayscale PNG of exactly the photo's size. A crafted file can carry anything, so the
    PNG's size is read from its header and checked against the loaded photo before any pixels are decoded: a mask
    without a photo, of another size, or that can't be read is left out with a warning."""
    if not entry:
        return
    photo = loaded.document.background
    if photo is None:
        loaded.warnings.append("The occlusion mask was left out because the project's photo couldn't be loaded")
        return
    try:
        data = base64.b64decode(entry.get("embedded_png", "") if isinstance(entry, dict) else "", validate=True)
    except (ValueError, TypeError):
        data = b""
    buffer = QBuffer()
    buffer.setData(QByteArray(data))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer, b"PNG")
    size = reader.size()  # from the header; nothing is decoded yet
    if not size.isValid():
        loaded.warnings.append("The occlusion mask in this project could not be read")
        return
    if (size.height(), size.width()) != photo.shape[:2]:
        loaded.warnings.append("The occlusion mask doesn't match the photo's size and was left out")
        return
    image = reader.read()
    if image.isNull() or (image.height(), image.width()) != photo.shape[:2]:
        loaded.warnings.append("The occlusion mask in this project could not be read")
        return
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width, height = image.width(), image.height()
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    loaded.document.mask = np.frombuffer(ptr, np.uint8).reshape(height, image.bytesPerLine())[:, :width].copy()


def is_allowed_photo_path(path: str, windows: bool = os.name == "nt") -> bool:
    """Allowlist for an absolute photo path read from a project: local drives only, never network locations.

    Windows: only drive-letter paths (C:\\...); this excludes UNC shares and device or extended-length prefixes.
    POSIX: absolute paths, except "//" roots and the /net and /Network automount roots.
    """
    if windows:
        return re.match(r"^[A-Za-z]:\\", path.replace("/", "\\")) is not None
    lowered = path.lower()  # macOS file names are case-insensitive
    if not path.startswith("/") or path.startswith("//"):
        return False
    return not any(lowered == root or lowered.startswith(root + "/") for root in ("/net", "/network"))


def _is_rooted(path: str) -> bool:
    """A relative_path that names a root or a drive isn't relative: treat it as untrusted."""
    return path.replace("\\", "/").startswith("/") or re.match(r"^[A-Za-z]:", path) is not None
