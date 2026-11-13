"""Looks: a material for the selected object plus the scene's light and shadow, applied in one step.

Built-in looks ship with the app. User looks are JSON files in a folder (one per look); a damaged or foreign
file is skipped with a warning. No Qt imports: the folder is passed in (see Preferences.looks_directory).
"""

import dataclasses
import json
import os
import re
import tempfile
from dataclasses import dataclass

from .materials import KINDS, PRESETS, SIDE_KINDS
from .model import LightParams, MaterialParams, ShadowParams

FORMAT_NAME = "text3d-studio-look"
FORMAT_VERSION = 1
LOOK_EXTENSION = ".json"
SECTIONS = {"material": MaterialParams, "light": LightParams, "shadow": ShadowParams}


@dataclass(frozen=True)
class Look:
    name: str
    material: dict
    light: dict
    shadow: dict
    builtin: bool = False
    path: str | None = None  # the user look's file


def _builtin(name, kind, material=None, light=None, shadow=None) -> Look:
    return Look(
        name=name,
        material={"kind": kind, **PRESETS[kind], **(material or {})},
        light=dataclasses.asdict(LightParams(**(light or {}))),
        shadow=dataclasses.asdict(ShadowParams(**(shadow or {}))),
        builtin=True,
    )


BUILTIN_LOOKS = (
    _builtin(
        "Gold at sunset", "metal",
        material=dict(face_color=(1.0, 0.78, 0.36), side_color=(0.78, 0.50, 0.20), roughness=0.25),
        light=dict(azimuth=60.0, elevation=12.0, intensity=1.2, color=(1.0, 0.72, 0.45),
                   ambient_color=(0.55, 0.45, 0.55), ambient_intensity=0.3),
        shadow=dict(opacity=0.65, softness=0.15, color=(0.18, 0.10, 0.12)),
    ),
    _builtin(
        "Chrome at noon", "metal",
        material=dict(face_color=(0.95, 0.95, 0.97), side_color=(0.80, 0.80, 0.82),
                      secondary_color=(0.80, 0.80, 0.82), roughness=0.08, side_darkening=0.35),
        light=dict(azimuth=20.0, elevation=70.0, intensity=1.1, color=(1.0, 1.0, 1.0),
                   ambient_color=(0.60, 0.70, 0.85), ambient_intensity=0.4),
        shadow=dict(opacity=0.75, softness=0.05, color=(0.05, 0.06, 0.08)),
    ),
    _builtin(
        "Frosted glass, overcast", "glass",
        material=dict(roughness=0.35),
        light=dict(azimuth=30.0, elevation=55.0, intensity=0.6, color=(0.92, 0.95, 1.0),
                   ambient_color=(0.75, 0.80, 0.88), ambient_intensity=0.6),
        shadow=dict(opacity=0.35, softness=0.6, color=(0.15, 0.17, 0.20)),
    ),
    _builtin(
        "Marble", "stone",
        material=dict(face_color=(0.93, 0.92, 0.90), side_color=(0.93, 0.92, 0.90),
                      secondary_color=(0.45, 0.45, 0.50), texture_strength=0.7, bump_strength=0.1,
                      roughness=0.3, varnish=0.4),
        light=dict(azimuth=35.0, elevation=45.0),
        shadow=dict(opacity=0.6, softness=0.25),
    ),
    _builtin(
        "Oak", "wood",
        material=dict(face_color=(0.72, 0.53, 0.32), side_color=(0.72, 0.53, 0.32),
                      secondary_color=(0.42, 0.26, 0.13), varnish=0.25),
        light=dict(azimuth=40.0, elevation=35.0, color=(1.0, 0.95, 0.86)),
        shadow=dict(opacity=0.6, softness=0.3),
    ),
)


def apply_look(vm, look: Look) -> None:
    """Sets the selected object's material and the scene's light and shadow as one undo step. Fields the look
    doesn't name (or this version doesn't know) are left as they are."""
    vm.begin_edit()
    for section, cls in SECTIONS.items():
        names = {f.name for f in dataclasses.fields(cls)}
        for name, value in getattr(look, section).items():
            if name in names:
                vm.set(section, name, value)
    vm.end_edit()


def capture_look(vm, name: str) -> Look:
    """The selected object's material and the scene's light and shadow, as a look named `name`."""
    return Look(
        name=name.strip(),
        material=dataclasses.asdict(vm.selected_object.material),
        light=dataclasses.asdict(vm.scene.light),
        shadow=dataclasses.asdict(vm.scene.shadow),
    )


def load_user_looks(folder: str) -> tuple[list[Look], list[str]]:
    """Every readable look in `folder`, sorted by name, and a warning for each file that was skipped."""
    if not os.path.isdir(folder):
        return [], []
    looks, warnings = [], []
    taken = {look.name.casefold() for look in BUILTIN_LOOKS}
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(LOOK_EXTENSION):
            continue
        path = os.path.join(folder, filename)
        try:
            look = _read_look(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"Skipped look file {filename}: {exc}")
            continue
        if look.name.casefold() in taken:
            warnings.append(f"Skipped look file {filename}: another look is already called “{look.name}”")
            continue
        taken.add(look.name.casefold())
        looks.append(look)
    return sorted(looks, key=lambda look: look.name.casefold()), warnings


def _read_look(path: str) -> Look:
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("not valid JSON") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise ValueError("not a Text3D Studio look")
    version = data.get("version")
    if not isinstance(version, int) or not 1 <= version <= FORMAT_VERSION:
        raise ValueError(f"unsupported look version {version!r}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("the look has no name")
    sections = {section: _section_from_dict(cls, data.get(section)) for section, cls in SECTIONS.items()}
    if sections["material"].get("kind", "metal") not in KINDS:
        raise ValueError(f"unknown material {sections['material']['kind']!r}")
    if sections["material"].get("side_kind", "same") not in SIDE_KINDS:
        raise ValueError(f"unknown side material {sections['material']['side_kind']!r}")
    return Look(name=name.strip(), path=path, **sections)


def _section_from_dict(cls, data) -> dict:
    """Known fields only, each checked against its default's type; colours become tuples."""
    if not isinstance(data, dict):
        raise ValueError(f"missing {cls.__name__}")
    defaults = dataclasses.asdict(cls())
    values = {}
    for name, value in data.items():
        if name not in defaults:
            continue  # written by a newer version
        default = defaults[name]
        if isinstance(default, tuple):
            if not (isinstance(value, list) and len(value) == 3 and all(_is_number(c) for c in value)):
                raise ValueError(f"{name} is not a colour")
            value = tuple(float(c) for c in value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} is not true or false")
        elif isinstance(default, (int, float)):
            if not _is_number(value):
                raise ValueError(f"{name} is not a number")
            value = type(default)(value)
        elif not isinstance(value, type(default)):
            raise ValueError(f"{name} has the wrong type")
        values[name] = value
    return values


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def save_user_look(folder: str, look: Look, replace: bool = False) -> Look:
    """Writes `look` into `folder` and returns it with its path.

    Raises ValueError for an empty name or a built-in's name, and FileExistsError when a user look already has
    the name (names compare case-insensitively) unless `replace`, which overwrites that look's file.
    """
    name = look.name.strip()
    if not name:
        raise ValueError("A look needs a name")
    if any(b.name.casefold() == name.casefold() for b in BUILTIN_LOOKS):
        raise ValueError(f"“{name}” is a built-in look; choose another name")
    existing = next((u for u in load_user_looks(folder)[0] if u.name.casefold() == name.casefold()), None)
    if existing is not None and not replace:
        raise FileExistsError(f"A look called “{existing.name}” already exists")
    os.makedirs(folder, exist_ok=True)
    path = existing.path if existing is not None else _new_look_path(folder, name)
    data = {"format": FORMAT_NAME, "version": FORMAT_VERSION, "name": name,
            "material": look.material, "light": look.light, "shadow": look.shadow}
    handle, temp_path = tempfile.mkstemp(prefix=".", suffix=LOOK_EXTENSION + ".tmp", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        os.replace(temp_path, path)
    except BaseException:
        os.unlink(temp_path)
        raise
    return dataclasses.replace(look, name=name, builtin=False, path=path)


def _new_look_path(folder: str, name: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "_", name).strip(" _") or "look"
    path = os.path.join(folder, stem + LOOK_EXTENSION)
    number = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}-{number}{LOOK_EXTENSION}")
        number += 1
    return path


def delete_user_look(look: Look) -> None:
    if look.builtin or look.path is None:
        raise ValueError(f"“{look.name}” is a built-in look and can't be deleted")
    os.remove(look.path)
