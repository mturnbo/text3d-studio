"""ViewModel: the single mutation point for the Document, exposing Qt signals to views."""

import dataclasses

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from . import ground as ground_model
from .materials import KINDS, PRESETS
from .model import Document, TextObject, TransformParams

# Fields that change the rasterized glyph mask (expensive, CPU) rather than just uniforms (cheap, GPU).
RASTER_FIELDS = {"content", "font_family", "bold", "italic", "bevel_width"}
OBJECT_SECTIONS = ("text", "transform", "material")  # per-object sections; the rest are shared by the scene


class SceneViewModel(QObject):
    scene_changed = pyqtSignal(str, str)  # (section, field)
    text_changed = pyqtSignal()
    background_changed = pyqtSignal()
    objects_changed = pyqtSignal()  # an object was added or deleted
    selection_changed = pyqtSignal(int)  # the selected object's index; its text, transform and material changed

    def __init__(self, document: Document | None = None):
        super().__init__()
        self.document = document or Document()
        self.selected = 0  # editor state: the object that "text", "transform" and "material" edits address
        # Ids are never reused, so nothing keyed by a deleted object's id (a cached raster) can match a new one.
        self._next_object_id = max(o.id for o in self.scene.objects) + 1

    @property
    def scene(self):
        return self.document.scene

    @property
    def selected_object(self) -> TextObject:
        return self.scene.objects[self.selected]

    def get(self, section: str, name: str):
        return getattr(self._section(section, name), name)

    def set(self, section: str, name: str, value) -> None:
        target = self._section(section, name)
        if getattr(target, name) == value:
            return
        old_ground = dataclasses.replace(self.scene.ground) if section == "ground" else None
        self._assign(target, section, name, value)
        if old_ground is not None:
            self._place_text_from(old_ground, dataclasses.replace(self.selected_object.transform))

    def set_ground_pose(self, anchor=None, **values) -> None:
        """Sets several ground fields together and re-places the text once, against the final pose.

        Setting them one by one could pass through a pose whose horizon sits above the text and lose it.
        `anchor` is an optional (ground, transform) to re-place the text from instead of the current state:
        drags pass the state from when they started, so a pose mid-drag that can't hold the text doesn't
        strand it for the rest of the drag.
        """
        old_ground = dataclasses.replace(self.scene.ground)
        base = anchor if anchor is not None else (old_ground, dataclasses.replace(self.selected_object.transform))
        for name, value in values.items():
            target = self._section("ground", name)
            if getattr(target, name) != value:
                self._assign(target, "ground", name, value)
        if self.scene.ground != old_ground:
            self._place_text_from(*base)

    def _assign(self, target, section: str, name: str, value) -> None:
        setattr(target, name, value)
        if section == "text" and name in RASTER_FIELDS:
            self.text_changed.emit()
        self.scene_changed.emit(section, name)

    def _place_text_from(self, base_ground, base_transform) -> None:
        """Aligning the ground to the photo shouldn't fling the text around: keep it where it appeared on
        `base_ground`. When that's impossible on the new ground, fall back to the base placement."""
        placement = ground_model.reanchor_text(base_ground, self.scene.ground, base_transform)
        if placement is None:
            placement = (base_transform.x, base_transform.z, base_transform.rot_y, base_transform.size)
        for name, value in zip(("x", "z", "rot_y", "size"), placement):
            self.set("transform", name, value)

    def apply_material(self, kind: str) -> None:
        """Switches the material kind and loads its preset values (replacing any adjustments)."""
        if kind not in KINDS:
            raise ValueError(f"Unknown material: {kind!r}")
        self.set("material", "kind", kind)
        for name, value in PRESETS[kind].items():
            self.set("material", name, value)

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.scene.objects):
            raise IndexError(f"No text object at index {index}")
        if index != self.selected:
            self.selected = index
            self.selection_changed.emit(index)

    def add_object(self) -> None:
        """Adds a default text object in front of the selected one and selects it."""
        base = ground_model.placement_in_front(self.scene.ground, self.selected_object.transform)
        transform = TransformParams(x=base.x, z=base.z, rot_y=base.rot_y, size=base.size)
        self._insert(TextObject(id=self._next_id(), transform=transform))

    def duplicate_object(self) -> None:
        """Adds a copy of the selected object in front of it and selects the copy."""
        source = self.selected_object
        self._insert(TextObject(
            id=self._next_id(),
            text=dataclasses.replace(source.text),
            transform=ground_model.placement_in_front(self.scene.ground, source.transform),
            material=dataclasses.replace(source.material),
        ))

    def delete_object(self) -> bool:
        """Deletes the selected object. Refuses (returns False) when it is the last one."""
        if len(self.scene.objects) == 1:
            return False
        del self.scene.objects[self.selected]
        self.selected = min(self.selected, len(self.scene.objects) - 1)
        self.objects_changed.emit()
        self.selection_changed.emit(self.selected)
        return True

    def _insert(self, obj: TextObject) -> None:
        self.scene.objects.append(obj)
        self.objects_changed.emit()
        self.select(len(self.scene.objects) - 1)

    def _next_id(self) -> int:
        self._next_object_id += 1
        return self._next_object_id - 1

    def nudge(self, section: str, name: str, delta: float) -> None:
        self.set(section, name, self.get(section, name) + delta)

    def set_background(self, pixels: np.ndarray, path: str | None = None) -> None:
        self.document.background = pixels
        self.document.background_path = path
        self.background_changed.emit()

    def _section(self, section: str, name: str):
        owner = self.selected_object if section in OBJECT_SECTIONS else self.scene
        target = getattr(owner, section, None)
        if target is None or not dataclasses.is_dataclass(target):
            raise KeyError(f"Unknown scene section: {section!r}")
        if name not in {f.name for f in dataclasses.fields(target)}:
            raise KeyError(f"Unknown field {name!r} in section {section!r}")
        return target
