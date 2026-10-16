"""ViewModel: the single mutation point for the Document, exposing Qt signals to views."""

import dataclasses

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from . import ground as ground_model
from .model import Document

# Fields that change the rasterized glyph mask (expensive, CPU) rather than just uniforms (cheap, GPU).
RASTER_FIELDS = {"content", "font_family", "bold", "italic", "bevel_width"}


class SceneViewModel(QObject):
    scene_changed = pyqtSignal(str, str)  # (section, field)
    text_changed = pyqtSignal()
    background_changed = pyqtSignal()

    def __init__(self, document: Document | None = None):
        super().__init__()
        self.document = document or Document()

    @property
    def scene(self):
        return self.document.scene

    def get(self, section: str, name: str):
        return getattr(self._section(section, name), name)

    def set(self, section: str, name: str, value) -> None:
        target = self._section(section, name)
        if getattr(target, name) == value:
            return
        old_ground = dataclasses.replace(self.scene.ground) if section == "ground" else None
        self._assign(target, section, name, value)
        if old_ground is not None:
            self._place_text_from(old_ground, dataclasses.replace(self.scene.transform))

    def set_ground_pose(self, anchor=None, **values) -> None:
        """Sets several ground fields together and re-places the text once, against the final pose.

        Setting them one by one could pass through a pose whose horizon sits above the text and lose it.
        `anchor` is an optional (ground, transform) to re-place the text from instead of the current state:
        drags pass the state from when they started, so a pose mid-drag that can't hold the text doesn't
        strand it for the rest of the drag.
        """
        old_ground = dataclasses.replace(self.scene.ground)
        base = anchor if anchor is not None else (old_ground, dataclasses.replace(self.scene.transform))
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

    def nudge(self, section: str, name: str, delta: float) -> None:
        self.set(section, name, self.get(section, name) + delta)

    def set_background(self, pixels: np.ndarray, path: str | None = None) -> None:
        self.document.background = pixels
        self.document.background_path = path
        self.background_changed.emit()

    def _section(self, section: str, name: str):
        target = getattr(self.document.scene, section, None)
        if target is None or not dataclasses.is_dataclass(target):
            raise KeyError(f"Unknown scene section: {section!r}")
        if name not in {f.name for f in dataclasses.fields(target)}:
            raise KeyError(f"Unknown field {name!r} in section {section!r}")
        return target
