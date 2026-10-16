"""ControlPanel: declarative slider/picker layout bound two-way to the SceneViewModel."""

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..ground import MAX_FOV, MAX_PITCH, MAX_ROLL, MIN_FOV
from ..materials import KINDS
from ..viewmodel import SceneViewModel

# (field, label, minimum, maximum, decimals)
SLIDERS = {
    "text": [("bevel_width", "Bevel width", 1, 20, 1)],
    "transform": [
        ("x", "Position X", -30, 30, 2),
        ("z", "Position Z", -150, -0.5, 2),
        ("rot_y", "Yaw", -180, 180, 1),
        ("size", "Size", 0.2, 30, 2),
        ("depth", "Extrude depth", 0, 2, 2),
        ("layers", "Layer count", 2, 256, 0),
    ],
    "ground": [
        ("pitch", "Tilt", -MAX_PITCH, MAX_PITCH, 1),
        ("roll", "Roll", -MAX_ROLL, MAX_ROLL, 1),
        ("yaw", "Grid rotation", -180, 180, 1),
        ("height", "Camera height", 0.05, 30, 2),
    ],
    "grid": [
        ("cell_size", "Cell size", 0.1, 10, 2),
        ("opacity", "Opacity", 0, 1, 2),
        ("fit_width", "Fit width", 0.1, 20, 2),
        ("fit_depth", "Fit depth", 0.1, 20, 2),
    ],
    "camera": [("fov", "Field of view", MIN_FOV, MAX_FOV, 1)],
    "light": [
        ("azimuth", "Azimuth", -180, 180, 1),
        ("elevation", "Elevation", 0, 90, 1),
        ("intensity", "Intensity", 0, 3, 2),
        ("ambient_intensity", "Ambient", 0, 1.5, 2),
        ("reflection_strength", "Reflections", 0, 2, 2),
    ],
    "material": [
        ("side_darkening", "Depth darkening", 0, 1, 2),
        ("bevel_strength", "Bevel strength", 0, 4, 2),
        ("metallic", "Metallic", 0, 1, 2),
        ("roughness", "Roughness", 0.02, 1, 2),
        ("texture_scale", "Pattern scale", 0.02, 5, 2),
        ("texture_strength", "Pattern strength", 0, 1, 2),
        ("grain_angle", "Grain direction", -90, 90, 0),
        ("bump_strength", "Bump", 0, 1, 2),
        ("varnish", "Varnish", 0, 1, 2),
        ("absorption", "Absorption", 0, 5, 2),
    ],
    "shadow": [
        ("opacity", "Opacity", 0, 1, 2),
        ("softness", "Softness", 0, 1, 2),
        ("falloff", "Falloff", 0, 3, 2),
    ],
}

CHECKBOXES = {
    "grid": [("visible", "Show grid"), ("fit_estimate_fov", "Fit: estimate field of view")],
    "light": [("reflect_photo", "Reflect the photo")],
    "shadow": [("enabled", "Cast shadow")],
}

COLORS = {
    "grid": [("color", "Grid color"), ("accent_color", "Path color")],
    "material": [("face_color", "Face color"), ("side_color", "Side color"), ("secondary_color", "Secondary color")],
    "light": [("color", "Light color"), ("ambient_color", "Ambient color")],
    "shadow": [("color", "Shadow color")],
}

# Material fields shown for each kind (the kind itself decides `metallic`, so that slider never shows).
_SOLID_COLOR = ("face_color", "side_color", "roughness", "side_darkening", "bevel_strength")
MATERIAL_CONTROLS = {
    "metal": _SOLID_COLOR,
    "plastic": _SOLID_COLOR,
    # Glass shows its roughness as frost and its thickness through the absorption tint.
    "glass": ("face_color", "side_color", "absorption", "roughness", "side_darkening", "bevel_strength"),
    # Textured materials color their side walls from the pattern too, so they have no side color.
    "stone": ("face_color", "secondary_color", "texture_scale", "texture_strength", "bump_strength",
              "roughness", "side_darkening", "bevel_strength"),
    "wood": ("face_color", "secondary_color", "texture_scale", "texture_strength", "grain_angle", "varnish",
             "bump_strength", "roughness", "side_darkening", "bevel_strength"),
}

# Per-kind label overrides where the generic name would mislead.
MATERIAL_LABELS = {
    "glass": {"face_color": "Tint", "side_color": "Edge tint", "absorption": "Tint depth", "roughness": "Frost"},
    "stone": {"face_color": "Stone color", "secondary_color": "Vein color", "texture_scale": "Grain size",
              "texture_strength": "Veining"},
    "wood": {"face_color": "Light wood", "secondary_color": "Dark wood", "texture_scale": "Ring spacing",
             "texture_strength": "Grain contrast"},
}

GROUP_TITLES = [
    ("text", "Text"),
    ("transform", "Text Placement"),
    ("ground", "Ground"),
    ("grid", "Grid"),
    ("camera", "Camera"),
    ("light", "Lighting"),
    ("material", "Material"),
    ("shadow", "Shadow"),
]


class WheelGuard(QObject):
    """Lets the wheel scroll the panel instead of changing whichever control is under the pointer.

    A guarded control only takes wheel input once it has focus (click it, or Tab to it). Otherwise the wheel
    goes to the panel's scroll bar.
    """

    def __init__(self, panel: QScrollArea):
        super().__init__(panel)
        self.panel = panel

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():
            QApplication.sendEvent(self.panel.verticalScrollBar(), event)
            return True
        return False


class SliderRow(QWidget):
    """A slider paired with a spin box, emitting floats (ints when decimals == 0)."""

    value_changed = pyqtSignal(object)

    def __init__(self, minimum: float, maximum: float, decimals: int):
        super().__init__()
        self.decimals = decimals
        self.factor = 10**decimals
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * self.factor), round(maximum * self.factor))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(1 / self.factor if decimals else 1)
        self.spin.setFixedWidth(78)
        self.spin.setKeyboardTracking(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(lambda v: self._emit(v / self.factor))
        self.spin.valueChanged.connect(self._emit)

    def _emit(self, value: float) -> None:
        value = int(round(value)) if self.decimals == 0 else round(value, self.decimals)
        self.set_value(value)
        self.value_changed.emit(value)

    def set_value(self, value: float) -> None:
        for widget, v in ((self.slider, round(value * self.factor)), (self.spin, value)):
            widget.blockSignals(True)
            widget.setValue(v)
            widget.blockSignals(False)


class ColorButton(QPushButton):
    color_changed = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(24)
        self._color = (1.0, 1.0, 1.0)
        self.clicked.connect(self._pick)

    def set_color(self, rgb: tuple[float, float, float]) -> None:
        self._color = tuple(rgb)
        r, g, b = (round(c * 255) for c in rgb)
        self.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;")

    def _pick(self) -> None:
        current = QColor.fromRgbF(*self._color)
        chosen = QColorDialog.getColor(current, self, "Choose color")
        if chosen.isValid():
            rgb = (chosen.redF(), chosen.greenF(), chosen.blueF())
            self.set_color(rgb)
            self.color_changed.emit(rgb)


class ControlPanel(QScrollArea):
    def __init__(self, viewmodel: SceneViewModel):
        super().__init__()
        self.vm = viewmodel
        self._setters: dict[tuple[str, str], callable] = {}
        self._material_rows: dict[str, tuple[QFormLayout, QWidget, str]] = {}  # field -> (form, widget, label)
        self.setWidgetResizable(True)
        self.setMinimumWidth(340)

        content = QWidget()
        column = QVBoxLayout(content)
        for section, title in GROUP_TITLES:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            if section == "text":
                self._add_text_inputs(form)
            if section == "material":
                self._add_material_picker(form)
            for field, label in CHECKBOXES.get(section, []):
                self._add_checkbox(form, section, field, label)
            for field, label, lo, hi, decimals in SLIDERS.get(section, []):
                row = self._add_slider(form, section, field, label, lo, hi, decimals)
                self._remember_material_row(section, field, label, form, row)
            for field, label in COLORS.get(section, []):
                button = self._add_color(form, section, field, label)
                self._remember_material_row(section, field, label, form, button)
            column.addWidget(group)
        column.addStretch(1)
        self.setWidget(content)
        self.wheel_guard = WheelGuard(self)
        for control in content.findChildren((QSlider, QAbstractSpinBox, QComboBox)):
            control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # focus by click or Tab, never by wheel
            control.installEventFilter(self.wheel_guard)
        self._show_material_controls(self.vm.get("material", "kind"))

        self.vm.scene_changed.connect(self._refresh)

    def _bind(self, section, field, setter, signal) -> None:
        setter(self.vm.get(section, field))
        self._setters[(section, field)] = setter
        signal.connect(lambda value: self.vm.set(section, field, value))

    def _add_text_inputs(self, form: QFormLayout) -> None:
        edit = QLineEdit()

        def set_text(value: str) -> None:
            # Only push text back when it differs, so refreshes don't move the cursor while typing.
            if edit.text() != value:
                _set_quietly(edit, edit.setText, value)

        self._bind("text", "content", set_text, edit.textChanged)
        form.addRow("Text", edit)

        fonts = QFontComboBox()
        set_font = lambda v: _set_quietly(fonts, fonts.setCurrentFont, QFont(v))
        set_font(self.vm.get("text", "font_family"))
        self._setters[("text", "font_family")] = set_font
        fonts.currentFontChanged.connect(lambda f: self.vm.set("text", "font_family", f.family()))
        form.addRow("Font", fonts)

        styles = QHBoxLayout()
        for field, label in (("bold", "Bold"), ("italic", "Italic")):
            box = QCheckBox(label)
            self._bind("text", field, lambda v, b=box: _set_quietly(b, b.setChecked, v), box.toggled)
            styles.addWidget(box)
        styles.addStretch(1)
        form.addRow("Style", styles)

    def _add_material_picker(self, form: QFormLayout) -> None:
        combo = QComboBox()
        combo.setObjectName("material.kind")
        for kind in KINDS:
            combo.addItem(kind.capitalize(), kind)

        def set_kind(kind: str) -> None:
            _set_quietly(combo, combo.setCurrentIndex, combo.findData(kind))

        set_kind(self.vm.get("material", "kind"))
        self._setters[("material", "kind")] = set_kind
        # `activated` fires only on user choice; picking the current material again resets it to its preset.
        combo.activated.connect(lambda index: self.vm.apply_material(combo.itemData(index)))
        form.addRow("Material", combo)

    def _add_checkbox(self, form, section, field, label) -> None:
        box = QCheckBox(label)
        self._bind(section, field, lambda v: _set_quietly(box, box.setChecked, v), box.toggled)
        form.addRow("", box)

    def _add_slider(self, form, section, field, label, lo, hi, decimals) -> QWidget:
        row = SliderRow(lo, hi, decimals)
        row.setObjectName(f"{section}.{field}")
        self._bind(section, field, row.set_value, row.value_changed)
        form.addRow(label, row)
        return row

    def _add_color(self, form, section, field, label) -> QWidget:
        button = ColorButton()
        button.setObjectName(f"{section}.{field}")
        self._bind(section, field, button.set_color, button.color_changed)
        form.addRow(label, button)
        return button

    def _remember_material_row(self, section, field, label, form, widget) -> None:
        if section == "material":
            self._material_rows[field] = (form, widget, label)

    def _show_material_controls(self, kind: str) -> None:
        labels = MATERIAL_LABELS.get(kind, {})
        for field, (form, widget, default_label) in self._material_rows.items():
            form.setRowVisible(widget, field in MATERIAL_CONTROLS[kind])
            form.labelForField(widget).setText(labels.get(field, default_label))

    def _refresh(self, section: str, field: str) -> None:
        setter = self._setters.get((section, field))
        if setter is not None:
            setter(self.vm.get(section, field))
        if (section, field) == ("material", "kind"):
            self._show_material_controls(self.vm.get(section, field))


def _set_quietly(widget: QWidget, setter, value) -> None:
    widget.blockSignals(True)
    setter(value)
    widget.blockSignals(False)
