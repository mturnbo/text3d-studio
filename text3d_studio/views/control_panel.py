"""ControlPanel: declarative slider/picker layout bound two-way to the SceneViewModel."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
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
    ],
    "material": [
        ("side_darkening", "Depth darkening", 0, 1, 2),
        ("bevel_strength", "Bevel strength", 0, 4, 2),
        ("metallic", "Metallic", 0, 1, 2),
        ("roughness", "Roughness", 0.02, 1, 2),
    ],
    "shadow": [
        ("opacity", "Opacity", 0, 1, 2),
        ("softness", "Softness", 0, 1, 2),
        ("falloff", "Falloff", 0, 3, 2),
    ],
}

CHECKBOXES = {
    "grid": [("visible", "Show grid"), ("fit_estimate_fov", "Fit: estimate field of view")],
    "shadow": [("enabled", "Cast shadow")],
}

COLORS = {
    "grid": [("color", "Grid color"), ("accent_color", "Path color")],
    "material": [("face_color", "Face color"), ("side_color", "Side color")],
    "light": [("color", "Light color"), ("ambient_color", "Ambient color")],
    "shadow": [("color", "Shadow color")],
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
            for field, label in CHECKBOXES.get(section, []):
                self._add_checkbox(form, section, field, label)
            for field, label, lo, hi, decimals in SLIDERS.get(section, []):
                self._add_slider(form, section, field, label, lo, hi, decimals)
            for field, label in COLORS.get(section, []):
                self._add_color(form, section, field, label)
            column.addWidget(group)
        column.addStretch(1)
        self.setWidget(content)

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

    def _add_checkbox(self, form, section, field, label) -> None:
        box = QCheckBox(label)
        self._bind(section, field, lambda v: _set_quietly(box, box.setChecked, v), box.toggled)
        form.addRow("", box)

    def _add_slider(self, form, section, field, label, lo, hi, decimals) -> None:
        row = SliderRow(lo, hi, decimals)
        row.setObjectName(f"{section}.{field}")
        self._bind(section, field, row.set_value, row.value_changed)
        form.addRow(label, row)

    def _add_color(self, form, section, field, label) -> None:
        button = ColorButton()
        self._bind(section, field, button.set_color, button.color_changed)
        form.addRow(label, button)

    def _refresh(self, section: str, field: str) -> None:
        setter = self._setters.get((section, field))
        if setter is not None:
            setter(self.vm.get(section, field))


def _set_quietly(widget: QWidget, setter, value) -> None:
    widget.blockSignals(True)
    setter(value)
    widget.blockSignals(False)
