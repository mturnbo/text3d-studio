"""Canvas tools for the wall, the sun and colour picking: real mouse and key events on a sized, unshown canvas."""

import dataclasses

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, QSettings, Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QMessageBox

from text3d_studio import ground as ground_model
from text3d_studio import math3d
from text3d_studio.model import Document
from text3d_studio.preferences import Preferences
from text3d_studio.viewmodel import SceneViewModel
from text3d_studio.views.image_canvas import TEXT_TOOL, WALL_TOOL
from text3d_studio.views.main_window import MainWindow
from test_views import drag, mouse, sized_canvas


@pytest.fixture
def vm(qapp):
    return SceneViewModel(Document())


@pytest.fixture
def window(vm, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "text3d_studio.views.main_window.QMessageBox.question", lambda *args: QMessageBox.StandardButton.Discard
    )
    win = MainWindow(vm, Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)))
    yield win
    win.close()


def wall_canvas(vm):
    vm.set("ground", "pitch", 15.0)  # a good view of the floor
    canvas = sized_canvas(vm)
    canvas.set_tool(WALL_TOOL)
    return canvas


def ground_at(canvas, xy):
    return canvas.ground_point_at(QPointF(*xy))


def on_wall_base(vm, ground_point):
    point = ground_model.ground_to_world(vm.scene.ground, *ground_point)
    plane = ground_model.wall_plane(vm.scene.ground, vm.scene.wall)
    return float(plane[:3] @ point + plane[3])


# ----- Set Wall ---------------------------------------------------------------------------------


@pytest.mark.parametrize("start, end", [((200, 400), (600, 350)), ((650, 450), (300, 300)), ((400, 480), (420, 300))])
def test_dragging_between_two_floor_points_puts_the_walls_base_through_both(vm, start, end):
    canvas = wall_canvas(vm)
    a, b = ground_at(canvas, start), ground_at(canvas, end)
    drag(canvas, start, end)
    assert on_wall_base(vm, a) == pytest.approx(0, abs=1e-9)
    assert on_wall_base(vm, b) == pytest.approx(0, abs=1e-9)
    assert vm.scene.wall.offset >= 0


def test_the_wall_base_follows_the_drag_as_it_goes(vm):
    canvas = wall_canvas(vm)
    mouse(canvas, QEvent.Type.MouseButtonPress, (200, 400), Qt.MouseButton.LeftButton)
    for end in ((500, 420), (600, 330)):
        mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
        assert on_wall_base(vm, ground_at(canvas, end)) == pytest.approx(0, abs=1e-9)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (600, 330), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)


@pytest.mark.parametrize("start, end", [((300, 40), (500, 400)), ((300, 400), (500, 40)), ((300, 40), (500, 60))])
def test_a_drag_into_or_from_the_sky_does_nothing(vm, start, end):
    canvas = wall_canvas(vm)
    assert ground_at(canvas, (400, 40)) is None  # the top of the image shows sky
    before = dataclasses.replace(vm.scene.wall)
    mouse(canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    if ground_at(canvas, start) is not None:
        assert vm.scene.wall == before
    mouse(canvas, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert vm.scene.wall == before


def test_a_click_without_dragging_leaves_the_wall(vm):
    canvas = wall_canvas(vm)
    vm.clear_history()
    before = dataclasses.replace(vm.scene.wall)
    drag(canvas, (300, 400), (300, 400))
    assert vm.scene.wall == before and not vm.can_undo


def test_hovering_in_the_wall_tool_changes_nothing(vm):
    canvas = wall_canvas(vm)
    before = dataclasses.replace(vm.scene.wall)
    mouse(canvas, QEvent.Type.MouseMove, (500, 400), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    assert vm.scene.wall == before


def test_a_wall_drag_is_one_undo_step(vm):
    canvas = wall_canvas(vm)
    vm.clear_history()
    before = dataclasses.replace(vm.scene.wall)
    mouse(canvas, QEvent.Type.MouseButtonPress, (200, 400), Qt.MouseButton.LeftButton)
    for end in ((400, 420), (500, 380), (600, 350)):
        mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (600, 350), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert vm.scene.wall != before
    assert vm.undo() and vm.scene.wall == before and not vm.can_undo


def test_wall_text_moves_with_a_new_wall(vm):
    canvas = wall_canvas(vm)
    vm.set("transform", "mount", "wall")
    drag(canvas, (200, 400), (600, 350))
    t = vm.selected_object.transform
    point = ground_model.text_position(vm.scene.ground, t, vm.scene.wall)
    plane = ground_model.wall_plane(vm.scene.ground, vm.scene.wall)
    assert plane[:3] @ point + plane[3] == pytest.approx(0, abs=1e-9)


def test_wall_preview_shows_only_in_the_wall_tool(vm):
    canvas = sized_canvas(vm)
    assert not canvas.tool_overlay.isVisibleTo(canvas)
    canvas.set_tool(WALL_TOOL)
    assert canvas.tool_overlay.isVisibleTo(canvas)
    canvas.set_tool(TEXT_TOOL)
    assert not canvas.tool_overlay.isVisibleTo(canvas)


def test_wall_preview_draws_the_base_line_on_the_floor_and_vertical_lines_up_the_wall(vm):
    canvas = wall_canvas(vm)
    drag(canvas, (200, 400), (600, 350))
    grid_lines, base = canvas.wall_preview_px()
    for point in base:  # the base line's ends sit on the floor, on the wall's base
        ground_point = ground_at(canvas, point)
        if ground_point is not None:
            assert on_wall_base(vm, ground_point) == pytest.approx(0, abs=1e-6)
    visible = [(a, b) for a, b in grid_lines if ground_at(canvas, a) is not None]
    verticals = [(a, b) for a, b in visible if abs(on_wall_base(vm, ground_at(canvas, a))) < 1e-6]
    assert len(verticals) >= 3
    assert all(b[1] < a[1] for a, b in verticals)  # they rise from the base line (pixel y decreases)


def test_wall_preview_stays_on_screen_when_part_of_the_wall_is_behind_the_camera(vm):
    canvas = sized_canvas(vm)
    vm.set("wall", "angle", 90.0)  # a wall off to the side, running from behind the camera into the distance
    vm.set("wall", "offset", 3.0)
    grid_lines, base = canvas.wall_preview_px()
    assert base is not None and grid_lines
    assert all(np.isfinite(p).all() for line in grid_lines for p in line)


def test_project_segment_clips_what_lies_behind_the_camera():
    size = (800, 500)
    assert math3d.project_segment((0, 0, 5), (1, 0, 2), 40, size) is None
    a, b = math3d.project_segment((0, -1, 1.0), (0, -1, -9.0), 40, size)
    np.testing.assert_allclose(b, math3d.project_to_image((0, -1, -9.0), 40, size))
    assert a[1] > b[1]  # the near end, clipped just in front of the camera, shows far below
    front = math3d.project_segment((1, 0, -2), (-1, 0, -3), 40, size)
    np.testing.assert_allclose(front, [math3d.project_to_image(p, 40, size) for p in ((1, 0, -2), (-1, 0, -3))])


def test_set_wall_is_in_the_tools_menu_with_shortcut_w(window):
    action = window.tool_actions[WALL_TOOL]
    assert action.shortcut() == QKeySequence("W")
    assert action in window.toolbar.actions()
    action.trigger()
    assert window.canvas.tool == WALL_TOOL
    assert "wall" in window.hint_label.text().lower()


# ----- Wall-mounted text ------------------------------------------------------------------------

OPTION = Qt.KeyboardModifier.AltModifier


def mounted_canvas(vm, wall_x=0.0, wall_y=1.0):
    vm.set("wall", "offset", 15.0)
    vm.set("transform", "size", 1.5)
    vm.set("transform", "mount", "wall")
    vm.set("transform", "wall_x", wall_x)
    vm.set("transform", "wall_y", wall_y)
    return sized_canvas(vm)


def wall_anchor_on_screen(canvas):
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    scene = canvas.vm.scene
    position = ground_model.text_position(scene.ground, canvas.vm.selected_object.transform, scene.wall)
    pixel = math3d.project_to_image(position, scene.camera.fov, (rect_w, rect_h))
    return (pixel[0] + rect_x, pixel[1] + rect_y)


def wall_apparent_size(vm):
    t = vm.selected_object.transform
    return t.size / np.linalg.norm(ground_model.text_position(vm.scene.ground, t, vm.scene.wall))


def on_wall_plane(vm):
    t = vm.selected_object.transform
    point = ground_model.text_position(vm.scene.ground, t, vm.scene.wall)
    plane = ground_model.wall_plane(vm.scene.ground, vm.scene.wall)
    return float(plane[:3] @ point + plane[3])


@pytest.mark.parametrize("target", [(500, 200), (250, 230), (420, 120)])
def test_plain_drag_moves_wall_text_on_the_wall_at_constant_on_screen_size(vm, target):
    canvas = mounted_canvas(vm)
    before = wall_apparent_size(vm)
    drag(canvas, wall_anchor_on_screen(canvas), target)
    t = vm.selected_object.transform
    assert t.mount == "wall" and t.wall_y > 1.0
    np.testing.assert_allclose(wall_anchor_on_screen(canvas), target, atol=1e-6)
    assert wall_apparent_size(vm) == pytest.approx(before)
    assert on_wall_plane(vm) == pytest.approx(0, abs=1e-9)
    assert (t.x, t.z) == (0.0, -12.0)  # the floor placement is untouched


def test_option_drag_slides_wall_text_along_the_wall_keeping_the_grabbed_point(vm):
    canvas = mounted_canvas(vm, wall_y=2.0)
    start = wall_anchor_on_screen(canvas)
    grabbed = canvas.wall_point_at(QPointF(*start))
    drag(canvas, start, (start[0] + 120, start[1] - 30), modifiers=OPTION)
    t = vm.selected_object.transform
    np.testing.assert_allclose(wall_anchor_on_screen(canvas), (start[0] + 120, start[1] - 30), atol=1e-6)
    assert t.size == 1.5  # sliding keeps the world size
    assert t.wall_x > grabbed[0] and t.wall_y > grabbed[1]
    assert on_wall_plane(vm) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("modifiers", [Qt.KeyboardModifier.NoModifier, OPTION])
def test_dragging_wall_text_below_the_floor_stops_at_the_base_line(vm, modifiers):
    canvas = mounted_canvas(vm, wall_y=1.0)
    start = wall_anchor_on_screen(canvas)
    drag(canvas, start, (start[0], start[1] + 150), modifiers=modifiers)
    assert vm.selected_object.transform.wall_y == 0.0
    assert on_wall_plane(vm) == pytest.approx(0, abs=1e-9)


def test_a_wall_text_drag_is_one_undo_step(vm):
    canvas = mounted_canvas(vm)
    vm.clear_history()
    before = dataclasses.replace(vm.selected_object.transform)
    start = wall_anchor_on_screen(canvas)
    mouse(canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    for dx in (20, 60, 100):
        mouse(canvas, QEvent.Type.MouseMove, (start[0] + dx, start[1] - dx / 2), Qt.MouseButton.NoButton,
              Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (start[0] + 100, start[1] - 50), Qt.MouseButton.LeftButton,
          Qt.MouseButton.NoButton)
    assert vm.selected_object.transform != before
    assert vm.undo() and vm.selected_object.transform == before and not vm.can_undo


def test_wall_text_can_be_picked(vm):
    vm.set("wall", "offset", 15.0)
    vm.add_object()  # floor text in front, selected
    vm.select(0)
    vm.set_mount("wall")
    vm.select(1)
    canvas = sized_canvas(vm)
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    scene = vm.scene
    model = ground_model.text_model_matrix(scene.ground, scene.objects[0].transform, scene.wall)
    raster = canvas._raster_of(scene.objects[0])
    # A point inside the first letter's ink: the left part of the H's stem, mid height.
    for u in np.linspace(raster.origin[0], raster.origin[0] + raster.size[0], 200):
        if raster.covers(u, 0.35):
            break
    pixel = math3d.project_to_image((model @ np.array([u, 0.35, 0.0, 1.0]))[:3], scene.camera.fov, (rect_w, rect_h))
    assert canvas.object_at(QPointF(pixel[0] + rect_x, pixel[1] + rect_y)) == 0


# ----- Mount control ----------------------------------------------------------------------------


def mount_combo(panel):
    from PyQt6.QtWidgets import QComboBox

    return panel.findChild(QComboBox, "transform.mount")


def row_visible(panel, name):
    from text3d_studio.views.control_panel import SliderRow

    row = panel.findChild(SliderRow, name)
    return row.isVisibleTo(panel)


def test_mount_control_moves_the_text_onto_the_wall_keeping_it_on_screen(vm):
    from text3d_studio.views.control_panel import ControlPanel

    vm.set("transform", "z", -30.0)  # behind the default wall, so its ray meets the wall above the floor
    panel = ControlPanel(vm)
    combo = mount_combo(panel)
    position = ground_model.text_position(vm.scene.ground, vm.selected_object.transform)
    combo.activated.emit(combo.findData("wall"))
    t = vm.selected_object.transform
    after = ground_model.text_position(vm.scene.ground, t, vm.scene.wall)
    assert t.mount == "wall"
    np.testing.assert_allclose(after / np.linalg.norm(after), position / np.linalg.norm(position), atol=1e-9)
    assert on_wall_plane(vm) == pytest.approx(0, abs=1e-9)


def test_mount_control_shows_the_fields_that_apply(vm):
    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    panel.show()
    assert row_visible(panel, "transform.x") and not row_visible(panel, "transform.wall_y")
    vm.set_mount("wall")
    assert mount_combo(panel).currentData() == "wall"
    assert not row_visible(panel, "transform.x") and not row_visible(panel, "transform.rot_y")
    assert row_visible(panel, "transform.wall_x") and row_visible(panel, "transform.wall_y")
    vm.undo()
    assert mount_combo(panel).currentData() == "floor" and row_visible(panel, "transform.x")
    panel.close()


def test_mount_control_follows_the_selection(vm):
    from text3d_studio.views.control_panel import ControlPanel

    vm.add_object()
    vm.set_mount("wall")
    panel = ControlPanel(vm)
    assert mount_combo(panel).currentData() == "wall"
    vm.select(0)
    assert mount_combo(panel).currentData() == "floor"


def test_wall_sliders_are_bound(vm):
    from text3d_studio.views.control_panel import ControlPanel, SliderRow

    panel = ControlPanel(vm)
    panel.findChild(SliderRow, "wall.angle").spin.setValue(30.0)
    panel.findChild(SliderRow, "transform.wall_y").spin.setValue(2.5)
    assert vm.scene.wall.angle == 30.0 and vm.selected_object.transform.wall_y == 2.5
