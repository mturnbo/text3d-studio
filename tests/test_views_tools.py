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
