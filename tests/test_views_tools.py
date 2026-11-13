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


# ----- Match Sun --------------------------------------------------------------------------------


def sun_canvas(vm):
    from text3d_studio.views.image_canvas import SUN_TOOL

    vm.set("ground", "pitch", 12.0)
    canvas = sized_canvas(vm)
    statuses = []
    canvas.tool_status.connect(statuses.append)
    canvas.set_tool(SUN_TOOL)
    return canvas, statuses


def post_clicks(canvas, azimuth, elevation, base_xz=(1.0, -8.0), height=1.0):
    """Widget positions of a vertical post's top and base and its shadow's tip under that sun."""
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    g, fov = canvas.vm.scene.ground, canvas.vm.scene.camera.fov
    base = ground_model.ground_to_world(g, *base_xz)
    top = base + height * ground_model.ground_up(g)
    toward_light = ground_model.light_frame(g) @ math3d.direction_from_angles(azimuth, 0.0)
    tip = base - toward_light * height / np.tan(np.radians(elevation))
    return [tuple(math3d.project_to_image(p, fov, (rect_w, rect_h)) + (rect_x, rect_y)) for p in (top, base, tip)]


def click(canvas, position):
    drag(canvas, position, position)


def test_three_clicks_match_the_sun_in_one_undo_step(vm):
    canvas, statuses = sun_canvas(vm)
    vm.clear_history()
    before = dataclasses.replace(vm.scene.light)
    for position in post_clicks(canvas, -70.0, 32.0):
        click(canvas, position)
    assert (vm.scene.light.azimuth, vm.scene.light.elevation) == pytest.approx((-70.0, 32.0), abs=1e-6)
    assert "matched" in statuses[-1].lower()
    assert vm.undo() and vm.scene.light == before and not vm.can_undo


def test_match_sun_prompts_for_each_click_and_shows_the_points(vm):
    canvas, statuses = sun_canvas(vm)
    assert canvas.tool_overlay.isVisibleTo(canvas)
    assert "top" in statuses[-1].lower()
    clicks = post_clicks(canvas, 40.0, 45.0)
    click(canvas, clicks[0])
    assert "base" in statuses[-1].lower()
    click(canvas, clicks[1])
    assert "shadow" in statuses[-1].lower()
    np.testing.assert_allclose(canvas.sun_points_px(), clicks[:2], atol=1e-6)
    click(canvas, clicks[2])
    np.testing.assert_allclose(canvas.sun_points_px(), clicks, atol=1e-6)  # all three stay shown


def test_a_fourth_click_starts_a_new_measurement(vm):
    canvas, _statuses = sun_canvas(vm)
    for position in post_clicks(canvas, 40.0, 45.0):
        click(canvas, position)
    click(canvas, (100, 100))
    assert canvas.sun_points_px() == [pytest.approx((100, 100))]
    assert canvas.sun_valid


@pytest.mark.parametrize("case, message", [("no_shadow", "no length"), ("top_below_base", "above the base")])
def test_impossible_clicks_leave_the_light_and_say_why(vm, case, message):
    canvas, statuses = sun_canvas(vm)
    before = dataclasses.replace(vm.scene.light)
    top, base, tip = post_clicks(canvas, 40.0, 45.0)
    clicks = {"no_shadow": (top, base, base), "top_below_base": ((base[0], base[1] + 30), base, tip)}[case]
    for position in clicks:
        click(canvas, position)
    assert vm.scene.light == before
    assert message in statuses[-1] and "Can't match the sun" in statuses[-1]
    assert not canvas.sun_valid


def test_clicks_off_the_image_are_ignored(vm):
    vm.set_background(np.full((200, 600, 4), 255, np.uint8))  # a wide photo letterboxed in the canvas
    canvas, _statuses = sun_canvas(vm)
    rect_x, rect_y, _, _ = canvas._image_rect()
    assert rect_y > 0
    click(canvas, (400, rect_y / 2))
    assert canvas.sun_points == []


def test_match_sun_clicks_move_nothing_else(vm):
    canvas, _statuses = sun_canvas(vm)
    before = dataclasses.replace(vm.selected_object.transform)
    top, base, _tip = post_clicks(canvas, 40.0, 45.0)
    drag(canvas, top, base)
    assert vm.selected_object.transform == before


def test_switching_to_match_sun_clears_old_points(vm):
    canvas, _statuses = sun_canvas(vm)
    click(canvas, post_clicks(canvas, 40.0, 45.0)[0])
    canvas.set_tool(TEXT_TOOL)
    assert not canvas.tool_overlay.isVisibleTo(canvas)
    from text3d_studio.views.image_canvas import SUN_TOOL

    canvas.set_tool(SUN_TOOL)
    assert canvas.sun_points == []


def test_match_sun_is_in_the_tools_menu_with_shortcut_l_and_reports_in_the_status_bar(window):
    from text3d_studio.views.image_canvas import SUN_TOOL

    action = window.tool_actions[SUN_TOOL]
    assert action.shortcut() == QKeySequence("L")
    action.trigger()
    assert window.canvas.tool == SUN_TOOL
    assert "top" in window.statusBar().currentMessage().lower()


# ----- Eyedropper -------------------------------------------------------------------------------


def photo(width=160, height=100):
    """Left half red, right half blue, with a 5x5 patch of known, varied colours around pixel (row 40, col 60)."""
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., 3] = 255
    pixels[:, : width // 2, :3] = (200, 30, 40)
    pixels[:, width // 2:, :3] = (20, 60, 220)
    rng = np.random.default_rng(3)
    pixels[38:43, 58:63, :3] = rng.integers(0, 256, (5, 5, 3))
    return pixels


def picking_canvas(vm, pixels=None):
    vm.set_background(photo() if pixels is None else pixels)
    canvas = sized_canvas(vm)  # 800 x 500: the 160 x 100 photo fills it at 5 widget pixels per photo pixel
    statuses = []
    canvas.tool_status.connect(statuses.append)
    return canvas, statuses


def widget_position(canvas, row, column):
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    height, width = canvas.vm.document.background.shape[:2]
    return (rect_x + (column + 0.5) * rect_w / width, rect_y + (row + 0.5) * rect_h / height)


@pytest.mark.parametrize("field", ["color", "ambient_color"])
def test_eyedropper_sets_the_light_colour_from_the_photo(vm, field):
    canvas, _statuses = picking_canvas(vm)
    canvas.start_eyedropper("light", field)
    click(canvas, widget_position(canvas, 70, 20))
    assert getattr(vm.scene.light, field) == pytest.approx((200 / 255, 30 / 255, 40 / 255))
    assert canvas.eyedropper is None


def test_eyedropper_averages_a_5x5_block(vm):
    pixels = photo()
    canvas, _statuses = picking_canvas(vm, pixels)
    canvas.start_eyedropper("light", "color")
    click(canvas, widget_position(canvas, 40, 60))
    expected = pixels[38:43, 58:63, :3].reshape(-1, 3).mean(axis=0) / 255.0
    assert vm.scene.light.color == pytest.approx(tuple(expected))
    assert vm.scene.light.color != pytest.approx(tuple(pixels[40, 60, :3] / 255.0))


def test_eyedropper_at_the_photos_corner_averages_what_is_there(vm):
    pixels = photo()
    pixels[0, 0, :3] = (255, 255, 255)
    canvas, _statuses = picking_canvas(vm, pixels)
    canvas.start_eyedropper("light", "color")
    click(canvas, widget_position(canvas, 0, 0))
    expected = pixels[0:3, 0:3, :3].reshape(-1, 3).mean(axis=0) / 255.0
    assert vm.scene.light.color == pytest.approx(tuple(expected))


def test_eyedropper_maps_through_the_letterboxed_image_rect(vm):
    pixels = np.zeros((100, 400, 4), np.uint8)  # wider than the canvas: bars above and below
    pixels[..., 3] = 255
    pixels[:, :200, :3] = (10, 200, 10)
    pixels[:, 200:, :3] = (250, 250, 0)
    canvas, _statuses = picking_canvas(vm, pixels)
    assert canvas._image_rect()[1] > 0
    canvas.start_eyedropper("light", "color")
    click(canvas, widget_position(canvas, 50, 300))
    assert vm.scene.light.color == pytest.approx((250 / 255, 250 / 255, 0.0))


def test_escape_cancels_the_eyedropper(vm):
    from PyQt6.QtTest import QTest

    canvas, statuses = picking_canvas(vm)
    before = vm.scene.light.color
    canvas.start_eyedropper("light", "color")
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas.eyedropper is None and statuses[-1] == ""
    click(canvas, widget_position(canvas, 70, 20))
    assert vm.scene.light.color == before


def test_a_click_off_the_photo_keeps_the_eyedropper_waiting(vm):
    pixels = np.zeros((100, 400, 4), np.uint8)
    pixels[..., 3] = 255
    canvas, statuses = picking_canvas(vm, pixels)
    before = vm.scene.light.color
    canvas.start_eyedropper("light", "color")
    click(canvas, (400, canvas._image_rect()[1] / 2))  # on the bar above the photo
    assert vm.scene.light.color == before and canvas.eyedropper == ("light", "color")
    assert "photo" in statuses[-1]


def test_eyedropper_without_a_photo_picks_nothing(vm):
    canvas = sized_canvas(vm)
    before = vm.scene.light.color
    canvas.start_eyedropper("light", "color")
    click(canvas, (400, 250))
    assert vm.scene.light.color == before


def test_eyedropper_click_does_not_move_the_text_and_is_one_undo_step(vm):
    canvas, _statuses = picking_canvas(vm)
    vm.clear_history()
    transform = dataclasses.replace(vm.selected_object.transform)
    before = vm.scene.light.color
    canvas.start_eyedropper("light", "color")
    drag(canvas, widget_position(canvas, 70, 20), widget_position(canvas, 30, 120))
    assert vm.selected_object.transform == transform
    assert vm.undo() and vm.scene.light.color == before and not vm.can_undo


def test_pick_buttons_sit_next_to_the_light_and_ambient_colours(window):
    from PyQt6.QtWidgets import QPushButton

    vm = window.vm
    vm.set_background(photo())
    for field in ("color", "ambient_color"):
        button = window.panel.findChild(QPushButton, f"light.{field}.pick")
        assert button is not None
        button.click()
        assert window.canvas.eyedropper == ("light", field)
        assert "Esc" in window.statusBar().currentMessage()
        window.canvas.cancel_eyedropper()
    assert window.panel.findChild(QPushButton, "shadow.color.pick") is None


def test_picked_colour_shows_on_the_panels_colour_button(window):
    from text3d_studio.views.control_panel import ColorButton

    window.vm.set_background(photo())
    canvas = window.canvas
    canvas.resize(800, 500)
    canvas.start_eyedropper("light", "color")
    click(canvas, widget_position(canvas, 70, 20))
    button = window.panel.findChild(ColorButton, "light.color")
    assert button._color == pytest.approx((200 / 255, 30 / 255, 40 / 255))


def test_wall_preview_is_clipped_to_the_photo(vm):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QImage, QRegion
    from PyQt6.QtWidgets import QWidget

    vm.set_background(np.full((100, 400, 4), 255, np.uint8))  # wide: bars above and below the photo
    canvas = wall_canvas(vm)
    canvas.tool_overlay.setGeometry(canvas.rect())
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(0)
    canvas.tool_overlay.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)  # no background
    alpha = np.array([[image.pixelColor(x, y).alpha() for x in range(0, 800, 4)] for y in range(0, 500, 4)])
    _, rect_y, _, rect_h = canvas._image_rect()
    rows = np.arange(0, 500, 4)
    inside = (rows >= rect_y) & (rows < rect_y + rect_h)
    assert alpha[inside].any()
    assert not alpha[~inside].any()


# ----- Mask tool --------------------------------------------------------------------------------


def mask_canvas(vm):
    from text3d_studio.views.image_canvas import MASK_TOOL

    canvas, _ = picking_canvas(vm)  # 160 x 100 photo at 5 widget pixels per photo pixel
    vm.set("mask", "brush_size", 0.1)  # 10 photo pixels across
    vm.set("mask", "feather", 0.0)
    vm.clear_history()
    canvas.set_tool(MASK_TOOL)
    return canvas


def test_dragging_with_the_mask_tool_paints_along_the_path_in_photo_pixels(vm):
    canvas = mask_canvas(vm)
    drag(canvas, widget_position(canvas, 50, 20), widget_position(canvas, 50, 120))
    mask = vm.document.mask
    assert mask.shape == (100, 160)
    assert (mask[50, 20:121] == 255).all()
    assert mask[50, 140] == 0 and mask[20, 70] == 0


def test_a_mask_stroke_is_one_undo_step_however_long(vm):
    canvas = mask_canvas(vm)
    mouse(canvas, QEvent.Type.MouseButtonPress, widget_position(canvas, 20, 20), Qt.MouseButton.LeftButton)
    for column in range(30, 140, 10):
        mouse(canvas, QEvent.Type.MouseMove, widget_position(canvas, 20 + column // 4, column), Qt.MouseButton.NoButton,
              Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, widget_position(canvas, 50, 130), Qt.MouseButton.LeftButton,
          Qt.MouseButton.NoButton)
    assert vm.document.mask[20, 20] == 255
    assert vm.undo() and vm.document.mask is None and not vm.can_undo
    assert vm.redo() and vm.document.mask[20, 20] == 255


def test_option_drag_erases_and_the_text_never_moves(vm):
    canvas = mask_canvas(vm)
    transform = dataclasses.replace(vm.selected_object.transform)
    drag(canvas, widget_position(canvas, 50, 20), widget_position(canvas, 50, 120))
    drag(canvas, widget_position(canvas, 50, 70), widget_position(canvas, 50, 70), modifiers=Qt.KeyboardModifier.AltModifier)
    assert vm.document.mask[50, 70] == 0 and vm.document.mask[50, 30] == 255
    assert vm.selected_object.transform == transform
    vm.undo()
    assert vm.document.mask[50, 70] == 255


def test_the_mask_tool_without_a_photo_paints_nothing(vm):
    from text3d_studio.views.image_canvas import MASK_TOOL

    canvas = sized_canvas(vm)
    canvas.set_tool(MASK_TOOL)
    drag(canvas, (100, 100), (300, 200))
    assert vm.document.mask is None and not vm.can_undo


def test_the_brush_outline_follows_the_cursor_and_the_brush_size(vm):
    canvas = mask_canvas(vm)
    assert canvas.tool_overlay.isVisibleTo(canvas)
    mouse(canvas, QEvent.Type.MouseMove, (300, 200), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    center, radius = canvas.brush_outline_px()
    assert center == (300, 200) and radius == pytest.approx(0.1 * 500 / 2)
    vm.set("mask", "brush_size", 0.2)
    assert canvas.brush_outline_px()[1] == pytest.approx(0.2 * 500 / 2)
    canvas.set_show_original(True)
    assert not canvas.tool_overlay.isVisibleTo(canvas)


def test_mask_tool_is_in_the_tools_menu_with_shortcut_m(window):
    from text3d_studio.views.image_canvas import MASK_TOOL

    action = window.tool_actions[MASK_TOOL]
    assert action.shortcut() == QKeySequence("M")
    assert action in window.toolbar.actions()
    action.trigger()
    assert window.canvas.tool == MASK_TOOL
    assert "erase" in window.hint_label.text().lower()


def test_the_panel_sets_the_brush_and_clears_the_mask(window):
    from PyQt6.QtWidgets import QPushButton

    from text3d_studio.views.control_panel import SliderRow

    vm = window.vm
    vm.set_background(photo())
    button = window.panel.findChild(QPushButton, "mask.clear")
    assert not button.isEnabled()
    window.panel.findChild(SliderRow, "mask.brush_size").spin.setValue(0.2)
    assert vm.scene.mask.brush_size == pytest.approx(0.2)
    vm.paint_mask((50, 50), (50, 50))
    assert button.isEnabled()
    button.click()
    assert vm.document.mask is None and not button.isEnabled()
    vm.undo()
    assert vm.document.mask is not None and button.isEnabled()


def test_a_mask_the_gpu_cannot_hold_is_reported_not_raised_while_painting(vm):
    canvas = sized_canvas(vm)
    errors = []
    canvas.error.connect(errors.append)
    canvas.show()
    canvas.grab()  # creates the GL context
    vm.document.mask = np.zeros((1, canvas.compositor.max_texture_size + 1), np.uint8)
    vm.mask_changed.emit(None)
    canvas.grab()  # paints; the upload's error goes to the signal
    assert errors and "Mask is" in errors[0]
    canvas.close()
