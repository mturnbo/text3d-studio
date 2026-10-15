import numpy as np
import pytest
from PyQt6.QtGui import QKeySequence

from text3d_studio.model import Document
from text3d_studio.viewmodel import SceneViewModel
from text3d_studio.views.image_canvas import ImageCanvas
from text3d_studio.views.main_window import MainWindow


@pytest.fixture
def vm(qapp):
    return SceneViewModel(Document())


@pytest.fixture
def window(vm):
    win = MainWindow(vm)
    yield win
    win.close()


def background():
    pixels = np.zeros((40, 60, 4), np.uint8)
    pixels[..., 3] = 255
    return pixels


def test_empty_state_visible_without_background(vm):
    canvas = ImageCanvas(vm)
    assert canvas.empty_state.isVisibleTo(canvas)


def test_empty_state_hides_once_background_loads(vm):
    canvas = ImageCanvas(vm)
    vm.set_background(background(), "photo.png")
    assert not canvas.empty_state.isVisibleTo(canvas)


def test_empty_state_returns_when_background_is_cleared(vm):
    canvas = ImageCanvas(vm)
    vm.set_background(background(), "photo.png")
    vm.set_background(None)
    assert canvas.empty_state.isVisibleTo(canvas)


def test_empty_state_message_shows_native_open_shortcut(vm):
    canvas = ImageCanvas(vm)
    shortcut = QKeySequence(QKeySequence.StandardKey.Open).toString(QKeySequence.SequenceFormat.NativeText)
    assert shortcut and shortcut in canvas.empty_state.message.text()


def test_empty_state_button_emits_open_requested_once(vm):
    canvas = ImageCanvas(vm)
    emitted = []
    canvas.open_requested.connect(lambda: emitted.append(True))
    canvas.empty_state.button.click()
    assert emitted == [True]


def test_empty_state_button_opens_dialog_without_a_path(window):
    calls = []
    window.open_image = lambda path=None: calls.append(path)
    window.canvas.empty_state.button.click()
    assert calls == [None]


def test_empty_state_is_centered_on_canvas(window, qapp):
    window.show()
    qapp.processEvents()
    canvas, overlay = window.canvas, window.canvas.empty_state
    assert abs(overlay.geometry().center().x() - canvas.width() // 2) <= 1
    assert abs(overlay.geometry().center().y() - canvas.height() // 2) <= 1


def test_toolbar_shares_open_and_export_actions_with_file_menu(window):
    assert window.toolbar.actions() == [window.open_action, window.export_action]
    file_menu = window.menuBar().actions()[0].menu()
    assert window.open_action in file_menu.actions()
    assert window.export_action in file_menu.actions()


def test_open_action_keeps_standard_shortcut(window):
    assert window.open_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)
