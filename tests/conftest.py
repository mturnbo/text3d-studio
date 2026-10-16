import gc

import pytest
from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext
from PyQt6.QtWidgets import QApplication

from text3d_studio.app import configure_opengl


@pytest.fixture(scope="session")
def qapp():
    configure_opengl()
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def gl_context(qapp):
    """A real OpenGL 4.1 core context on an offscreen surface; skips if the machine has no GPU access."""
    context = QOpenGLContext()
    surface = QOffscreenSurface()
    surface.create()
    if not context.create() or not context.makeCurrent(surface):
        pytest.skip("OpenGL 4.1 core context unavailable")
    yield context
    context.doneCurrent()


@pytest.fixture(autouse=True)
def collect_leftover_widgets():
    """Frees widgets dropped by the previous test now. Left to Python's collector, they could die in the middle
    of a later canvas's paint, where destroying a GL widget switches the current OpenGL context."""
    gc.collect()
    yield
