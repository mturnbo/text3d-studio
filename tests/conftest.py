import pytest
from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
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
