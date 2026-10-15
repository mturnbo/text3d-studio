"""Application entry point: configures the OpenGL surface, wires Model -> ViewModel -> Views."""

import sys

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

from .model import Document
from .preferences import Preferences
from .viewmodel import SceneViewModel
from .views.main_window import MainWindow


def configure_opengl() -> None:
    """Must run before QApplication is created. 4.1 core is the highest profile macOS offers."""
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> int:
    configure_opengl()
    app = QApplication(sys.argv)
    app.setOrganizationName("Text3D Studio")
    app.setApplicationName("Text3D Studio")

    window = MainWindow(SceneViewModel(Document()), Preferences(QSettings()))
    window.show()
    if len(sys.argv) > 1:
        window.open_image(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
