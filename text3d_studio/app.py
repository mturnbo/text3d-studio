"""Application entry point: configures the OpenGL surface, wires Model -> ViewModel -> Views."""

import os
import sys

from PySide6.QtCore import QEvent, QObject, QSettings
from PySide6.QtGui import QIcon, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .model import Document
from .preferences import Preferences
from .viewmodel import SceneViewModel
from .views.main_window import MainWindow

ICON_PATH = os.path.join(os.path.dirname(__file__), "resources", "icon.png")


def configure_opengl() -> None:
    """Must run before QApplication is created. 4.1 core is the highest profile macOS offers."""
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


def open_command_line_file(window: MainWindow, args: list[str]) -> None:
    """Opens the first argument, a photo or a .t3d project; absolute so Open Recent and saves keep working."""
    if args:
        window.open_path(os.path.abspath(args[0]))


class FileOpenFilter(QObject):
    """Opens the files macOS hands a running app: Finder double-clicks and Open With arrive as FileOpen events."""

    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.window = window

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.FileOpen:
            self.window.open_path(event.file())
            return True
        return False


def main() -> int:
    configure_opengl()
    app = QApplication(sys.argv)
    app.setOrganizationName("Text3D Studio")
    app.setApplicationName("Text3D Studio")
    app.setWindowIcon(QIcon(ICON_PATH))

    window = MainWindow(SceneViewModel(Document()), Preferences(QSettings()))
    app.installEventFilter(FileOpenFilter(window))
    window.show()
    open_command_line_file(window, sys.argv[1:])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
