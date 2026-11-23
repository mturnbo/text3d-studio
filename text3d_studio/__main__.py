# Nuitka build options, shared by local and CI builds: `uv run --group build python -m nuitka text3d_studio`
# nuitka-project: --mode=app
# nuitka-project: --enable-plugin=pyside6
# nuitka-project: --output-dir=build
# nuitka-project: --output-filename=text3d-studio
# nuitka-project: --product-name=Text3D Studio
# nuitka-project: --company-name=Text3D Studio
# nuitka-project: --macos-app-name=Text3D Studio
# nuitka-project: --assume-yes-for-downloads
# nuitka-project: --include-data-files={MAIN_DIRECTORY}/resources/icon.png=text3d_studio/resources/icon.png
# nuitka-project: --macos-app-icon={MAIN_DIRECTORY}/resources/icon.icns
# nuitka-project: --windows-icon-from-ico={MAIN_DIRECTORY}/resources/icon.ico
# nuitka-project: --linux-app-icon={MAIN_DIRECTORY}/resources/icon.png
# nuitka-project-if: {OS} == "Windows":
#    nuitka-project: --windows-console-mode=disable
from text3d_studio.app import main

raise SystemExit(main())
