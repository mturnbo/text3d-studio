# Nuitka build options, shared by local and CI builds: `uv run --group build python -m nuitka text3d_studio`
# nuitka-project: --mode=app
# nuitka-project: --enable-plugin=pyside6
# nuitka-project: --output-dir=build
# nuitka-project: --output-filename=text3d-studio
# nuitka-project: --product-name=Text3D Studio
# nuitka-project: --company-name=Text3D Studio
# nuitka-project: --macos-app-name=Text3D Studio
# nuitka-project: --assume-yes-for-downloads
# nuitka-project: --macos-app-icon=none
# nuitka-project-if: {OS} == "Windows":
#    nuitka-project: --windows-console-mode=disable
from text3d_studio.app import main

raise SystemExit(main())
