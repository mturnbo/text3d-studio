#!/bin/sh
# Installs Text3D Studio for the current user: the app, its menu entry and icon, and the .t3d project type.
#   ./install.sh               install (or update)
#   ./install.sh --uninstall   remove everything it installed
set -eu

here=$(cd "$(dirname "$0")" && pwd)
data="${XDG_DATA_HOME:-$HOME/.local/share}"
bin="$HOME/.local/bin/text3d-studio"
desktop="$data/applications/text3d-studio.desktop"
mime="$data/mime/packages/text3d-studio.xml"
icon="$data/icons/hicolor/256x256/apps/text3d-studio.png"

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$bin" "$desktop" "$mime" "$icon"
    echo "Removed Text3D Studio."
else
    install -Dm755 "$here/text3d-studio" "$bin"
    install -Dm644 "$here/text3d-studio.png" "$icon"
    install -Dm644 "$here/text3d-studio.xml" "$mime"
    mkdir -p "$(dirname "$desktop")"
    sed "s|@BIN@|$bin|g" "$here/text3d-studio.desktop" > "$desktop"
    echo "Installed Text3D Studio to $bin"
fi

# Refresh the caches that file managers read; desktops without these tools pick changes up on their own.
if command -v update-mime-database >/dev/null 2>&1; then
    update-mime-database "$data/mime"
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$data/applications"
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t "$data/icons/hicolor" || true
fi
