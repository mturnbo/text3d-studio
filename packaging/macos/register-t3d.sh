#!/bin/sh
# Registers the .t3d project type in a built app's Info.plist (Nuitka has no option for document types) and gives
# it a reverse-DNS bundle identifier (Nuitka uses the app name, spaces included). Then re-signs the bundle ad hoc,
# since editing Info.plist breaks the signature Nuitka applied.
#   packaging/macos/register-t3d.sh build/text3d_studio.app
set -eu

app="$1"
plist="$app/Contents/Info.plist"
bundle_id="io.github.mturnbo.text3d-studio"
uti="$bundle_id.project"
icon=$(plutil -extract CFBundleIconFile raw "$plist")

plutil -replace CFBundleIdentifier -string "$bundle_id" "$plist"

plutil -replace UTExportedTypeDeclarations -json "[{
  \"UTTypeIdentifier\": \"$uti\",
  \"UTTypeDescription\": \"Text3D Studio Project\",
  \"UTTypeConformsTo\": [\"public.json\"],
  \"UTTypeIconFile\": \"$icon\",
  \"UTTypeTagSpecification\": {\"public.filename-extension\": [\"t3d\"]}
}]" "$plist"

plutil -replace CFBundleDocumentTypes -json "[{
  \"CFBundleTypeName\": \"Text3D Studio Project\",
  \"CFBundleTypeRole\": \"Editor\",
  \"CFBundleTypeIconFile\": \"$icon\",
  \"LSHandlerRank\": \"Owner\",
  \"LSItemContentTypes\": [\"$uti\"]
}]" "$plist"

plutil -lint "$plist"
codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"
