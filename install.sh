#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:-/}"
BUILD_DIR="${BUILD_DIR:-build}"

test -x "$BUILD_DIR/mono_depth_rescaler"
install -d "$TARGET/usr/bin"
install -m 755 "$BUILD_DIR/mono_depth_rescaler" "$TARGET/usr/bin/"
install -d "$TARGET/etc/mono_depth_rescaler"
cp -R config/. "$TARGET/etc/mono_depth_rescaler/"

echo "Installed mono_depth_rescaler to $TARGET/usr/bin"
