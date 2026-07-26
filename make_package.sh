#!/bin/bash
#   ./make_package.sh    build the .deb (run inside voxl-cross after ./build.sh qrb5165)
set -e
cd "$(dirname "$0")"

VERSION=$(grep "Version" pkg/control/control | cut -d' ' -f2)
PACKAGE=$(grep "Package" pkg/control/control | cut -d' ' -f2)
DATA_DIR=pkg/data
DEB_DIR=pkg/DEB

sudo rm -rf "$DATA_DIR" "$DEB_DIR" ./*.deb
mkdir -p "$DATA_DIR"

# stage /usr/bin, /etc/systemd, /etc/mono_depth_rescaler via the CMake install rules
sudo make -C build DESTDIR="$(pwd)/$DATA_DIR" install

mkdir -p "$DEB_DIR"
cp -rf pkg/control "$DEB_DIR/DEBIAN"
cp -rf "$DATA_DIR"/* "$DEB_DIR"

dpkg-deb --build "$DEB_DIR" "${PACKAGE}_${VERSION}_arm64.deb"
