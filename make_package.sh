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

if [ ! -f "$DATA_DIR/usr/bin/mono_depth_rescaler" ]; then
    echo "no binary staged; run ./install_build_deps.sh then ./build.sh qrb5165" >&2
    exit 1
fi

mkdir -p "$DEB_DIR"
cp -rf pkg/control "$DEB_DIR/DEBIAN"
cp -rf "$DATA_DIR"/* "$DEB_DIR"

dpkg-deb --build "$DEB_DIR" "${PACKAGE}_${VERSION}_arm64.deb"
