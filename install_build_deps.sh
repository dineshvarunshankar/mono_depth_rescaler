#!/bin/bash
# Install build dependencies into the voxl-cross sysroot from the ModalAI feed.
#   ./install_build_deps.sh qrb5165 dev      # 1.x system image
#   ./install_build_deps.sh qrb5165-2 dev    # 2.x system image

DEPS="
libmodal-json
libmodal-pipe"

if [ "$#" -ne 2 ]; then
    echo "usage: ./install_build_deps.sh {platform} {section}   e.g. qrb5165 dev"
    echo "sections/platforms: http://voxl-packages.modalai.com/dists/"
    exit 1
fi

PLATFORM=$(echo "$1" | tr '[:upper:]' '[:lower:]')
SECTION=$(echo "$2" | tr '[:upper:]' '[:lower:]')
echo "using $PLATFORM $SECTION debian repo"

LINE="deb [trusted=yes] http://voxl-packages.modalai.com/ ./dists/$PLATFORM/$SECTION/binary-arm64/"
sudo echo "${LINE}" > /etc/apt/sources.list.d/modalai.list

sudo apt-get update \
    -o Dir::Etc::sourcelist="sources.list.d/modalai.list" \
    -o Dir::Etc::sourceparts="-" \
    -o APT::Get::List-Cleanup="0"

echo -e "\nINSTALLING: $DEPS\n"
sudo apt install -y $DEPS
