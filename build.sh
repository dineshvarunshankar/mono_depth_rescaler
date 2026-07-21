#!/bin/bash
#   ./build.sh qrb5165    cross-compile for VOXL2 (run inside: voxl-docker -i voxl-cross)
#   ./build.sh native     local build + tests (dev machine)
set -e

TOOLCHAIN_QRB5165="/opt/cross_toolchain/qrb5165_ubun1_18.04_aarch64.toolchain.cmake"

print_usage() {
    echo ""
    echo " Build for a specific target."
    echo " Usage:"
    echo "   ./build.sh qrb5165    cross-compile for VOXL2 (inside voxl-cross docker)"
    echo "   ./build.sh native     build + run tests on this machine"
    echo ""
}

case "$1" in
    qrb5165)
        mkdir -p build
        cd build
        cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_QRB5165} \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_CXX_STANDARD=17 \
              -DBUILD_TESTING=OFF ..
        make -j$(nproc)
        ;;
    native)
        mkdir -p build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)
        ctest --output-on-failure
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
