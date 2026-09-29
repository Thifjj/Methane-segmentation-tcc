#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Na placa: ./build.sh. Cross compile: SYSROOT=... CXX=aarch64-linux-gnu-g++ ./build.sh
if [[ -n "${SYSROOT:-}" ]]; then
    export PKG_CONFIG_SYSROOT_DIR="$SYSROOT"
    export PKG_CONFIG_LIBDIR="$SYSROOT/usr/lib/pkgconfig:$SYSROOT/usr/share/pkgconfig"
    sysroot_arg=("--sysroot=$SYSROOT")
else
    sysroot_arg=()
fi

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra "${sysroot_arg[@]}" benchmark_arm.cpp power.cpp \
    -o benchmark_arm $(pkg-config --cflags --libs opencv4 libonnxruntime) -pthread
