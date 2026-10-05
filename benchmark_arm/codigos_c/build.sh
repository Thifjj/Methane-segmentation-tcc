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

# Pacote oficial ORT (ORT_ROOT) ou o pkg-config libonnxruntime da placa.
if [[ -n "${ORT_ROOT:-}" ]]; then
    ort_flags=("-I$ORT_ROOT/include" "-L$ORT_ROOT/lib" -lonnxruntime)
else
    ort_text=$(pkg-config --cflags --libs libonnxruntime)
    read -r -a ort_flags <<< "$ort_text"
fi
opencv_text=$(pkg-config --cflags --libs opencv4)
read -r -a opencv_flags <<< "$opencv_text"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra "${sysroot_arg[@]}" \
    benchmark_arm.cpp dataset.cpp postprocess.cpp power.cpp \
    -o benchmark_arm "${opencv_flags[@]}" "${ort_flags[@]}" -pthread
