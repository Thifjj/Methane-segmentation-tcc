#!/bin/bash
set -e

# Present in the PetaLinux image/sysroot captured by this project.
if [ -f /etc/profile.d/xilinx.sh ]; then
  source /etc/profile.d/xilinx.sh
fi

SRC="benchmark_zcu104.cpp"
OUT="benchmark_zcu104"
SWEEP_SRC="sweep_zcu104.cpp"
SWEEP_OUT="sweep_zcu104"

if pkg-config --exists opencv4; then
  OPENCV_PKG="opencv4"
elif pkg-config --exists opencv; then
  OPENCV_PKG="opencv"
else
  echo "ERROR: OpenCV pkg-config not found"
  exit 1
fi

echo "Building $SRC -> $OUT"

${CXX:-g++} \
  -O3 \
  -DNDEBUG \
  -std=c++17 \
  -Wall \
  -Wextra \
  "$SRC" \
  -o "$OUT" \
  $(pkg-config --cflags --libs "$OPENCV_PKG") \
  -lvart-runner \
  -lvitis_ai_library-graph_runner \
  -lxir \
  -lpthread

echo
echo "Built:"
ls -lh "$OUT"

echo
echo "Building $SWEEP_SRC -> $SWEEP_OUT"

${CXX:-g++} \
  -O3 \
  -DNDEBUG \
  -std=c++17 \
  -Wall \
  -Wextra \
  "$SWEEP_SRC" \
  -o "$SWEEP_OUT"

echo
echo "Built:"
ls -lh "$SWEEP_OUT"

echo
echo "Example:"
echo "./$OUT --model MODEL.xmodel --dataset DATASET --csv DATASET/test.csv --profile all"

echo
echo "Automatic staged sweep (up to 16 concurrent workers/slots):"
echo "./$SWEEP_OUT --model MODEL.xmodel --dataset DATASET --csv DATASET/test.csv --resume"
