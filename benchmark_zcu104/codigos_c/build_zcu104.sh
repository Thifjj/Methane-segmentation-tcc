#!/bin/bash
set -euo pipefail

if [ -f /etc/profile.d/xilinx.sh ]; then
    source /etc/profile.d/xilinx.sh
fi

if pkg-config --exists opencv4; then
    opencv=opencv4
elif pkg-config --exists opencv; then
    opencv=opencv
else
    echo 'OpenCV nao encontrado via pkg-config' >&2
    exit 1
fi

${CXX:-g++} -O3 -DNDEBUG -std=c++17 -Wall -Wextra \
    benchmark_vitis.cpp dataset.cpp preprocess.cpp xmodel_runner.cpp \
    postprocess.cpp metricas.cpp pipeline.cpp power.cpp estatisticas.cpp resultados.cpp \
    validacao.cpp \
    -o benchmark_vitis $(pkg-config --cflags --libs "$opencv") \
    -lvart-runner -lvart-util -lvitis_ai_library-graph_runner -lxir -pthread

${CXX:-g++} -O2 -DNDEBUG -std=c++17 -Wall -Wextra \
    sweep.cpp -o sweep_vitis -pthread

${CXX:-g++} -O2 -std=c++17 -Wall -Wextra \
    self_test_support.cpp dataset.cpp preprocess.cpp metricas.cpp postprocess.cpp estatisticas.cpp \
    -o self_test_support $(pkg-config --cflags --libs "$opencv") -pthread
