#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="benchmark_nao_embarcado/resultados_remaining_all_dpu_retrain"
DATA_ROOT="/media/jacques/games/Datasets/test/STARCOP_test"
mkdir -p "$OUTPUT_DIR"

for DEVICE in cpu cuda; do
    for MODEL in attentiongates_dpu_bce_remaining_all attentiongates_dpu_focaldice_remaining_all; do
        echo "Iniciando $MODEL em $DEVICE"
        .venv/bin/python -u -m benchmark_nao_embarcado.benchmark_geral \
            --attention-dpu "$MODEL" \
            --device "$DEVICE" \
            --dataset test \
            --data-root "$DATA_ROOT" \
            --num-threads 4 \
            --patch-size 512 \
            --patch-batch-size 1 \
            --output-dir "$OUTPUT_DIR"
    done
done
