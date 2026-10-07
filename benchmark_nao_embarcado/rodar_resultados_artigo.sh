#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-cpu}"
DATA_ROOT="${2:-/media/jacques/games/Datasets/test/STARCOP_test}"

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
    echo "Uso: $0 {cpu|cuda} [diretorio_STARCOP_test]" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
OUTPUT_DIR="benchmark_nao_embarcado/resultados_artigo"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    attentiongates_dpu_focaldice_artigo
    attentiongates_dpu_bce_artigo
    attentiongates_bce
    attentiongates_focaldice_artigo
    mobilenet_v3_bce_artigo
    mobilenet_v3_focaldice_artigo
)

for MODEL in "${MODELS[@]}"; do
    "$PROJECT_ROOT/.venv/bin/python" -m benchmark_nao_embarcado.benchmark_geral \
        --attention-dpu "$MODEL" \
        --device "$DEVICE" \
        --dataset test \
        --data-root "$DATA_ROOT" \
        --num-threads 4 \
        --patch-size 512 \
        --patch-batch-size 1 \
        --output-dir "$OUTPUT_DIR"
done
