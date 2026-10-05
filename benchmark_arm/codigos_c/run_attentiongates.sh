#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
models_dir="$script_dir/../exportar_to_onnx/modelos_convertidos_onnx"
dataset=""
batch=1
threads=4
while (($#)); do
    case "$1" in
        --dataset) dataset=${2:?Falta pasta do dataset}; shift 2 ;;
        --models-dir) models_dir=${2:?Falta pasta dos modelos}; shift 2 ;;
        --patch-batch-size) batch=${2:?Falta batch}; shift 2 ;;
        --threads) threads=${2:?Falta threads}; shift 2 ;;
        --help|-h) echo 'Uso: ./run_attentiongates.sh --dataset PASTA [--models-dir PASTA] [--patch-batch-size 1|16] [--threads N]'; exit 0 ;;
        *) echo "Opcao desconhecida: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$dataset" ]] || { echo 'Informe --dataset PASTA' >&2; exit 2; }
[[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch deve ser inteiro positivo' >&2; exit 2; }
suffix=""
[[ "$batch" == 1 ]] || suffix="_batch_dynamic"
cd -- "$script_dir"
for model in attentiongates_dpu_easy_remaining attentiongates_dpu_only_remaining; do
    ./benchmark_arm --model "$models_dir/$model$suffix.onnx" --dataset "$dataset" \
        --threads "$threads" --patch-batch-size "$batch" --mode all
done
