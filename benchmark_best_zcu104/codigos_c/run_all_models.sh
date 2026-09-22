#!/bin/bash
set -euo pipefail

models_dir=
dataset=

while (($#)); do
    case "$1" in
        --models-dir) models_dir=$2; shift 2 ;;
        --dataset) dataset=$2; shift 2 ;;
        *) echo "Opcao desconhecida: $1" >&2; exit 1 ;;
    esac
done

if [[ -z $models_dir || -z $dataset ]]; then
    echo 'Uso: run_all_models.sh --models-dir PASTA --dataset PASTA' >&2
    exit 1
fi

mapfile -d '' models < <(find "$models_dir" -type f -name 'methane_*.xmodel' -print0 | sort -z)
if ((${#models[@]} == 0)); then
    echo "Nenhum methane_*.xmodel em $models_dir" >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for model in "${models[@]}"; do
    "$script_dir/sweep_vitis" --model "$model" --dataset "$dataset"
done
