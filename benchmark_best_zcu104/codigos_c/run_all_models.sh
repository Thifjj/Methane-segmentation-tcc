#!/bin/bash
set -euo pipefail

models_dir=
dataset=
out=
csv=
extra=()

while (($#)); do
    case "$1" in
        --models-dir) models_dir=$2; shift 2 ;;
        --dataset) dataset=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --csv) csv=$2; shift 2 ;;
        *) extra+=("$1"); shift ;;
    esac
done

if [[ -z $models_dir || -z $dataset || -z $out ]]; then
    echo 'Uso: run_all_models.sh --models-dir PASTA --dataset PASTA --out PASTA [opcoes do sweep]' >&2
    exit 1
fi
if [[ -z $csv ]]; then csv=$dataset/test.csv; fi

mapfile -d '' models < <(find "$models_dir" -type f -name 'methane_*.xmodel' -print0 | sort -z)
if ((${#models[@]} == 0)); then
    echo "Nenhum methane_*.xmodel em $models_dir" >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for model in "${models[@]}"; do
    name=$(basename "$model" .xmodel)
    "$script_dir/sweep_vitis" --model "$model" --dataset "$dataset" \
        --csv "$csv" --out "$out/$name" "${extra[@]}"
done
