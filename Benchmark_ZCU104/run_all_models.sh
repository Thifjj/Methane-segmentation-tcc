#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../build/compiled_zcu104"
[[ -d "$MODELS_DIR" ]] || MODELS_DIR="$SCRIPT_DIR/../build/vitis_ai/compiled_zcu104"
MODEL=""
DATASET=""
CSV=""
OUT="$SCRIPT_DIR/results_all_models"
EXTRA=()

usage() {
  cat <<'EOF'
Usage:
  ./run_all_models.sh --dataset DIR [--csv FILE] [--models-dir DIR] [sweep options]
  ./run_all_models.sh --model FILE.xmodel --dataset DIR [--csv FILE] [sweep options]

Known wrapper options: --model, --models-dir, --dataset, --csv, --out.
All other options are passed unchanged to sweep_zcu104.
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "$DATASET" ]] || { echo "ERROR: --dataset is required" >&2; exit 2; }
[[ -n "$CSV" ]] || CSV="$DATASET/test.csv"
[[ -x "$SCRIPT_DIR/sweep_zcu104" ]] || {
  echo "ERROR: build first with $SCRIPT_DIR/build_zcu104.sh" >&2
  exit 2
}

if [[ -n "$MODEL" ]]; then
  MODELS=("$MODEL")
else
  mapfile -t MODELS < <(find "$MODELS_DIR" -type f -name '*.xmodel' | sort)
fi
((${#MODELS[@]})) || { echo "ERROR: no .xmodel found" >&2; exit 2; }

mkdir -p "$OUT"
for xmodel in "${MODELS[@]}"; do
  name="$(basename "$xmodel" .xmodel)"
  "$SCRIPT_DIR/sweep_zcu104" \
    --binary "$SCRIPT_DIR/benchmark_zcu104" \
    --model "$xmodel" \
    --dataset "$DATASET" \
    --csv "$CSV" \
    --out "$OUT/$name" \
    "${EXTRA[@]}"
done

for output_name in benchmark_geral.csv benchmark_overhead.csv metricas_globais.csv; do
  destination="$OUT/$output_name"
  first=1
  : > "$destination"
  for xmodel in "${MODELS[@]}"; do
    source_csv="$OUT/$(basename "$xmodel" .xmodel)/$output_name"
    [[ -f "$source_csv" ]] || continue
    if ((first)); then
      head -n 1 "$source_csv" >> "$destination"
      first=0
    fi
    tail -n +2 "$source_csv" >> "$destination"
  done
done

echo "Consolidated results: $OUT/benchmark_geral.csv"
