#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET=/dataset_STARCOP
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
OUT=build/vitis_ai/quantize

test -f "$DATASET/train.csv" || { echo "Dataset ausente: $DATASET/train.csv" >&2; exit 1; }
test -f "$ARCH" || { echo "arch.json ausente: $ARCH" >&2; exit 1; }
test -f ../Modelos_treinados/HyperSTARCOP_oficial/final_checkpoint_model.ckpt || {
  echo "Checkpoint do HyperSTARCOP ausente" >&2
  exit 1
}

python -c 'import rasterio, segmentation_models_pytorch, omegaconf, pytorch_nndct' || {
  echo "Dependencias ausentes no ambiente vitis-ai-pytorch" >&2
  exit 1
}

echo "=== Calibrando o HyperSTARCOP com 1000 imagens ==="
python quantize_model.py \
  --model hyperstarcop --quant-mode calib \
  --csv "$DATASET/train.csv" --data-root "$DATASET" \
  --subset-len 1000 --seed 12345 \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir "$OUT"

echo "=== Exportando o XModel INT8 ==="
python quantize_model.py \
  --model hyperstarcop --quant-mode test --deploy \
  --csv "$DATASET/train.csv" --data-root "$DATASET" \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir "$OUT"

XMODEL="$(find "$OUT/hyperstarcop" -maxdepth 1 -name '*_int.xmodel' -print -quit)"
test -n "$XMODEL" || { echo "XModel INT8 ausente" >&2; exit 1; }

echo "=== Compilando para a ZCU104 ==="
python compile_xmodel.py \
  --xmodel "$XMODEL" --arch "$ARCH" \
  --output-dir build/vitis_ai/compiled_zcu104/hyperstarcop \
  --name methane_hyperstarcop

echo "=== Modelo pronto ==="
ls -lh build/vitis_ai/compiled_zcu104/hyperstarcop/methane_hyperstarcop.xmodel
