#!/usr/bin/env bash
# Run the six remaining article checkpoints with the project's Vitis AI tools.
set -u
export PYTHONDONTWRITEBYTECODE=1

cd /workspace/VitisAI || exit 1

build=build/vitis_ai
quant_root=$build/quantize_artigo
compiled_root=$build/compiled_artigo
evaluation_root=$build/evaluation_artigo
log_root=$build/logs_artigo
status=$log_root/batch_status.tsv
arch=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json

mkdir -p "$quant_root" "$compiled_root" "$evaluation_root" "$log_root"
if [[ ! -f "$status" ]]; then
  printf 'utc\tmodel\tstage\tstatus\tlog\n' > "$status"
fi

record() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" | tee -a "$status"
}

finish() {
  chown -R 1000:1000 "$quant_root" "$compiled_root" "$evaluation_root" "$log_root" || true
}
trap finish EXIT

if [[ ! -f /dataset_STARCOP/train.csv || ! -f /dataset_test/test.csv || ! -f "$arch" ]]; then
  record batch preflight failed missing_dataset_or_arch
  exit 1
fi

run_stage() {
  local model=$1 stage=$2 log=$3 code
  shift 3
  record "$model" "$stage" running "$log"
  if "$@" > "$log" 2>&1; then
    record "$model" "$stage" ok "$log"
    return 0
  else
    code=$?
    record "$model" "$stage" "failed:$code" "$log"
    return "$code"
  fi
}

while IFS='|' read -r model architecture checkpoint products; do
  [[ -n "$model" ]] || continue
  source_checkpoint=/workspace/Modelos_treinados/$checkpoint
  model_logs=$log_root/$model
  model_quant=$quant_root/$model
  model_compiled=$compiled_root/$model/$model.xmodel
  model_xmodel=$model_quant/${architecture}_int.xmodel
  model_summary=$evaluation_root/$model/summary.csv
  mkdir -p "$model_logs"

  if [[ ! -f "$source_checkpoint" ]]; then
    record "$model" preflight failed:missing_checkpoint "$source_checkpoint"
    continue
  fi

  if [[ -f "$model_quant/calibration_manifest.json" ]]; then
    record "$model" calib already_present "$model_quant/calibration_manifest.json"
  elif ! run_stage "$model" calib "$model_logs/calib.log" \
      python quantize_model.py --model "$model" --architecture "$architecture" \
      --checkpoint "$source_checkpoint" --quant-mode calib \
      --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
      --products "$products" --no-patching --subset-len 100 \
      --range-samples 512 --refine-layers 0 --num-workers 0 \
      --output-dir "$quant_root"; then
    continue
  fi

  if [[ -f "$model_xmodel" ]]; then
    record "$model" export already_present "$model_xmodel"
  else
    run_stage "$model" export "$model_logs/export.log" \
      python quantize_model.py --model "$model" --architecture "$architecture" \
      --checkpoint "$source_checkpoint" --quant-mode test --deploy \
      --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
      --products "$products" --no-patching --subset-len 1 \
      --range-samples 512 --refine-layers 0 --num-workers 0 \
      --output-dir "$quant_root" || true
  fi

  if [[ -f "$model_compiled" ]]; then
    record "$model" compile already_present "$model_compiled"
  elif [[ -f "$model_xmodel" ]]; then
    run_stage "$model" compile "$model_logs/compile.log" \
      python compile_xmodel.py --xmodel "$model_xmodel" --arch "$arch" \
      --output-dir "$compiled_root/$model" --name "$model" || true
  else
    record "$model" compile skipped:missing_xmodel "$model_xmodel"
  fi

  if [[ -f "$model_summary" ]]; then
    record "$model" evaluate already_present "$model_summary"
  else
    run_stage "$model" evaluate "$model_logs/evaluate.log" \
      python evaluate_quantized.py --model "$model" --architecture "$architecture" \
      --checkpoint "$source_checkpoint" --dataset test \
      --csv /dataset_test/test.csv --data-root /dataset_test \
      --quant-dir "$quant_root" --output-dir "$evaluation_root" \
      --num-threads 4 || true
  fi
done <<'MODELS'
hyperstarcop|HyperSTARCOPOficial|HyperSTARCOP_oficial/final_checkpoint_model.ckpt|mag1c,TOA_AVIRIS_640nm,TOA_AVIRIS_550nm,TOA_AVIRIS_460nm
attentiongates_dpu_focaldice_artigo|UNetMobileNetV3AttentionGatesDPU|UnetMobilenetV3AttentionGates_dpu_FocalDiceLossmag1c_rgb.pth|mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm
attentiongates_bce|UNetMobileNetV3AttentionGates|MobileNetV3_AttentionGates_BCE_mag1c_rgb.pth|mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm
attentiongates_focaldice_artigo|UNetMobileNetV3AttentionGates|MobileNetV3_AttentionGates_FocalDiceLossmag1c_rgb.pth|mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm
mobilenet_v3_bce_artigo|UNetMobileNetV3|Mobile_Net_v3_BCELoss_mag1c_rgb.pth|mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm
mobilenet_v3_focaldice_artigo|UNetMobileNetV3|MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth|mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm
MODELS

record batch all complete "$status"
