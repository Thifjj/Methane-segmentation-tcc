#!/bin/bash
# Quantiza/avalia/compila os 4 SegFormer DPU (perfil de calibracao do artigo, 512x512). Rodar no container.
source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch
cd /workspace/VitisAI
B=build/vitis_ai
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
# nome | arquitetura | checkpoint
MODELS=(
 "segformer_dpu_bce|SegFormerB0_DPU|UNet_SegFormer_DPU_BCE_mag1c_rgb.pth"
 "segformer_dpu_focaldice|SegFormerB0_DPU|UNet_SegFormer_DPU_FD_mag1c_rgb.pth"
 "segformer_attgate_dpu_bce|SegFormerB0_Attention_DPU|UNet_SegFormer_AttGate_DPU_BCE_mag1c_rgb.pth"
 "segformer_attgate_dpu_focaldice|SegFormerB0_Attention_DPU|UNet_SegFormer_AttGate_DPU_FD_mag1c_rgb.pth"
)
for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME ARCHN CKPT <<< "$entry"
  CK=../Modelos_treinados/$CKPT
  LOG=$B/logs_segformer/$NAME; mkdir -p $LOG
  COMMON="--model $NAME --architecture $ARCHN --checkpoint $CK"
  echo "=== $NAME ==="
  python quantize_model.py $COMMON --quant-mode calib --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
    --no-patching --subset-len 100 --range-samples 512 --refine-layers 0 --output-dir $B/quantize_segformer > $LOG/calib.log 2>&1 || { echo "FALHOU calib $NAME"; continue; }
  python evaluate_quantized.py $COMMON --dataset test --csv /dataset_test/test.csv --data-root /dataset_test \
    --quant-dir $B/quantize_segformer --output-dir $B/evaluation_segformer --num-threads 4 > $LOG/eval.log 2>&1 || echo "FALHOU eval $NAME"
  python quantize_model.py $COMMON --quant-mode test --deploy --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
    --no-patching --range-samples 512 --refine-layers 0 --output-dir $B/quantize_segformer > $LOG/export.log 2>&1 || { echo "FALHOU export $NAME"; continue; }
  X=$(ls $B/quantize_segformer/$NAME/*_int.xmodel | head -1)
  python compile_xmodel.py --xmodel $X --arch $ARCH --output-dir $B/compiled_segformer/$NAME --name $NAME > $LOG/compile.log 2>&1 || echo "FALHOU compile $NAME"
done
echo FIM
