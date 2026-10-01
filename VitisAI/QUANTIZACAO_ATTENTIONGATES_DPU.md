# Quantização INT8 do AttentionGates DPU com patches

## Configuração

- Modelo: `mobilenet_v3_attention_gates_dpu` / `UNetMobileNetV3AttentionGatesDPU`.
- Checkpoint FP32: `Modelos_treinados/UNetMobileNetV3AttentionGatesDPU_mag1c_rgb.pth`.
- Target: `DPUCZDX8G_ISA1_B4096`, Vitis AI 3.5.
- Entrada exportada: `[1, 4, 128, 128]`.
- Recortes: patches de 128×128 com passo 64, como no treinamento; 49 patches por imagem de 512×512.
- Ordem dos canais: `mag1c,460nm,550nm,640nm`.
- Normalização: `Utils.DataLoader.DataNormalizer`, a mesma do treino e do teste.

A calibração sorteia 1.000 patches entre os patches das 2.911 imagens de treino, com seed 12345. As 514 imagens de validação estão excluídas. Portanto, são 1.000 patches, não 1.000 imagens inteiras. Os lotes de calibração contêm 16 patches.

A calibração usa o modelo em modo `eval`, sem augmentações aleatórias, máscaras, mapas de loss ou atualização de pesos. Mantém a geometria dos recortes e a normalização do treinamento, para medir a inferência que será usada no teste e na placa.

## Calibrar e exportar

No host:

```bash
docker start methane-vitis-ai-dpu
docker exec -it -w /workspace/VitisAI methane-vitis-ai-dpu bash
```

Dentro do container:

```bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

Calibração:

```bash
python quantize_model.py \
  --model mobilenet_v3_attention_gates_dpu \
  --quant-mode calib \
  --csv build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930/calibration_train.csv \
  --data-root /dataset_STARCOP \
  --products mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm \
  --subset-len 0 --patching --patch-count 1000 \
  --batch-size 16 --num-workers 2 --seed 12345 \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930
```

Exportação, após concluir a calibração:

```bash
python quantize_model.py \
  --model mobilenet_v3_attention_gates_dpu \
  --quant-mode test --deploy --patching \
  --csv build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930/calibration_train.csv \
  --data-root /dataset_STARCOP \
  --products mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm \
  --num-workers 0 \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930
```

O deploy usa batch 1 e uma única inferência de um patch. O aplicativo na placa deverá recortar os patches e processá-los usando a mesma ordem de canais e normalização.

## Avaliar FP32 e INT8 nos mesmos patches

No host, o container `methane-vitis-ai-dpu-eval` também monta o TEST SET em `/dataset_test`, somente para leitura:

```bash
docker start methane-vitis-ai-dpu-eval
docker exec -it -w /workspace/VitisAI methane-vitis-ai-dpu-eval bash
```

Ative o mesmo ambiente conda e execute:

```bash
python evaluate_quantized.py \
  --quant-dir build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930/mobilenet_v3_attention_gates_dpu \
  --input-mode patches \
  --csv /dataset_test/test.csv --data-root /dataset_test \
  --output-dir build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930/metrics_test_cpu
```

A comparação é feita em CPU, com INT8 simulado pelo PyTorch do Vitis AI. Não mede a vazão da DPU. Usa o mesmo threshold, abertura morfológica, contagens acumuladas e AUPRC média por imagem do teste original.

## Artefatos e referência anterior

Os arquivos da nova execução ficam em `build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930/`. O manifesto `mobilenet_v3_attention_gates_dpu/calibration_patches.json` registra os patches efetivamente selecionados no container.

Os números da comparação anterior com entrada 512 estão preservados em `referencia_512/`, para documentar a mudança. O XModel, a configuração e os arquivos gerados da quantização anterior de 512 pixels foram removidos. Apenas os resultados numéricos e o relatório de comparação foram preservados como referência.

A exportação INT8 ainda exige compilação com o `arch.json` correspondente à DPU da placa e avaliação na placa.

## Resultado concluído

Calibração dos 1.000 patches concluída em 140 segundos, XModel de 128×128 exportado e avaliação nas 342 imagens do TEST SET concluída. F1 Global FP32: 0.6463; INT8: 0.5908. F1 Weak FP32: 0.5581; INT8: 0.4945. Ainda há perda de qualidade com INT8. Relatório: `build/vitis_ai/quantize/attentiongates_dpu_patches128_20260930/metrics_test_cpu/COMPARACAO_FP32_INT8.md`.
