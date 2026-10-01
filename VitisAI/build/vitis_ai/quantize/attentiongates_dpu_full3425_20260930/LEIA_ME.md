# Calibração com as 3.425 imagens do conjunto completo

- Dataset: `STARCOP_train_remaining_all/train.csv`, todas as 3.425 linhas.
- Patches: 128×128, passo 64; um patch sorteado entre os 49 de cada imagem, total de 3.425 patches. Os 167.825 patches permanecem elegíveis.
- Quantização: INT8, Vitis AI 3.5, target `DPUCZDX8G_ISA1_B4096`.
- Canais: mag1c, 460nm, 550nm, 640nm; mesma DataNormalizer do treinamento.
- Lotes de calibração: 16 patches. Modelo em modo eval, sem atualização dos pesos.
- Inclui as 514 imagens da antiga divisão de validação, conforme a solicitação de usar o conjunto completo. O TEST SET externo não participa da calibração.
- O checkpoint foi copiado para `checkpoint_fp32.pth`, com SHA256 registrado, para preservar a referência durante esta execução longa.

A calibração rápida anterior usou 1.000 patches. Conforme a escolha do usuário, esta usa um patch de cada uma das 3.425 imagens; o tempo estimado é de cerca de 8 minutos. A tentativa de usar todos os patches foi interrompida e sua configuração incompleta foi descartada.

## Calibração dentro do container

```bash
python quantize_model.py \
  --model mobilenet_v3_attention_gates_dpu \
  --checkpoint build/vitis_ai/quantize/attentiongates_dpu_full3425_20260930/checkpoint_fp32.pth \
  --quant-mode calib \
  --csv build/vitis_ai/quantize/attentiongates_dpu_full3425_20260930/calibration_full.csv \
  --data-root /dataset_STARCOP \
  --products mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm \
  --subset-len 0 --patching --patches-per-image 1 \
  --batch-size 16 --num-workers 2 --progress-every 128 \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/quantize/attentiongates_dpu_full3425_20260930
```

`--subset-len 0` usa todas as imagens; `--patches-per-image 1` garante um patch sorteado de cada imagem, sem deixar nenhuma de fora. Na execução inicial, o modelo foi carregado do checkpoint original, com o mesmo SHA256 da cópia acima.

## Exportação após a calibração terminar

```bash
python quantize_model.py \
  --model mobilenet_v3_attention_gates_dpu \
  --checkpoint build/vitis_ai/quantize/attentiongates_dpu_full3425_20260930/checkpoint_fp32.pth \
  --quant-mode test --deploy --patching \
  --csv build/vitis_ai/quantize/attentiongates_dpu_full3425_20260930/calibration_full.csv \
  --data-root /dataset_STARCOP \
  --products mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm \
  --num-workers 0 \
  --target DPUCZDX8G_ISA1_B4096 \
  --output-dir build/vitis_ai/quantize/attentiongates_dpu_full3425_20260930
```

O log da calibração fica em `calib.log`. A versão anterior com 1.000 patches foi preservada em sua própria pasta para comparação.

## Resultado concluído

Calibração: 8 min 51 s; avaliação: 9 min 27 s. F1 global INT8: 0,6213; F1 weak: 0,5138. Comparação completa em [COMPARACAO_FP32_INT8.md](metrics_test_cpu/COMPARACAO_FP32_INT8.md). Artefatos e checkpoint original conferidos por SHA-256.
