# MobileNet V3 DPU 512x512: calibracao e comparacao

Configuracao aceita: perda maxima de F1 global de 1 ponto percentual.

| Modo | F1 global | IoU | AUPRC | F1 forte | F1 fraco | FPR sem pluma |
|---|---:|---:|---:|---:|---:|---:|
| FP32_512 | 0.516919 | 0.348544 | 0.745439 | 0.510009 | 0.562549 | 0.000094 |
| INT8_512 | 0.512693 | 0.344712 | 0.722560 | 0.514948 | 0.518384 | 0.000039 |

Perda de F1 global: 0.4226 ponto percentual. A meta foi atingida nas mesmas 342 imagens de teste.

O F1 nas plumas fracas caiu de 0,562549 para 0,518384, e a AUPRC de 0,745439 para 0,722560. A proximidade do F1 global nao significa preservacao identica de todas as metricas.

A calibracao usa 300 imagens completas de treinamento (sem IDs em comum com o teste), seed 12345, batch 1, layer_mse, clipping 0,1%, 2048 amostras por camada e refinamento de 12 camadas. A entrada de 512x512 preserva o contexto usado na referencia FP32. O refinamento utiliza apenas as imagens de calibracao e os logits FP32.

A comparacao manteve canais mag1c, 460, 550, 640 nm, normalizacao do treinamento, limiar logit > 0 e abertura morfologica com cruz 3x3. Os totais de pixels, o F1 recalculado e os hashes dos pesos e da quantizacao foram conferidos. A referencia FP32 reproduziu exatamente as contagens do teste anterior.

Compilado para DPUCZDX8G_ISA1_B4096 da ZCU104: compiled_zcu104/mobilenet_v3_dpu_512/mobilenet_v3_dpu_512.xmodel, relativo a build/vitis_ai. XIR confirmou um subgrafo DPU com 276 operacoes, entrada INT8 NHWC [1,512,512,4] fix_point 5 e saida INT8 [1,512,512,1] fix_point 2. Existe uma conversao final fix2float na CPU.

Os valores INT8 sao simulacao Vitis AI 3.5 em CPU. A execucao fisica na placa ainda nao foi validada. O runner deve usar entrada 512x512 para este artefato.

Perfil reutilizavel: VitisAI/configs/mobilenet_v3_dpu_512.json. Comandos no README do VitisAI. evaluation_manifest.json registra o criterio e os hashes; compilation_manifest.json registra as interfaces compiladas.
