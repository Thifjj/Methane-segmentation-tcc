# MobileNet V3 DPU: FP32 e INT8

Mesmas 342 imagens, checkpoint e arquitetura com align_corners=False em todos os modos.

| Modo | F1 global | IoU | AUPRC | F1 forte | F1 fraco | FPR sem pluma |
|---|---:|---:|---:|---:|---:|---:|
| FP32_512 | 0.516919 | 0.348544 | 0.745439 | 0.510009 | 0.562549 | 0.000094 |
| FP32_patches128 | 0.114279 | 0.060602 | 0.363282 | 0.110600 | 0.142192 | 0.000155 |
| INT8_patches128 | 0.118082 | 0.062746 | 0.361373 | 0.122146 | 0.108275 | 0.000027 |

Delta INT8 menos FP32 nos mesmos patches (pontos percentuais):

- F1-Global: +0.3803
- IoU: +0.2143
- AUPRC: -0.1909
- F1-Strong: +1.1546
- F1-Weak: -3.3917
- FPR (No-Plume): -0.0128

A queda de F1 entre FP32 na imagem inteira e FP32 nos patches e muito maior que a alteracao pela quantizacao. O teste evidencia um impacto do modo de inferencia por patches; nao identifica isoladamente a causa da baixa qualidade.

Protocolo: canais mag1c, 460, 550, 640 nm; mag1c / 1750, RGB / 60, clip [0,2]; logits > 0; abertura morfologica com cruz 3x3. Avaliacao em patches 128x128 sem sobreposicao. INT8 por simulacao Vitis AI 3.5 em CPU. Sem execucao ou medicao de velocidade na placa.

Compatibilidade conferida no XModel compilado: um subgrafo DPU com 276 operacoes, um no USER de entrada data-fix e uma conversao final fix2float na CPU. O processamento da rede esta na DPU; o grafo completo inclui essa conversao na CPU.

Os CSVs por imagem, summary.csv e evaluation_manifest.json registram resultados e hashes.
