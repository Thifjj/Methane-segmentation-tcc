# MobileNet V3 Focal Dice: FP32 e INT8, 512x512

**Qualidade reprovada no criterio BCE:** perda permitida de 1 ponto percentual de F1 global; perda observada de 14.6824 pontos percentuais.

| Metrica | FP32 | INT8 |
|---|---:|---:|
| F1-Global | 0.495362992 | 0.348538494 |
| IoU | 0.329224251 | 0.211048513 |
| AUPRC | 0.413685547 | 0.352075652 |
| F1-Strong | 0.485071094 | 0.351009337 |
| F1-Weak | 0.563067608 | 0.351707411 |
| FPR (No-Plume) | 0.000112850 | 0.000038932 |

Checkpoint `MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth`; arquitetura `UNetMobileNetV3_dpu`, cinco upsamplings com `align_corners=False`.

Protocolo BCE preservado: mesmas 300 imagens de calibracao, na mesma ordem, CSV e seed 12345; batch 1, entrada inteira 512x512, MAG1C/1750 e RGB 460/550/640 nm divididos por 60, clamp [0,2]. Faixas por camada, clipping 0,1%, 2048 amostras e refinamento de 12 camadas. As mesmas 342 imagens TEST, dificuldades e contagens positivas dos labels foram conferidas. Calibracao e TEST nao se sobrepoem.

Metricas: logit > 0, abertura em cruz 3x3, forte/fraco por difficulty, AUPRC como media de AP das imagens positivas e FPR nos pixels sem pluma. Nao houve ajuste de limiar nem troca de imagens para melhorar o resultado.

Compilado para DPUCZDX8G_ISA1_B4096: um subgrafo DPU, 276 operacoes, entrada INT8 NHWC [1,512,512,4] fix_point 5; saida [1,512,512,1] fix_point 1. CPU somente fix2float. Arch.json e fingerprint coincidem com o BCE.

Compatibilidade do grafo validada. Qualidade reprovada no mesmo limite BCE; necessita melhoria de calibracao/quantizacao antes de aprovar o modelo. INT8 avaliado em simulacao Vitis AI 3.5 na CPU, sem execucao fisica na placa.

[Resumo CSV](summary.csv). Os CSVs FP32/INT8 desta pasta contem IDs e contagens por imagem. [Manifesto de avaliacao](evaluation_manifest.json) registra hashes e criterio.
