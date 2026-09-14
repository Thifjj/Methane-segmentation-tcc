# Comparação de resultados: CPU e ZCU104

Os dados CPU vêm do registro CPU mais recente de cada modelo em `Resultados_testes/historico_testes.csv`. Os dados ZCU104 usam as execuções finais de 4 núcleos em `results_all_models`, com 342 imagens, dois runners e potência habilitada. FPS da CPU foi calculado como `1000 / latência`.

## Resultados na CPU (PRECISAM SER ARRUMADOS RODAR DATASET INTEIRO)

| Modelo | Teste | Parâmetros | Tamanho (MB) | Latência (ms/img) | FPS | F1 global | F1 forte | F1 fraco | IoU | AUPRC | FPR sem pluma |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 1 | 31038209 | 118.49 | 1481.75 | 0.675 | 84.11% | 84.11% | 0.00% | 72.57% | 92.59% | 0.00% |
| Depth Reduced | 3 | 1864705 | 7.15 | 972.49 | 1.028 | 92.55% | 92.55% | 0.00% | 86.13% | 97.15% | 0.00% |
| Skip Connections | 2 | 1680385 | 6.44 | 835.64 | 1.197 | 90.09% | 90.09% | 0.00% | 81.97% | 97.07% | 0.00% |
| MobileNet V2 | 1 | 4132865 | 16.02 | 146.97 | 6.804 | 86.43% | 86.43% | 0.00% | 76.10% | 94.39% | 0.00% |
| MobileNet V3 | 1 | 1651953 | 6.45 | 57.64 | 17.349 | 64.99% | 64.99% | 0.00% | 48.14% | 68.95% | 0.00% |

## Resultados na ZCU104 (PRECISAM SER ARRUMADOS RODAR DATASET INTEIRO)

Todas as latências estão em milissegundos, a potência em watts, o tempo em segundos e a energia em joules. `MO` significa model-only e `E2E` significa end-to-end.

| Modelo | Imagens | Config. | MO FPS | MO média | MO mediana | MO mín. | MO máx. | MO P95 | MO P99 | MO DPU | MO tempo | MO W méd. | MO W mín. | MO W máx. | MO energia | E2E FPS | E2E média | E2E mediana | E2E mín. | E2E máx. | E2E P95 | E2E P99 | E2E DPU | Pré | Pós | I/O | E2E tempo | E2E W méd. | E2E W mín. | E2E W máx. | E2E energia | TP | FP | FN | TN | Precisão | Recall | F1 global | F1 forte | F1 fraco | IoU | AUPRC | FPR sem pluma | Acurácia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 342 | r2/pre1/post1/spr1 | 5.926 | 337.076 | 337.664 | 334.497 | 341.822 | 339.454 | 340.290 | 337.076 | 57.713 | 27.125 | 15.412 | 28.600 | 1565.473 | 3.928 | 596.390 | 596.528 | 462.714 | 737.043 | 704.378 | 726.642 | 336.656 | 161.748 | 6.129 | 121.443 | 87.067 | 23.054 | 15.050 | 28.175 | 2007.287 | 29002 | 27990 | 161650 | 89434606 | 50.89% | 15.21% | 23.42% | 26.63% | 23.65% | 13.26% | 28.71% | 0.06% | 99.79% |
| Depth Reduced | 342 | r2/pre1/post1/spr1 | 10.945 | 182.578 | 182.631 | 180.687 | 185.100 | 183.005 | 183.693 | 182.578 | 31.248 | 26.627 | 15.287 | 28.337 | 832.030 | 5.644 | 364.286 | 362.039 | 308.638 | 426.198 | 391.963 | 415.306 | 181.450 | 162.284 | 6.150 | 121.651 | 60.593 | 20.879 | 14.912 | 27.400 | 1265.115 | 37602 | 19511 | 153050 | 89443085 | 65.84% | 19.72% | 30.35% | 35.22% | 13.54% | 17.89% | 29.82% | 0.04% | 99.81% |
| Skip Connections | 342 | r2/pre2/post1/spr1 | 12.384 | 161.097 | 161.810 | 157.688 | 168.107 | 166.227 | 167.548 | 161.097 | 27.615 | 25.889 | 15.125 | 28.700 | 714.930 | 6.112 | 492.509 | 495.696 | 337.248 | 518.392 | 510.983 | 513.826 | 160.077 | 160.720 | 6.181 | 120.813 | 55.954 | 20.532 | 14.925 | 28.450 | 1148.824 | 6406 | 5339 | 184246 | 89457257 | 54.54% | 3.36% | 6.33% | 7.15% | 1.88% | 3.27% | 23.92% | 0.01% | 99.79% |
| MobileNet V2 | 342 | r2/pre1/post1/spr1 | 8.501 | 234.572 | 234.880 | 232.927 | 236.629 | 235.502 | 236.154 | 234.572 | 40.230 | 15.571 | 14.450 | 18.987 | 626.401 | 4.973 | 441.626 | 440.994 | 364.124 | 527.849 | 486.540 | 511.045 | 235.004 | 158.937 | 6.145 | 119.539 | 68.777 | 15.365 | 14.437 | 18.312 | 1056.732 | 133907 | 88474 | 56745 | 89374122 | 60.22% | 70.24% | 64.84% | 80.25% | 49.81% | 47.97% | 60.17% | 0.14% | 99.84% |
| MobileNet V3 | 342 | r2/pre1/post1/spr1 | 9.437 | 211.500 | 210.920 | 209.263 | 216.928 | 213.630 | 214.435 | 211.500 | 36.241 | 15.369 | 14.437 | 17.875 | 556.987 | 5.270 | 406.603 | 405.281 | 336.693 | 484.226 | 444.034 | 473.391 | 211.299 | 158.699 | 6.118 | 119.279 | 64.898 | 15.159 | 14.425 | 16.987 | 983.816 | 60171 | 14422 | 130481 | 89448174 | 80.67% | 31.56% | 45.37% | 43.06% | 59.26% | 29.34% | 41.92% | 0.00% | 99.84% |

## Comparação CPU e ZCU104

O ganho usa latência: valores acima de `1×` indicam que a ZCU104 foi mais rápida. A comparação de qualidade deve ser tratada como indicativa porque o histórico CPU não registra o dataset nem o número de imagens usados, enquanto a ZCU104 usou explicitamente 342 imagens do `STARCOP_test`.

| Modelo | CPU ms | CPU FPS | ZCU MO ms | ZCU MO FPS | Ganho MO | ZCU E2E ms | ZCU E2E FPS | Ganho E2E | CPU F1 | ZCU F1 | Δ F1 | CPU IoU | ZCU IoU | CPU AUPRC | ZCU AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 1481.75 | 0.675 | 337.076 | 5.926 | 4.40× | 596.390 | 3.928 | 2.48× | 84.11% | 23.42% | -60.69 pp | 72.57% | 13.26% | 92.59% | 28.71% |
| Depth Reduced | 972.49 | 1.028 | 182.578 | 10.945 | 5.33× | 364.286 | 5.644 | 2.67× | 92.55% | 30.35% | -62.20 pp | 86.13% | 17.89% | 97.15% | 29.82% |
| Skip Connections | 835.64 | 1.197 | 161.097 | 12.384 | 5.19× | 492.509 | 6.112 | 1.70× | 90.09% | 6.33% | -83.76 pp | 81.97% | 3.27% | 97.07% | 23.92% |
| MobileNet V2 | 146.97 | 6.804 | 234.572 | 8.501 | 0.63× | 441.626 | 4.973 | 0.33× | 86.43% | 64.84% | -21.59 pp | 76.10% | 47.97% | 94.39% | 60.17% |
| MobileNet V3 | 57.64 | 17.349 | 211.500 | 9.437 | 0.27× | 406.603 | 5.270 | 0.14× | 64.99% | 45.37% | -19.62 pp | 48.14% | 29.34% | 68.95% | 41.92% |

`Model-only` é a comparação temporal mais próxima da inferência CPU registrada. `End-to-end` inclui leitura, normalização, filas, modelo e pós-processamento. Os valores de energia representam o consumo total estimado durante cada região medida, por isso dependem também da duração completa do teste.
