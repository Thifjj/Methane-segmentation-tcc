# AttentionGates DPU no TEST: CPU × GPU × ZCU104

Comparativo atualizado com as execuções CPU/CUDA de 02/10/2026 e os resultados existentes da ZCU104. Cada execução mede 342 imagens 512×512, divididas em 16 patches 128×128, na ordem MAG1C, 460, 550, 640 nm. FPS representa imagens completas por segundo, não patches por segundo.

CPU: AMD Ryzen 5 5600X, PyTorch FP32, 4 threads. GPU: NVIDIA GeForce RTX 4070 SUPER, PyTorch 2.5.1+cu124, CUDA 12.4, FP32. CPU/GPU usam batches de patches 1 e 16, com execução sequencial de imagens. DPU: ZCU104, XModel INT8, Vitis AI 3.0, 2 runners concorrentes, 4 núcleos CPU, 2 workers de pré-processamento e 1 de pós-processamento. Warm-up de 10 imagens fora das medidas.

`throughput_fps = imagens / duração medida`; `fps_latencia = 1000 / latência média em ms`. A concorrência da placa aumenta a vazão sem reduzir na mesma proporção a latência por imagem. As razões abaixo comparam as configurações medidas, incluindo precisão e concorrência diferentes.

## FPS de todas as configurações

| Variante / modo | CPU batch 1 | CPU batch 16 | GPU batch 1 | GPU batch 16 | DPU ZCU104 |
|---|---:|---:|---:|---:|---:|
| easy_remaining / model-only | 10.603 | 14.774 | 13.354 | 188.544 | 47.358 |
| easy_remaining / end-to-end | 8.396 | 10.939 | 11.132 | 40.060 | 14.787 |
| only_remaining / model-only | 10.780 | 15.632 | 13.550 | 186.606 | 47.335 |
| only_remaining / end-to-end | 8.816 | 11.455 | 11.077 | 39.698 | 14.795 |

Valores da tabela: throughput medido em imagens completas por segundo. DPU: medição existente INT8 com 2 runners; CPU/GPU: FP32. A DPU não foi medida novamente com batch 16.

## attentiongates_dpu_easy_remaining

| Métrica / modo | CPU batch 1 | CPU batch 16 | GPU batch 1 | GPU batch 16 | DPU ZCU104 |
|---|---:|---:|---:|---:|---:|
| Throughput (imagens/s) — model-only | 10.603 | 14.774 | 13.354 | 188.544 | 47.358 |
| FPS por latência — model-only | 10.604 | 14.780 | 13.355 | 188.833 | 23.684 |
| Latência média (ms/imagem) — model-only | 94.305 | 67.658 | 74.877 | 5.296 | 42.223 |
| Latência P95 (ms/imagem) — model-only | 94.996 | 71.317 | 76.611 | 5.728 | 44.404 |
| Throughput (imagens/s) — end-to-end | 8.396 | 10.939 | 11.132 | 40.060 | 14.787 |
| FPS por latência — end-to-end | 8.397 | 10.942 | 11.133 | 40.084 | 5.197 |
| Latência média (ms/imagem) — end-to-end | 119.086 | 91.393 | 89.821 | 24.947 | 192.414 |
| Latência P95 (ms/imagem) — end-to-end | 123.082 | 94.757 | 93.373 | 27.187 | 231.923 |

Qualidade: validação oficial com abertura em cruz 3×3; grupo forte definido por label positivo e `difficulty=easy`; AUPRC é a média de average precision nas imagens positivas.

| Métrica | CPU batch 1 | CPU batch 16 | GPU batch 1 | GPU batch 16 | DPU ZCU104 |
|---|---:|---:|---:|---:|---:|
| Precisão | 0.651539 | 0.651539 | 0.651573 | 0.651644 | 0.672809 |
| Recall | 0.639773 | 0.639773 | 0.639742 | 0.639773 | 0.580671 |
| F1 global | 0.645602 | 0.645602 | 0.645603 | 0.645654 | 0.623353 |
| IoU | 0.476671 | 0.476671 | 0.476672 | 0.476727 | 0.452806 |
| AUPRC | 0.630064 | 0.630064 | 0.630022 | 0.630075 | 0.560959 |
| F1 fortes | 0.773928 | 0.773928 | 0.773901 | 0.773928 | 0.741348 |
| F1 fracas | 0.586412 | 0.586412 | 0.586468 | 0.586427 | 0.553665 |
| FPR global por pixel | 0.000729 | 0.000729 | 0.000729 | 0.000729 | 0.000602 |
| FPR sem pluma por pixel | 0.000956 | 0.000956 | 0.000956 | 0.000955 | 0.000831 |
| FPR por tile | 0.034286 | 0.034286 | 0.034286 | 0.034286 | 0.028571 |
| Acurácia | 0.998506 | 0.998506 | 0.998506 | 0.998507 | 0.998508 |

Fontes desta variante:

- CPU batch 1: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cpu_20261002_181922_147241/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cpu_20261002_181922_147241/validacao_oficial/metricas_globais.csv).
- CPU batch 16: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cpu_20261002_185314_772913/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cpu_20261002_185314_772913/validacao_oficial/metricas_globais.csv).
- GPU batch 1: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cuda_20261002_184036_060767/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cuda_20261002_184036_060767/validacao_oficial/metricas_globais.csv).
- GPU batch 16: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cuda_20261002_185733_976854/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_easy_remaining_test_cuda_20261002_185733_976854/validacao_oficial/metricas_globais.csv).
- DPU ZCU104: [desempenho](../resultados_zcu104/attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/benchmark_geral.csv) e [validação oficial](../resultados_zcu104/attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/validacao_oficial/metricas_globais.csv).

[CSV batch 1](comparativo_test_attentiongates_dpu_easy_remaining_zcu104_cpu_gpu.csv) · [CSV batch 16](comparativo_test_attentiongates_dpu_easy_remaining_batch16_zcu104_cpu_gpu.csv).


## attentiongates_dpu_only_remaining

| Métrica / modo | CPU batch 1 | CPU batch 16 | GPU batch 1 | GPU batch 16 | DPU ZCU104 |
|---|---:|---:|---:|---:|---:|
| Throughput (imagens/s) — model-only | 10.780 | 15.632 | 13.550 | 186.606 | 47.335 |
| FPS por latência — model-only | 10.781 | 15.639 | 13.552 | 186.881 | 23.686 |
| Latência média (ms/imagem) — model-only | 92.752 | 63.944 | 73.790 | 5.351 | 42.219 |
| Latência P95 (ms/imagem) — model-only | 94.145 | 66.652 | 77.957 | 5.774 | 44.397 |
| Throughput (imagens/s) — end-to-end | 8.816 | 11.455 | 11.077 | 39.698 | 14.795 |
| FPS por latência — end-to-end | 8.818 | 11.458 | 11.078 | 39.722 | 5.218 |
| Latência média (ms/imagem) — end-to-end | 113.411 | 87.272 | 90.266 | 25.175 | 191.629 |
| Latência P95 (ms/imagem) — end-to-end | 116.392 | 91.131 | 93.229 | 27.996 | 234.997 |

Qualidade: validação oficial com abertura em cruz 3×3; grupo forte definido por label positivo e `difficulty=easy`; AUPRC é a média de average precision nas imagens positivas.

| Métrica | CPU batch 1 | CPU batch 16 | GPU batch 1 | GPU batch 16 | DPU ZCU104 |
|---|---:|---:|---:|---:|---:|
| Precisão | 0.634522 | 0.634522 | 0.634584 | 0.634516 | 0.664063 |
| Recall | 0.691910 | 0.691910 | 0.691957 | 0.691810 | 0.657853 |
| F1 global | 0.661975 | 0.661975 | 0.662030 | 0.661926 | 0.660944 |
| IoU | 0.494740 | 0.494740 | 0.494802 | 0.494685 | 0.493589 |
| AUPRC | 0.631302 | 0.631302 | 0.631249 | 0.631281 | 0.531214 |
| F1 fortes | 0.795546 | 0.795546 | 0.795582 | 0.795486 | 0.791310 |
| F1 fracas | 0.596744 | 0.596744 | 0.596678 | 0.596694 | 0.579119 |
| FPR global por pixel | 0.000849 | 0.000849 | 0.000849 | 0.000849 | 0.000709 |
| FPR sem pluma por pixel | 0.000998 | 0.000998 | 0.000997 | 0.000997 | 0.000911 |
| FPR por tile | 0.040000 | 0.040000 | 0.040000 | 0.040000 | 0.040000 |
| Acurácia | 0.998497 | 0.998497 | 0.998498 | 0.998497 | 0.998565 |

Fontes desta variante:

- CPU batch 1: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cpu_20261002_182145_627052/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cpu_20261002_182145_627052/validacao_oficial/metricas_globais.csv).
- CPU batch 16: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cpu_20261002_185512_024418/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cpu_20261002_185512_024418/validacao_oficial/metricas_globais.csv).
- GPU batch 1: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cuda_20261002_184234_192003/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cuda_20261002_184234_192003/validacao_oficial/metricas_globais.csv).
- GPU batch 16: [desempenho](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cuda_20261002_185829_090807/benchmark_geral.csv) e [validação oficial](../benchmark_nao_embarcado/resultados_dpu/attentiongates_dpu_only_remaining_test_cuda_20261002_185829_090807/validacao_oficial/metricas_globais.csv).
- DPU ZCU104: [desempenho](../resultados_zcu104/attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/benchmark_geral.csv) e [validação oficial](../resultados_zcu104/attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/validacao_oficial/metricas_globais.csv).

[CSV batch 1](comparativo_test_attentiongates_dpu_only_remaining_zcu104_cpu_gpu.csv) · [CSV batch 16](comparativo_test_attentiongates_dpu_only_remaining_batch16_zcu104_cpu_gpu.csv).


## Limites da comparação

Model-only reutiliza quatro entradas preparadas em todas as plataformas. End-to-end inclui leitura, preparação, inferência e saída; label, métricas e escrita dos CSVs ficam fora. Na ZCU104 há sobreposição e esperas do pipeline; CPU/GPU processam uma imagem por vez.

CPU/GPU têm hashes de CSV e checkpoint iguais entre si para cada variante. A placa registra o caminho do CSV e a geometria, mas esses CSVs de desempenho não registram os hashes do dataset/checkpoint; portanto, a identidade exata dos arquivos da placa não foi confirmada por hash neste comparativo.

Batch 1 usa 16 chamadas ao modelo por imagem; batch 16 agrupa os 16 patches em uma chamada. Todas as novas execuções concluíram a validação 342/342. São medições únicas, sem repetições independentes; ganhos entre batches incluem variação de carga e leitura. Não devem ser confundidos com o forward histórico de uma imagem inteira 512×512. Este relatório usa somente os novos benchmarks CPU/GPU para desempenho e qualidade.
