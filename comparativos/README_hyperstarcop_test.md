# HyperSTARCOP no STARCOP_test (342 imagens)

Resultados locais: CPU e GPU com PyTorch, CPU ARM com ONNX Runtime e DPU da ZCU104 com XModel. As tabelas reúnem os valores agregados; os CSVs por imagem e por amostra preservam os dados individuais. Para a DPU, as tabelas usam a execução validada `manual_1637343199461` (2 runners, 342 inferências), que registrou 49,035 FPS em `model_only`. O [CSV comparativo](comparativo_test_hyperstarcop_zcu104_cpu_gpu.csv) contém outras configurações da ZCU104.

## Qualidade da segmentação

Os números locais abaixo estão na escala de 0 a 1. O artigo publica percentuais como média ± desvio padrão de cinco treinamentos; as colunas do artigo mantêm essa unidade. `n/d` significa que o CSV não fornece a métrica (ou registra `nan`). O FPR por tile de ARM e DPU foi recalculado dos respectivos CSVs por imagem com o limiar do código oficial: mais de 640 pixels positivos por imagem 512×512 (equivalente a mais de 10 por 64×64). Os CSVs globais antigos usam mais de 10 pixels por 512×512 e, portanto, seus FPRs por tile não são comparáveis aos do artigo.

| Métrica | CPU PyTorch | GPU CUDA | ARM ONNX | DPU XModel | Artigo: só MAG1C | Artigo: MAG1C + RGB |
|---|---:|---:|---:|---:|---:|---:|
| Imagens avaliadas | 342 | 342 | 342 | 342 | n/d | n/d |
| TP (pixels) | 158319 | 158311 | 158319 | 151288 | n/d | n/d |
| FP (pixels) | 353050 | 353002 | 353050 | 327279 | n/d | n/d |
| FN (pixels) | 32333 | 32341 | 32333 | 39364 | n/d | n/d |
| TN (pixels) | 89109546 | 89109594 | 89109546 | 89135317 | n/d | n/d |
| Precisão global | 0.309598 | 0.309617 | 0.309598 | 0.316127 | n/d | n/d |
| Recall global | 0.830408 | 0.830366 | 0.830408 | 0.793530 | n/d | n/d |
| F1 global | 0.451038 | 0.451051 | 0.451038 | 0.452133 | n/d | n/d |
| F1 plumas fortes | 0.830791 | 0.830791 | 0.830791 | 0.824950 | 74.15 ± 6.10% | 81.96 ± 3.71% |
| F1 plumas fracas | 0.537245 | 0.537194 | 0.537245 | 0.536552 | 47.57 ± 4.17% | 43.42 ± 5.72% |
| IoU global | 0.291187 | 0.291198 | 0.291187 | 0.292101 | n/d | n/d |
| FPR global por pixel | 0.003946342 | 0.003945805 | 0.003946342 | 0.003658277 | n/d | n/d |
| FPR por pixel em tiles sem pluma | n/d | n/d | 0.006013716 | 0.005662957 | n/d | n/d |
| AUPRC | 0.506149 | 0.506082 | 0.506112 | n/d | 49.41 ± 5.49% | 51.99 ± 2.76% |
| Acurácia global | 0.995701¹ | 0.995702¹ | 0.995701 | 0.995910 | n/d | n/d |
| Tiles com falso positivo (>640 pixels) | n/d | n/d | 23 | 22 | n/d | n/d |
| Tiles sem falso positivo (≤640 pixels) | n/d | n/d | 153 | 154 | n/d | n/d |
| FPR por tile (`FP tiles / (FP tiles + TN tiles)`) | n/d | n/d | 0.130682 | 0.125000 | 52.11 ± 10.98% | 43.66 ± 7.36% |
| Tiles com FP / 342 imagens | n/d | n/d | 0.067251 | 0.064327 | n/d | n/d |

¹ A acurácia CPU/GPU foi calculada como `(TP + TN) / (TP + FP + FN + TN)`, pois esses dois CSVs não têm a coluna. O AUPRC da DPU aparece como `nan` por uma falha no acumulador de histograma: após o último bin ocupado, o cálculo tenta dividir `0/0`, e o `nan` se propaga. O código foi corrigido, mas o CSV não guarda os scores necessários para recuperar a AUPRC sem repetir a validação. CPU/GPU também exigem nova validação para obter o FPR por tile com o limiar corrigido, pois seus CSVs só guardam os totais. `FP tiles / 342` usa todas as imagens no denominador e não é o FPR do artigo. A diferença entre os valores locais corrigidos e a média publicada permanece sem causa comprovada; o artigo agrega cinco treinamentos, enquanto os benchmarks locais avaliam modelos exportados. Fonte: [artigo, Tabela 2](https://www.nature.com/articles/s41598-023-44918-6#Tab2) e [regra de classificação do código oficial](https://github.com/spaceml-org/STARCOP/blob/main/starcop/models/model_module.py). A variante local usa MAG1C + RGB.

### Métricas por grupo disponíveis nos CSVs embarcados

Os CSVs CPU/GPU registram somente o F1 dos grupos forte e fraco. ARM e DPU registram todas as contagens e métricas abaixo.

| Grupo | Métrica | ARM ONNX | DPU XModel |
|---|---|---:|---:|
| Forte | TP / FP / FN / TN | 125287 / 31646 / 19389 / 14765886 | 119697 / 25819 / 24979 / 14771713 |
| Forte | Precisão / recall / F1 | 0.798347 / 0.865983 / 0.830791 | 0.822569 / 0.827345 / 0.824950 |
| Forte | IoU / FPR / acurácia | 0.710558 / 0.002138600 / 0.996585 | 0.702056 / 0.001744818 / 0.996600 |
| Fraca | TP / FP / FN / TN | 32913 / 43949 / 12750 / 28484084 | 31476 / 40188 / 14187 / 28487845 |
| Fraca | Precisão / recall / F1 | 0.428209 / 0.720781 / 0.537245 | 0.439216 / 0.689311 / 0.536552 |
| Fraca | IoU / FPR / acurácia | 0.367283 / 0.001540550 / 0.998016 | 0.366635 / 0.001408720 / 0.998097 |
| Sem pluma | TP / FP / FN / TN | 119 / 277455 / 194 / 45859576 | 115 / 261272 / 198 / 45875759 |
| Sem pluma | Precisão / recall / F1 | 0.000429 / 0.380192 / 0.000856 | 0.000440 / 0.367412 / 0.000879 |
| Sem pluma | IoU / FPR / acurácia | 0.000428 / 0.006013716 / 0.993982 | 0.000440 / 0.005662957 / 0.994333 |

## Desempenho: comparações duas a duas

CPU/GPU fornecem `model_fps` e `e2e_fps` em seus CSVs; ARM/DPU fornecem `throughput_fps`. Os fatores comparam esses campos como reportados, sem recalcular FPS a partir da latência. Os quatro benchmarks usaram 342 imagens. ARM e DPU rodaram na ZCU104; o ONNX usou 4 threads e a DPU usou 2 runners, 4 núcleos CPU, 2 workers de pré-processamento, 1 de pós-processamento e 2 slots por runner. Ambos usaram 10 warm-ups.

| Modo | CPU × GPU (FPS) | GPU / CPU | ARM × DPU (throughput, FPS) | DPU / ARM |
|---|---:|---:|---:|---:|
| `model_only` | 5.637294 × 165.920443 | 29.43× | 0.353271 × 49.034568 | 138.80× |
| `end_to_end` | 5.010424 × 45.512032 | 9.08× | 0.329222 × 14.632430 | 44.45× |

### Dados completos dos dois modos

| Modo e campo | CPU PyTorch | GPU CUDA | ARM ONNX | DPU XModel |
|---|---:|---:|---:|---:|
| `model_only`: inferências | 342 | 342 | 342 | 342 |
| `model_only`: duração (s) | n/d | n/d | 968.095161 | 6.974671 |
| `model_only`: FPS informado pelo benchmark | 5.637294 | 165.920443 | 0.353312 | 24.536744 |
| `model_only`: throughput medido (FPS) | n/d | n/d | 0.353271 | 49.034568 |
| `model_only`: latência média (ms) | 177.390069 | 6.026985 | 2830.363382 | 40.755203 |
| `model_only`: latência mediana (ms) | 176.824412 | 5.901922 | 2842.640850 | 40.452405 |
| `model_only`: latência mínima / máxima (ms) | n/d | n/d | 2742.564700 / 2888.101000 | 38.944750 / 43.412690 |
| `model_only`: latência P95 / P99 (ms) | 186.045058 / 211.431945 | 6.621629 / 6.992191 | 2871.550518 / 2884.052937 | 42.507895 / 43.349221 |
| `model_only`: desvio da latência (ms) | n/d | n/d | 35.058571 | 1.050766 |
| `end_to_end`: inferências | 342 | 342 | 342 | 342 |
| `end_to_end`: duração (s) | n/d | n/d | 1038.814161 | 23.372741 |
| `end_to_end`: FPS informado pelo benchmark | 5.010424 | 45.512032 | 0.330032 | 5.418988 |
| `end_to_end`: throughput medido (FPS) | n/d | n/d | 0.329222 | 14.632430 |
| `end_to_end`: latência média (ms) | 199.583903 | 21.972211 | 3030.009249 | 184.536298 |
| `end_to_end`: latência mediana (ms) | 198.737548 | 21.842740 | 3032.877455 | 183.833970 |
| `end_to_end`: latência mínima / máxima (ms) | n/d | n/d | 2919.632940 / 3269.345660 | 138.398820 / 242.941760 |
| `end_to_end`: latência P95 / P99 (ms) | 209.767139 / 234.423149 | 23.675849 / 24.390649 | 3088.776454 / 3134.344446 | 218.395371 / 230.012281 |
| `end_to_end`: desvio da latência (ms) | n/d | n/d | 38.705118 | 19.001470 |
| `end_to_end`: inferência média (ms) | n/d | n/d | 2821.850917 | 39.751581 |
| `end_to_end`: leitura média (ms) | 20.988380 | 14.299878 | 179.246061 | 123.243719 |
| `end_to_end`: pré-processamento médio (ms) | 0.959790 | 1.485289 | 26.441060 | 13.076590 |
| `end_to_end`: pós-processamento médio (ms) | 0.233450 | 0.144408 | 2.471210 | 1.674550 |

### Distribuição das etapas E2E no ARM e na DPU

Cada célula apresenta `média / mediana / mínima / máxima / P95 / P99 / desvio`, em ms, conforme `benchmark_estagios.csv`. A latência total já consta na tabela anterior.

| Etapa | ARM ONNX | DPU XModel |
|---|---:|---:|
| Espera por slot | n/d | 0.001495 / 0.001415 / 0.000480 / 0.007010 / 0.002299 / 0.003472 / 0.000575 |
| Leitura | 179.246 / 181.346 / 114.910 / 229.416 / 216.906 / 226.383 / 23.9173 | 123.243719 / 126.973625 / 85.156440 / 152.484000 / 137.299993 / 142.852792 / 12.199863 |
| Pré-processamento | 26.4411 / 26.4485 / 26.2283 / 26.8792 / 26.5808 / 26.6842 / 0.0900222 | 13.076590 / 12.828140 / 12.648270 / 14.995070 / 14.023093 / 14.677905 / 0.496774 |
| Espera pelo runner | n/d | 6.607999 / 0.059410 / 0.031350 / 40.488450 / 34.466525 / 39.075977 / 11.798935 |
| Sincronização da entrada | n/d | 0.000362 / 0.000300 / 0.000150 / 0.001130 / 0.000630 / 0.000861 / 0.000139 |
| Inferência | 2821.85 / 2822.85 / 2742.39 / 3018.22 / 2862.00 / 2939.87 / 30.1057 | 39.751581 / 39.958460 / 38.509590 / 44.418580 / 41.829464 / 43.866630 / 1.084511 |
| Sincronização da saída | n/d | 0.000604 / 0.000580 / 0.000280 / 0.001540 / 0.000880 / 0.001108 / 0.000160 |
| Espera pelo pós-processamento | n/d | 0.179398 / 0.048945 / 0.027140 / 1.215730 / 0.998697 / 1.133476 / 0.327878 |
| Pós-processamento | 2.47121 / 2.47356 / 2.44043 / 2.65435 / 2.56156 / 2.63599 / 0.0396509 | 1.674550 / 1.660350 / 1.614430 / 2.679490 / 1.753806 / 1.920804 / 0.075454 |

Os CSVs ARM/DPU também repetem `inferencia_media_ms = latencia_media_ms` em `model_only` e registram leitura, pré e pós-processamento iguais a zero nesse modo. A DPU preparou 4 entradas no `model_only`; em E2E, `entradas_preparadas = 0`. Há outra execução DPU com 4 runners e 51.235044 FPS `model_only`, e buscas E2E com 80 inferências; elas não foram misturadas às tabelas da execução de 2 runners.

## Potência e energia medidas no trilho INA226

Os CSVs históricos de CPU/GPU não têm medições de potência. O benchmark não embarcado agora registra energia e potência do pacote CPU via RAPL e, em CUDA, da GPU via NVML; é preciso executar novamente para preencher essas colunas. Essas fontes medem domínios diferentes do trilho `ina226:power1` usado por ARM e DPU. A energia embarcada é a estimativa `potência média × duração`, e não uma leitura de contador.

| Modo e campo | ARM ONNX | DPU XModel |
|---|---:|---:|
| `model_only`: amostras do sensor | 4827 | 36 |
| `model_only`: potência média / mínima / máxima (W) | 15.583 / 14.537 / 16.437 | 23.318889 / 14.687 / 26.550 |
| `model_only`: energia total (J) | 15085.8 | 162.641588 |
| `model_only`: energia por inferência (J) | 44.110700 | 0.475560 |
| `end_to_end`: amostras do sensor | 5180 | 118 |
| `end_to_end`: potência média / mínima / máxima (W) | 15.536300 / 14.525 / 16.450 | 17.960051 / 14.650 / 26.575 |
| `end_to_end`: energia total (J) | 16139.3 | 419.775622 |
| `end_to_end`: energia por inferência (J) | 47.190900 | 1.227414 |

## Arquivos usados

- [CPU PyTorch](../benchmark_nao_embarcado/resultado_test_cpu_hyperstarcop_20260922_1811.csv) e [GPU CUDA](../benchmark_nao_embarcado/resultado_test_cuda_hyperstarcop_20260922_2129.csv).
- [ARM: qualidade](../resultados_arm/hyperstarcop_test/metricas_globais.csv), [grupos](../resultados_arm/hyperstarcop_test/metricas_grupos.csv), [desempenho](../resultados_arm/hyperstarcop_test/benchmark_geral.csv), [etapas](../resultados_arm/hyperstarcop_test/benchmark_estagios.csv) e [potência](../resultados_arm/hyperstarcop_test/benchmark_power_rails.csv).
- [DPU: qualidade](../resultados_zcu104/methane_hyperstarcop_STARCOP_test_manual_1637343199461/metricas_globais.csv), [grupos](../resultados_zcu104/methane_hyperstarcop_STARCOP_test_manual_1637343199461/metricas_grupos.csv), [desempenho](../resultados_zcu104/methane_hyperstarcop_STARCOP_test_manual_1637343199461/benchmark_geral.csv), [etapas](../resultados_zcu104/methane_hyperstarcop_STARCOP_test_manual_1637343199461/benchmark_estagios.csv) e [potência](../resultados_zcu104/methane_hyperstarcop_STARCOP_test_manual_1637343199461/benchmark_power_rails.csv).
