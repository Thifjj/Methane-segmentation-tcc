# HyperSTARCOP no STARCOP_test (342 imagens)

Resultados locais: CPU e GPU com PyTorch, CPU ARM com ONNX Runtime e DPU da ZCU104 com XModel. As tabelas reúnem os valores agregados; os CSVs por imagem e por amostra preservam os dados individuais. Para a DPU, as tabelas usam a execução validada `manual_1637346267285` (2 runners, 342 inferências), que registrou 48,960 FPS em `model_only`. O [CSV comparativo](comparativo_test_hyperstarcop_zcu104_cpu_gpu.csv) contém outras configurações da ZCU104.

## Qualidade da segmentação

Os números locais abaixo estão na escala de 0 a 1. O artigo publica percentuais como média ± desvio padrão de cinco treinamentos; as colunas do artigo mantêm essa unidade. `n/d` significa que o CSV não fornece a métrica. Há uma divergência entre as fontes: o [texto do artigo](https://www.nature.com/articles/s41598-023-44918-6) descreve **mais de 10 pixels ativos por tile**; a [função `pred_classification` do código oficial](https://github.com/spaceml-org/STARCOP/blob/main/starcop/models/model_module.py) escala 10 pixels por 64×64, resultando em **mais de 640 pixels** para 512×512. Os benchmarks ARM e DPU gravaram o limiar de 640. A tabela mostra também o recálculo com 10 a partir de `metricas_por_imagem.csv` para permitir a comparação com o texto do artigo.

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
| AUPRC | 0.506149 | 0.506082 | 0.506112 | 0.471906 | 49.41 ± 5.49% | 51.99 ± 2.76% |
| Acurácia global | 0.995701¹ | 0.995702¹ | 0.995701 | 0.995910 | n/d | n/d |
| Tiles com falso positivo (>10 pixels; texto do artigo) | 73 | 73 | 73 | 75 | n/d | n/d |
| FPR por tile (>10; `FP tiles / 176 sem pluma`) | 0.414773 | 0.414773 | 0.414773 | 0.426136 | 52.11 ± 10.98% | 43.66 ± 7.36% |
| Tiles com falso positivo (>640 pixels; código oficial) | n/d | n/d | 23 | 22 | n/d | n/d |
| FPR por tile (>640; `FP tiles / 176 sem pluma`) | n/d | n/d | 0.130682 | 0.125000 | n/d | n/d |

¹ A acurácia CPU/GPU foi calculada como `(TP + TN) / (TP + FP + FN + TN)`, pois esses dois CSVs não têm a coluna. O FPR por tile usa as 176 imagens marcadas `sem_pluma` como denominador; o FPR por pixel usa `FP / (FP + TN)` e mede outra coisa. Os CSVs CPU/GPU já registravam `73/176` com limiar de 10; as contagens ARM/DPU com esse limiar foram recalculadas somando `tp + fp` por imagem. Com o mesmo limiar do texto do artigo, o FPR da DPU fica 42,61%, próximo dos 43,66% da Tabela 2 (média de cinco treinamentos). O código oficial publicado usa 640 para imagens 512×512; não é possível afirmar apenas com essas fontes qual limiar gerou a tabela do artigo. A variante local usa MAG1C + RGB.

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
| `model_only` | 5.637294 × 165.920443 | 29.43× | 0.358934 × 48.960472 | 136.41× |
| `end_to_end` | 5.010424 × 45.512032 | 9.08× | 0.335090 × 7.005313 | 20.91× |

### Dados completos dos dois modos

| Modo e campo | CPU PyTorch | GPU CUDA | ARM ONNX | DPU XModel |
|---|---:|---:|---:|---:|
| `model_only`: inferências | 342 | 342 | 342 | 342 |
| `model_only`: duração (s) | n/d | n/d | 952.821349 | 6.985227 |
| `model_only`: FPS informado pelo benchmark | 5.637294 | 165.920443 | 0.358974 | 24.494160 |
| `model_only`: throughput medido (FPS) | n/d | n/d | 0.358934 | 48.960472 |
| `model_only`: latência média (ms) | 177.390069 | 6.026985 | 2785.716991 | 40.826058 |
| `model_only`: latência mediana (ms) | 176.824412 | 5.901922 | 2783.730205 | 40.613875 |
| `model_only`: latência mínima / máxima (ms) | n/d | n/d | 2747.107430 / 2865.365150 | 39.073240 / 43.612640 |
| `model_only`: latência P95 / P99 (ms) | 186.045058 / 211.431945 | 6.621629 / 6.992191 | 2806.923476 / 2851.620326 | 42.672430 / 43.271072 |
| `model_only`: desvio da latência (ms) | n/d | n/d | 13.054801 | 1.065820 |
| `end_to_end`: inferências | 342 | 342 | 342 | 342 |
| `end_to_end`: duração (s) | n/d | n/d | 1020.621338 | 48.820085 |
| `end_to_end`: FPS informado pelo benchmark | 5.010424 | 45.512032 | 0.336060 | 3.043083 |
| `end_to_end`: throughput medido (FPS) | n/d | n/d | 0.335090 | 7.005313 |
| `end_to_end`: latência média (ms) | 199.583903 | 21.972211 | 2975.661221 | 328.614150 |
| `end_to_end`: latência mediana (ms) | 198.737548 | 21.842740 | 2985.796695 | 339.908480 |
| `end_to_end`: latência mínima / máxima (ms) | n/d | n/d | 2894.514610 / 3051.152550 | 177.090100 / 401.573420 |
| `end_to_end`: latência P95 / P99 (ms) | 209.767139 / 234.423149 | 23.675849 / 24.390649 | 3017.504853 / 3026.175611 | 374.997465 / 393.055672 |
| `end_to_end`: desvio da latência (ms) | n/d | n/d | 30.523152 | 37.852112 |
| `end_to_end`: inferência média (ms) | n/d | n/d | 2825.460576 | 39.380454 |
| `end_to_end`: leitura média (ms) | 20.988380 | 14.299878 | 121.386648 | 271.988945 |
| `end_to_end`: pré-processamento médio (ms) | 0.959790 | 1.485289 | 26.342757 | 13.006024 |
| `end_to_end`: pós-processamento médio (ms) | 0.233450 | 0.144408 | 2.471241 | 1.657022 |

### Distribuição das etapas E2E no ARM e na DPU

Cada célula apresenta `média / mediana / mínima / máxima / P95 / P99 / desvio`, em ms, conforme `benchmark_estagios.csv`. A latência total já consta na tabela anterior.

| Etapa | ARM ONNX | DPU XModel |
|---|---:|---:|
| Espera por slot | n/d | 0.001673 / 0.001610 / 0.000650 / 0.005180 / 0.002337 / 0.003095 / 0.000418 |
| Leitura | 121.387 / 123.876 / 78.2474 / 156.844 / 146.289 / 151.713 / 14.3363 | 271.988945 / 285.357245 / 123.725320 / 338.387770 / 310.511829 / 329.533994 / 37.288024 |
| Pré-processamento | 26.3428 / 26.2946 / 26.2149 / 26.9361 / 26.6475 / 26.7322 / 0.126065 | 13.006024 / 12.767960 / 12.627820 / 16.366410 / 14.000604 / 15.733068 / 0.591568 |
| Espera pelo runner | n/d | 2.500287 / 0.048435 / 0.032600 / 39.982230 / 21.118126 / 31.437480 / 6.983578 |
| Sincronização da entrada | n/d | 0.000365 / 0.000320 / 0.000180 / 0.000850 / 0.000620 / 0.000772 / 0.000124 |
| Inferência | 2825.46 / 2832.24 / 2771.47 / 2883.39 / 2865.07 / 2875.53 / 27.8079 | 39.380454 / 39.764675 / 38.299380 / 42.603800 / 40.227196 / 40.725123 / 0.701089 |
| Sincronização da saída | n/d | 0.000474 / 0.000450 / 0.000260 / 0.000980 / 0.000700 / 0.000750 / 0.000118 |
| Espera pelo pós-processamento | n/d | 0.078905 / 0.045410 / 0.027480 / 1.178820 / 0.081490 / 0.968397 / 0.160044 |
| Pós-processamento | 2.47124 / 2.47174 / 2.43984 / 2.81087 / 2.53867 / 2.68001 / 0.0462274 | 1.657022 / 1.648045 / 1.607070 / 1.896060 / 1.751556 / 1.832431 / 0.044226 |

Os CSVs ARM/DPU também repetem `inferencia_media_ms = latencia_media_ms` em `model_only` e registram leitura, pré e pós-processamento iguais a zero nesse modo. A DPU preparou 4 entradas no `model_only`; em E2E, `entradas_preparadas = 0`. As buscas E2E com 80 inferências não foram misturadas às tabelas da execução validada de 2 runners.

## Potência e energia medidas no trilho INA226

Os CSVs históricos de CPU/GPU não têm medições de potência. O benchmark não embarcado agora registra energia e potência do pacote CPU via RAPL e, em CUDA, da GPU via NVML; é preciso executar novamente para preencher essas colunas. Essas fontes medem domínios diferentes do trilho `ina226:power1` usado por ARM e DPU. A energia embarcada é a estimativa `potência média × duração`, e não uma leitura de contador.

| Modo e campo | ARM ONNX | DPU XModel |
|---|---:|---:|
| `model_only`: amostras do sensor | 4751 | 36 |
| `model_only`: potência média / mínima / máxima (W) | 15.595700 / 14.537 / 16.487 | 22.936556 / 14.700 / 26.437 |
| `model_only`: energia total (J) | 14859.9 | 160.217043 |
| `model_only`: energia por inferência (J) | 43.449900 | 0.468471 |
| `end_to_end`: amostras do sensor | 5089 | 245 |
| `end_to_end`: potência média / mínima / máxima (W) | 15.573800 / 14.537 / 16.475 | 16.157522 / 14.537 / 20.937 |
| `end_to_end`: energia total (J) | 15894.9 | 788.811619 |
| `end_to_end`: energia por inferência (J) | 46.476400 | 2.306467 |

## Arquivos usados

- [CPU PyTorch](../benchmark_nao_embarcado/resultado_test_cpu_hyperstarcop_20260922_1811.csv) e [GPU CUDA](../benchmark_nao_embarcado/resultado_test_cuda_hyperstarcop_20260922_2129.csv).
- [ARM: qualidade](../resultados_arm/hyperstarcop_test/metricas_globais.csv), [grupos](../resultados_arm/hyperstarcop_test/metricas_grupos.csv), [desempenho](../resultados_arm/hyperstarcop_test/benchmark_geral.csv), [etapas](../resultados_arm/hyperstarcop_test/benchmark_estagios.csv) e [potência](../resultados_arm/hyperstarcop_test/benchmark_power_rails.csv).
- [DPU: qualidade](../resultados_zcu104/methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/metricas_globais.csv), [grupos](../resultados_zcu104/methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/metricas_grupos.csv), [desempenho](../resultados_zcu104/methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/benchmark_geral.csv), [etapas](../resultados_zcu104/methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/benchmark_estagios.csv) e [potência](../resultados_zcu104/methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/benchmark_power_rails.csv).
