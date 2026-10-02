# AttentionGates DPU: CPU × GPU × ZCU104

Comparativo das variantes **easy_remaining** e **only_remaining** de `UnetMobilenetV3AttentionGates_dpu_*_mag1c_rgb`. Ambas têm 1.690.777 parâmetros; o histórico registra checkpoints de 6,64 MB. Valores de qualidade na escala de 0 a 1; `n/d` indica dado não registrado.

## Qualidade no STARCOP_test — 342 imagens

CPU: reavaliação PyTorch FP32 com imagem 512×512. GPU: primeiras duas linhas do histórico CUDA, identificadas como TEST pelo usuário. ZCU104: XModel INT8, 16 patches de 128×128 por imagem, com métricas de `validacao_oficial`.

### easy_remaining

| Métrica | CPU FP32 | GPU CUDA FP32 (histórico) | ZCU104 INT8 |
|---|---:|---:|---:|
| F1 global | 0.647798 | 0.647800 | 0.623353 |
| IoU | 0.479069 | 0.479100 | 0.452806 |
| AUPRC | 0.514058 | 0.663400 | 0.560959 |
| F1 plumas fortes | 0.832135 | 0.832200 | 0.741348 |
| F1 plumas fracas | 0.537973 | 0.537800 | 0.553665 |
| FPR por pixel em imagens sem pluma | 0.001368 | 0.001367 | 0.000831 |
| Acurácia por pixel | 0.998350 | n/d | 0.998508 |

Variação de F1 ZCU104 − GPU: **-2.44 pontos percentuais**.

### only_remaining

| Métrica | CPU FP32 | GPU CUDA FP32 (histórico) | ZCU104 INT8 |
|---|---:|---:|---:|
| F1 global | 0.676553 | 0.676500 | 0.660944 |
| IoU | 0.511205 | 0.511200 | 0.493589 |
| AUPRC | 0.610146 | 0.669100 | 0.531214 |
| F1 plumas fortes | 0.812526 | 0.812500 | 0.791310 |
| F1 plumas fracas | 0.606033 | 0.606000 | 0.579119 |
| FPR por pixel em imagens sem pluma | 0.000993 | 0.000993 | 0.000911 |
| Acurácia por pixel | 0.998508 | n/d | 0.998565 |

Variação de F1 ZCU104 − GPU: **-1.56 pontos percentuais**.

**AUPRC:** CPU e GPU FP32 têm F1/IoU praticamente iguais, mas os valores de AUPRC registrados divergem. A reavaliação CPU/ZCU104 usa média de average precision nas imagens positivas. Sem confirmar a equivalência do cálculo e dos artefatos do histórico CUDA, a diferença de AUPRC não pode ser atribuída apenas ao hardware.

**Protocolo da ZCU104:** abertura morfológica em cruz 3×3 e grupo forte definido por rótulo positivo com `difficulty=easy`. O FPR da tabela é por pixel em imagens sem pluma, não FPR por tile. Os números exibidos no terminal (F1 0,628873 / 0,664310) pertencem ao relatório bruto, com outro protocolo; por isso a tabela usa `validacao_oficial`.

## Referência CPU com os mesmos patches da DPU

As linhas INT8 abaixo são simulação do Vitis AI na CPU. Elas permitem conferir a quantização sem misturar a mudança de tamanho da entrada com a mudança de precisão.

| Variante | CPU FP32 patches128: F1 / IoU / AUPRC | CPU INT8 patches128: F1 / IoU / AUPRC | ZCU104 INT8: F1 / IoU / AUPRC |
|---|---:|---:|---:|
| easy_remaining | 0.645602 / 0.476671 / 0.630064 | 0.623353 / 0.452806 / 0.560959 | 0.623353 / 0.452806 / 0.560959 |
| only_remaining | 0.661975 / 0.494740 / 0.631302 | 0.660944 / 0.493589 / 0.531214 | 0.660944 / 0.493589 / 0.531214 |

A ZCU104 reproduz as métricas agregadas da simulação INT8 na CPU, na precisão apresentada. A perda de F1 com a quantização, comparando patches128 FP32 → INT8, é de **2,22 p.p.** em easy_remaining e **0,10 p.p.** em only_remaining.

## Desempenho no TEST

Os tempos GPU são de forward do histórico, com imagem 512×512; o FPS GPU abaixo é calculado como `1000 / ms`. Na ZCU104, throughput é o número de imagens dividido pela duração medida, com **2 runners concorrentes**. Latência é por imagem completa (16 patches), e não por patch. São protocolos distintos; estes dados não estabelecem um speedup CPU/GPU/DPU controlado.

| Variante | Plataforma / modo | Latência média (ms/imagem)q | Imagens/s |
|---|---|---:|---:|
| easy_remaining | CPU FP32, TEST | n/d | n/d |
| easy_remaining | GPU CUDA, forward | 8.500 | 117.647 (calculado) |
| easy_remaining | ZCU104 `model_only` | 42.223 | 47.358 (medido) |
| easy_remaining | ZCU104 `end_to_end` | 192.414 | 14.787 (medido) |
| only_remaining | CPU FP32, TEST | n/d | n/d |
| only_remaining | GPU CUDA, forward | 6.630 | 150.830 (calculado) |
| only_remaining | ZCU104 `model_only` | 42.219 | 47.335 (medido) |
| only_remaining | ZCU104 `end_to_end` | 191.629 | 14.795 (medido) |

Não há tempo de CPU para estas variantes no TEST nem tempo end-to-end GPU nos arquivos usados. Na ZCU104, end-to-end inclui leitura, pré-processamento, inferência, pós-processamento e esperas do pipeline.

## Potência e energia na ZCU104

| Variante | Modo | Potência média (W) | Energia (J/imagem) |
|---|---|---:|---:|
| easy_remaining | `model_only` | 17.210 | 0.3634 |
| easy_remaining | `end_to_end` | 15.906 | 1.0757 |
| only_remaining | `model_only` | 17.145 | 0.3622 |
| only_remaining | `end_to_end` | 15.868 | 1.0725 |

Medição do trilho INA226 `power1`, com integração trapezoidal no intervalo do benchmark. O sensor não tem rótulo de domínio; não representa uma medição isolada da DPU. Não há potência/energia CPU ou GPU disponível para estes modelos.

## Histórico FULL remaining+easy — separado do TEST

As demais duas linhas do histórico usam FULL remaining+easy, conforme informado pelo usuário. A medição CPU disponível pertence a esse conjunto, portanto não preenche a lacuna de tempo CPU no TEST.

| Variante | Plataforma | F1 | IoU | AUPRC | Inferência (ms/imagem) | Imagens/s (calculado) |
|---|---|---:|---:|---:|---:|---:|
| only_remaining | CUDA | 0.648000 | 0.479300 | 0.665300 | 6.280 | 159.236 |
| easy_remaining | CPU | 0.623400 | 0.452800 | 0.531500 | 66.070 | 15.135 |

Imagens/s calculado como `1000 / inferência média em ms`, para execução sequencial do modelo. Não é throughput end-to-end medido. As linhas usam variantes diferentes e o dataset FULL; não permitem calcular um speedup CPU/GPU/ZCU104 no TEST.

## Resumo

- **only_remaining** tem maior F1/IoU na GPU histórica e na ZCU104; easy_remaining tem maior AUPRC na validação oficial INT8.
- Na ZCU104, ambas atingem aproximadamente **47,35 imagens/s em model_only** e **14,79 imagens/s end-to-end**.
- Para completar uma comparação de velocidade no mesmo conjunto, falta registrar o tempo CPU no TEST com estas variantes.

## Fontes locais

- [Histórico CPU/GPU](../Resultados_testes/historico_testes.csv).
- [Reavaliação CPU FP32/INT8 no TEST](../VitisAI/build/vitis_ai/official/evaluation/summary.csv).
- [Código da reavaliação CPU](../VitisAI/evaluate_quantized.py) e [carregamento dos modelos](../VitisAI/common.py).
- [easy_remaining — validação oficial](../resultados_zcu104/attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/validacao_oficial/metricas_globais.csv).
- [easy_remaining — desempenho](../resultados_zcu104/attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/benchmark_geral.csv).
- [easy_remaining — potência](../resultados_zcu104/attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/benchmark_power_rails.csv).
- [only_remaining — validação oficial](../resultados_zcu104/attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/validacao_oficial/metricas_globais.csv).
- [only_remaining — desempenho](../resultados_zcu104/attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/benchmark_geral.csv).
- [only_remaining — potência](../resultados_zcu104/attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/benchmark_power_rails.csv).
