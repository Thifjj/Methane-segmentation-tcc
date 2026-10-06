# AttentionGates — resultados 512×512

Comparação dos checkpoints easy_remaining e only_remaining em imagens completas 512×512. Teste: 342 imagens; entrada MAG1C, 460, 550 e 640 nm. A pasta `benchmark_nao_embarcado/resultados_cpu/` contém os benchmarks FP32 de CPU. A quantização INT8 foi avaliada no simulador Vitis AI 3.5.

## Qualidade

As métricas `FP32 oficial` e `INT8 simulado` usam abertura cruz 3×3 e AUPRC como AP médio por imagem positiva. `CPU bruto` usa `logit > 0`, sem abertura, e AUPRC global por pixels; os protocolos diferem.

| Variante / protocolo | F1 global | F1 strong | F1 weak | IoU | AUPRC |
|---|---:|---:|---:|---:|---:|
| easy_remaining — FP32 oficial | 0.647798 | 0.832135 | 0.537973 | 0.479069 | 0.514058 |
| easy_remaining — INT8 simulado | 0.619683 | 0.815807 | 0.498174 | 0.448943 | 0.436448 |
| easy_remaining — CPU bruto | 0.649430 | 0.834465 | 0.590729 | 0.480857 | 0.663496 |
| only_remaining — FP32 oficial | 0.676553 | 0.812526 | 0.606033 | 0.511205 | 0.610146 |
| only_remaining — INT8 simulado | 0.652105 | 0.802400 | 0.572711 | 0.483796 | 0.473214 |
| only_remaining — CPU bruto | 0.676564 | 0.813724 | 0.608497 | 0.511218 | 0.669160 |

## Desempenho CPU FP32

Ryzen 9 5980HX, 4 threads. FPS é imagens completas por segundo.

| Variante | FPS modelo | P99 modelo (ms) | FPS E2E | Latência média E2E (ms) | P99 E2E (ms) | Potência E2E (W) | Energia E2E (J/imagem) |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy_remaining | 15.098 | 74.555 | 11.732 | 85.219 | 103.628 | 39.964 | 3.406 |
| only_remaining | 15.165 | 79.948 | 6.687 | 149.504 | 236.437 | 51.067 | 7.637 |

Inferência representa 77.29% do E2E no easy_remaining e 79.39% no only_remaining; leitura representa 20.46% e 17.81%, respectivamente. Os CSVs por estágio listados nas fontes contêm médias e P99 completos.

## ZCU104 e conjunto full

Os XModels INT8 512×512 estão compilados em `VitisAI/build/vitis_ai/compiled_zcu104/`. A geometria foi confirmada com XIR. FPS, latência, P99, potência, energia e estágios físicos da DPU em test/full ainda são `nan`; não há benchmark CPU de 3.425 imagens registrado.

## Fontes

- easy_remaining CPU 512×512: `benchmark_nao_embarcado/resultados_cpu/attentiongates_dpu_easy_remaining_test_cpu_20261005_195540_045944/`
- only_remaining CPU 512×512: `benchmark_nao_embarcado/resultados_cpu/attentiongates_dpu_only_remaining_test_cpu_20261005_195736_887288/`
- Métricas FP32/INT8 do simulador: `VitisAI/build/vitis_ai/evaluation/attentiongates_summary.csv`
- Modelos compilados: `VitisAI/build/vitis_ai/compiled_zcu104/attentiongates_dpu_easy_remaining/` e `.../attentiongates_dpu_only_remaining/`
