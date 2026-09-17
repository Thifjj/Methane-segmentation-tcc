# Contexto para continuar este projeto

Este arquivo é um ponto de entrada para outras conversas/modelos. Leia os arquivos citados antes de alterar código ou interpretar resultados. O objetivo do projeto é comparar cinco redes de segmentação de plumas de metano no STARCOP, em PyTorch/CPU e na FPGA ZCU104 com Vitis AI.

## Projeto e fontes de verdade

- Arquiteturas: `Modelos/UNet_baseline.py`, `UNet_depth_reduced.py`, `UNet_SkipConnections.py`, `UNet_MobileNet_v2.py` e `UNet_MobileNet_v3.py`. Pesos em `Modelos_treinados/`. O registro modelo → checkpoint e a ordem dos canais estão em `VitisAI/common.py`.
- Dados: `Utils/DataLoader.py` lê os TIFFs conforme CSV e normaliza `mag1c`, `TOA_AVIRIS_640nm`, `TOA_AVIRIS_550nm`, `TOA_AVIRIS_460nm`. Máscara de referência: `labelbinary.tif`. A avaliação PyTorch original está em `Testes/Teste_Unet.py`.
- Conversão para a placa: `VitisAI/inspect_model.py`, `quantize_model.py` e `compile_xmodel.py`. O fluxo, ambiente Docker, target e comandos estão em [VitisAI/README.md](VitisAI/README.md). Os cinco modelos compilados ficam em `build/vitis_ai/compiled_zcu104/<modelo>/methane_<modelo>.xmodel`.
- Benchmark da placa: `Benchmark_ZCU104/benchmark_zcu104.cpp` mede um modelo; `sweep_zcu104.cpp` testa configurações e executa as medições finais; `run_all_models.sh` percorre os `.xmodel`; `build_zcu104.sh` compila na placa. Consulte [Benchmark_ZCU104/README.md](Benchmark_ZCU104/README.md) e [GUIA_PROJETO_ZCU104.md](GUIA_PROJETO_ZCU104.md).
- Benchmark manual de CPU/GPU: `benchmark_manual/`. A implementação modular atual da placa é `benchmark_best_zcu104/codigos_c/`; consulte [o README](benchmark_best_zcu104/codigos_c/README.md) e [a referência de comandos](benchmark_best_zcu104/codigos_c/comandos.md). `Benchmark_ZCU104/` é a implementação anterior e os resultados em `results_all_models/` descritos abaixo pertencem a ela.

## Benchmark modular atual na ZCU104

Layout confirmado na placa (`root@192.168.2.100`, prompt `root@xilinx-zcu104-20222`):

| Conteúdo | Caminho na placa |
| --- | --- |
| Código e executáveis | `/home/root/thiago/benchmark/codigos_c/` |
| XModels | `/home/root/thiago/benchmark/modelos/<modelo>/methane_<modelo>.xmodel` |
| Modelos | `baseline`, `depth_reduced`, `mobilenet_v2`, `mobilenet_v3`, `skip_connections` |
| TIFFs e CSV | `/home/root/thiago/dataset_starcop/` e `/home/root/thiago/dataset_starcop/train.csv` |
| Saídas | `/home/root/resultados/` |

No computador, copie a versão local do código a partir da raiz do repositório. Na placa, compile e confira a quantização NEON contra a referência escalar:

```bash
scp -O -r benchmark_best_zcu104/codigos_c root@192.168.2.100:/home/root/thiago/benchmark/
ssh root@192.168.2.100
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
./self_test_support
```

Comando para medir a MobileNetV3 com leitura OpenCV e pré-processamento NEON, mantendo a configuração de desempenho das execuções anteriores:

```bash
./benchmark_vitis \
  --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel \
  --dataset /home/root/thiago/dataset_starcop \
  --csv /home/root/thiago/dataset_starcop/train.csv \
  --out /home/root/resultados/mobilenet_v3_neon_opencv_p2 \
  --mode all --runners 2 --cpu-cores 4 \
  --pre-workers 2 --post-workers 1 --slots 2 \
  --iterations 0 --warmup 20 --power --no-validate
```

Para varrer os cinco modelos sequencialmente:

```bash
./run_all_models.sh \
  --models-dir /home/root/thiago/benchmark/modelos \
  --dataset /home/root/thiago/dataset_starcop \
  --csv /home/root/thiago/dataset_starcop/train.csv \
  --out /home/root/resultados/todos --max-runners 2 --resume
```

O programa principal aceita `--runners`, `--cpu-cores` (ou `--threads`), `--pre-workers`, `--post-workers` e `--slots` de 1 a 4; `--mode model_only|end_to_end|all`, `--samples`, `--iterations`, `--warmup`, `--pin|--no-pin`, `--power|--no-power` e `--validate|--no-validate`. `--samples 0` usa todo o CSV; `--iterations 0` faz uma inferência por amostra. O terminal mostra `concluídas/total` aproximadamente uma vez por segundo. `sweep_vitis` busca configurações; `run_all_models.sh` chama o sweep para cada XModel. O limite de dois runners é prudente porque a MobileNetV2 já travou a placa com três; a causa continua desconhecida.

`model_only` mede `execute_async + wait` com entrada preparada. `end_to_end` inclui TIFF, pré-processamento, filas, runner e pós-processamento; validação ocorre fora das regiões temporizadas. Nas MobileNets, o runner também executa subgrafos de CPU, então `model_only` não equivale a tempo puro da DPU. As saídas incluem `benchmark_geral.csv` (FPS, latências, estágios e duração), `benchmark_estagios.csv`, `benchmark_samples.csv`, `benchmark_power_rails.csv`, `metricas_globais.csv`, `metricas_por_imagem.csv`, `metricas_grupos.csv` e `config.txt`. A potência por trilho vem dos sensores da placa; não some trilhos que possam medir partes sobrepostas.

### Referência MobileNetV3 e gargalo

`resultados/baseline_unico/` foi nomeado por engano: [config.txt](resultados/baseline_unico/config.txt) identifica **MobileNetV3**, 3.425 imagens de `train.csv`, dois runners, quatro núcleos, dois workers de pré, um de pós, dois slots, warm-up 20, sem afinidade, potência e validação habilitadas. O grafo registrou seis subgrafos DPU e seis CPU. Resultados em [benchmark_geral.csv](resultados/baseline_unico/benchmark_geral.csv):

| Modo | Duração | Throughput | Latência média | P99 |
| --- | ---: | ---: | ---: | ---: |
| `model_only` | 361,646 s | 9,471 FPS | 211,135 ms | 214,187 ms |
| `end_to_end` | 561,446 s | 6,100 FPS | 580,675 ms | 653,965 ms |

No E2E, médias por imagem: leitura **282,846 ms**, pré **43,680 ms**, inferência **246,262 ms**, pós **1,662 ms**. Dois workers de leitura/pré ficaram praticamente saturados: `(leitura + pré) × 3425 / (2 × 561,446 s) ≈ 99,6%`; capacidade calculada ≈ 6,125 FPS, próxima dos 6,100 FPS medidos. Portanto, o gargalo principal desta configuração é a alimentação dos runners, especialmente a leitura dos TIFFs; o pós-processamento é pequeno. Há duas sequências de pausas de 26–30 s nos índices 284–287 e 2495–2498 de `benchmark_samples.csv`; aparecem também na inferência. A causa dessas pausas não foi confirmada. Não atribua automaticamente ao swap ou ao armazenamento.

Qualidade da placa em [metricas_globais.csv](resultados/baseline_unico/metricas_globais.csv): precisão 0,798848, recall 0,422202, F1 0,552435, IoU 0,381631, AUPRC 0,521139. Potência média no trilho `ina226:power1`: 15,452 W (`model_only`) e 15,660 W (E2E). A execução manual da MobileNetV3 em [benchmark_manual/resultado_mobilenet_v3_20260916_2119.csv](benchmark_manual/resultado_mobilenet_v3_20260916_2119.csv) usou as mesmas 3.425 máscaras de referência (totais de pixels positivos/negativos coincidem), com F1 0,721894 e IoU 0,564816; média de 58,634 ms no modelo e 75,710 ms E2E na CPU de desktop. O FPS manual é `1000/latência média`, não throughput concorrente como na placa. O resultado manual salvo não prova qual checkpoint foi usado; não atribua toda a diferença de qualidade à quantização sem verificar checkpoint, entrada e pós-processamento. `Testes/Teste_Unet.py` usa abertura morfológica e não é uma comparação direta.

### Otimização de pré-processamento e tentativa de leitura direta

`resultados/mobilenet_v3_otimizado_p2/` contém a execução com leitura direta via libtiff **e** quantização NEON; a configuração é idêntica à de `resultados/baseline_unico/`. `model_only` ficou em 211,21 ms versus 211,14 ms. O pré-processamento médio caiu de 43,68 para 13,10 ms; a latência E2E mediana caiu de 517,58 para 503,01 ms; a mediana da leitura subiu de 255,33 para 271,29 ms. O throughput total caiu de 6,100 para 4,481 FPS porque houve uma pausa de aproximadamente 251 s que aparece ao mesmo tempo na leitura e na inferência, além de outra de cerca de 30 s. A referência anterior teve pausas de cerca de 30 e 26 s. A causa da pausa de 251 s é desconhecida; não atribua automaticamente à libtiff, swap ou DPU. As métricas de segmentação das duas execuções são idênticas.

Por solicitação do usuário, `dataset.cpp` continua usando `cv::imread` + `convertTo(CV_32F)` e `preprocess.cpp` voltou a usar NEON AArch64, com fallback escalar fora da placa. O link explícito com libtiff foi removido. **A combinação OpenCV + NEON ainda não foi medida na ZCU104.** Para novos testes de desempenho, use `--no-validate`, pois a validação completa é demorada e já confirmou métricas idênticas nas duas execuções anteriores.

Para trazer os resultados ao computador:

```bash
scp -O -r root@192.168.2.100:/home/root/resultados/mobilenet_v3_neon_opencv_p2 .
```

A placa tem 1,9 GiB de RAM; foi ativado 1 GiB de swap em `/home/root/swapfile_1g`. Se reiniciar, use `swapon /home/root/swapfile_1g` e confirme com `free -h`. Swap é apoio para memória, não evidência de que tenha sido usado durante o benchmark; monitore seu uso ao interpretar timings.

## Execução histórica do `Benchmark_ZCU104`

Placa: ZCU104, acesso conhecido `root@192.168.2.100`; modelo em `/home/root/models/compiled_zcu104`, dataset em `/home/root/STARCOP_test`, CSV padrão `/home/root/STARCOP_test/test.csv`. O script fonteia `/etc/profile.d/xilinx.sh` quando existe. A compilação na placa foi feita com `./build_zcu104.sh`, seguida de `./benchmark_zcu104 --self-test`.

```bash
cd /home/root/Benchmark_ZCU104
./run_all_models.sh \
  --models-dir /home/root/models/compiled_zcu104 \
  --dataset /home/root/STARCOP_test \
  --out /home/root/Benchmark_ZCU104/results_all_models \
  --quick-sweep --skip-baseline --sweep-samples 5 \
  --max-runners 2 --max-concurrency 2 --pin-modes no-pin \
  --search-iterations 5 --candidate-iterations 5 \
  --warmup 2 --final-warmup 20 \
  --full-samples 0 --final-iterations 0 --final-repeats 1 \
  --power --power-sample-ms 200 --resume
```

`--sweep-samples 5` seleciona cinco imagens distintas para a busca; `--search-iterations 5` e `--candidate-iterations 5` fazem cinco inferências por configuração nas respectivas etapas. `--full-samples 0` usa todas as imagens válidas no final e `--final-iterations 0` faz uma inferência por imagem. O final testa separadamente 1, 2, 3 e 4 núcleos de CPU. `--skip-baseline` pula a etapa de referência do sweep, **não** o modelo `baseline`. `--resume` só reutiliza execuções válidas no mesmo diretório de saída e com o mesmo identificador/configuração. `run_all_models.sh` percorre sempre todos os modelos desde o início; para continuar somente um, passe `--model /caminho/modelo.xmodel`. Quando chamado para um único modelo, o wrapper reescreve os CSVs consolidados da raiz do `--out` apenas com os modelos da chamada atual; os CSVs por modelo continuam disponíveis.

O limite de dois runners foi adotado após travamento da placa durante a MobileNet V2 com três runners. Não há diagnóstico conclusivo da causa do travamento. Antes de desconectar o terminal, execute dentro de `tmux` ou com `nohup`; fechar uma sessão interativa normal pode interromper o processo.

Para copiar os resultados da placa ao computador, execute no computador:

```bash
scp -O -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -r root@192.168.2.100:/home/root/Benchmark_ZCU104/results_all_models .
```

## Resultados históricos do `Benchmark_ZCU104`

`model-only` usa entradas já preparadas e mede execução/espera do runner; `end-to-end` inclui leitura dos TIFFs, pré-processamento, filas, runner e pós-processamento. Nas MobileNets, há operações de `upsample` fora da partição DPU: `model-only` desses modelos inclui essas operações internas do grafo e não deve ser descrito como tempo puro da DPU. Inicialização e warm-up ficam fora das regiões temporizadas.

Para cada modelo, `results_all_models/methane_<modelo>/benchmark_geral.csv` contém FPS, latências média/mínima/máxima/mediana/P95/P99, tempos de DPU/pré/pós/I/O, tempo total, potência e energia para os quatro limites de CPU; `benchmark_overhead.csv` compara os dois modos; `metricas_globais.csv` e `metricas_por_imagem.csv` contêm TP, FP, FN, TN, precisão, recall, F1 global/forte/fraco, IoU, AUPRC, FPR sem pluma e acurácia. `all_runs.csv`, `ranking_search.csv`, `ranking_final_runs.csv` e `best_config.json` documentam a busca. Cada pasta `runs/` guarda logs, `config.txt`, latências individuais e `benchmark_power_rails.csv`. Potência vem dos sensores `power*_input` em `/sys/class/hwmon`; joules são estimados pela potência média × duração medida.

Os cinco modelos têm arquivos de resultado finais locais em `results_all_models/`. A comparação já produzida está em [RESULTADOS_CPU_ZCU104.md](RESULTADOS_CPU_ZCU104.md). Ela usa 342 imagens da ZCU104, execuções finais com quatro núcleos e dois runners, e o registro **CPU histórico mais recente** de cada modelo em `Resultados_testes/historico_testes.csv`. Atenção: o próprio relatório está marcado como precisando ser refeito com o dataset completo. O CSV histórico da CPU não registra dataset nem quantidade de imagens. Portanto, as diferenças de F1/IoU/AUPRC e os ganhos de velocidade são apenas indicativos; não atribua a queda de qualidade à quantização ou à DPU sem avaliar CPU e ZCU104 sobre as mesmas amostras e pré/pós-processamento. FPS CPU no relatório foi derivado como `1000 / latência média por imagem`, enquanto o FPS da placa veio de throughput medido com concorrência; essas grandezas não são equivalentes quando há vários runners.

Para verificar novas execuções, prefira os CSVs por modelo e filtre `status=ok`, `stage=70_full_model_only` ou `80_full_end_to_end`, `cpu_cores` desejado. Não interprete linhas com `status=failed` ou zeros como medições. Execute novamente a comparação quando houver resultados CPU e placa no mesmo conjunto de dados.

## Cuidados ao continuar

- Leia o estado atual do repositório antes de editar. Em 2026-09-17 havia alterações do usuário em `benchmark_manual/benchmark_cpu.py`, `benchmark_manual/model_loader.py` e diretórios não rastreados `.vscode/`, `baseline_unico/` e `benchmark_best_zcu104/`; não reverta nem sobrescreva essas mudanças sem solicitação.
- A instrução local fornecida pelo usuário é `/home/thiago/.codex/RTK.md`: prefixe comandos shell com `rtk`.
- O usuário prefere respostas em português, comandos concretos para a placa e soluções simples. Quando pedir somente um comando ou resposta curta, respeite isso.
