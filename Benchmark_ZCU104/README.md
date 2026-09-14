# Benchmark dos modelos de metano na ZCU104

Este diretório contém um benchmark C++17 para os `.xmodel` do projeto. O modelo e o dataset são escolhidos somente por argumentos; nenhum caminho precisa ser alterado no código.

A implementação parte do runner e do sweep do [Projeto_VITISAI_hyperstarcop](https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop/tree/main/to_zcu). Ela mantém runners independentes, slots alinhados e reutilizáveis, filas limitadas, workers separados de pré-processamento/DPU/pós-processamento, afinidade, warm-up e a busca em estágios. A seleção usa a mediana do throughput das confirmações finais e P99 como desempate.

## O que é medido

- `model-only`: os inputs já estão lidos, normalizados, quantizados, residentes em buffers pré-alocados e sincronizados antes do relógio. A região medida contém apenas `execute_async()` e `wait()`. Modelos totalmente atribuídos à DPU usam o runner DPU direto; grafos particionados usam o GraphRunner.
- `end-to-end`: mede leitura dos quatro TIFFs, normalização `mag1c + RGB`, quantização INT8, filas, sincronização dos buffers, execução do modelo e threshold com abertura morfológica.
- Energia: detecta automaticamente os trilhos `power*_input` em `/sys/class/hwmon`, amostra potência durante cada região medida e estima energia em joules. Use `--power-sample-ms N` para mudar o intervalo ou `--no-power` para desativar a telemetria.
- Qualidade: TP, FP, FN, TN, precisão, recall, F1 global, F1 forte, F1 fraca, IoU, AUPRC, FPR em tiles sem pluma e acurácia. O critério forte/fraco e a abertura morfológica são os de `Testes/Teste_Unet.py`.

Leitura dos labels, cálculo das métricas e escrita dos resultados ficam fora das regiões temporizadas. A AUPRC é exata para a saída INT8: ela acumula os 256 valores possíveis sem guardar todos os pixels na memória.

Nos modelos `baseline`, `depth_reduced` e `skip_connections`, o GraphRunner contém somente a partição DPU. Os `.xmodel` MobileNet atuais possuem cinco `upsample` com `align_corners=true` atribuídos à CPU pelo compilador. Neles, `model-only` continua excluindo I/O e pré/pós-processamento, mas inclui essas partições internas obrigatórias do grafo. O arquivo `config.txt` de cada execução registra `dpu_subgraphs` e `cpu_fallback_subgraphs` para evitar interpretar esse tempo como DPU puro.

## Compilar na placa

Copie `Benchmark_ZCU104/`, os modelos e o dataset para a ZCU104. A imagem PetaLinux 2022.2 capturada em `zcu104_sysroot/` fornece VART 3.0, XIR, GraphRunner e OpenCV. O script usa automaticamente o mesmo source existente na placa:

```bash
cd /home/root/Benchmark_ZCU104
chmod +x build_zcu104.sh run_all_models.sh
./build_zcu104.sh
./benchmark_zcu104 --self-test
```

Não é necessário Conda nem Python na placa.

## Executar um modelo

O comando abaixo faz primeiro o sweep rápido e depois executa o benchmark completo, separadamente, restringindo o processo a 1, 2, 3 e 4 núcleos da CPU:

```bash
./sweep_zcu104 \
  --model /home/root/models/compiled_zcu104/depth_reduced/methane_depth_reduced.xmodel \
  --dataset /home/root/STARCOP_test \
  --csv /home/root/STARCOP_test/test.csv \
  --out /home/root/results/depth_reduced \
  --sweep-samples 8 \
  --search-iterations 24 \
  --full-samples 0 \
  --candidate-iterations 5 \
  --final-iterations 0 \
  --warmup 20 \
  --final-repeats 3 \
  --max-concurrency 16 \
  --dpu-cores 2 \
  --pin-modes both \
  --resume
```

`--full-samples 0` usa todas as linhas válidas do CSV. `--candidate-iterations` controla a confirmação rápida das melhores configurações; `0` reutiliza `--search-iterations`. `--final-iterations 0` executa uma inferência por imagem disponível no benchmark completo. O runner aceita ainda `--runners`, `--pre-workers`, `--post-workers`, `--slots-per-runner`, `--iterations`, `--samples`, `--cpu-cores` (também controla `--threads`), `--pin`, `--no-pin`, `--validate` e `--no-validate`.

## Executar todos os modelos

O wrapper encontra recursivamente todos os `.xmodel`. Ele aceita tanto `build/compiled_zcu104/` quanto o caminho atual `build/vitis_ai/compiled_zcu104/`:

```bash
./run_all_models.sh \
  --models-dir /home/root/models/compiled_zcu104 \
  --dataset /home/root/STARCOP_test \
  --csv /home/root/STARCOP_test/test.csv \
  --out /home/root/results/all_models \
  --sweep-samples 8 \
  --search-iterations 24 \
  --candidate-iterations 5 \
  --final-iterations 0 \
  --final-repeats 3 \
  --resume
```

Para um único artefato, troque `--models-dir` por `--model /caminho/modelo.xmodel`.

## Resultados

Cada modelo recebe um diretório próprio. Os principais arquivos são:

- `benchmark_geral.csv`: medições completas `model-only` e `end-to-end` para 1, 2, 3 e 4 núcleos, com cada repetição separada;
- `benchmark_overhead.csv`: diferença de latência e perda de throughput entre os dois modos;
- `metricas_globais.csv` e `metricas_por_imagem.csv`: métricas de qualidade;
- `all_runs.csv`, `ranking_search.csv` e `ranking_final_runs.csv`: todas as configurações do sweep;
- `best_config.json`: configuração selecionada;
- `benchmark_samples.csv`: latência e tempos de estágio por inferência;
- `benchmark_power_rails.csv`: potência média, mínima, máxima e energia estimada de cada sensor/trilho da placa;
- `config.txt`: modelo, dataset, afinidade e partições executadas.

Ao executar vários modelos, `run_all_models.sh` também cria `benchmark_geral.csv` e `benchmark_overhead.csv` consolidados na raiz do diretório de saída, permitindo comparação direta.

Para uma busca curta, use `--quick-sweep --skip-baseline --max-runners 3 --max-concurrency 2 --pin-modes no-pin --final-candidates 1 --warmup 2 --final-warmup 20`. Esse modo ajusta apenas runners e 1–2 workers de pré-processamento; pós-processamento permanece em um worker e cada runner usa um slot.
