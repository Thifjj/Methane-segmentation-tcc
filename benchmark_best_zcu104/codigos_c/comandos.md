# Comandos do benchmark modular na ZCU104

Execute os comandos na placa em `/home/root/thiago/benchmark/codigos_c`, exceto `scp`, que é executado no computador. Informe `--csv` explicitamente: o padrão do programa é `DATASET/test.csv`, mas o CSV desta instalação é `train.csv`.

## Copiar, compilar e testar

No computador, a partir da raiz do repositório:

```bash
scp -O -r benchmark_best_zcu104/codigos_c root@192.168.2.100:/home/root/thiago/benchmark/
```

Na ZCU104:

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
./self_test_support
```

Na placa, o autoteste compara a quantização NEON com a referência escalar e confere o pós-processamento e as métricas.

## Medir um modelo

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

Esse exemplo usa a configuração de `resultados/baseline_unico/`, que contém MobileNetV3 apesar do nome, com leitura OpenCV e quantização NEON. `--no-validate` evita repetir a validação, que já produziu métricas idênticas nas duas execuções anteriores. Use uma pasta `--out` nova para comparar o desempenho.

### Opções de `benchmark_vitis`

| Opção | Valores aceitos; padrão | O que influencia |
| --- | --- | --- |
| `--model` | Arquivo `.xmodel` existente; obrigatório | Seleciona a rede compilada. |
| `--dataset` | Pasta existente; obrigatório | Raiz das pastas das amostras e dos TIFFs. |
| `--csv` | CSV existente; padrão: `DATASET/test.csv` | Seleciona e ordena as amostras. Nesta placa, use `.../dataset_starcop/train.csv`. |
| `--out` | Caminho de pasta; obrigatório | Guarda `config.txt` e os CSVs. O programa cria a pasta. |
| `--run-id` | Texto; padrão: `manual_<timestamp>` | Identifica linhas dos resultados; não altera a inferência. |
| `--mode` | `model_only`, `end_to_end` ou `all`; padrão: `all` | `model_only` mede runner com entrada preparada; `end_to_end` inclui leitura, pré, filas, runner e pós; `all` executa os dois modos. |
| `--samples` | Inteiro ≥ 0; padrão: `0` | Quantas amostras distintas carregar do CSV. `0` usa todas; também determina quantas imagens validar. |
| `--iterations` | Inteiro ≥ 0; padrão: `0` | Quantas inferências executar **em cada modo**. `0` faz uma por amostra carregada; outro valor percorre ou repete amostras. Não muda a quantidade validada. |
| `--runners` | `1` a `4`; padrão: `1` | Quantos runners do XModel executam em paralelo. Mais runners aumentam concorrência e uso de memória. A MobileNetV2 já travou a placa com três; a causa é desconhecida. |
| `--cpu-cores` ou `--threads` | `1` a `4`; padrão: `4` | Limita o processo aos primeiros N núcleos permitidos pela afinidade do sistema, inclusive com `--no-pin`. Não é a quantidade de threads criadas. |
| `--pre-workers` | `1` a `4`; padrão: `1` | Threads de leitura TIFF e pré-processamento no E2E. Mais workers podem alimentar melhor os runners, mas disputam CPU, memória e armazenamento. |
| `--post-workers` | `1` a `4`; padrão: `1` | Threads de pós-processamento no E2E. |
| `--slots` | `1` a `4`; padrão: `2` | Buffers/trabalhos por runner. O E2E mantém até `runners × slots` trabalhos em circulação; mais slots usam mais memória. |
| `--warmup` | Inteiro de `0` a `2147483647`; padrão: `20` | Inferências de aquecimento antes da região medida de cada modo; também é passado à validação. |
| `--pin` / `--no-pin` | Sem valor; padrão: `--no-pin` | `--pin` fixa cada worker a um dos núcleos permitidos; `--no-pin` deixa o escalonador mover as threads entre eles. |
| `--power` / `--no-power` | Sem valor; padrão: `--power` | Ativa ou desativa a amostragem de potência durante cada modo medido. |
| `--power-sample-ms` | Inteiro de `1` a `2147483647`; padrão: `200` | Intervalo em milissegundos entre leituras de potência; valores menores geram mais amostras e trabalho de monitoramento. |
| `--validate` / `--no-validate` | Sem valor; padrão: `--validate` | Calcula métricas de segmentação em todas as amostras carregadas **depois** dos modos medidos. Aumenta o tempo total do comando, mas não entra no FPS medido. |
| `--help` / `-h` | Sem valor | Mostra a ajuda. |

`--samples` e `--iterations` aceitam somente inteiros decimais não negativos. `--pre-workers` e `--post-workers` não mudam a região medida de `model_only`. Mesmo com apenas um modo, `--validate` executa a validação. Nas MobileNets, o runner contém subgrafos CPU; `model_only` não é tempo puro da DPU.

## Buscar a melhor configuração para um modelo

```bash
./sweep_vitis \
  --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel \
  --dataset /home/root/thiago/dataset_starcop \
  --csv /home/root/thiago/dataset_starcop/train.csv \
  --out /home/root/resultados/sweep_mobilenet_v3 \
  --max-runners 2 --resume
```

O sweep testa 1 a 4 núcleos e 1 a `--max-runners` runners. Para os melhores candidatos E2E, testa 1 a 4 workers de pré/pós, 1 a 4 slots e afinidade ligada/desligada. Ordena por FPS maior e desempata por P99 menor. Busca e confirmação dispensam potência/validação; finais medem potência, e a primeira repetição E2E valida a segmentação. As saídas ficam em `ranking_search.csv`, `best_config.txt`, `runs/` e `final/`.

### Opções de `sweep_vitis`

| Opção | Valores aceitos; padrão | O que influencia |
| --- | --- | --- |
| `--model` | Arquivo `.xmodel` existente; obrigatório | Rede usada em todas as configurações. |
| `--dataset` | Pasta existente; obrigatório | Amostras e TIFFs. |
| `--csv` | CSV existente; padrão: `DATASET/test.csv` | Lista de amostras; informe `train.csv` nesta placa. |
| `--out` | Pasta; obrigatório | Campanha e resultados. Uma campanha existente exige os mesmos parâmetros. |
| `--benchmark` | Arquivo existente; padrão: `benchmark_vitis` ao lado de `sweep_vitis` | Binário chamado para cada medição. |
| `--max-runners` | `1` a `4`; padrão: `4` | Maior número de runners testado. Recomenda-se `2` até investigar o travamento com `3` na MobileNetV2. |
| `--search-iterations` | Inteiro ≥ 1; padrão: `50` | Inferências por configuração nas grades e ajustes de workers/slots; valores maiores aumentam duração e estabilidade da busca. |
| `--candidate-iterations` | Inteiro ≥ 1; padrão: `150` | Inferências em cada repetição da confirmação dos candidatos. |
| `--confirm-repeats` | Inteiro ≥ 1; padrão: `3` | Repetições por candidato/afinidade; a mediana seleciona o vencedor. |
| `--final-repeats` | Inteiro ≥ 1; padrão: `3` | Repetições finais do dataset inteiro **em cada modo**. |
| `--warmup` | Inteiro ≥ 0; padrão: `5` | Aquecimento de buscas e confirmações. |
| `--final-warmup` | Inteiro ≥ 0; padrão: `20` | Aquecimento das execuções finais. |
| `--timeout-seconds` | Inteiro ≥ 1; padrão: `3600` | Tempo máximo de **cada processo** `benchmark_vitis`; ao exceder, o processo é encerrado e o sweep falha. |
| `--resume` | Sem valor; desativado por padrão | Reutiliza runs válidos da mesma campanha. Pastas incompletas são preservadas com sufixo `.incompleta_N` antes de refazer a execução. |
| `--help` | Sem valor | Mostra a ajuda. |

Os inteiros do sweep são limitados pelo tipo C++ `int`. Ele escolhe `--cpu-cores`, `--runners`, workers e slots; não recebe essas opções diretamente. A fase final usa `--iterations 0`, ou uma inferência por amostra.

## Varrer todos os modelos

```bash
./run_all_models.sh \
  --models-dir /home/root/thiago/benchmark/modelos \
  --dataset /home/root/thiago/dataset_starcop \
  --csv /home/root/thiago/dataset_starcop/train.csv \
  --out /home/root/resultados/todos \
  --max-runners 2 --resume
```

`run_all_models.sh` encontra os arquivos `methane_*.xmodel` dentro de `--models-dir` e chama `sweep_vitis` **sequencialmente**. `--models-dir`, `--dataset` e `--out` são obrigatórios; `--csv` usa o mesmo padrão `DATASET/test.csv`. Cada modelo recebe uma subpasta de `--out`. As outras opções são repassadas ao sweep e seguem a tabela anterior. Para somente um modelo, chame `sweep_vitis` diretamente.

## Trazer resultados ao computador

```bash
scp -O -r root@192.168.2.100:/home/root/resultados/mobilenet_v3_neon_opencv_p2 .
```

Troque a pasta de origem pelo `--out` usado. `-r` copia a pasta inteira; `-O` usa o protocolo SCP clássico necessário nesta placa.
