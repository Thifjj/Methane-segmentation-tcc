# Benchmark de XModels na ZCU104

Compila e mede modelos de segmentação com VART/GraphRunner. O programa lê os
quatro TIFFs do STARCOP, normaliza mag1c por 1750 e RGB por 60, limita os
valores a [0, 2], quantiza para INT8 e gera máscara com `logit > 0`, igual ao
limiar efetivo do `benchmark_nao_embarcado/benchmark_geral.py`.

## Compilar na placa

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
```

São necessários OpenCV, VART, XIR e GraphRunner da imagem da placa.

## Executar o sweep

### Todos os modelos nos dois datasets

Os cinco XModels compilados para a ZCU104 estão em
`build/vitis_ai/compiled_zcu104/`: `baseline`, `depth_reduced`,
`skip_connections`, `mobilenet_v2` e `mobilenet_v3`. Na placa, considerando que
os arquivos foram copiados para `/home/root/thiago/benchmark/modelos` e os
datasets estão em `/home/root/thiago`, compile e execute todos os modelos nos
dois conjuntos: `STARCOP_test` e o full, chamado `dataset_starcop` na placa:

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh

MODELS=/home/root/thiago/benchmark/modelos
for DATASET in \
  /home/root/thiago/STARCOP_test \
  /home/root/thiago/dataset_starcop; do
  ./run_all_models.sh --models-dir "$MODELS" --dataset "$DATASET"
done
```

`run_all_models.sh` aceita `--models-dir` e `--dataset`, encontra recursivamente os cinco arquivos
`methane_*.xmodel` e roda `sweep_vitis` e o benchmark final para cada um.

### Somente o baseline nos dois datasets

Para rodar o sweep somente do baseline nos dois datasets, copie e execute na
placa:

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
./sweep_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/STARCOP_test
./sweep_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/dataset_starcop
```

O sweep grava `ranking_search.csv`, `best_config.txt` e a execução final em
`resultados_zcu104/methane_baseline_<dataset>_sweep_*/`.

## Executar sem sweep

### Um modelo por vez: STARCOP_test

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/depth_reduced/methane_depth_reduced.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/skip_connections/methane_skip_connections.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v2/methane_mobilenet_v2.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/STARCOP_test
```

### Um modelo por vez: dataset full (`dataset_starcop`)

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/depth_reduced/methane_depth_reduced.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/skip_connections/methane_skip_connections.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v2/methane_mobilenet_v2.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/dataset_starcop
```

O programa usa `test.csv` ou `train.csv`, conforme o CSV presente no dataset.
Se ambos estiverem na mesma pasta, defina `NOME_CSV` em `benchmark_vitis.cpp`
para selecionar o conjunto sem ambiguidade. Para este uso, mantenha `test.csv`
em `STARCOP_test` e `train.csv` em `dataset_starcop`.
As pastas das amostras vêm da coluna `folder` do CSV, como no benchmark Python.
Os resultados são criados em `resultados_zcu104/` no diretório atual, em uma
pasta com o nome do modelo, do dataset e horário da execução. Por exemplo,
execuções do baseline ficam em pastas `methane_baseline_STARCOP_test_*` e
`methane_baseline_dataset_starcop_*`, mantendo os resultados do test e do full
separados.

## Configuração no código

- `pipeline.hpp`: runners, quatro núcleos de CPU, workers de pré e pós,
  slots, warm-up, inferências e afinidade da execução direta.
- `benchmark_vitis.cpp`, estrutura `Opcoes`: modo, limite de amostras,
  potência, intervalo de amostragem e validação. `NOME_CSV` resolve datasets
  que contenham os dois CSVs; a pasta de saída também é escolhida nesse arquivo.
- `sweep.cpp`, constantes no início: runners 2–4 (2 para MobileNetV2), 80 inferências por candidato
  na busca, warm-up e limites de tempo. O sweep mantém quatro núcleos,
  um worker de pós-processamento e dois slots por runner; compara dois e quatro
  workers de pré-processamento. Para MobileNetV2, limita a busca a dois runners
  porque três já travaram a placa.

Cada candidato mede 80 inferências em `end_to_end`. Só a configuração
vencedora passa à execução final sobre todas as amostras. O modo `all` mede
`model_only` e `end_to_end` e valida a segmentação em etapas separadas. A busca
serve para escolher candidatos; os CSVs finais trazem as medidas
completas. O `sweep_vitis` chama
`benchmark_vitis` com opções internas para automatizar cada candidato; elas
não são necessárias na execução manual.

## Resultados e comparação

- `benchmark_geral.csv`: configuração usada, CSV do dataset, distribuição e
  kernel Linux, arquitetura e versões instaladas de Vitis AI Library, VART,
  XIR e OpenCV, além de throughput, FPS por latência e tempos médios das
  etapas. O FPS por latência (`1000 / média em ms`) segue a
  definição do benchmark Python. Como o pipeline da placa executa imagens em
  paralelo, a latência E2E ainda representa uma execução diferente da
  sequência usada no Python.
- `benchmark_estagios.csv` e `benchmark_samples.csv`: distribuição e tempos
  individuais de leitura, pré-processamento, espera, sincronização, inferência
  e pós-processamento.
- `metricas_globais.csv`: configuração vencedora, CSV do dataset e ambiente
  também aparecem junto de TP, FP, FN, TN, precision, recall, F1, IoU, FPR por
  pixel, F1 strong/weak, AUPRC e FPR por tile. `fp_tiles` conta tiles sem pluma
  com mais de dez pixels previstos; `fpr_tile` divide pelos tiles sem pluma e
  `fpr_tile_tabela` divide por todas as imagens.
- `metricas_por_imagem.csv` e `metricas_grupos.csv`: métricas por amostra e
  grupo. A validação usa todas as amostras fora das regiões cronometradas.
- `benchmark_power_rails.csv`: potência e energia por trilho, incluindo
  joules por inferência. A energia por inferência é a energia total do trilho
  dividida pelo número de inferências; não é uma leitura individual de cada
  inferência. Cada linha informa `sensor_chip` e os caminhos `fonte_name`,
  `fonte_label` e `fonte_power_input` usados nessa execução. Não some trilhos
  que possam se sobrepor.

## Origem das leituras de potência

O monitor percorre `/sys/class/hwmon/hwmon*/`. Para cada trilho, lê o chip em
`name`, o rótulo (quando existe) em `powerN_label` e a potência em
`powerN_input`. O CSV grava esses caminhos exatos para permitir conferir na
própria placa, por exemplo:

```bash
cat /sys/class/hwmon/hwmonN/name
cat /sys/class/hwmon/hwmonN/powerM_label
cat /sys/class/hwmon/hwmonN/powerM_input
```

Substitua `hwmonN` e `powerM` pelos nomes registrados no CSV; os índices podem
mudar entre inicializações. Esses arquivos são interfaces virtuais do driver
do kernel, não logs gravados diretamente pelo sensor. Segundo a
[interface hwmon do Linux](https://docs.kernel.org/hwmon/sysfs-interface.html),
`powerN_input` fornece potência instantânea em microwatts. O código divide o
valor por `1.000.000` para obter watts. A `energia_j` reportada é uma **estimativa**
`media_w × duracao_s`, e `energia_por_inferencia_j = energia_j / inferencias`;
o programa não lê um contador direto de energia.

`model_only` mede `execute_async + wait` com as entradas preparadas nos slots,
reutilizando essas entradas. `entradas_preparadas` no CSV registra quantos
slots foram montados. O `end_to_end` mede o pipeline concorrente de leitura
até pós-processamento. Leitura dos labels e cálculo da qualidade ficam fora
dos tempos, como no benchmark Python. Nas MobileNets, o runner pode conter
subgrafos de CPU.

A AUPRC usa a área trapezoidal da curva precisão–revocação, como no Python,
mas agrupa scores nos 256 níveis INT8 do XModel. Diferenças residuais em
relação ao PyTorch são esperadas por causa da quantização. O código conserva
OpenCV para ler TIFF e usa NEON no pré-processamento da ZCU104.

As versões de Vitis AI Library, VART e XIR são lidas dos arquivos de versão
instalados na placa. `vitis_ai_library_versao` identifica a biblioteca de
execução, não a versão do compilador usada para gerar o XModel. Um campo fica
`indisponivel` quando a informação não existe na imagem da placa.
