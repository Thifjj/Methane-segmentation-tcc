# Benchmark de XModels na ZCU104

## Seleção atual do artigo

Os cinco XModels INT8 da seleção atual são AttentionGates DPU BCE e Focal Dice, MobileNetV3 DPU BCE e Focal Dice, e HyperSTARCOP. As coletas de referência ficam em `benchmark_zcu104/resultados_zcu104/`; os artefatos correspondentes ficam em `VitisAI/build/vitis_ai/compiled_artigo/`. As seções abaixo documentam também o fluxo e os oito modelos históricos em `compiled_zcu104/`. Para comparação com CPU/GPU FP32, veja [`../../benchmark_nao_embarcado/README.md`](../../benchmark_nao_embarcado/README.md).

Compila e mede modelos de segmentação com VART/GraphRunner. O programa lê os
quatro TIFFs do STARCOP, normaliza mag1c por 1750 e RGB por 60, limita os
valores a [0, 2], quantiza para INT8 e gera máscara com `logit > 0`, igual ao
limiar efetivo do `benchmark_nao_embarcado/benchmark_geral.py`.

## Caminhos usados e artefatos restaurados

| Conteúdo na placa | Caminho |
|---|---|
| Fontes e executáveis | `/home/root/thiago/benchmark/codigos_c/` |
| Compilados dos oito modelos | `/home/root/thiago/benchmark/modelos/` |
| Dataset test (342 imagens) | `/home/root/thiago/STARCOP_test/` |
| Dataset full remaining+easy | `/home/root/thiago/dataset_starcop/` |
| Resultados | `resultados_zcu104/` dentro da pasta em que o programa é executado |

Os seis modelos antigos foram restaurados do Git/Git LFS nos caminhos
originais do repositório, incluindo **HyperSTARCOP**:

| Modelo | Compilado em `VitisAI/build/vitis_ai/compiled_zcu104/` | Quantizado em `VitisAI/build/vitis_ai/quantize/` |
|---|---|---|
| Baseline | `baseline/methane_baseline.xmodel` | `baseline/UNetBaseline_int.xmodel` |
| Depth Reduced | `depth_reduced/methane_depth_reduced.xmodel` | `depth_reduced/UNetDepthReduced_int.xmodel` |
| Skip Connections | `skip_connections/methane_skip_connections.xmodel` | `skip_connections/UNetElementWise_int.xmodel` |
| MobileNetV2 | `mobilenet_v2/methane_mobilenet_v2.xmodel` | `mobilenet_v2/UNetMobileNetV2_int.xmodel` |
| MobileNetV3 | `mobilenet_v3/methane_mobilenet_v3.xmodel` | `mobilenet_v3/UNetMobileNetV3_int.xmodel` |
| HyperSTARCOP | `hyperstarcop/methane_hyperstarcop.xmodel` | `hyperstarcop/HyperSTARCOPOficial_int.xmodel` |

Cada compilado tem `meta.json` e `md5sum.txt`. Cada quantizado tem
`quant_info.json` e `bias_corr.pth`. A restauração recuperou os artefatos
originais sem recalibração, quantização ou recompilação.
O manifesto de restauração citado no fluxo antigo não está presente nesta
árvore de trabalho; confira os manifestos dos modelos atuais em `compiled_artigo/`.
Os seis MD5 e o target `DPUCZDX8G_ISA1_B4096` foram conferidos;
os seis quantizados e seis compilados abriram no XIR.

Os dois AttentionGates continuam em
`VitisAI/build/vitis_ai/compiled_zcu104/` e
`VitisAI/build/vitis_ai/quantize/`.
A pasta de modelos da placa recebe os arquivos de **compiled_zcu104**;
os grafos em **quantize** são os artefatos anteriores à compilação.

## Copiar para a placa — no computador

Execute os comandos abaixo na **raiz do repositório Methane-segmentation-tcc**.
O IP dos exemplos é `192.168.2.100`; ajuste-o se o endereço da placa for outro.
Os destinos correspondem às pastas que já existem na placa.

```bash
# Copia os fontes atualizados para /home/root/thiago/benchmark/codigos_c/.
scp -O -r benchmark_zcu104/codigos_c root@192.168.2.100:/home/root/thiago/benchmark/

# Copia todos os compilados, incluindo HyperSTARCOP, AttentionGates e MobileNet DPU.
# Os subdiretórios baseline, hyperstarcop etc. ficam diretamente em modelos/.
scp -O -r VitisAI/build/vitis_ai/compiled_zcu104/* root@192.168.2.100:/home/root/thiago/benchmark/modelos/


# Opcional: envia o test set se ele ainda não estiver na placa.
# A pasta deve conter test.csv e os TIFFs das 342 imagens.
scp -O -r STARCOP_test root@192.168.2.100:/home/root/thiago/
```

`-O` usa o protocolo SCP tradicional. Depois da cópia dos fontes, compile
na placa conforme a próxima seção. Para o full, mantenha `train.csv` e os TIFFs
em `/home/root/thiago/dataset_starcop/`.

## Compilar na placa

```bash
# Na placa: entre na pasta dos códigos antes de compilar/executar.
cd /home/root/thiago/benchmark/codigos_c
# Recompila benchmark_vitis, sweep_vitis e self_test_support com os fontes atualizados.
./build_zcu104.sh

# Confere os cálculos em CPU; o resultado esperado é Self-test OK.
./self_test_support
```

São necessários OpenCV, VART, XIR e GraphRunner da imagem da placa.

## Validação dos dois AttentionGates nas 342 imagens — na placa

O comando normal mede `model_only`, mede `end_to_end` e depois valida
a segmentação em uma etapa separada. Usa todas as amostras válidas do CSV;
com o test set completo, confira **342 amostras** na inicialização e
**validacao: 342/342** ao terminar. `config.txt` deve registrar `amostras=342`.

```bash
# Na placa: parta da pasta dos executáveis já recompilados.
cd /home/root/thiago/benchmark/codigos_c

# Executa os dois AttentionGates em sequência, usando o mesmo test set.
for MODELO in attentiongates_dpu_easy_remaining attentiongates_dpu_only_remaining; do
  ./benchmark_vitis \
    --model "/home/root/thiago/benchmark/modelos/$MODELO/$MODELO.xmodel" \
    --dataset /home/root/thiago/STARCOP_test || break
done
```

Cada modelo gera sua própria pasta em `resultados_zcu104/`. Para comparar
com a validação recente, leia os CSVs de `validacao_oficial/` dessa pasta.

### Todos os oito modelos diretamente, com a configuração atual

Este bloco executa cada modelo uma vez com a configuração de `pipeline.hpp`.
Para o full, altere somente `DATASET`.

```bash
cd /home/root/thiago/benchmark/codigos_c
MODELS=/home/root/thiago/benchmark/modelos
DATASET=/home/root/thiago/STARCOP_test
# Para usar o full: DATASET=/home/root/thiago/dataset_starcop

# Primeiro, os seis modelos anteriores, incluindo HyperSTARCOP.
for MODELO in baseline depth_reduced skip_connections mobilenet_v2 mobilenet_v3 hyperstarcop; do
  ./benchmark_vitis \
    --model "$MODELS/$MODELO/methane_$MODELO.xmodel" \
    --dataset "$DATASET" || break
done

# Depois, os AttentionGates, que usam outro padrão de nome do XModel.
for MODELO in attentiongates_dpu_easy_remaining attentiongates_dpu_only_remaining; do
  ./benchmark_vitis \
    --model "$MODELS/$MODELO/$MODELO.xmodel" \
    --dataset "$DATASET" || break
done
```

O script `run_all_models.sh` da seção seguinte faz **sweep + execução final**
de cada modelo. Os comandos acima usam diretamente a configuração atual.

## Executar o sweep

### Todos os modelos nos dois datasets

Os modelos anteriores são `baseline`, `depth_reduced`,
`skip_connections`, `mobilenet_v2`, `mobilenet_v3` e `hyperstarcop`.
Os dois AttentionGates oficiais estão no repositório em
`VitisAI/build/vitis_ai/compiled_zcu104/`. Na placa, considerando que
os arquivos foram copiados para `/home/root/thiago/benchmark/modelos` e os
datasets estão em `/home/root/thiago`, compile e execute todos os modelos nos
dois conjuntos: `STARCOP_test` e o full, chamado `dataset_starcop` na placa:

```bash
# Na placa: entre na pasta dos códigos antes de compilar/executar.
cd /home/root/thiago/benchmark/codigos_c
# Recompila benchmark_vitis, sweep_vitis e self_test_support com os fontes atualizados.
./build_zcu104.sh

# Pasta dos oito compilados na placa.
MODELS=/home/root/thiago/benchmark/modelos
# Repete a busca de configuração no test e no full, separando os resultados.
for DATASET in \
  /home/root/thiago/STARCOP_test \
  /home/root/thiago/dataset_starcop; do
# Faz sweep + execução final de cada compilado encontrado (os oito modelos).
  ./run_all_models.sh --models-dir "$MODELS" --dataset "$DATASET"
done
```

`run_all_models.sh` aceita `--models-dir` e `--dataset`, encontra recursivamente
todos os arquivos `methane_*.xmodel` e `attentiongates_dpu_*.xmodel` na pasta e executa `sweep_vitis` para cada
um. Portanto, inclua nessa pasta somente os modelos que deseja medir. O sweep
avalia configurações de pipeline e executa a configuração vencedora sobre todas
as amostras; ele não é apenas uma execução única com configuração fixa.

### Somente o baseline nos dois datasets

Para rodar o sweep somente do baseline nos dois datasets, copie e execute na
placa:

```bash
# Na placa: entre na pasta dos códigos antes de compilar/executar.
cd /home/root/thiago/benchmark/codigos_c
# Recompila benchmark_vitis, sweep_vitis e self_test_support com os fontes atualizados.
./build_zcu104.sh
# Baseline: Busca a melhor configuração e depois mede/valida o conjunto completo; test.csv, 342 imagens do test completo.
./sweep_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/STARCOP_test
# Baseline: Busca a melhor configuração e depois mede/valida o conjunto completo; train.csv do full remaining+easy.
./sweep_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/dataset_starcop
```

O sweep grava `ranking_search.csv`, `best_config.txt` e a execução final em
`resultados_zcu104/methane_baseline_<dataset>_sweep_*/`.

### Sweep do HyperSTARCOP

```bash
# Na placa: procura a melhor configuração e valida o HyperSTARCOP no test.
cd /home/root/thiago/benchmark/codigos_c
./sweep_vitis --model /home/root/thiago/benchmark/modelos/hyperstarcop/methane_hyperstarcop.xmodel --dataset /home/root/thiago/STARCOP_test

# Repete a busca e a execução final no full remaining+easy.
./sweep_vitis --model /home/root/thiago/benchmark/modelos/hyperstarcop/methane_hyperstarcop.xmodel --dataset /home/root/thiago/dataset_starcop
```

## Executar sem sweep

### Um modelo por vez: STARCOP_test

```bash
# Baseline: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# Depth Reduced: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/depth_reduced/methane_depth_reduced.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# Skip Connections: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/skip_connections/methane_skip_connections.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# MobileNetV2: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v2/methane_mobilenet_v2.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# MobileNetV3: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# HyperSTARCOP: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/hyperstarcop/methane_hyperstarcop.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# Na placa: AttentionGates easy + remaining, todas as imagens do test (342).
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/attentiongates_dpu_easy_remaining/attentiongates_dpu_easy_remaining.xmodel --dataset /home/root/thiago/STARCOP_test
```

```bash
# Na placa: AttentionGates only remaining, todas as imagens do test (342).
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/attentiongates_dpu_only_remaining/attentiongates_dpu_only_remaining.xmodel --dataset /home/root/thiago/STARCOP_test
```

### Um modelo por vez: dataset full (`dataset_starcop`)

```bash
# Baseline: Mede desempenho e valida todas as amostras; train.csv do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# Depth Reduced: Mede desempenho e valida todas as amostras; train.csv do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/depth_reduced/methane_depth_reduced.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# Skip Connections: Mede desempenho e valida todas as amostras; train.csv do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/skip_connections/methane_skip_connections.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# MobileNetV2: Mede desempenho e valida todas as amostras; train.csv do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v2/methane_mobilenet_v2.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# MobileNetV3: Mede desempenho e valida todas as amostras; train.csv do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# HyperSTARCOP: Mede desempenho e valida todas as amostras; train.csv do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/hyperstarcop/methane_hyperstarcop.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# Na placa: AttentionGates easy + remaining, todas as imagens do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/attentiongates_dpu_easy_remaining/attentiongates_dpu_easy_remaining.xmodel --dataset /home/root/thiago/dataset_starcop
```

```bash
# Na placa: AttentionGates only remaining, todas as imagens do full remaining+easy.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/attentiongates_dpu_only_remaining/attentiongates_dpu_only_remaining.xmodel --dataset /home/root/thiago/dataset_starcop
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
  com mais de 640 pixels previstos em cada imagem 512×512, seguindo o limiar
  oficial de 10 pixels por 64×64; `fpr_tile` divide pelos tiles sem pluma e
  `fpr_tile_tabela` divide por todas as imagens (não é o FPR da Tabela 2).
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
# Na placa: use o caminho do sensor registrado no CSV; N/M são exemplos.
cat /sys/class/hwmon/hwmonN/name
# Na placa: use o caminho do sensor registrado no CSV; N/M são exemplos.
cat /sys/class/hwmon/hwmonN/powerM_label
# Na placa: use o caminho do sensor registrado no CSV; N/M são exemplos.
cat /sys/class/hwmon/hwmonN/powerM_input
```

Substitua `hwmonN` e `powerM` pelos nomes registrados no CSV; os índices podem
mudar entre inicializações. Esses arquivos são interfaces virtuais do driver
do kernel, não logs gravados diretamente pelo sensor. Segundo a
[interface hwmon do Linux](https://docs.kernel.org/hwmon/sysfs-interface.html),
`powerN_input` fornece potência instantânea em microwatts. O código divide o
valor por `1.000.000` para obter watts. A `energia_j` reportada é uma **estimativa**
da integral trapezoidal das leituras com seus horários, restrita ao intervalo
medido do pipeline. `media_w = energia_j / duracao_s` é uma média ponderada
pelo tempo, e `energia_por_inferencia_j = energia_j / inferencias`;
o programa não lê um contador direto de energia.

`model_only` mede `execute_async + wait` com as entradas preparadas nos slots,
reutilizando essas entradas. `entradas_preparadas` no CSV registra quantos
slots foram montados. O `end_to_end` mede o pipeline concorrente de leitura
até pós-processamento. Leitura dos labels e cálculo da qualidade ficam fora
dos tempos, como no benchmark Python. Nas MobileNets, o runner pode conter
subgrafos de CPU.

A AUPRC usa a área trapezoidal da curva precisão–revocação, como no Python,
agrupando os 256 níveis INT8 pela probabilidade FLOAT32 após sigmoid
(inclusive empates por saturação). A escala vem do tensor quantizado; para
saída FLOAT32 de GraphRunner, vem da entrada do operador `fix2float`.
Não se usa uma escala presumida, e saídas fora dessa grade são rejeitadas. Diferenças residuais em
relação ao PyTorch são esperadas por causa da quantização. O código conserva
OpenCV para ler TIFF e usa NEON no pré-processamento da ZCU104.

As versões de Vitis AI Library, VART e XIR são lidas dos arquivos de versão
instalados na placa. `vitis_ai_library_versao` identifica a biblioteca de
execução, não a versão do compilador usada para gerar o XModel. Um campo fica
`indisponivel` quando a informação não existe na imagem da placa.


## AttentionGates oficiais e conferência de qualidade

Os AttentionGates usam entrada INT8 NHWC `[1,512,512,4]`, uma inferência por
imagem, na mesma ordem de canais de `VitisAI/evaluate_quantized.py`. O runner
rejeita XModels AttentionGates com outra resolução.
O nome do grafo `UNetMobileNetV3AttentionGatesDPU` seleciona a ordem
`mag1c,460nm,550nm,640nm` do treinamento e da calibração, mesmo se o arquivo
for renomeado. Os modelos anteriores continuam usando `mag1c,640nm,550nm,460nm`.
Para AttentionGates, a seleção de amostras também exige os quatro TIFFs de
entrada e o label, como o avaliador Python. A entrada usa arredondamento
`floor(x + 0.5)`, correspondente ao operador Vitis AI com `method=2`.

Os arquivos de métricas existentes continuam usando máscara `logit > 0`,
grupos por `has_plume/qplume` e AUPRC global trapezoidal, para comparação com
`benchmark_nao_embarcado/benchmark_geral.py`. Na subpasta
`validacao_oficial/` são acrescentados os mesmos três CSVs, seguindo
`VitisAI/evaluate_quantized.py` e o histórico recente:

- Abertura morfológica com cruz 3×3 depois de reconstruir a imagem.
- AP calculado com sigmoid dos logits originais, por imagem com label positivo;
  AUPRC é a média desses APs. Imagens sem positivos ficam com AP vazio.
- Strong: label positivo e `difficulty == easy`; demais positivos são weak.
- FPR No-Plume usa FP/TN das imagens sem pixels positivos no label.
- Sem imagens positivas, a AUPRC oficial fica indefinida (NaN), como no Python.

Para comparar com a avaliacao FP32/INT8, use os CSVs 512x512 de
`VitisAI/build/vitis_ai/evaluation/attentiongates_dpu_<variante>/`.
Os XModels e a avaliacao usam entrada 512x512 e uma inferencia por imagem.

`config.txt` registra geometria, ordem de canais, escalas, seleção de amostras
e protocolos. Os CSVs acrescentam identificação da amostra e o método de
cada métrica. Throughput, latência e energia por inferência representam
**uma imagem 512×512**; cada AttentionGate executa uma inferência por imagem.
`model_only` mede apenas `execute_async + wait`, sem abertura.
`end_to_end` mantém o pós-processamento original; a abertura do protocolo
oficial é calculada somente na validação fora da região cronometrada.
A concorrência continua diferente da execução sequencial em CPU/GPU.

Energia é estimada por trilho, dentro da mesma janela de desempenho, excluindo
warm-up e o tempo de parada do monitor. A leitura inicial/final delimita a
interpolação; quando um sensor não cobre uma borda temporal, usa-se sua leitura
mais próxima. Min/max e quantidade de amostras descrevem as leituras coletadas,
incluindo essas bordas. Não existe leitura direta de energia nem subtração
de consumo em repouso. Sensores ausentes continuam marcados como indisponíveis.

### Comandos com os caminhos da placa

Depois de copiar os fontes atualizados para
`/home/root/thiago/benchmark/codigos_c`:

```bash
# Na placa: entre na pasta dos códigos antes de compilar/executar.
cd /home/root/thiago/benchmark/codigos_c
# Recompila benchmark_vitis, sweep_vitis e self_test_support com os fontes atualizados.
./build_zcu104.sh
# Confere os cálculos em CPU; resultado esperado: Self-test OK.
./self_test_support
# AttentionGates easy + remaining: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/attentiongates_dpu_easy_remaining/attentiongates_dpu_easy_remaining.xmodel --dataset /home/root/thiago/STARCOP_test
# AttentionGates only remaining: Mede desempenho e valida todas as amostras; test.csv, 342 imagens do test completo.
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/attentiongates_dpu_only_remaining/attentiongates_dpu_only_remaining.xmodel --dataset /home/root/thiago/STARCOP_test
```

Para o full, substitua o dataset por `/home/root/thiago/dataset_starcop`.
O script de todos os modelos inclui os oito modelos das subpastas mostradas.

Conferência local: compilação C++ no contêiner Vitis AI e `self_test_support`.
Uma comparação temporária com três amostras reais confirmou leitura
OpenCV/Rasterio, normalização/quantização Vitis AI, montagem de patches,
abertura, contagens, AP/AUPRC e formato dos CSVs. Os logits dessa comparação
eram sintéticos; a execução da DPU e os sensores reais precisam ser
conferidos na placa.


## Estrutura dos resultados e cópia de volta

Executando na pasta dos códigos, a execução direta cria:

```text
resultados_zcu104/<nome_xmodel>_<dataset>_manual_<horario>/
├── config.txt
├── benchmark_geral.csv
├── benchmark_estagios.csv
├── benchmark_samples.csv
├── benchmark_power_rails.csv
├── metricas_globais.csv
├── metricas_por_imagem.csv
├── metricas_grupos.csv
└── validacao_oficial/
    ├── metricas_globais.csv
    ├── metricas_por_imagem.csv
    └── metricas_grupos.csv
```

No sweep, os resultados finais ficam em
`<modelo>_<dataset>_sweep_<horario>/final/final_all_r<N>_pre<N>/`.
A mesma estrutura de CSVs está dentro dessa pasta. O diretório do sweep
também contém `ranking_search.csv`, `best_config.txt` e os candidatos em `runs/`.

```bash
# No computador: execute na pasta local em que deseja guardar os resultados.
# Copia tanto as execuções diretas quanto os sweeps e suas subpastas.
scp -O -r root@192.168.2.100:/home/root/thiago/benchmark/codigos_c/resultados_zcu104 .
```

### Conferir os arquivos restaurados — no computador

```bash
# Execute na raiz do repositório; confere tamanhos e SHA-256 do manifesto.
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

raiz = Path("VitisAI/build/vitis_ai")
manifesto = json.loads((raiz / "restoration_manifest.json").read_text())
for item in manifesto["files"]:
    arquivo = raiz / item["path"]
    h = hashlib.sha256()
    with arquivo.open("rb") as entrada:
        for bloco in iter(lambda: entrada.read(1024 * 1024), b""):
            h.update(bloco)
    assert arquivo.stat().st_size == item["bytes"], arquivo
    assert h.hexdigest() == item["sha256"], arquivo
print(f'{len(manifesto["files"])} arquivos restaurados conferidos.')
PY
```

A conferência dos hashes e dos grafos comprova a restauração dos artefatos.
A execução real dos modelos e a coleta dos sensores devem ser conferidas
na ZCU104 com os comandos acima.
