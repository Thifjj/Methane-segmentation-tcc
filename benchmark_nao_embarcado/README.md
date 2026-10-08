# Benchmark manual em CPU e GPU

## Coletas CUDA atuais do artigo

Em `resultados_artigo/` há cinco coletas CUDA concluídas: `attentiongates_dpu_bce_artigo`, `attentiongates_dpu_focaldice_artigo`, `mobilenet_v3_dpu_bce_artigo`, `mobilenet_v3_dpu_focaldice_artigo` e `hyperstarcop`. A coleta CUDA de `attentiongates_dpu_bce_artigo` usa o checkpoint anterior; o alias agora carrega `UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth`, cujo benchmark CPU/GPU está pendente. Cada execução usa as 342 imagens do TEST, 512×512, FP32, batch 1 e dez aquecimentos.

Para medir os cinco modelos na GPU com os checkpoints atuais, na raiz do projeto:

```bash
for modelo in attentiongates_dpu_bce_artigo attentiongates_dpu_focaldice_artigo mobilenet_v3_dpu_bce_artigo mobilenet_v3_dpu_focaldice_artigo hyperstarcop; do
  venv/bin/python -m benchmark_nao_embarcado.benchmark_geral \
    --attention-dpu "$modelo" --device cuda --dataset test \
    --data-root /media/jacques/games/Datasets/test/STARCOP_test \
    --num-threads 4 --patch-size 512 --patch-batch-size 1 \
    --output-dir benchmark_nao_embarcado/resultados_artigo || break
done
```

`benchmark_geral.csv` grava `cpu_energia_*`, `gpu_energia_*` e `cpu_gpu_energia_*` nas execuções CUDA. A soma recebe status `ok` somente com leituras válidas dos dois sensores. Ela cobre o pacote CPU (RAPL) e a GPU (NVML), não a energia total do PC na tomada. `benchmark_power_rails.csv` mantém cada fonte separada.

## Potencia media real da CPU

O benchmark le o contador RAPL do pacote CPU e calcula **potencia media (W) =
energia medida (J) / duracao medida (s)**. Registra separadamente `model_only`
(inferencia) e `end_to_end` (fluxo completo), excluindo o warm-up.
A medida inclui o consumo do pacote CPU durante o intervalo, inclusive outros
processos; nao estima o consumo a partir de TDP ou porcentagem de utilizacao.

Na extensao, `benchmark_geral.csv` e o console mostram `cpu_potencia_media_w`,
`cpu_energia_j`, `cpu_energia_por_inferencia_j` e `cpu_energia_status`.
`benchmark_power_rails.csv` preserva as medidas por pacote CPU/GPU. No fluxo
manual, os campos recebem prefixos `model_only_` e `end_to_end_`.
Sem leitura real, os novos campos numericos ficam `nan`; leituras incompletas
recebem status `parcial`.

Neste computador, o Linux exige permissao para o contador. Para liberar somente
a leitura do pacote CPU ate a reinicializacao, execute no seu terminal:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
```

Depois execute o benchmark normalmente, sem sudo. Se outro computador usar um
caminho diferente, o benchmark informa o caminho exato bloqueado.

## Novos checkpoints: MobileNet Focal Dice e Attention Gate BCE

Na raiz do projeto, para CPU e as 342 imagens do TEST com entrada inteira 512×512:

```bash
for MODELO in mobilenet_v3_focaldice attentiongates_bce; do
  venv/bin/python -m benchmark_nao_embarcado.benchmark_geral \
    --attention-dpu "$MODELO" --device cpu --dataset test \
    --num-threads 4 --patch-size 512 \
    --output-dir benchmark_nao_embarcado/resultados_cpu || break
done
```

`mobilenet_v3_focaldice` usa `MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth`
com `UNetMobileNetV3_dpu`, conforme o `main.ipynb` atual.
`attentiongates_bce` usa `MobileNetV3_AttentionGates_BCE_mag1c_rgb.pth`
com `UNetMobileNetV3AttentionGates`. A opção `--attention-dpu` seleciona
o modelo na extensão existente; estas execuções são PyTorch FP32 na CPU.

## AttentionGates DPU: CPU/GPU comparáveis com a ZCU104

A extensão `benchmark_dpu.py` adiciona as duas variantes DPU ao menu de
`benchmark_geral.py` (opções 7 e 8). O fluxo dos seis modelos anteriores permanece
o mesmo. Para executar sem perguntas, use os comandos abaixo **na raiz do projeto**.
O alias `attentiongates_dpu_bce_artigo` usa agora o checkpoint BCE principal
medido no ARM e na DPU; as coletas CPU/GPU antigas desse alias são históricas.

### Dois modelos na CPU — TEST completo, 342 imagens válidas

```bash
source venv/bin/activate
for MODELO in attentiongates_dpu_easy_remaining attentiongates_dpu_only_remaining; do
  python -m benchmark_nao_embarcado.benchmark_geral \
    --attention-dpu "$MODELO" --device cpu --dataset test \
    --num-threads 4 --patch-size 128 --patch-batch-size 1 || break
done
```

### Dois modelos na GPU CUDA — mesmo conjunto e geometria

```bash
source venv/bin/activate
for MODELO in attentiongates_dpu_easy_remaining attentiongates_dpu_only_remaining; do
  python -m benchmark_nao_embarcado.benchmark_geral \
    --attention-dpu "$MODELO" --device cuda --dataset test \
    --num-threads 4 --patch-size 128 --patch-batch-size 1 || break
done
```

Para uma conferência curta, acrescente `--limit 2 --warmup 1`. Sem `--limit`,
executa todas as imagens válidas do CSV. Confira a quantidade mostrada na
inicialização e o término `Validacao: 342/342` no TEST completo.

### Entrada, medição e métricas da extensão

- Reutiliza `STARCOPDataset` e `DataNormalizer` do treinamento: janelas do CSV,
  MAG1C/1750, bandas/60, clamp `[0,2]` e ordem **MAG1C, 460, 550, 640 nm**.
- Por padrão, cada imagem 512×512 vira **16 patches 128×128 sem sobreposição**,
  processados com batch 1 e reconstruídos na ordem usada na ZCU104.
- `model_only` reutiliza até quatro entradas preparadas, excluindo leitura,
  pré/pós-processamento e transferência; `end_to_end` lê cada imagem. Warm-up
  fica fora das medidas; há sincronização CUDA nas fronteiras das etapas.
- E2E inclui leitura, normalização/organização dos patches, transferência de
  entrada, modelo, transferência/reconstrução da saída e máscara `logit > 0`.
  Abertura, leitura do label, métricas e escrita dos CSVs ficam na validação
  separada, como no benchmark da placa.
- `throughput_fps = imagens / duração medida` e
  `fps_latencia = 1000 / latência média`. O pipeline CPU/GPU é sequencial;
  a placa pode executar runners concorrentes. Os campos têm a mesma definição,
  mas as configurações de execução diferem.
- Registra média, mediana, mínimo, máximo, P95, P99, desvio e tempos por imagem
  e etapa. Potência/energia reutilizam os contadores RAPL/NVML existentes, com
  leituras nos limites de cada fase e a cada 200 ms. Potência mínima/máxima
  corresponde às médias dos intervalos, não a valores instantâneos. Fontes
  indisponíveis ou com leituras parciais ficam identificadas;
  estes domínios diferem do trilho INA226 da placa.
- Validação bruta: sem abertura, grupos por `has_plume/qplume` e AUC da curva PR
  global por pixel. `validacao_oficial`: abertura em cruz 3×3, grupos por pixels
  positivos do label e `difficulty=easy`, AUPRC como média de AP das imagens
  positivas. Ambas registram TP/FP/FN/TN, precisão, recall, F1, IoU, acurácia,
  FPR global/sem pluma e FPR por tile (**mais de 640 pixels** por imagem 512×512).
- Registra hash do checkpoint/CSV, IDs das imagens, geometria, batch, threads,
  versões, CPU/GPU e pico de VRAM. Confere o hash do checkpoint com o manifesto
  oficial de calibração quando esse manifesto está disponível.

`threads_pytorch` é a quantidade de threads configurada e `cpus_permitidas` é
a quantidade de CPUs permitida pelo sistema; nenhuma afinidade é imposta pela
extensão. `runners=0` indica que esta execução PyTorch não utiliza runners VART;
há uma instância do modelo, com uma imagem por vez.

`--patch-size 512` avalia a imagem inteira para comparar com o histórico FP32.
`--patch-batch-size 16` agrupa patches em uma chamada; muda o protocolo de
desempenho em relação às 16 chamadas da placa. Ambos são registrados na saída.
Para FULL ou outros caminhos, use `--dataset full --data-root /caminho/dataset`
e, se necessário, `--csv /caminho/train.csv`.

### Resultados da extensão

Cada execução cria uma pasta única, sem sobrescrever os CSVs antigos:

```text
benchmark_nao_embarcado/resultados_dpu/<modelo>_<dataset>_<cpu|cuda>_<horario>/
├── config.json
├── amostras.csv
├── benchmark_geral.csv
├── benchmark_estagios.csv
├── benchmark_samples.csv
├── benchmark_power_rails.csv
├── metricas_globais.csv
├── metricas_grupos.csv
├── metricas_por_imagem.csv
└── validacao_oficial/
    ├── metricas_globais.csv
    ├── metricas_grupos.csv
    └── metricas_por_imagem.csv
```

Compare vazão em `benchmark_geral.csv` e qualidade recente em
`validacao_oficial/metricas_globais.csv`, com os arquivos equivalentes da ZCU104.
FP32 CPU/GPU pode diferir do INT8 da placa pela quantização. A AUPRC bruta usa
arquivos temporários para scores/labels; sua ordenação global ainda exige RAM,
principalmente no FULL. Para executar os comandos da extensão, o ambiente
também precisa das dependências do treinamento, incluindo torchvision, Kornia
e scikit-learn. A extensão não executa calibração nem quantização.

Verificação curta dos patches e das métricas, sem dataset/checkpoints:

```bash
python -m benchmark_nao_embarcado.test_benchmark_dpu
```

## Benchmark existente

Este diretório executa os modelos PyTorch do projeto em CPU ou GPU CUDA. O
benchmark mede latência, FPS e qualidade da segmentação usando as amostras do
STARCOP no dataset selecionado (`full` ou `test`).

## Arquivos

| Arquivo | Função |
|---|---|
| `benchmark_geral.py` | Executa e mede o modelo em CPU ou GPU CUDA, escolhidos no programa. |
| `model_loader.py` | Cria a arquitetura, carrega os pesos e move o modelo para o device recebido. |
| `dataset.py` | Lê o CSV do dataset escolhido, os quatro canais TIFF e o label. |
| `preprocess.py` | Normaliza, limita os valores e monta o tensor NCHW. |
| `postprocess.py` | Aplica sigmoid e limiar de 0,5. |
| `metricas.py` | Calcula métricas globais e F1 por intensidade da pluma. |

## Requisitos

- Python 3.12 ou uma versão compatível com as dependências do projeto;
- PyTorch;
- NumPy;
- pandas;
- rasterio;
- tqdm;
- uma GPU NVIDIA, driver e instalação CUDA do PyTorch para o benchmark GPU.

Na raiz do repositório, crie um ambiente e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Confira o ambiente:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

O benchmark CPU não exige CUDA. O benchmark GPU exige que `CUDA` apareça como
`True`.

## Estrutura do dataset

O benchmark aceita dois conjuntos locais:

- `full`: `/home/thiago/Documents/STARCOP_DATASET`, com `train.csv`;
- `test`: `STARCOP_test` dentro da raiz do projeto, com `test.csv`.

Cada CSV deve apontar para uma pasta por amostra:

```text
STARCOP_DATASET/ (ou STARCOP_test/)
├── train.csv (ou test.csv)
├── amostra_0001/
│   ├── mag1c.tif
│   ├── TOA_AVIRIS_640nm.tif
│   ├── TOA_AVIRIS_550nm.tif
│   ├── TOA_AVIRIS_460nm.tif
│   └── labelbinary.tif
└── amostra_0002/
    └── ...
```

O `train.csv` precisa ter estas colunas:

| Coluna | Uso |
|---|---|
| `folder` | Caminho original ou nome da pasta da amostra. O benchmark usa o último componente do caminho e o procura dentro da raiz selecionada. |
| `has_plume` | Aceita `true`/`false` ou `1`/`0`. Define se a amostra participa dos F1 strong e weak. |
| `qplume` | Intensidade usada para separar strong e weak. É obrigatória quando `has_plume=true`. |

Os quatro canais e o label devem possuir as mesmas dimensões esperadas pelo
modelo, normalmente `512 × 512`.

## Configuração

### Caminhos dos pesos

Edite o dicionário `modelos` em `benchmark_nao_embarcado/model_loader.py`. Cada item
deve conter a classe e o caminho do respectivo arquivo `.pth`:

```python
modelos = {
    "baseline": (UNetBaseline, "/caminho/UNET_mag1c_rgb.pth"),
    "depth_reduced": (UNetDepthReduced, "/caminho/UNET_depth_reduced_mag1c_rgb.pth"),
    "mobilenet_v2": (UNetMobileNetV2, "/caminho/Mobile_Net_v2_mag1c_rgb.pth"),
    "mobilenet_v3": (UNetMobileNetV3, "/caminho/Mobile_Net_v3_mag1c_rgb.pth"),
    "skip": (UNetElementWise, "/caminho/UNET_SkipConnections_mag1c_rgb.pth"),
}
```

Os modelos apresentados no menu são:

| Valor | Arquitetura |
|---|---|
| `baseline` | U-Net baseline |
| `depth_reduced` | U-Net com profundidade reduzida |
| `mobilenet_v2` | U-Net MobileNetV2 |
| `mobilenet_v3` | U-Net MobileNetV3 |
| `skip` | U-Net com skip connections element-wise |

## Execução

Execute os comandos a partir da raiz do repositório. O uso de `python3 -m` é
necessário porque os scripts usam imports relativos do pacote
`benchmark_nao_embarcado`.

### Benchmark geral

```bash
source .venv/bin/activate
python3 -m benchmark_nao_embarcado.benchmark_geral
```

Ao iniciar, o programa pergunta o modelo, o dataset (`full` ou `test`) e o
dispositivo (CPU ou GPU CUDA). A GPU exige CUDA disponível no PyTorch. Depois,
pergunta quantas imagens executar (`0` para todas ou um número entre `1` e o
total de amostras).

`full` usa `STARCOP_DATASET/train.csv`; `test` usa
`STARCOP_test/test.csv`.

Para executar outro modelo, inicie o programa novamente e selecione-o no menu.

## Pipeline executado

Para cada amostra, o benchmark faz:

1. leitura dos quatro canais TIFF;
2. normalização de `mag1c` por `1750` e RGB por `60`;
3. clamp de cada canal no intervalo `[0, 2]`;
4. montagem do tensor `[1, 4, H, W]`;
5. inferência PyTorch;
6. sigmoid e limiar estrito `> 0,5` (equivalente a `logit > 0`);
7. leitura do label e cálculo das métricas por pixel.

Antes das medições são executadas dez inferências de warm-up usando a primeira
amostra. O warm-up não entra nos tempos reportados.

## Tempos medidos

| Campo | Conteúdo |
|---|---|
| `model_*` | Somente a chamada do modelo. Na GPU há sincronização CUDA antes e depois da inferência. |
| `e2e_*` | Leitura dos quatro canais, pré-processamento, transferência para a GPU quando aplicável, modelo e pós-processamento. |
| `carregamento_ms` | Média da leitura dos quatro canais TIFF. |
| `preprocess_ms` | Média da normalização e montagem da entrada. Na GPU também inclui a transferência da entrada ao device. |
| `posprocess_ms` | Média do sigmoid e da criação da máscara binária. |

A leitura de `labelbinary.tif`, o cálculo das métricas e a escrita do CSV ficam
fora do tempo E2E.

Para `model_only` e E2E são mostrados média, mediana, mínimo, máximo, P95, P99
e FPS. O FPS é calculado como:

```text
FPS = 1000 / latência média em milissegundos
```

Esse FPS representa a execução sequencial de uma imagem por vez. Ele não é
throughput com várias inferências concorrentes.

### Potência e energia

O benchmark lê o contador RAPL do pacote CPU e, quando usa CUDA, o contador
acumulado de energia da GPU via NVML. Para `model_only` e `end_to_end`, o CSV
registra a fonte (`energia_fontes`), número de imagens medidas, energia total
(J), energia por inferência (J) e potência média (W). Mínimo e máximo de
potência são as médias de cada imagem, não amostras instantâneas. O pacote CPU
e a GPU são domínios separados; não representam a potência total da tomada.
O E2E inclui leitura, pré-processamento e pós-processamento, mas exclui a
validação das métricas. Se RAPL/NVML não existir, não permitir leitura ou não
suportar contador de energia, a fonte fica ausente do CSV; `energia_fontes`
mostra `indisponivel` quando nenhuma fonte pôde ser medida. Os CSVs antigos
não ganham esses valores retroativamente.

## Métricas de qualidade

As contagens são acumuladas por pixel sobre todas as amostras do grupo antes
do cálculo das métricas, ou seja, são métricas globais agregadas.

| Métrica | Definição |
|---|---|
| `tp`, `fp`, `fn`, `tn` | Contagens globais de pixels. |
| `precision` | `TP / (TP + FP)`. |
| `recall` | `TP / (TP + FN)`. |
| `f1_global` | F1 calculado com todas as amostras, inclusive `has_plume=false`. |
| `f1_strong_plume` | F1 das amostras com `has_plume=true` e `qplume > 1000`. |
| `f1_weak_plume` | F1 das amostras com `has_plume=true` e `qplume <= 1000`. |
| `iou` | `TP / (TP + FP + FN)` usando todas as amostras. |
| `fpr` | `FP / (FP + TN)` usando todas as amostras. |

A quantidade de pixels positivos da máscara não é usada para classificar uma
amostra como strong ou weak. Amostras com `has_plume=false` continuam no
`f1_global`, mas não entram em `f1_strong_plume` nem em `f1_weak_plume`.

O pós-processamento deste benchmark não executa abertura morfológica. O loop
principal usa `> 0,5`; o helper isolado `postprocess.py` usa `>= 0,5` e não é
chamado pelo loop atual.

## Resultado gerado

Ao terminar, o programa cria:

```text
benchmark_nao_embarcado/resultado_<dataset>_<device>_<modelo>_AAAAMMDD_HHMM.csv
```

Exemplo:

```text
benchmark_nao_embarcado/resultado_test_cpu_mobilenet_v3_20260921_1430.csv
```

O CSV contém uma linha com:

- data, modelo e quantidade de imagens;
- latências e FPS de model-only;
- latências e FPS E2E;
- médias de carregamento, pré-processamento e pós-processamento;
- precision, recall, `f1_global`, `f1_strong_plume`, `f1_weak_plume`, IoU e FPR;
- TP, FP, FN e TN globais.
- potência e energia de CPU/GPU, quando os contadores estiverem disponíveis.

O nome identifica dataset, dispositivo e modelo, com precisão de um minuto.
Duas execuções da mesma combinação iniciadas no mesmo minuto usam o mesmo
caminho e a segunda sobrescreve a primeira.

## Comparações reproduzíveis

Para comparar CPU, GPU e ZCU104:

1. use o mesmo conjunto, CSV e as mesmas amostras, na mesma ordem;
2. use checkpoints correspondentes às versões quantizadas compiladas para a placa;
3. confirme que a ordem dos canais, a normalização e o limiar são iguais;
4. feche cargas concorrentes no computador e registre PyTorch, runtime Vitis AI,
   arquitetura da DPU e configuração do pipeline;
5. compare qualidade pelas métricas globais e por amostra. As diferenças
   residuais podem vir da quantização INT8;
6. compare desempenho separando latência de inferência, latência E2E e throughput.

A entrada e as métricas de qualidade são comparáveis quando CPU e ZCU104 usam
exatamente o mesmo CSV e conjunto. O CPU mede PyTorch FP32 sequencial; a placa
usa XModel quantizado e pode executar várias inferências em paralelo. Por isso,
FPS sequencial, throughput e latência do pipeline são medidas distintas, mesmo
com os mesmos dados.

## Verificação rápida da classificação

```bash
python3 -m benchmark_nao_embarcado.metricas
```

A saída esperada é:

```text
Self-test OK
```

## Erros comuns

### `ImportError: attempted relative import with no known parent package`

O arquivo foi executado diretamente. Volte à raiz do repositório e use:

```bash
python3 -m benchmark_nao_embarcado.benchmark_geral
```

### `FileNotFoundError` para `train.csv` ou TIFF

Confira `DATASET_PATH`, os nomes dos cinco TIFFs e se as pastas listadas em
`train.csv` existem dentro da raiz configurada.

### Erro ao carregar os pesos

Confira se cada entrada de `modelos` possui exatamente a classe e um caminho
válido para o `.pth` correspondente. O checkpoint deve ser compatível com a
arquitetura selecionada.

### `CUDA não disponível`

Use o benchmark CPU ou instale uma versão do PyTorch compatível com a GPU, o
driver NVIDIA e a versão CUDA do ambiente.

### Coluna ausente ou valor inválido em `train.csv`

Confirme a existência de `folder`, `has_plume` e `qplume`. Para amostras com
`has_plume=true`, `qplume` não pode estar vazio.
