# Benchmark manual em CPU e GPU

Este diretório executa os modelos PyTorch do projeto em CPU ou GPU CUDA. O
benchmark mede latência, FPS e qualidade da segmentação usando as amostras do
STARCOP descritas por `train.csv`.

## Arquivos

| Arquivo | Função |
|---|---|
| `benchmark_cpu.py` | Executa e mede o modelo em CPU. |
| `benchmark_gpu.py` | Executa e mede o modelo em GPU CUDA. |
| `model_loader.py` | Cria a arquitetura, carrega os pesos e move o modelo para o device recebido. |
| `dataset.py` | Lê `train.csv`, os quatro canais TIFF e o label. |
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

O caminho configurado em `DATASET_PATH` deve conter `train.csv` e uma pasta
por amostra:

```text
STARCOP_DATASET/
├── train.csv
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
| `folder` | Caminho original ou nome da pasta da amostra. O benchmark usa o último componente do caminho e o procura dentro de `DATASET_PATH`. |
| `has_plume` | Aceita `true`/`false` ou `1`/`0`. Define se a amostra participa dos F1 strong e weak. |
| `qplume` | Intensidade usada para separar strong e weak. É obrigatória quando `has_plume=true`. |

Os quatro canais e o label devem possuir as mesmas dimensões esperadas pelo
modelo, normalmente `512 × 512`.

## Configuração

### 1. Caminho do dataset

Edite `DATASET_PATH` nos dois executáveis:

- `benchmark_manual/benchmark_cpu.py`;
- `benchmark_manual/benchmark_gpu.py`.

Exemplo:

```python
DATASET_PATH = "/caminho/para/STARCOP_DATASET"
```

### 2. Caminhos dos pesos

Edite o dicionário `modelos` em `benchmark_manual/model_loader.py`. Cada item
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

Os nomes aceitos por `--modelo` são:

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
`benchmark_manual`.

### CPU

```bash
source .venv/bin/activate
python3 -m benchmark_manual.benchmark_cpu --modelo mobilenet_v3
```

### GPU

```bash
source .venv/bin/activate
python3 -m benchmark_manual.benchmark_gpu --modelo mobilenet_v3
```

O programa pergunta:

```text
Quantas imagens deseja usar? (0 = todas, máximo N):
```

- digite `0` para executar todas as amostras encontradas;
- digite um valor entre `1` e `N` para executar somente as primeiras amostras,
  na ordem de `train.csv`.

Exemplo rápido com 20 imagens:

```text
Quantas imagens deseja usar? (0 = todas, máximo 3425): 20
```

### Executar todos os modelos

Todas as amostras em CPU:

```bash
for modelo in baseline depth_reduced mobilenet_v2 mobilenet_v3 skip; do
    printf '0\n' | python3 -m benchmark_manual.benchmark_cpu --modelo "$modelo"
done
```

Todas as amostras em GPU:

```bash
for modelo in baseline depth_reduced mobilenet_v2 mobilenet_v3 skip; do
    printf '0\n' | python3 -m benchmark_manual.benchmark_gpu --modelo "$modelo"
done
```

## Pipeline executado

Para cada amostra, o benchmark faz:

1. leitura dos quatro canais TIFF;
2. normalização de `mag1c` por `1750` e RGB por `60`;
3. clamp de cada canal no intervalo `[0, 2]`;
4. montagem do tensor `[1, 4, H, W]`;
5. inferência PyTorch;
6. sigmoid e limiar `>= 0,5`;
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

O pós-processamento deste benchmark não executa abertura morfológica.

## Resultado gerado

Ao terminar, o programa cria:

```text
benchmark_manual/resultado_<modelo>_AAAAMMDD_HHMM.csv
```

Exemplo:

```text
benchmark_manual/resultado_mobilenet_v3_20260921_1430.csv
```

O CSV contém uma linha com:

- data, modelo e quantidade de imagens;
- latências e FPS de model-only;
- latências e FPS E2E;
- médias de carregamento, pré-processamento e pós-processamento;
- precision, recall, `f1_global`, `f1_strong_plume`, `f1_weak_plume`, IoU e FPR;
- TP, FP, FN e TN globais.

O nome possui precisão de um minuto. Duas execuções do mesmo modelo iniciadas
no mesmo minuto usam o mesmo caminho e a segunda sobrescreve a primeira. O CSV
também não possui uma coluna indicando CPU ou GPU; renomeie o arquivo após a
execução quando precisar manter ambos:

```bash
mv benchmark_manual/resultado_mobilenet_v3_20260921_1430.csv \
   benchmark_manual/resultado_mobilenet_v3_gpu_20260921_1430.csv
```

## Comparações reproduzíveis

Para comparar CPU, GPU e ZCU104:

1. use o mesmo checkpoint e a mesma ordem de canais;
2. use o mesmo `train.csv` e a mesma quantidade de amostras;
3. confirme que o limiar e o pós-processamento são iguais;
4. feche cargas concorrentes no computador;
5. registre as versões do PyTorch, CUDA e driver;
6. compare separadamente model-only, E2E e qualidade.

## Verificação rápida da classificação

```bash
python3 -m benchmark_manual.metricas
```

A saída esperada é:

```text
Self-test OK
```

## Erros comuns

### `ImportError: attempted relative import with no known parent package`

O arquivo foi executado diretamente. Volte à raiz do repositório e use:

```bash
python3 -m benchmark_manual.benchmark_cpu --modelo mobilenet_v3
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
