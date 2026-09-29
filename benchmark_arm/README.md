# Benchmark ONNX no Cortex-A53 da ZCU104

O exportador usa os checkpoints em `Modelos_treinados`; o programa C++ mede
ONNX Runtime em CPU (sem DPU), usando os quatro TIFFs, normalização e limiar
`logit > 0` dos benchmarks existentes. Execute a exportação no computador com
PyTorch, torchvision, segmentation-models-pytorch e `onnx` instalados.

```bash
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model all
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model hyperstarcop
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model resnet34 --checkpoint /caminho/pesos.pth
```

Modelos com checkpoint atual: `baseline`, `depth_reduced`, `mobilenet_v2`,
`mobilenet_v3`, `skip`, `resnet34`, `segformer`, `hyperstarcop`.
`attention_gates` e `psa` ainda não têm checkpoint no projeto. Use um
checkpoint explícito para exportá-los:

```bash
python benchmark_arm/exportar_to_onnx/scripts/exportar.py \
  --model attention_gates --checkpoint /caminho/pesos.pth
```

Para uma arquitetura futura com construtor `(in_channels=4, out_channels=1)` e
pesos `state_dict`:

```bash
python benchmark_arm/exportar_to_onnx/scripts/exportar.py \
  --model Modelos.NovoModelo:MinhaClasse --checkpoint /caminho/modelo.pth \
  --output benchmark_arm/exportar_to_onnx/modelos_convertidos_onnx/novo.onnx
```

## Execução na ZCU104

Com a estrutura informada na placa (`/home/root/thiago/benchmark_arm`,
`/home/root/thiago/STARCOP_test` e `/home/root/thiago/dataset_starcop`), entre
na pasta do código, compile o executável e rode o self-contained benchmark.
Os ONNX devem estar em
`/home/root/thiago/benchmark_arm/exportar_to_onnx/modelos_convertidos_onnx/`.

### Compilar

```bash
cd /home/root/thiago/benchmark_arm/codigos_c
./build.sh
```

Cada execução abaixo é independente. Primeiro compile uma vez; depois copie e
cole somente o comando do modelo e dataset que deseja medir. Cada execução
roda `model_only`, `end_to_end` e validação, salvando em sua própria pasta.
O terminal mostra o avanço de `warmup`, `model_only`, `end_to_end` e `validacao`
em contagem e porcentagem, atualizado após cada inferência.

### Baseline

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/baseline.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/baseline_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/baseline.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/baseline_full
```

### Depth reduced

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/depth_reduced.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/depth_reduced_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/depth_reduced.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/depth_reduced_full
```

### Skip connections

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/skip.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/skip_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/skip.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/skip_full
```

### MobileNetV2

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/mobilenet_v2.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/mobilenet_v2_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/mobilenet_v2.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/mobilenet_v2_full
```

### MobileNetV3

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/mobilenet_v3.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/mobilenet_v3_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/mobilenet_v3.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/mobilenet_v3_full
```

### ResNet34

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/resnet34.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/resnet34_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/resnet34.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/resnet34_full
```

### SegFormer

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/segformer.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/segformer_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/segformer.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/segformer_full
```

### HyperSTARCOP

**STARCOP_test**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/hyperstarcop.onnx --dataset /home/root/thiago/STARCOP_test --threads 4 --output resultados_arm/hyperstarcop_test
```

**Dataset full**

```bash
cd /home/root/thiago/benchmark_arm/codigos_c && ./benchmark_arm --model ../exportar_to_onnx/modelos_convertidos_onnx/hyperstarcop.onnx --dataset /home/root/thiago/dataset_starcop --threads 4 --output resultados_arm/hyperstarcop_full
```

Os resultados ficam em `/home/root/thiago/benchmark_arm/codigos_c/resultados_arm/`,
em diretórios separados por modelo e dataset. Para copiar os resultados da placa
ao computador, execute no computador:

```bash
scp -r root@IP_DA_ZCU104:/home/root/thiago/benchmark_arm/codigos_c/resultados_arm .
```

Para cross compile no computador, use o sysroot existente e informe o
compilador AArch64 instalado:

```bash
SYSROOT="$PWD/zcu104_sysroot" CXX=aarch64-linux-gnu-g++ benchmark_arm/codigos_c/build.sh
```

`--csv` escolhe `test.csv` ou `train.csv` quando ambos existem; `--limit`,
`--warmup`, `--threads`, `--mode`, `--power-interval-ms`, `--no-power` e
`--output` controlam a execução. Por padrão, cada execução grava uma pasta
própria em `resultados_arm/`; `--mode all` é o padrão. O benchmark exige entrada/saída float32 com shapes
`[1,4,512,512]` e `[1,1,512,512]`. ONNX Runtime 1.14 do sysroot suporta o
opset 16 usado na exportação. Os ONNX gerados são mantidos localmente e
ignorados pelo Git por causa do tamanho.

O executável usa os nomes `input` e `logits` definidos pelo exportador e
reutiliza buffers de entrada e saída com esses shapes. Para modelos futuros,
exporte pelo script desta pasta para manter esse contrato.

## Resultados

- `benchmark_geral.csv`: latência média, mediana, mínimo, máximo, P95, P99,
  desvio, FPS, throughput e versões do ambiente para `model_only` e `end_to_end`.
- `benchmark_estagios.csv` e `benchmark_samples.csv`: leitura, preprocessamento,
  inferência e pós-processamento. `model_only` repete a primeira entrada já
  preparada, como carga de inferência isolada.
- `metricas_globais.csv`, `metricas_grupos.csv` e `metricas_por_imagem.csv`:
  TP, FP, FN, TN, precision, recall, F1, IoU, acurácia, FPR, F1 forte/fraca,
  AUPRC e FPR por tile. AUPRC usa 256 bins dos logits, como o benchmark DPU;
  a escala de 0,1 é fixa no ARM, portanto pode diferir da escala do XModel.
- `benchmark_power_rails.csv`: potência média, mínima e máxima por trilho,
  energia estimada em joules e joules por inferência para cada modo. O sensor
  vem de `/sys/class/hwmon/*/power*_input` em microwatts. A energia é
  `potência média × duração`; não é uma leitura de contador de energia. O CSV
  registra chip, label e arquivo de cada trilho para conferência. Quando não
  há sensor legível, registra `indisponivel`, sem inventar um valor.

Os modos de desempenho excluem a validação com labels da janela medida. Os
trilhos representam domínios elétricos da placa e podem incluir outros
componentes além do Cortex-A53; não some trilhos que se sobrepõem. Tempos de
sincronização DPU, runners e slots não se aplicam à inferência ONNX em CPU.

O ONNX Runtime usa otimizações básicas, sem arena de memória nem padrão de
alocação, para reduzir o uso de RAM na placa. Se ocorrer `std::bad_alloc`, a
mensagem informa a etapa; a exceção sozinha não prova falta de RAM. Consulte
`free -m` na placa; `--threads 1` reduz o consumo por threads quando necessário
e deve ser registrado como uma configuração de benchmark diferente.
