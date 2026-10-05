# Benchmark ARM / ONNX na ZCU104

O executável mede **ONNX Runtime CPU no Cortex-A53**, com os mesmos modos e
protocolos de qualidade dos benchmarks CPU/GPU e DPU: `model_only`,
`end_to_end`, validação bruta e `validacao_oficial`. A inferência continua em
FP32 no ARM; exportar a arquitetura compatível com DPU para ONNX não a torna
uma execução INT8 nem usa a DPU.

## Attention Gates DPU: modelos prontos

Em `exportar_to_onnx/modelos_convertidos_onnx/`:

| Arquivo | Entrada | Uso |
|---|---|---|
| `attentiongates_dpu_easy_remaining.onnx` | `[1,4,128,128]` | 16 chamadas por imagem |
| `attentiongates_dpu_only_remaining.onnx` | `[1,4,128,128]` | 16 chamadas por imagem |
| `attentiongates_dpu_easy_remaining_batch_dynamic.onnx` | `[B,4,128,128]` | B de 1 a 16 |
| `attentiongates_dpu_only_remaining_batch_dynamic.onnx` | `[B,4,128,128]` | B de 1 a 16 |

Os checkpoints são os dois `UnetMobilenetV3AttentionGates_dpu_*_mag1c_rgb.pth`
em `Modelos_treinados/`; ambos usam `UNetMobileNetV3AttentionGatesDPU` com
1.690.777 parâmetros. Cada ONNX tem metadados com ordem dos canais, checkpoint,
SHA256, normalização e geometria; o `.json` ao lado registra SHA256 do ONNX e a
comparação PyTorch/ONNX Runtime. Não use os pesos da arquitetura original de
atenção com um único canal para esta variante.

Os arquivos usam **opset 16, IR 8**, nomes `input`/`logits` e saída de logits
NCHW. A exportação foi conferida com `onnx.checker` e ONNX Runtime **1.14.1**
CPU, contra PyTorch, em batch 1 e, nos modelos dinâmicos, batch 16
(`rtol=1e-4`, `atol=1e-4`). Isso verifica a conversão no computador; desempenho,
memória e sensores do Cortex-A53 precisam ser medidos na placa.

## Compilar e executar na placa

Copie **toda a pasta `benchmark_arm`** para `/home/root/thiago/benchmark_arm/`,
ou extraia o pacote `entrega_zcu104_attentiongates.tar.gz` nesse diretório pai.
O pacote contém código, README e os quatro ONNX com seus manifestos; não inclui
um executável x86 nem os pesos PyTorch. São necessárias as bibliotecas e
headers OpenCV e ONNX Runtime da placa.

```bash
cd /home/root/thiago/benchmark_arm/codigos_c
./build.sh

# Executa os dois checkpoints, TEST completo, patch 128, batch 1:
./run_attentiongates.sh --dataset /home/root/thiago/STARCOP_test --threads 4

# Agrupa os 16 patches em uma chamada (maior uso de memória):
./run_attentiongates.sh --dataset /home/root/thiago/STARCOP_test --threads 4 --patch-batch-size 16

# FULL, mantendo a mesma geometria:
./run_attentiongates.sh --dataset /home/root/thiago/dataset_starcop --threads 4
```

Execução individual e teste curto:

```bash
./benchmark_arm \
  --model ../exportar_to_onnx/modelos_convertidos_onnx/attentiongates_dpu_only_remaining.onnx \
  --dataset /home/root/thiago/STARCOP_test --threads 4 --limit 2 --warmup 1
```

Sem `--limit`, processa todas as amostras válidas. O CSV é escolhido quando há
apenas `test.csv` ou `train.csv`; se ambos existirem, passe `--csv test.csv`.
O CSV deve incluir `folder` ou `id`, `has_plume`, `qplume` e `difficulty`;
as quatro colunas de janela são opcionais, mas devem estar completas. Pastas
sem os cinco TIFFs são filtradas como no benchmark Attention Gates CPU/GPU.
Labels devem ser binários, finitos e 512×512 depois do recorte.

| Opção | Padrão / finalidade |
|---|---|
| `--mode all\|model_only\|end_to_end` | `all`; a validação separada é executada em qualquer modo |
| `--threads N` | 4 threads intra-op, 1 inter-op |
| `--warmup N` | 10 imagens, fora das medidas |
| `--limit N` | 0 = todas as imagens válidas |
| `--inferencias N` | 0 = uma por amostra; N repete entradas no desempenho, sem repetir a validação |
| `--patch-size 128\|512` | Detectado pelo ONNX; se informado, deve coincidir |
| `--patch-batch-size N` | 1; para N maior, use ONNX dinâmico ou batch fixo compatível |
| `--channel-order auto\|rgb\|legacy` | `auto`: metadados; ONNX antigo sem metadados usa `legacy` |
| `--power-interval-ms N` | 200 ms |
| `--no-power` | Desativa sensores, registrando `desativada` |
| `--output PASTA` | Pasta única em `resultados_arm/`; rejeita pasta com arquivos |

Batch dinâmico aceita grupos parciais, por exemplo 16 patches com batch 3 =
seis chamadas (3+3+3+3+3+1). O executável valida tipo, nomes e shapes reais do
ONNX antes das medições. Modelos antigos de 512×512 continuam funcionando.
Para ONNX antigos sem metadados que esperem MAG1C/460/550/640, informe
`--channel-order rgb`; `legacy` usa MAG1C/640/550/460. A exportação atual dos
Attention Gates grava a ordem RGB automaticamente. As bandas são divididas
por 60 e MAG1C por 1750, com clamp `[0,2]`.

O build usa `pkg-config opencv4 libonnxruntime`. Para um pacote oficial ORT,
use `ORT_ROOT=/caminho/onnxruntime ./build.sh` e configure o loader para sua
pasta `lib` se ela ainda não estiver no sistema. Cross compile:

```bash
SYSROOT=/caminho/zcu104_sysroot CXX=aarch64-linux-gnu-g++ ./build.sh
```

O ONNX Runtime mantém otimização básica, arena e padrão de memória desativados,
como no benchmark ARM anterior. Batch 16 pode usar mais RAM que batch 1; um
`std::bad_alloc` informa a fase. Resultados parcialmente gravados têm
`config.json` ainda `em_execucao` e um `falha.txt`; somente `concluido` representa
uma execução terminada.

## Métricas e protocolos

- **Desempenho:** duração, inferências, throughput (imagens/duração), FPS por
  latência (1000/média), média/mediana/mínimo/máximo/P95/P99/desvio populacional,
  tempos de leitura, preparo dos patches, inferência e reconstrução/máscara.
- **Somente modelo:** reutiliza até quatro entradas preparadas, sem leitura,
  preparo, reconstrução ou transferências. **E2E:** lê cada imagem e inclui
  preparo, modelo e saída. Labels, abertura morfológica, métricas e escrita de
  CSV ficam na validação fora das medidas. Os CSVs são escritos após cada fase.
- **Qualidade bruta:** limiar estrito `logit > 0`, sem abertura; strong por
  `has_plume && qplume > 1000`, weak pelas demais plumas; AUPRC é área
  trapezoidal da curva PR global dos scores sigmoid FP32.
- **Qualidade oficial:** abertura em cruz 3×3 com bordas neutras, strong por
  label positivo e `difficulty=easy`, weak pelas demais imagens positivas;
  AUPRC é a média de average precision das imagens positivas.
- Ambos registram TP/FP/FN/TN, precision, recall, F1 global/strong/weak, IoU,
  acurácia, FPR global por pixel, FPR por pixel sem pluma, FPR por tile e
  `fpr_tile_tabela`, FP/TN tiles e quantidade de imagens positivas. Tile
  positivo significa **mais de 640 pixels** preditos na imagem 512×512.
- FPR por tile divide FP tiles pelos tiles negativos; `fpr_tile_tabela` divide
  FP tiles por todas as imagens. AUPRC bruta sem positivos é 0,5; oficial sem
  imagens positivas é `nan`, seguindo os benchmarks de referência.
- **Potência:** média/mínimo/máximo por trilho hwmon, joules por integração
  trapezoidal dos timestamps e J/imagem na mesma janela de desempenho, com
  chip/label/caminho do sensor. Ausência de leitura vira `indisponivel`.
  Trilhos distintos podem se sobrepor e não representam apenas a CPU ou a
  tomada; não são diretamente equivalentes aos domínios RAPL/NVML.

A PR não re-quantiza logits em 256 bins. Ordena uma imagem de cada vez, agrega
scores iguais e faz merge em disco com até 32 arquivos abertos. Memória do
cálculo limitada a uma imagem; o uso de disco cresce com o número de scores
únicos (até cerca de 6 MiB/imagem, com espaço adicional durante merges).
Temporários `.scores_pr` são removidos no sucesso ou em erros capturados;
uma interrupção abrupta pode deixá-los na pasta incompleta.

ARM é sequencial e FP32; a DPU mede pipeline concorrente INT8. Use o mesmo
checkpoint, CSV, patches, canais e protocolo de qualidade; compare latência e
throughput separadamente. Pequenas diferenças numéricas FP32 entre runtimes
podem alterar pixels próximos do limiar zero. Campos de sincronização ficam em
zero, porque não há transferência CPU/GPU/DPU nesse executável; runners e slots
ficam zero, com uma instância ONNX.

## Arquivos de saída

```text
resultados_arm/<modelo>_<dataset>_<run_id>/
├── config.json                 # execução, shapes, versões, hashes, estado
├── amostras.csv                # IDs e classificações efetivamente usadas
├── benchmark_geral.csv
├── benchmark_estagios.csv
├── benchmark_samples.csv
├── benchmark_power_rails.csv
├── metricas_globais.csv         # protocolo bruto
├── metricas_grupos.csv
├── metricas_por_imagem.csv
└── validacao_oficial/
    ├── metricas_globais.csv
    ├── metricas_grupos.csv
    └── metricas_por_imagem.csv
```

`onnx_sha256`/`csv_sha256` são calculados pelo executável; o hash do checkpoint
vem dos metadados do exportador. Compare-o com o manifesto de calibração da
DPU. CSVs incluem `run_id`, geometria e protocolo; `config.json` registra
threads, CPUs permitidas, versões e configurações. Resultados antigos não são
alterados nem recebem métricas novas retroativamente.

## Reexportar no computador

No ambiente PyTorch do projeto, instale `onnxruntime==1.14.1` para `--verify`
(com NumPy 1.x para essa versão do runtime). A exportação sem `--verify` exige
apenas PyTorch, as dependências da arquitetura e ONNX.

```bash
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model attentiongates_dpu_easy_remaining --verify
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model attentiongates_dpu_only_remaining --verify
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model attentiongates_dpu_easy_remaining --dynamic-batch --verify
python benchmark_arm/exportar_to_onnx/scripts/exportar.py --model attentiongates_dpu_only_remaining --dynamic-batch --verify
```

`--patch-size 512` exporta uma versão separada com sufixo `_512`. Outros modelos
continuam disponíveis: `baseline`, `depth_reduced`, `mobilenet_v2`,
`mobilenet_v3`, `skip`, `resnet34`, `segformer`, `hyperstarcop`, `all`.
`attention_gates` e `psa` exigem checkpoint explícito. Arquitetura customizada:

```bash
python benchmark_arm/exportar_to_onnx/scripts/exportar.py \
  --model Modelos.NovoModelo:MinhaClasse --checkpoint /caminho/pesos.pth \
  --output /caminho/novo.onnx --patch-size 128 --verify
```

Fontes compartilhadas de dataset, pós-processamento e potência foram adaptadas
do benchmark ZCU104 para manter o ARM autocontido ao copiar a pasta.

## Verificações

```bash
g++ -std=c++17 -O2 tests/test_core.cpp codigos_c/power.cpp -pthread -o /tmp/test_arm_core
/tmp/test_arm_core /tmp/arm_core_fixtures
python tests/test_integracao.py --binary codigos_c/benchmark_arm
```

O teste de integração usa ONNX/TIFFs sintéticos e referência independente
PyTorch/Kornia/sklearn. Cobre recortes, ordem de bandas, patch 128/512,
batch 1/3/16, grupos divergentes de `has_plume`, abertura, AP/PR, merge de
35 imagens, hashes, schemas CSV, datasets sem positivos e proteção das saídas.
Os ONNX e o pacote de entrega ficam localmente, ignorados pelo Git; os
manifestos JSON acompanham a exportação.
