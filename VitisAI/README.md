# Vitis AI — quantizacao e compilacao dos modelos

Os dois modelos oficiais foram escolhidos pelo maior F1 global INT8 nas
342 imagens de `STARCOP_test/test.csv`. A avaliacao usa patches 128x128 sem
sobreposicao, logits > 0 e abertura morfologica, com comparacao FP32 equivalente.

| Modelo | Imagens selecionadas | Patches por grupo | Amostras por camada | Refinamento | F1 INT8 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| `attentiongates_dpu_easy_remaining` | 300 | 900 | 2048 | 12 camadas | 0.623353 | 0.560959 |
| `attentiongates_dpu_only_remaining` | 100 | 300 | 512 | desativado | 0.660944 | 0.531214 |

O easy usa 2700 patches provenientes de 298 das 300 imagens selecionadas.
O only usa 900 patches das 100 imagens. A selecao oficial prioriza F1;
no easy, a configuracao escolhida tinha AUPRC inferior a outra configuracao.
Os resultados sao de simulacao INT8 no Vitis AI 3.5, ainda sem validacao na placa.

## Arquivos oficiais

Os artefatos ficam em `build/vitis_ai/`, agrupados por etapa e modelo:

- `quantize/<modelo>/`: configuracao INT8, bias correction, manifesto e faixas.
- `evaluation/<modelo>/`: resultados por imagem, resumo e protocolo da avaliacao.
- `evaluation/summary.csv`: comparacao FP32/INT8 de todos os modelos avaliados.
- `evaluation/attentiongates_selection.json`: criterio de selecao e metricas dos AttentionGates.
- `inspect/`: inspecoes disponiveis dos dois checkpoints.
- `logs/<modelo>/`: logs de calibracao, exportacao, compilacao e avaliacao.

Os modelos compilados ficam em `build/vitis_ai/compiled_zcu104/<modelo>/`,
com os demais compilados ZCU104.

Os checkpoints FP32 ficam em `../Modelos_treinados/`. `common.py` registra
os checkpoints conhecidos e importa as arquiteturas pelo `Modelos/__init__.py`.
`OFFICIAL_CALIBRATION` preserva as configuracoes dos dois modelos oficiais.

## Pre-processamento e calibracao

Canais: `mag1c, TOA_AVIRIS_460nm, TOA_AVIRIS_550nm, TOA_AVIRIS_640nm`.
Usa `STARCOPDataset` e `DataNormalizer` do treinamento: mag1c / 1750,
RGB / 60, clip [0, 2], patches 128x128 com passo 64 na calibracao.
Os grupos forte, fraco e fundo recebem a mesma quantidade de patches.
Os rotulos selecionam patches e nunca entram como entrada do modelo.

A politica `layer_mse` escolhe faixas por camada com clipping amostrado de
ate 0,1%, antes dos ajustes nativos da DPU. No easy, o refinamento testa
posicoes vizinhas nas camadas de maior erro e aceita reducoes do erro dos
logits frente ao FP32 em 24 patches de calibracao. Preserva as restricoes
nativas e as faixas dos parametros. O test nao participa do refinamento.
O limite de clipping e uma estimativa; ajustes obrigatorios da DPU podem
excede-lo. `layer_ranges.json` e `refinement.json` registram os resultados.

O manifesto v2 confere hashes SHA-256 dos pesos e artefatos, canais,
normalizacao, dimensoes, patching e target antes de avaliar ou exportar.
Diretorios ja calibrados sao protegidos contra sobrescrita.

## Uso no container

Ambiente: Vitis AI 3.5, `conda activate vitis-ai-pytorch`, diretorio
`/workspace/VitisAI`. Dependencias adicionais usadas: rasterio 1.3.11 e
kornia 0.6.12. O dataset de calibracao fica em `/dataset_STARCOP`.

Para avaliar os modelos oficiais:

```bash
python evaluate_quantized.py --dataset test
```

Para reproduzir calibracao e avaliacao com as configuracoes oficiais,
use um diretorio novo:

```bash
python run_layer_validation.py --output-dir build/vitis_ai/nova_execucao
```

Para exportar um modelo oficial (batch 1, uma inferencia):

```bash
python quantize_model.py --model attentiongates_dpu_easy_remaining \
  --quant-mode test --deploy --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP
```

Repita com `attentiongates_dpu_only_remaining`. Os padroes por modelo ja
reproduzem a configuracao oficial; a saida e `quantize/<modelo>`.
Para inspecionar, use `inspect_model.py --model MODELO --target DPUCZDX8G_ISA1_B4096`.
Para compilar o XModel exportado, use `compile_xmodel.py --xmodel ARQUIVO \
--arch ARCH_JSON --name NOME`, informando o arch.json do bitstream da placa.

## Compilacao oficial concluida

Os dois XModels em `build/vitis_ai/compiled_zcu104/` foram compilados para
`DPUCZDX8G_ISA1_B4096`, usando o arch.json da ZCU104 do Vitis AI 3.5.
Cada modelo tem um subgrafo DPU com 324 operacoes. A interface DPU e INT8
NHWC: entrada `[1, 128, 128, 4]` com fix_point 5 e saida
`[1, 128, 128, 1]` com fix_point 1. O grafo inclui a conversao final
`fix2float` na CPU. `compilation_manifest.json` registra interfaces e hashes;
o arch.json utilizado tambem foi copiado para esse diretorio.
A compilacao foi conferida com XIR; a execucao na placa ainda precisa ser validada.


## Artefatos anteriores restaurados

Os modelos `baseline`, `depth_reduced`, `skip_connections`, `mobilenet_v2`,
`mobilenet_v3` e `hyperstarcop` foram restaurados do Git/Git LFS:

- `build/vitis_ai/compiled_zcu104/<modelo>/`: compilado ZCU104, meta e MD5.
- `build/vitis_ai/quantize/<modelo>/`: XModel quantizado, configuração e bias.
- `build/vitis_ai/restoration_manifest.json`: commit, tamanhos e SHA-256.

Foram conferidos os hashes, o target ISA1 B4096 e a abertura dos 12 XModels
no XIR. São os arquivos históricos, sem recalibração ou recompilação.
Os scripts atuais tambem aceitam os demais modelos exportados por `Modelos`.
Os comandos de cópia e execução dos oito modelos estão no
[README do benchmark](../benchmark_zcu104/codigos_c/README.md).


## Usar outros checkpoints

`quantize_model.py`, `inspect_model.py`, `evaluate_quantized.py` e
`run_layer_validation.py` aceitam `--checkpoint CAMINHO`. Para os nomes de
checkpoints existentes em `Modelos_treinados`, a arquitetura e identificada
pelo nome do arquivo e importada sob demanda de `Modelos`. Tambem e possivel
usar `--model` com um alias de `common.MODEL_REGISTRY` ou uma classe exportada.
Para um arquivo renomeado ou uma arquitetura nova, informe
`--architecture NOME_DA_CLASSE`, exportada pelo `Modelos/__init__.py`.
Os pesos sao carregados com `strict=True`; arquiteturas incompativeis falham
antes da quantizacao. HyperSTARCOP usa seu carregador especifico.

Os novos modelos recebem os mesmos padroes do AttentionGates easy:
300 imagens, 900 patches por grupo, 2048 amostras por camada, refinamento de
12 camadas, clipping 0,1%, seed 12345, batch 1, dois workers, patches 128x128
e target `DPUCZDX8G_ISA1_B4096`. O perfil only preserva seus padroes originais.
A normalizacao e a ordem dos canais continuam as documentadas acima.
O nome do checkpoint identifica a arquitetura; os pesos `.pth` sozinhos nao
codificam configuracoes como `align_corners`. A variante `mobilenet_v3_dpu`
usa `UNetMobileNetV3_dpu`, com `align_corners=False`.

Dentro do container, em `/workspace/VitisAI` com `vitis-ai-pytorch` ativo:

```bash
python quantize_model.py \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode calib --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP

python quantize_model.py \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode test --deploy --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP

python compile_xmodel.py \
  --xmodel build/vitis_ai/quantize/mobilenet_v3_dpu/UNetMobileNetV3_dpu_int.xmodel \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output-dir build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu \
  --name mobilenet_v3_dpu
```

Para outro checkpoint, troque o caminho em `--checkpoint`. Os artefatos sao
separados por modelo; uma calibracao existente continua protegida contra
sobrescrita. O compilador aceita qualquer XModel exportado, sujeito aos
operadores suportados pelo Vitis AI e pela DPU; aceitar uma arquitetura no
carregador nao garante que todas as suas operacoes sejam executadas na placa.

### MobileNet V3 DPU: execucao concluida

Checkpoint: `Mobile_Net_v3_dpu_mag1c_rgb.pth`; arquitetura:
`Modelos.UNetMobileNetV3_dpu`. A calibracao usou 300 imagens e 2700 patches
equilibrados, com todos os padroes acima. Foram exportados:

- `build/vitis_ai/quantize/mobilenet_v3_dpu/UNetMobileNetV3_dpu_int.xmodel`.
- `build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu/mobilenet_v3_dpu.xmodel`.

A verificacao XIR confirmou um subgrafo DPU com 276 operacoes, entrada INT8
NHWC `[1,128,128,4]` com fix_point 5 e saida INT8 `[1,128,128,1]` com fix_point 2.
Existe uma conversao final `fix2float` na CPU. O diretorio compilado contem
`compilation_manifest.json` com hashes e interfaces e uma copia do `arch.json`.
Logs: `build/vitis_ai/logs/mobilenet_v3_dpu/{calib,export,compile}.log`.
A comparacao de qualidade FP32/INT8 foi concluida nas 342 imagens do test set:
[relatorio](build/vitis_ai/evaluation/mobilenet_v3_dpu/comparacao_fp32_int8.md).
F1 global: FP32 imagem inteira 0,516919; FP32 patches 128x128 0,114279;
INT8 nos mesmos patches 0,118082. O INT8 foi simulado em CPU pelo Vitis AI 3.5.
A execucao fisica na placa ainda nao foi validada.

Para avaliar com o conjunto de teste montado em outro caminho, use
`evaluate_quantized.py --checkpoint CAMINHO --dataset test --csv /dataset_test/test.csv
--data-root /dataset_test --output-dir DIRETORIO_NOVO`.

Testes do carregador e do perfil JSON: `python -m unittest discover -s tests -v`,
a partir de `VitisAI/`.

## Calibracao da MobileNet V3 DPU em imagens completas

O perfil [`configs/mobilenet_v3_dpu_512.json`](configs/mobilenet_v3_dpu_512.json)
usa entrada de 512x512, sem dividir a imagem em patches. Isso preserva o
contexto espacial da inferencia FP32 de referencia. A selecao de 300 imagens
do treinamento e aleatoria e reproduzivel (seed 12345), com batch 1,
faixas por camada, clipping de 0,1% e refinamento de 12 camadas.
Argumentos CLI tem prioridade sobre o JSON. A quantizacao fica em
`build/vitis_ai/quantize/mobilenet_v3_dpu_512/` e a avaliacao em
`build/vitis_ai/evaluation/mobilenet_v3_dpu_512/`.
O compilado fica em `build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu_512/`,
separado da versao de 128x128 em `compiled_zcu104/mobilenet_v3_dpu/`.

No container com os dois datasets montados, execute em `/workspace/VitisAI`:

```bash
python quantize_model.py --config configs/mobilenet_v3_dpu_512.json \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode calib --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP

python evaluate_quantized.py --model mobilenet_v3_dpu_512 \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --dataset test --csv /dataset_test/test.csv --data-root /dataset_test \
  --quant-dir build/vitis_ai/quantize \
  --output-dir build/vitis_ai/evaluation

python quantize_model.py --config configs/mobilenet_v3_dpu_512.json \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode test --deploy --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP

python compile_xmodel.py \
  --xmodel build/vitis_ai/quantize/mobilenet_v3_dpu_512/UNetMobileNetV3_dpu_int.xmodel \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output-dir build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu_512 \
  --name mobilenet_v3_dpu_512
```

O avaliador le as dimensoes no manifesto e compara FP32 e INT8 na imagem
inteira para este perfil. O limiar `logit > 0` e a abertura morfologica sao
os mesmos em ambos. Os dados de teste nao entram na calibracao nem no
refinamento de faixas.

### Resultado aceito em 512x512

Nas mesmas 342 imagens de teste, o F1 global foi **0,516919 em FP32** e
**0,512693 em INT8**, com perda de **0,4226 ponto percentual**, abaixo do
criterio de 1 ponto. A referencia FP32 reproduziu as contagens do teste
anterior. Os 300 IDs de calibracao nao aparecem no conjunto de teste.
AUPRC: 0,745439 / 0,722560; F1 fraco: 0,562549 / 0,518384 (FP32 / INT8).

O arquivo `build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu_512/mobilenet_v3_dpu_512.xmodel`
foi compilado para a ZCU104. XIR confirmou um unico subgrafo DPU com 276
operacoes e entrada INT8 NHWC `[1,512,512,4]`; a CPU faz apenas `fix2float`
na saida. O runner deve usar a entrada 512x512 deste artefato.

[Relatorio e protocolo](build/vitis_ai/evaluation/mobilenet_v3_dpu_512/comparacao_fp32_int8.md).
Os resultados INT8 sao de simulacao Vitis AI 3.5 na CPU; a execucao fisica
na placa ainda nao foi validada.
