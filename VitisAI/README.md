# Vitis AI — quantizacao e compilacao dos modelos

## Seleção atual do artigo

Os cinco XModels da seleção atual estão em `build/vitis_ai/compiled_artigo/`: AttentionGates DPU BCE e Focal Dice, MobileNetV3 DPU BCE e Focal Dice, e HyperSTARCOP. Os manifestos de quantização atuais ficam em `build/vitis_ai/quantize_artigo/`. As seções sobre `easy_remaining`, `only_remaining` e `compiled_zcu104/` abaixo documentam o fluxo anterior e não substituem os resultados físicos atuais em `../benchmark_zcu104/resultados_zcu104/`. O host GPU FP32 usa checkpoints PyTorch e resultados em `../benchmark_nao_embarcado/resultados_artigo/`.

Os AttentionGates easy_remaining e only_remaining usam entrada 512x512, sem
dividir imagens em patches. A avaliacao compara FP32 e INT8 nas mesmas 342
imagens de `STARCOP_test/test.csv`, com logits > 0 e abertura morfologica.

| Modelo | F1 FP32 | F1 INT8 | AUPRC FP32 | AUPRC INT8 |
|---|---:|---:|---:|---:|
| `attentiongates_dpu_easy_remaining` | 0.647798 | 0.619683 | 0.514058 | 0.436448 |
| `attentiongates_dpu_only_remaining` | 0.676553 | 0.652105 | 0.610146 | 0.473214 |

Os resultados desta tabela são simulações INT8 históricas do Vitis AI 3.5 em CPU.
Os XModels 512×512 foram compilados para `DPUCZDX8G_ISA1_B4096`.
As coletas físicas atuais do artigo estão em `../benchmark_zcu104/resultados_zcu104/`;
os manifestos guardam os perfis usados em cada rodada.

## Arquivos oficiais

Os artefatos ficam em `build/vitis_ai/`, agrupados por etapa e modelo:

- `quantize/<modelo>/`: configuracao INT8, bias correction, manifesto e faixas.
- `evaluation/<modelo>/`: resultados por imagem, resumo e protocolo da avaliacao.
- `evaluation/summary.csv`: comparacao FP32/INT8 de todos os modelos avaliados.
- `evaluation/attentiongates_summary.csv`: resultados FP32/INT8 512x512 dos dois AttentionGates.
- `inspect/`: inspecoes disponiveis dos dois checkpoints.
- `logs/<modelo>/`: logs de calibracao, exportacao, compilacao e avaliacao.

Os modelos compilados ficam em `build/vitis_ai/compiled_zcu104/<modelo>/`,
com os demais compilados ZCU104.

Os checkpoints FP32 ficam em `../Modelos_treinados/`. `common.py` registra
os checkpoints conhecidos e importa as arquiteturas pelo `Modelos/__init__.py`.
`CALIBRATION_PROFILE` fixa um unico perfil para todas as novas calibracoes.

## Pre-processamento e calibracao

Canais: `mag1c, TOA_AVIRIS_460nm, TOA_AVIRIS_550nm, TOA_AVIRIS_640nm`.
Usa `STARCOPDataset` e `DataNormalizer` do treinamento: mag1c / 1750,
RGB / 60, clip [0, 2], imagens completas 512x512 na calibracao.
Os rotulos nao entram como entrada do modelo.

A politica `layer_mse` escolhe faixas por camada com clipping amostrado de
ate 0,1%, antes dos ajustes nativos da DPU. Todas as novas calibracoes usam
100 imagens, 512 ativacoes amostradas por camada/imagem e nenhum refinamento.
O perfil e validado pelo CLI e valores diferentes sao rejeitados. O test nao
participa da calibracao.
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
Cada modelo tem um subgrafo DPU com 324 operacoes e entrada INT8 NHWC
`[1, 512, 512, 4]`. A compilacao foi conferida com XIR; a execucao na placa
esta pendente.


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

Os AttentionGates usam entrada completa 512x512, batch 1, sem refinamento,
clipping maximo estimado de 0,1% e target `DPUCZDX8G_ISA1_B4096`. As rodadas
anteriores usaram perfis diferentes (easy: 300/2048; only: 100/512), conforme
seus manifestos; novas calibracoes de ambos usam o perfil unico 100/512/0.
A normalizacao e a ordem dos canais continuam as documentadas acima.
O nome do checkpoint identifica a arquitetura; os pesos `.pth` sozinhos nao
codificam configuracoes como `align_corners`. A variante `mobilenet_v3_dpu`
usa `UNetMobileNetV3_dpu`, com `align_corners=False`.

Dentro do container, em `/workspace/VitisAI` com `vitis-ai-pytorch` ativo:

```bash
python quantize_model.py \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode calib --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP --subset-len 100 \
  --range-samples 512 --refine-layers 0

python quantize_model.py \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode test --deploy --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP --range-samples 512 --refine-layers 0

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
`Modelos.UNetMobileNetV3_dpu`. Este artefato e historico: foi calibrado com
300 imagens e 2700 patches equilibrados. Novas calibracoes deste alias tambem
usam 100 imagens e 512 amostras por camada/imagem, mantendo a geometria do
modelo selecionado. Foram exportados:

- `build/vitis_ai/quantize/mobilenet_v3_dpu/UNetMobileNetV3_dpu_int.xmodel`.
- `build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu/mobilenet_v3_dpu.xmodel`.

A verificacao XIR confirmou um subgrafo DPU com 276 operacoes, entrada INT8
NHWC `[1,128,128,4]` com fix_point 5 e saida INT8 `[1,128,128,1]` com fix_point 2.
Existe uma conversao final `fix2float` na CPU. O diretorio compilado contem
`compilation_manifest.json` com hashes e interfaces e uma copia do `arch.json`.
Logs: `build/vitis_ai/logs/mobilenet_v3_dpu/{calib,export,compile}.log`.
A comparacao de qualidade FP32/INT8 foi concluida nas 342 imagens do test set:
relatório histórico de avaliação (arquivo não presente nesta árvore de trabalho).
F1 global: FP32 imagem inteira 0,516919; FP32 patches 128x128 0,114279;
INT8 nos mesmos patches 0,118082. O INT8 foi simulado em CPU pelo Vitis AI 3.5.
A execucao fisica na placa ainda nao foi validada.

Para avaliar com o conjunto de teste montado em outro caminho, use
`evaluate_quantized.py --checkpoint CAMINHO --dataset test --csv /dataset_test/test.csv
--data-root /dataset_test --output-dir DIRETORIO_NOVO`.

Testes do carregador e do perfil de calibracao: `python -m unittest discover -s tests -v`,
a partir do container Vitis AI em `VitisAI/`.

## Calibracao da MobileNet V3 DPU em imagens completas

Usa entrada de 512x512, sem dividir a imagem em patches. Isso preserva o
contexto espacial da inferencia FP32 de referencia. Novas calibracoes usam
100 imagens de treinamento selecionadas de forma reproduzivel (seed 12345),
512 ativacoes amostradas por camada/imagem, batch 1, faixas por camada,
clipping de 0,1% e nenhum refinamento. Os argumentos de calibracao de um JSON
ou CLI que contrariem esse perfil sao rejeitados. A quantizacao fica em
`build/vitis_ai/quantize/mobilenet_v3_dpu_512/` e a avaliacao em
`build/vitis_ai/evaluation/mobilenet_v3_dpu_512/`.
O compilado fica em `build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu_512/`.

No container com os dois datasets montados, execute em `/workspace/VitisAI`:

```bash
python quantize_model.py --model mobilenet_v3_dpu_512 \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode calib --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
  --no-patching --subset-len 100 --range-samples 512 --refine-layers 0

python evaluate_quantized.py --model mobilenet_v3_dpu_512 \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --dataset test --csv /dataset_test/test.csv --data-root /dataset_test \
  --quant-dir build/vitis_ai/quantize \
  --output-dir build/vitis_ai/evaluation

python quantize_model.py --model mobilenet_v3_dpu_512 \
  --checkpoint ../Modelos_treinados/Mobile_Net_v3_dpu_mag1c_rgb.pth \
  --quant-mode test --deploy --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
  --no-patching --subset-len 1 --range-samples 512 --refine-layers 0

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

### Resultado historico em 512x512

Nas mesmas 342 imagens de teste, o F1 global foi **0,516919 em FP32** e
**0,512693 em INT8**, com perda de **0,4226 ponto percentual**, abaixo do
criterio de 1 ponto. A referencia FP32 reproduziu as contagens do teste
anterior. Esse artefato foi calibrado com o perfil anterior (300/2048 e
refinamento de 12 camadas), conforme o manifesto; seus resultados nao usam o
perfil padronizado novo. Os IDs de calibracao nao aparecem no conjunto de teste.
AUPRC: 0,745439 / 0,722560; F1 fraco: 0,562549 / 0,518384 (FP32 / INT8).

O arquivo `build/vitis_ai/compiled_zcu104/mobilenet_v3_dpu_512/mobilenet_v3_dpu_512.xmodel`
foi compilado para a ZCU104. XIR confirmou um unico subgrafo DPU com 276
operacoes e entrada INT8 NHWC `[1,512,512,4]`; a CPU faz apenas `fix2float`
na saida. O runner deve usar a entrada 512x512 deste artefato.

Relatório e protocolo históricos (arquivo não presente nesta árvore de trabalho).
Os resultados INT8 sao de simulacao Vitis AI 3.5 na CPU; a execucao fisica
na placa ainda nao foi validada.

## MobileNetV3 Focal Dice — quantizacao direta, 512x512

Este fluxo aplica a quantizacao PTQ normal do Vitis AI ao checkpoint
`MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth`, com a arquitetura
`UNetMobileNetV3_dpu`. Nao faz reconstrucao de camadas nem ajuste posterior.
O perfil de calibracao replica o `attentiongates_dpu_only_remaining`:

- `--subset-len 100`: seleciona 100 imagens do CSV de treinamento, de forma
  reproduzivel com `--seed 12345`. Sao imagens usadas para calibrar, nao para
  medir a qualidade final.
- `--range-samples 512`: para cada camada e imagem de calibracao, amostra ate
  512 ativacoes para estimar erro e clipping e escolher a faixa INT8. Esse
  valor nao e o tamanho da imagem nem a quantidade de imagens.
- `--max-clipping-percent 0.1`: limita a saturacao estimada por camada a 0,1%.
- `--refine-layers 0`: desativa refinamento posterior; usa diretamente as
  configuracoes da ferramenta Vitis AI.
- `--no-patching`: calibra e avalia imagens completas de 512x512.

Os valores 100/512 fazem parte do perfil usado pelo `only_remaining`; mante-los
e essencial para reproduzir essa calibracao e seus resultados. A ordem dos
quatro canais e a normalizacao sao as mesmas do modelo DPU: MAG1C e bandas
460, 550 e 640 nm. A calibracao usa somente `/dataset_STARCOP/train.csv`; o
test set nao participa da escolha das faixas.

No container Vitis AI, em `/workspace/VitisAI`, execute:

```bash
python quantize_model.py \
  --model mobilenet_v3_focaldice_dpu_512 \
  --architecture UNetMobileNetV3_dpu \
  --checkpoint ../Modelos_treinados/MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth \
  --quant-mode calib \
  --csv /dataset_STARCOP/train.csv --data-root /dataset_STARCOP \
  --no-patching --subset-len 100 --range-samples 512 \
  --refine-layers 0 --max-clipping-percent 0.1 --num-workers 2

python evaluate_quantized.py \
  --model mobilenet_v3_focaldice_dpu_512 \
  --architecture UNetMobileNetV3_dpu \
  --checkpoint ../Modelos_treinados/MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth \
  --dataset test --csv /workspace/STARCOP_test/test.csv \
  --data-root /workspace/STARCOP_test --num-threads 2

python quantize_model.py \
  --model mobilenet_v3_focaldice_dpu_512 \
  --architecture UNetMobileNetV3_dpu \
  --checkpoint ../Modelos_treinados/MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth \
  --quant-mode test --deploy --csv /dataset_STARCOP/train.csv \
  --data-root /dataset_STARCOP --no-patching --subset-len 1 \
  --range-samples 512 --refine-layers 0 --num-workers 0
```

Para compilar o XModel para a placa, use um container Vitis AI que inclua
`vai_c_xir` e o `arch.json` correspondente ao bitstream da ZCU104:

```bash
python compile_xmodel.py \
  --xmodel build/vitis_ai/quantize/mobilenet_v3_focaldice_dpu_512/UNetMobileNetV3_dpu_int.xmodel \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output-dir build/vitis_ai/compiled_zcu104/mobilenet_v3_focaldice_dpu_512 \
  --name mobilenet_v3_focaldice_dpu_512
```

A calibracao e o XModel INT8 ficam em
`build/vitis_ai/quantize/mobilenet_v3_focaldice_dpu_512/`. O avaliador grava
as linhas por imagem e o resumo em
`build/vitis_ai/evaluation/mobilenet_v3_focaldice_dpu_512/`.

### Resultado no test set

No test set de 342 imagens, a avaliacao padrao obteve F1 global **0,495363
FP32** e **0,492145 INT8** (queda de **0,32 ponto percentual**). AUPRC:
**0,413686 FP32** e **0,412671 INT8**. A mesma execucao tambem reportou F1
strong **0,485071 / 0,531231** e F1 weak **0,563068 / 0,530190** (FP32 / INT8).
As metricas INT8 sao simulacao do Vitis AI em CPU; nao representam uma medicao
fisica na DPU.

O XModel foi exportado. A compilacao para a ZCU104 ainda precisa ser feita em
um ambiente que tenha `vai_c_xir`; o container usado para quantizar nao dispoe
desse compilador.
