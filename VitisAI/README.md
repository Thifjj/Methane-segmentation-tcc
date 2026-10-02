# Vitis AI — modelos oficiais

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

Os artefatos dos dois AttentionGates ficam em `build/vitis_ai/official/`:

- `quantize/<modelo>/`: configuracao INT8, bias correction, manifesto e faixas.
- `evaluation/`: resultados por imagem e resumo da avaliacao dos dois modelos.
- `summary.csv`: comparacao FP32/INT8 dos modelos oficiais nas 342 imagens.
- `official.json`: criterio de selecao, configuracoes e metricas oficiais.
- `compiled_zcu104/<modelo>/<modelo>.xmodel`: modelos oficiais compilados.
- `inspect/`: inspecoes disponiveis dos dois checkpoints.
- Logs de calibracao dos modelos escolhidos.

Os checkpoints FP32 ficam em `../Modelos_treinados/`. `common.py` registra
somente os dois modelos oficiais e suas configuracoes em `OFFICIAL_CALIBRATION`.

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
reproduzem a configuracao oficial; a saida e `official/quantize/<modelo>`.
Para inspecionar, use `inspect_model.py --model MODELO --target DPUCZDX8G_ISA1_B4096`.
Para compilar o XModel exportado, use `compile_xmodel.py --xmodel ARQUIVO \
--arch ARCH_JSON --name NOME`, informando o arch.json do bitstream da placa.

## Compilacao oficial concluida

Os dois XModels em `official/compiled_zcu104/` foram compilados para
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
Os scripts atuais continuam configurados para os dois AttentionGates.
Os comandos de cópia e execução dos oito modelos estão no
[README do benchmark](../benchmark_zcu104/codigos_c/README.md).
