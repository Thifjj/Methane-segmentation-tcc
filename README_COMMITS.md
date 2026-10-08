# Histórico de modificações do projeto

Este é um registro histórico dos commits até 05/10/2026. Para o fluxo e os resultados atuais, consulte o [README principal](README.md).

Histórico do Git, do commit mais recente para o mais antigo. Os identificadores abaixo são os hashes reais dos commits.

## Commit cd0b856 — 2026-10-05

Mensagem: `Adding new mobilinet_v3_dpu to work 4 dpu compiled to zcu104`

Modificações:

* Generaliza o carregamento de checkpoints e arquiteturas da pasta `Modelos` para o fluxo do Vitis AI.
* Adiciona a configuração de calibração da MobileNet V3 DPU em imagens completas de 512×512.
* Registra F1 global de 0,516919 em FP32 e 0,512693 em INT8 nas 342 imagens de teste, com INT8 simulado no Vitis AI.
* Exporta e compila a versão recalibrada para a ZCU104, com um único subgrafo DPU para a rede.
* Organiza quantização, compilados, avaliações, inspeções e logs por modelo; remove as divisões `official` e `full512`.
* Atualiza caminhos, manifestos, documentação e testes de carregamento e configuração.
* Adiciona este README com o histórico de commits.

## Commit 03d7029 — 2026-10-02

Mensagem: `new dpu model`

Modificações:

* Adiciona os artefatos de quantização, avaliação e compilação dos dois AttentionGates DPU.
* Implementa calibração por camada e avaliação FP32/INT8 no Vitis AI.
* Atualiza benchmarks, métricas, pós-processamento e comparativos da ZCU104.

## Commit 0578803 — 2026-10-01

Mensagem: `add dpu version of attgate`

Modificações:

* Adiciona a arquitetura MobileNet V3 com AttentionGates adaptada para DPU.
* Inclui os checkpoints easy remaining e only remaining.
* Atualiza treinamento, testes e cálculo de AUPRC.

## Commit 92c21bc — 2026-10-01

Mensagem: `treinamento full dataset  validacao test set`

Modificações:

* Ajusta o treinamento para o dataset completo e a validação no conjunto de teste.

## Commit e53e953 — 2026-09-30

Mensagem: `add new trainconfig from joao`

Modificações:

* Atualiza a configuração de treinamento e o notebook principal.
* Ajusta MobileNet V3 com AttentionGates e adiciona um checkpoint treinado.
* Documenta incompatibilidades com o Vitis AI.

## Commit 048f275 — 2026-09-29

Mensagem: `adding notebook and train modifications from joao_github`

Modificações:

* Adiciona a arquitetura MobileNet V3 com AttentionGates e a FocalDiceLoss.
* Atualiza treinamento, carregamento dos dados, testes e notebook.

## Commit 9e08d63 — 2026-09-29

Mensagem: `adding results and refining benchmarks`

Modificações:

* Adiciona AttentionGates, PSA, ResNet34 e SegFormer às arquiteturas.
* Inclui benchmark ARM em C++ e exportação para ONNX.
* Atualiza resultados e comparativos entre CPU, GPU e ZCU104.

## Commit 562b01f — 2026-09-28

Mensagem: `adding new resultados`

Modificações:

* Atualiza o histórico de testes e os resultados da ZCU104.
* Atualiza a documentação dos benchmarks e do Vitis AI.

## Commit e239536 — 2026-09-23

Mensagem: `adding hyperstarcop_xmodel`

Modificações:

* Integra HyperSTARCOP ao carregamento e ao fluxo do Vitis AI.
* Adiciona artefatos XModel do HyperSTARCOP e script de execução.

## Commit c565df3 — 2026-09-23

Mensagem: `fixing md`

Modificações:

* Atualiza os guias e a documentação da ZCU104.
* Atualiza arquivos do benchmark C++ e instruções de execução.

## Commit b2f880c — 2026-09-23

Mensagem: `modified some file name`

Modificações:

* Reorganiza nomes e caminhos dos arquivos de benchmark.
* Inclui a reorganização dos resultados CPU/GPU.

## Commit fb86eef — 2026-09-23

Mensagem: `adding zcu104_testset_benchmark`

Modificações:

* Atualiza o benchmark e o sweep da ZCU104 para o conjunto de teste.
* Adiciona resultados CUDA e comparativo de métricas entre dispositivos.

## Commit 6b37e1d — 2026-09-22

Mensagem: `Adding fixes all benchmarks`

Modificações:

* Integra a arquitetura e o checkpoint oficial do HyperSTARCOP.
* Corrige componentes dos benchmarks, incluindo métricas, dados e pós-processamento.

## Commit d202bf0 — 2026-09-21

Mensagem: `adding cpu_results and some non complete zcu104 dpu results`

Modificações:

* Adiciona resultados de CPU e resultados parciais da DPU na ZCU104.
* Atualiza os benchmarks e a documentação de resultados.

## Commit b9abb5d — 2026-09-17

Mensagem: `adding cpu results`

Modificações:

* Adiciona resultados de CPU para baseline e MobileNet V2.

## Commit 1025c3f — 2026-09-17

Mensagem: `Investigating dpu execution`

Modificações:

* Adiciona componentes do benchmark C++ para investigar a execução na DPU.
* Inclui módulos de pipeline, métricas, pré-processamento e medição de energia.

## Commit d5b5cc9 — 2026-09-16

Mensagem: `adding new benchmark`

Modificações:

* Adiciona benchmark em CPU/GPU com carregamento de modelos, métricas e processamento dos dados.
* Registra resultados da MobileNet V3.

## Commit af8d4f0 — 2026-09-14

Mensagem: `adding results from zcu104`

Modificações:

* Adiciona resultados de benchmark da ZCU104.
* Inclui guias de execução e documentação dos resultados CPU/ZCU104.

## Commit 30f763d — 2026-09-14

Mensagem: `Adding the benchmark file for ZCU104`

Modificações:

* Adiciona o benchmark C++ inicial da ZCU104.
* Inclui scripts de compilação, execução dos modelos e sweep de configurações.

## Commit 628c4b0 — 2026-09-10

Mensagem: `add xmodels`

Modificações:

* Adiciona XModels compilados para baseline, depth reduced, skip connections e MobileNet V2/V3.
* Atualiza calibração, compilação, carregamento dos dados e instruções do container.

## Commit ff79e93 — 2026-09-10

Mensagem: `add quantized models`

Modificações:

* Adiciona modelos quantizados INT8, configurações de quantização e correção de bias.
* Registra os artefatos de baseline, depth reduced, skip connections e MobileNet V2/V3.

## Commit fa8ce97 — 2026-09-09

Mensagem: `Store PyTorch models with Git LFS`

Modificações:

* Armazena os modelos PyTorch usando Git LFS.

## Commit ec3dbcc — 2026-09-09

Mensagem: `Configure Git LFS for PyTorch models`

Modificações:

* Configura o Git LFS para os arquivos de modelos PyTorch.

## Commit 6ccb3c1 — 2026-09-09

Mensagem: `Add VitisAI files/scripts and config`

Modificações:

* Adiciona scripts de inspeção, quantização e compilação do Vitis AI.
* Inclui relatórios de inspeção e notebook do pipeline.
* Atualiza exports de modelos, pesos e histórico de testes.

## Commit 68b3b13 — 2026-09-04

Mensagem: `Merge joao/main into main`

Modificações:

* Integra a branch joao/main à branch main.

## Commit 44e3e25 — 2026-09-04

Mensagem: `save_state`

Modificações:

* Salva ajustes nas arquiteturas baseline e depth reduced.
* Atualiza exports, notebook e regras do Git.

## Commit 72d64a5 — 2026-09-04

Mensagem: `Pesos dos modelos`

Modificações:

* Adiciona pesos treinados de MobileNet V2/V3 e UNet com skip connections.

## Commit b903042 — 2026-09-04

Mensagem: `adicionado algun modelos`

Modificações:

* Adiciona arquiteturas MobileNet V2/V3 e UNet com skip connections.
* Atualiza a avaliação dos modelos e a configuração de arquivos.

## Commit 5129ec2 — 2026-09-03

Mensagem: `add results_test`

Modificações:

* Adiciona resultados e relatório de teste da UNet depth reduced.
* Ajusta o nome do checkpoint, os testes e o notebook.

## Commit a6fc593 — 2026-09-03

Mensagem: `Adding the trained model`

Modificações:

* Adiciona o checkpoint treinado da UNet depth reduced.

## Commit bd8ae56 — 2026-09-03

Mensagem: `Configura treinamento com dataset STARCOP completo`

Modificações:

* Configura o treinamento com o dataset STARCOP completo.
* Adiciona documentação de treinamento e atualiza dependências e notebook.

## Commit 2eb4967 — 2026-09-01

Mensagem: `Adicionado primeira otimização da UNET e consertado alguns erros no codigo`

Modificações:

* Adiciona a primeira otimização da UNet: a arquitetura depth reduced.
* Reorganiza a arquitetura baseline e corrige testes e notebook.

## Commit 9a899f1 — 2026-09-01

Mensagem: `Ajustes gitfiles`

Modificações:

* Ajusta o .gitignore e cria marcadores para as pastas de dados, pesos e resultados.

## Commit 8f40dfe — 2026-09-01

Mensagem: `Initial commit`

Modificações:

* Cria a estrutura inicial do projeto com UNet, treinamento, testes e carregamento dos dados.
* Inclui o notebook principal e utilitários.
