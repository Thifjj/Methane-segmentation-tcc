# Guia do fluxo ZCU104

Este arquivo resume o fluxo atual; os comandos completos ficam nos READMEs
específicos para evitar instruções duplicadas e divergentes.

## Fluxo

1. Treine os modelos PyTorch e mantenha os checkpoints em `Modelos_treinados/`.
2. Inspecione, calibre, exporte e compile os XModels conforme
   [`VitisAI/README.md`](VitisAI/README.md). A calibração reproduzível usa 1000
   imagens do `train.csv` do STARCOP full, com seed 12345.
3. Copie os modelos de `build/vitis_ai/compiled_zcu104/` e o código de
   `benchmark_zcu104/codigos_c/` para a placa.
4. Compile e rode os benchmarks conforme
   [`benchmark_zcu104/codigos_c/README.md`](benchmark_zcu104/codigos_c/README.md).

## Dados na placa

- Test set: `/home/root/thiago/STARCOP_test` (normalmente `test.csv`).
- Dataset full: `/home/root/thiago/dataset_starcop` (normalmente `train.csv`).
- XModels: `/home/root/thiago/benchmark/modelos/<modelo>/methane_<modelo>.xmodel`.
- Código e executáveis: `/home/root/thiago/benchmark/codigos_c/`.

O README do benchmark tem comandos separados para cada dataset e modelo,
além de uma opção para executar todos. Os resultados são separados pelo nome
do dataset e gravados em `resultados_zcu104/` a partir do diretório de execução.

## Comparação CPU/GPU/ZCU104

O benchmark não embarcado está em `benchmark_nao_embarcado/`. A tabela CSV
consolidada do test set está em
[`comparativos/comparativo_test_zcu104_cpu_gpu.csv`](comparativos/comparativo_test_zcu104_cpu_gpu.csv).
Compare `throughput_fps` da placa com FPS sequencial da CPU/GPU com cuidado:
o throughput da placa usa runners concorrentes, enquanto CPU/GPU medem uma
imagem por vez.
