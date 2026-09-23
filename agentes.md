# Contexto do projeto

Este arquivo registra caminhos e instruções atuais para continuar o trabalho.
Antes de editar, confira o estado do repositório e preserve alterações locais do
usuário.

## Estrutura atual

- `Modelos/`, `Modelos_treinados/`, `Utils/`: arquiteturas, pesos e dados.
- `VitisAI/`: inspeção, quantização INT8 e compilação para ZCU104.
- `build/vitis_ai/compiled_zcu104/`: XModels compilados.
- `benchmark_zcu104/codigos_c/`: benchmark atual C++/VART da placa.
- `benchmark_nao_embarcado/`: benchmark PyTorch CPU/GPU e resultados.
- `comparativos/comparativo_test_zcu104_cpu_gpu.csv`: tabela consolidada do
  test set para CPU, GPU e ZCU104.
- `Treinos/README.md`: notas sobre treinamento.

Os cinco modelos usados no benchmark da placa são `baseline`,
`depth_reduced`, `skip_connections`, `mobilenet_v2` e `mobilenet_v3`. O runner
espera quatro canais na ordem mag1c, 640 nm, 550 nm e 460 nm. A entrada divide
mag1c por 1750 e RGB por 60 e limita os valores a [0, 2]. A máscara binária usa
limiar estrito `logit > 0`, equivalente a `sigmoid(logit) > 0.5`.

## Calibração e execução na placa

O README de [`VitisAI/README.md`](VitisAI/README.md) contém o procedimento de
calibração; o padrão documentado para calibração representativa é 1000 imagens
reprodutíveis do dataset full, seed 12345. O guia
[`GUIA_PROJETO_ZCU104.md`](GUIA_PROJETO_ZCU104.md) resume o fluxo.

Na placa, os caminhos usados são:

| Conteúdo | Caminho |
| --- | --- |
| Código | `/home/root/thiago/benchmark/codigos_c/` |
| XModels | `/home/root/thiago/benchmark/modelos/<modelo>/` |
| Test set | `/home/root/thiago/STARCOP_test/` |
| Dataset full | `/home/root/thiago/dataset_starcop/` |
| Saídas | `resultados_zcu104/` no diretório corrente |

Consulte [`benchmark_zcu104/codigos_c/README.md`](benchmark_zcu104/codigos_c/README.md)
para comandos individuais e execução em lote. O script `run_all_models.sh`
recebe apenas `--models-dir` e `--dataset`; o sweep recebe apenas `--model` e
`--dataset`. O CSV é escolhido automaticamente quando a pasta contém um único
`test.csv` ou `train.csv`; se houver ambos, ajuste `NOME_CSV` em
`benchmark_vitis.cpp` e recompile.

## Comparação e interpretação

O comparativo detalhado das execuções CPU está em
[`benchmark_nao_embarcado/comparativo_benchmarks_starcop.md`](benchmark_nao_embarcado/comparativo_benchmarks_starcop.md).
A tabela entre dispositivos está em
[`comparativos/comparativo_test_zcu104_cpu_gpu.csv`](comparativos/comparativo_test_zcu104_cpu_gpu.csv).
CPU/GPU reportam FPS sequencial por latência média; o `throughput_fps` da placa
é medido com pipeline concorrente. Não trate esses valores como medições com
protocolos idênticos.

## Cuidados

- Prefixe todo comando shell com `rtk`, conforme `/home/thiago/.codex/RTK.md`.
- Preserve resultados anteriores e alterações do usuário; verifique `git status`
  antes de editar ou mover arquivos.
- Não atribua diferenças de qualidade apenas à quantização sem verificar o
  checkpoint, as amostras, o pré-processamento e o pós-processamento usados.
