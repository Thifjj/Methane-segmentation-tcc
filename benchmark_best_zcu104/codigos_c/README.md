# Benchmark modular dos XModels na ZCU104

Este diretório separa leitura, pré-processamento, runner, pós-processamento,
métricas, validação, pipeline, energia e escrita dos resultados. O
`benchmark_vitis.cpp` conecta esses componentes; `sweep_vitis` procura a
configuração de melhor desempenho.

## Execução de `benchmark_vitis`

```text
benchmark_vitis --model ARQUIVO.xmodel --dataset PASTA [--csv ARQUIVO.csv]
  --out PASTA [--run-id ID] [--mode model_only|end_to_end|all] [--samples N]
  --runners 1..4 --cpu-cores 1..4 --pre-workers 1..4 --post-workers 1..4
  --slots 1..4 --iterations N --warmup N --pin|--no-pin
  --power|--no-power [--power-sample-ms N] --validate|--no-validate
```

`--iterations 0` usa uma inferência por amostra do CSV. `--samples 0` usa
todas as amostras. Por padrão, os dois modos rodam e a validação percorre
todas as amostras separadamente, fora das regiões temporizadas. `--cpu-cores`
também aceita o nome `--threads`.
O terminal mostra o progresso `concluídas/total` em cada modo e na validação,
atualizado aproximadamente a cada segundo.

## Compilação na placa

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
./self_test_support
```

É necessário OpenCV, VART, XIR e GraphRunner da imagem da placa. O build
também pode ser feito com CMake. Na placa, o autoteste confere a quantização
NEON contra a referência escalar, além do pós-processamento e das métricas.

## Busca de desempenho

```bash
./sweep_vitis --model /home/root/thiago/benchmark/modelos/baseline/methane_baseline.xmodel \
  --dataset /home/root/thiago/dataset_starcop \
  --csv /home/root/thiago/dataset_starcop/train.csv \
  --out /home/root/resultados/baseline --resume
```

O sweep testa runners 1–4 e núcleos de CPU 1–4 para `model_only` e
`end_to_end`. Ajusta workers de pré e pós-processamento e 1–4 slots dos três
melhores candidatos end-to-end, confirma os finalistas com e sem afinidade e executa o
dataset completo três vezes em cada modo. Os resultados ficam em `runs/` e
`final/`, com `ranking_search.csv` e `best_config.txt` na pasta de saída.

Na MobileNet V2 a placa já travou com três runners; use `--max-runners 2` até
decidir testar 3 e 4 com acesso para reiniciá-la. `--resume` reutiliza runs
válidos e preserva pastas de execuções interrompidas.

## Métricas e interpretação

`model_only` mede `execute_async + wait` com entradas já preparadas.
`end_to_end` mede leitura dos TIFFs, pré-processamento, filas, sincronização,
inferência e pós-processamento. Nas MobileNets o tempo do runner inclui
operações de CPU dentro do grafo. Throughput é inferências concluídas divididas
pelo tempo real; latência é calculada por imagem. Potência é registrada por
trilho, sem somar sensores que podem se sobrepor.

A leitura dos TIFFs usa `cv::imread` e conversão para `CV_32F`. A quantização
usa NEON na ZCU104 e o caminho escalar nas outras arquiteturas. Na execução
com NEON e leitura direta via libtiff, o pré-processamento médio caiu de
43,68 para 13,10 ms e a latência E2E mediana caiu de 517,58 para 503,01 ms.
O FPS total caiu por uma pausa de cerca de 251 s vista simultaneamente na
leitura e na inferência; a causa não foi determinada. A leitura direta via
libtiff foi removida. A combinação atual OpenCV + NEON ainda precisa ser
medida na placa.

A máscara usa `logit >= 0`, equivalente ao limiar do `benchmark_manual`.
O `Teste_Unet.py` usa abertura morfológica, portanto suas métricas não são
diretamente equivalentes. AUPRC usa os 256 níveis INT8: é exata para saída
INT8 e agrupada nesses níveis para saída FLOAT32.
