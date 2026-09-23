# Comandos na ZCU104

No computador, a partir da raiz do repositório:

```bash
scp -O -r benchmark_zcu104/codigos_c root@192.168.2.100:/home/root/thiago/benchmark/
```

Na placa:

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
./self_test_support
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/dataset_starcop
./sweep_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/dataset_starcop
```

Para usar o test set (`test.csv`):

```bash
./benchmark_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/STARCOP_test
./sweep_vitis --model /home/root/thiago/benchmark/modelos/mobilenet_v3/methane_mobilenet_v3.xmodel --dataset /home/root/thiago/STARCOP_test
```

Para todos os modelos, em sequência:

```bash
./run_all_models.sh --models-dir /home/root/thiago/benchmark/modelos --dataset /home/root/thiago/dataset_starcop
```

Para rodar todos os modelos no test set, troque o dataset:

```bash
./run_all_models.sh --models-dir /home/root/thiago/benchmark/modelos --dataset /home/root/thiago/STARCOP_test
```

Os programas escolhem `test.csv` ou `train.csv` dentro do dataset quando há
apenas um deles. Se ambos existirem, defina `NOME_CSV` em `benchmark_vitis.cpp`
antes de compilar. A saída
fica em `resultados_zcu104/` abaixo do diretório de execução. Para copiar os
resultados ao computador:

```bash
scp -O -r root@192.168.2.100:/home/root/thiago/benchmark/codigos_c/resultados_zcu104 .
```

Edite os valores de execução em `pipeline.hpp` e `benchmark_vitis.cpp`, e os
valores da busca no início de `sweep.cpp`. O README explica cada métrica.
