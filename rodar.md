# Rodar os quatro modelos na ZCU104

Os comandos abaixo usam o benchmark existente em `benchmark_zcu104/codigos_c` e executam cada modelo uma vez no `STARCOP_test` (342 imagens). Os resultados são salvos em `resultados_zcu104/` dentro da pasta dos códigos.

## 1. Enviar os fontes e XModels

Execute no computador, na raiz do projeto:

```bash
# Envia o benchmark atualizado, incluindo xmodel_runner.cpp.
scp -O -r benchmark_zcu104/codigos_c root@192.168.2.100:/home/root/thiago/benchmark/

# Envia os quatro XModels compilados para a ZCU104.
for MODELO in \
  attentiongates_dpu_easy_remaining \
  attentiongates_dpu_only_remaining \
  mobilenet_v3_focaldice_dpu_512 \
  mobilenet_v3_dpu_512; do
  scp -O -r "VitisAI/build/vitis_ai/compiled_zcu104/$MODELO" \
    "root@192.168.2.100:/home/root/thiago/benchmark/modelos/" || break
done
```

## 2. Executar na ZCU104

No terminal da placa, recompile o benchmark para garantir que o executável usa os fontes recém-enviados:

```bash
cd /home/root/thiago/benchmark/codigos_c
./build_zcu104.sh
./self_test_support
```

Depois rode os quatro modelos no test set:

```bash
MODELS=/home/root/thiago/benchmark/modelos
DATASET=/home/root/thiago/STARCOP_test

for MODELO in \
  attentiongates_dpu_easy_remaining \
  attentiongates_dpu_only_remaining \
  mobilenet_v3_focaldice_dpu_512 \
  mobilenet_v3_dpu_512; do
  ./benchmark_vitis \
    --model "$MODELS/$MODELO/$MODELO.xmodel" \
    --dataset "$DATASET" || break
done
```

O benchmark executa `model_only`, `end_to_end` e a validação do conjunto. Para usar o dataset full em vez do test, altere somente `DATASET` para:

```bash
DATASET=/home/root/thiago/dataset_starcop
```
