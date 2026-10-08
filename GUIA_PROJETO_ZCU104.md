# Guia do fluxo ZCU104

Este guia aponta para as instruções mantidas em cada componente. A seleção atual do artigo tem cinco XModels INT8 em `VitisAI/build/vitis_ai/compiled_artigo/` e resultados físicos em `benchmark_zcu104/resultados_zcu104/`.

1. Confira os checkpoints em `Modelos_treinados/` e a arquitetura correspondente.
2. Para calibração, quantização e compilação, siga o [README do Vitis AI](VitisAI/README.md). O perfil atual do artigo usa 100 imagens de treino, seed 12345, sem amostras do test set na calibração.
3. Envie os XModels e os fontes de `benchmark_zcu104/codigos_c/` para a placa.
4. Compile e execute conforme o [README do benchmark ZCU104](benchmark_zcu104/codigos_c/README.md).

Na placa, o test set costuma ficar em `/home/root/thiago/STARCOP_test/` e os modelos em `/home/root/thiago/benchmark/modelos/`. Confirme os caminhos reais antes de executar. O benchmark mede `model_only`, `end_to_end` e a validação das 342 imagens.

## Comparação entre dispositivos

A DPU usa XModel INT8 com pipeline concorrente. O [benchmark ARM](benchmark_arm/README.md) usa ONNX Runtime FP32, e o [benchmark host](benchmark_nao_embarcado/README.md) usa PyTorch FP32 em CPU ou GPU. Compare FPS, latência e energia apenas com o modo e o domínio do sensor identificados. A soma RAPL+NVML do host cobre CPU e GPU monitoradas; não é consumo total na tomada.
