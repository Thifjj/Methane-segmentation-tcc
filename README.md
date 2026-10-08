# Methane Segmentation

Este repositório reúne treinamento, quantização Vitis AI e benchmarks de segmentação de metano no STARCOP. Os resultados atuais do artigo usam 342 imagens de `STARCOP_test/test.csv`, com resolução 512×512.

## Onde encontrar cada etapa

- [Benchmark PyTorch CPU e GPU](benchmark_nao_embarcado/README.md): execução FP32, métricas e energia RAPL/NVML.
- [Benchmark ARM](benchmark_arm/README.md): ONNX Runtime FP32 no Cortex-A53 da ZCU104.
- [Benchmark DPU](benchmark_zcu104/codigos_c/README.md): XModels INT8, VART e medições na ZCU104.
- [Vitis AI](VitisAI/README.md): calibração, quantização e compilação.
- [Guia da ZCU104](GUIA_PROJETO_ZCU104.md): caminhos e sequência de execução.
- [Comparativo histórico CPU](benchmark_nao_embarcado/comparativo_benchmarks_starcop.md): coletas de setembro, fora da seleção atual do artigo.

## Seleção atual do artigo

As cinco variantes com resultados GPU CUDA atuais são `attentiongates_dpu_bce_artigo`, `attentiongates_dpu_focaldice_artigo`, `mobilenet_v3_dpu_bce_artigo`, `mobilenet_v3_dpu_focaldice_artigo` e `hyperstarcop`. Cada pasta concluída em `benchmark_nao_embarcado/resultados_artigo/` contém 342 medições `model_only`, 342 `end_to_end` e validação separada. A soma de energia CPU+GPU inclui apenas o pacote CPU medido por RAPL e a GPU medida por NVML; não equivale ao consumo do computador na tomada.

Os resultados ARM e DPU usam protocolos e dispositivos diferentes. Confira checkpoint, ordem de canais, precisão e definição de FPS antes de comparar números entre plataformas.
