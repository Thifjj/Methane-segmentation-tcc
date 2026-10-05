# Comparativo das métricas — ZCU104

## 1. Escopo e critérios

Foram analisadas **10 execuções locais**: **9 no STARCOP_test (342 imagens)** e **1 no dataset_starcop/train.csv (3425 imagens)**. São oito modelos distintos; HyperSTARCOP tem execuções com 1 e 2 runners. Os IDs das nove execuções de test são os mesmos.

Os valores vêm dos CSVs de cada execução e de config.txt. As contagens por imagem foram somadas e conferidas contra os resumos globais. Os CSVs de origem foram preservados. **N/D** representa valor ausente ou NaN, não zero.

Ambiente registrado: **PetaLinux 2022.2**, kernel **5.15.36-xilinx-v2022.2**, arquitetura **aarch64**, Vitis AI Library/VART/XIR **3.0.0** e OpenCV **4.5.2**. Todas as execuções usam 4 núcleos, 2 workers de pré, 1 de pós, 2 slots por runner e warm-up 10. HyperSTARCOP tem uma execução com 1 runner; as demais usam 2.

Os AttentionGates usam **16 patches 128×128** por imagem 512×512; os modelos anteriores usam entrada 512×512. Latência, FPS e energia por inferência são por **imagem 512×512**, não por patch.

### Protocolos de qualidade

- **Relatórios principais:** máscara sem abertura, F1 strong/weak por has_plume/qplume; AUPRC global trapezoidal onde disponível. Só os AttentionGates identificam esse protocolo explicitamente nos CSVs. Os relatórios antigos não têm os campos de identificação do método.
- **validacao_oficial/:** abertura com cruz 3×3, strong/weak por label/difficulty; AUPRC é a média do AP por imagem positiva. Disponível apenas nos dois AttentionGates.
- AUPRC global e AP médio por imagem são definições distintas; suas diferenças não indicam, por si, mudança de qualidade.

## 2. Qualidade no test — relatórios principais

Tabela ordenada por F1 global. Para HyperSTARCOP, usa-se aqui a execução com **2 runners e AUPRC disponível**; a execução com 1 runner está detalhada nas seções de desempenho e auditoria.

| Modelo | Precision | Recall | F1 global | IoU | AUPRC global registrada | Acurácia pixel (%) |
| --- | --- | --- | --- | --- | --- | --- |
| AttentionGates only remaining | 0.659992 | 0.668684 | 0.664310 | 0.497353 | 0.645172 | 99.8563 |
| MobileNetV2 | 0.600893 | 0.704965 | 0.648782 | 0.480146 | N/D | 99.8377 |
| AttentionGates easy + remaining | 0.669711 | 0.592729 | 0.628873 | 0.458654 | 0.599712 | 99.8512 |
| HyperSTARCOP (2 runners) | 0.316127 | 0.793530 | 0.452133 | 0.292101 | 0.471906 | 99.5910 |
| MobileNetV3 | 0.808336 | 0.310671 | 0.448838 | 0.289356 | N/D | 99.8377 |
| Depth Reduced | 0.639329 | 0.208285 | 0.314206 | 0.186385 | N/D | 99.8067 |
| Baseline | 0.509496 | 0.160979 | 0.244657 | 0.139378 | N/D | 99.7886 |
| Skip Connections | 0.547398 | 0.034256 | 0.064477 | 0.033313 | N/D | 99.7886 |


### Grupos e falsos positivos por pixel

Strong/weak nesta tabela seguem os grupos dos relatórios principais. FPR está em **porcentagem**; quanto menor, melhor.

| Modelo | F1 strong | F1 weak | FPR pixel (%) | FPR sem pluma (%) |
| --- | --- | --- | --- | --- |
| AttentionGates only remaining | 0.794754 | 0.609928 | 0.0734 | 0.0978 |
| MobileNetV2 | 0.811711 | 0.610898 | 0.0998 | 0.1383 |
| AttentionGates easy + remaining | 0.746904 | 0.583165 | 0.0623 | 0.0885 |
| HyperSTARCOP (2 runners) | 0.824950 | 0.536552 | 0.3658 | 0.5663 |
| MobileNetV3 | 0.449203 | 0.456183 | 0.0157 | 0.0020 |
| Depth Reduced | 0.405609 | 0.104536 | 0.0250 | 0.0436 |
| Baseline | 0.308800 | 0.163199 | 0.0330 | 0.0607 |
| Skip Connections | 0.083594 | 0.009536 | 0.0060 | 0.0115 |


### Contagens globais

| Modelo | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| AttentionGates only remaining | 127486 | 65677 | 63166 | 89396919 |
| MobileNetV2 | 134403 | 89269 | 56249 | 89373327 |
| AttentionGates easy + remaining | 113005 | 55732 | 77647 | 89406864 |
| HyperSTARCOP (2 runners) | 151288 | 327279 | 39364 | 89135317 |
| MobileNetV3 | 59230 | 14044 | 131422 | 89448552 |
| Depth Reduced | 39710 | 22402 | 150942 | 89440194 |
| Baseline | 30691 | 29547 | 159961 | 89433049 |
| Skip Connections | 6531 | 5400 | 184121 | 89457196 |


O test tem **190652 pixels positivos em 89653248 pixels (0.2127%)**. Uma máscara sempre vazia já obteria **99.7873% de acurácia**; por isso F1, recall, IoU e falsos positivos são mais informativos que acurácia isolada.

## 3. AttentionGates — protocolo da validação recente

Use estes resultados para comparar com VitisAI/evaluate_quantized.py e com o histórico do mesmo dataset. O grupo forte e a definição de AUPRC diferem da seção 2.

| Modelo | Precision | Recall | F1 | IoU | AP médio | F1 strong | F1 weak | FPR sem pluma (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AttentionGates only remaining | 0.664063 | 0.657853 | 0.660944 | 0.493589 | 0.531214 | 0.791310 | 0.579119 | 0.0911 |
| AttentionGates easy + remaining | 0.672809 | 0.580671 | 0.623353 | 0.452806 | 0.560959 | 0.741348 | 0.553665 | 0.0831 |


**Only remaining tem maior F1 (+3.7590 pontos percentuais), IoU e recall. Easy + remaining tem maior AP médio (+2.9745 pontos percentuais) e menor FPR sem pluma.** Portanto, o modelo com maior F1 não é o de maior AP médio.

### Conferência com a simulação INT8

| Modelo | Imagens conferidas por ID | Imagens com TP/FP/FN/TN diferentes | Maior diferença absoluta de AP |
| --- | --- | --- | --- |
| AttentionGates only remaining | 342 | 0 | 4.933e-13 |
| AttentionGates easy + remaining | 342 | 0 | 4.911e-13 |


As contagens coincidiram **em todas as 342 imagens de cada modelo**. As diferenças de AP são inferiores a 5×10⁻¹³, compatíveis com arredondamento na gravação do CSV. Esta conferência é da qualidade medida nesses arquivos; não compara o tempo da placa com o tempo do simulador.

### Referências FP32 e INT8 do mesmo test

| Modelo | Modo no avaliador | F1 | IoU | AP médio |
| --- | --- | --- | --- | --- |
| AttentionGates only remaining | FP32_512 | 0.676553 | 0.511205 | 0.610146 |
| AttentionGates only remaining | FP32_patches128 | 0.661975 | 0.494740 | 0.631302 |
| AttentionGates only remaining | INT8_patches128 | 0.660944 | 0.493589 | 0.531214 |
| AttentionGates easy + remaining | FP32_512 | 0.647798 | 0.479069 | 0.514058 |
| AttentionGates easy + remaining | FP32_patches128 | 0.645602 | 0.476671 | 0.630064 |
| AttentionGates easy + remaining | INT8_patches128 | 0.623353 | 0.452806 | 0.560959 |


A comparação FP32_patches128 → INT8_patches128 isola a quantização mantendo a avaliação por patches. FP32_512 também altera a forma de inferência. Esses números são referências da avaliação local, não novas execuções em CPU/GPU feitas para este documento.

## 4. Desempenho no test

As tabelas apresentam as execuções medidas, incluindo as duas configurações do HyperSTARCOP. **Throughput** é imagens concluídas/duração do pipeline. **FPS por latência** é 1000/latência média em ms. Há concorrência; esses dois valores não são intercambiáveis. Cada configuração tem uma execução disponível, sem repetição para estimar estabilidade.

### Model only

| Modelo | Runners | Duração (s) | Throughput (img/s) | Latência média (ms/img) | P95 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| AttentionGates only remaining | 2 | 7.225 | 47.335 | 42.219 | 44.397 | 44.460 |
| MobileNetV2 | 2 | 40.247 | 8.497 | 234.835 | 235.778 | 236.289 |
| AttentionGates easy + remaining | 2 | 7.222 | 47.358 | 42.223 | 44.404 | 44.483 |
| HyperSTARCOP (2 runners) | 2 | 6.985 | 48.960 | 40.826 | 42.672 | 43.271 |
| HyperSTARCOP (1 runner) | 1 | 13.465 | 25.399 | 39.369 | 39.436 | 39.478 |
| MobileNetV3 | 2 | 36.212 | 9.444 | 211.311 | 213.427 | 214.003 |
| Depth Reduced | 2 | 31.249 | 10.944 | 182.593 | 182.980 | 183.503 |
| Baseline | 2 | 57.704 | 5.927 | 337.076 | 339.407 | 340.348 |
| Skip Connections | 2 | 27.614 | 12.385 | 161.112 | 165.783 | 168.514 |


Model only usa entradas já preparadas nos slots, reutilizadas; exclui leitura, pré-processamento e recomposição da saída. O tempo de inferência inclui as operações CPU do grafo quando executadas por GraphRunner.

### End to end

| Modelo | Throughput (img/s) | FPS por latência | Latência média (ms/img) | P95 (ms) | P99 (ms) | Duração (s) |
| --- | --- | --- | --- | --- | --- | --- |
| AttentionGates only remaining | 14.795 | 5.218 | 191.629 | 234.997 | 239.979 | 23.116 |
| MobileNetV2 | 8.392 | 1.747 | 572.309 | 649.105 | 689.732 | 40.751 |
| AttentionGates easy + remaining | 14.787 | 5.197 | 192.414 | 231.923 | 241.448 | 23.129 |
| HyperSTARCOP (2 runners) | 7.005 | 3.043 | 328.614 | 374.997 | 393.056 | 48.820 |
| HyperSTARCOP (1 runner) | 11.403 | 4.621 | 216.417 | 259.319 | 264.205 | 29.991 |
| MobileNetV3 | 9.359 | 1.997 | 500.745 | 558.146 | 589.730 | 36.542 |
| Depth Reduced | 10.814 | 2.489 | 401.775 | 445.846 | 478.272 | 31.626 |
| Baseline | 5.875 | 1.323 | 755.702 | 874.383 | 995.213 | 58.212 |
| Skip Connections | 11.959 | 2.955 | 338.457 | 362.407 | 388.293 | 28.598 |


### Etapas médias do end to end

| Modelo | Leitura (ms) | Pré (ms) | Inferência (ms) | Pós (ms) |
| --- | --- | --- | --- | --- |
| AttentionGates only remaining | 120.923 | 13.881 | 43.414 | 3.198 |
| MobileNetV2 | 123.288 | 13.573 | 236.555 | 1.685 |
| AttentionGates easy + remaining | 120.858 | 13.920 | 43.291 | 3.289 |
| HyperSTARCOP (2 runners) | 271.989 | 13.006 | 39.380 | 1.657 |
| HyperSTARCOP (1 runner) | 120.480 | 13.184 | 38.587 | 1.640 |
| MobileNetV3 | 122.783 | 13.290 | 212.308 | 1.889 |
| Depth Reduced | 133.136 | 15.479 | 183.292 | 1.849 |
| Baseline | 242.173 | 15.121 | 338.047 | 1.854 |
| Skip Connections | 141.399 | 16.471 | 162.678 | 1.864 |


A latência total também inclui filas e sincronizações, portanto não é apenas a soma dessas quatro etapas. Nos AttentionGates, a abertura morfológica do protocolo oficial é feita na validação, fora da janela de desempenho.

## 5. Potência e energia no test

Todas as linhas registram o sensor **ina226:power1**, sem rótulo do trilho, em **/sys/class/hwmon/hwmon0/power1_input**. Os valores representam esse sensor; não há identificação suficiente para atribuí-los exclusivamente à DPU. Energia por imagem inclui o consumo do trilho durante a janela medida, sem subtração do repouso.

| Modelo | Potência model only (W) | Energia model only (J/img) | Potência E2E (W) | Energia E2E (J/img) | Método registrado |
| --- | --- | --- | --- | --- | --- |
| AttentionGates only remaining | 17.145 | 0.3622 | 15.868 | 1.0725 | integral_trapezoidal_hwmon |
| MobileNetV2 | 15.630 | 1.8394 | 15.976 | 1.9036 | não informado no CSV |
| AttentionGates easy + remaining | 17.210 | 0.3634 | 15.906 | 1.0757 | integral_trapezoidal_hwmon |
| HyperSTARCOP (2 runners) | 22.937 | 0.4685 | 16.158 | 2.3065 | não informado no CSV |
| HyperSTARCOP (1 runner) | 18.997 | 0.7479 | 16.936 | 1.4852 | não informado no CSV |
| MobileNetV3 | 15.305 | 1.6205 | 15.673 | 1.6746 | não informado no CSV |
| Depth Reduced | 26.132 | 2.3878 | 26.969 | 2.4939 | não informado no CSV |
| Baseline | 26.593 | 4.4869 | 27.316 | 4.6495 | não informado no CSV |
| Skip Connections | 25.566 | 2.0643 | 26.229 | 2.1933 | não informado no CSV |


Só os AttentionGates registram explicitamente **integral_trapezoidal_hwmon**. As execuções anteriores não identificam o método de integração nos arquivos; a comparação de energia entre versões deve considerar essa diferença de coleta.

## 6. Auditoria do FPR por tile

Regra vigente: em cada imagem 512×512 classificada como sem pluma, considerar falso alarme se **TP+FP > 640 pixels**. FPR tile = imagens com falso alarme / imagens sem pluma. Recalculou-se esse indicador a partir de metricas_por_imagem.csv, mantendo a classificação de fundo dos relatórios principais.

| Execução | Fundo (imagens) | FP tiles salvo | FPR tile salvo (%) | FP tiles >640 | FPR tile >640 (%) |
| --- | --- | --- | --- | --- | --- |
| AttentionGates only remaining | 176 | 8 | 4.5455 | 8 | 4.5455 |
| MobileNetV2 | 176 | 19 | 10.7955 | 8 | 4.5455 |
| AttentionGates easy + remaining | 176 | 6 | 3.4091 | 6 | 3.4091 |
| HyperSTARCOP (2 runners) | 176 | 22 | 12.5000 | 22 | 12.5000 |
| HyperSTARCOP (1 runner) | 176 | 75 | 42.6136 | 22 | 12.5000 |
| MobileNetV3 | 176 | 2 | 1.1364 | 1 | 0.5682 |
| Depth Reduced | 176 | 13 | 7.3864 | 4 | 2.2727 |
| Baseline | 176 | 6 | 3.4091 | 2 | 1.1364 |
| Skip Connections | 176 | 5 | 2.8409 | 1 | 0.5682 |
| Baseline — full | 1713 | 19 | 1.1092 | 2 | 0.1168 |


**Os registros antigos de Baseline, Depth Reduced, Skip Connections, MobileNetV2, MobileNetV3 e HyperSTARCOP com 1 runner reproduzem a contagem de TP+FP >10 pixels por imagem, em vez de >640.** Isso é uma inferência baseada na reprodução das contagens salvas, não na recuperação do código executado na placa. AttentionGates e HyperSTARCOP com AUPRC disponível já coincidem com >640. Use a coluna recalculada para comparar o FPR tile entre essas execuções. Os CSVs originais não foram alterados.

As duas execuções HyperSTARCOP têm **as mesmas contagens TP/FP/FN/TN por ID** e o mesmo F1. A diferença de FPR tile salvo (42,6136% versus 12,5000%) é reproduzida pela troca do limiar de contagem; o recálculo dá **12,5000% nas duas**. A conferência disponível é das contagens por imagem, sem comparação das máscaras pixel a pixel.

Nos relatórios principais do test há **176 imagens classificadas como sem pluma pelos metadados**. No protocolo oficial dos AttentionGates há **175 imagens sem pixels positivos no label e 167 positivas**. Essa diferença também afeta os denominadores e os grupos; compare indicadores dentro do mesmo protocolo.

## 7. Baseline no full — 3425 imagens

Este resultado pertence a dataset_starcop/train.csv e fica separado do ranking de test. Há somente um modelo medido no full nesta pasta; não é possível formar um ranking entre modelos nesse conjunto.

| Imagens | Precision | Recall | F1 | IoU | AUPRC | F1 strong | F1 weak | Acurácia (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3425 | 0.803542 | 0.205128 | 0.326824 | 0.195332 | N/D | 0.377661 | 0.303298 | 99.7827 |


| TP | FP | FN | TN | FPR pixel (%) | FPR sem pluma (%) | FPR tile >640 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 473638 | 115800 | 1835350 | 895418412 | 0.0129 | 0.0032 | 0.1168 |


| Modo | Throughput (img/s) | Latência média (ms/img) | P99 (ms) | Duração (s) | Potência (W) | Energia (J/img) |
| --- | --- | --- | --- | --- | --- | --- |
| model_only | 5.934 | 337.032 | 340.308 | 577.230 | 27.717 | 4.6712 |
| end_to_end | 5.895 | 771.016 | 989.208 | 580.965 | 28.399 | 4.8172 |


## 8. Principais observações

- **Maior F1 no test:** AttentionGates only remaining, 0,664310 no relatório principal e 0,660944 no protocolo oficial. No relatório principal, MobileNetV2 é o segundo, com 0,648782.
- **Entre os dois AttentionGates**, easy + remaining tem maior AP médio oficial e menor FPR sem pluma; only remaining tem maior F1 e recall. Os valores impressos no terminal (F1 0,628873/0,664310 e AUPRC 0,599712/0,645172) são do relatório principal, não de validacao_oficial/.
- **Desempenho E2E medido:** os AttentionGates ficaram próximos de 14,79 img/s, o maior throughput observado nas execuções de test. A diferença entre eles é pequena e há apenas uma medição de cada configuração.
- **HyperSTARCOP:** recall elevado (0,793530), mas precision baixa (0,316127) e muitos falsos positivos (327279). Com 2 runners, model only sobe de 25,40 para 48,96 img/s, enquanto E2E cai de 11,40 para 7,01 img/s. A leitura média passa de 120,48 para 271,99 ms; os registros mostram contribuição de I/O, mas não isolam a causa dessa mudança.
- **MobileNetV3:** precision alta (0,808336), porém recall de 0,310671. **Skip Connections:** recall de apenas 0,034256; poucos falsos alarmes devem ser lidos junto dessa perda de detecção.
- **Energia E2E:** os AttentionGates registraram aproximadamente 1,07 J/img, o menor valor entre as execuções de test. O método antigo não é identificado nos CSVs, impedindo tratar todas as diferenças como efeito exclusivo do modelo.
- **Conferência da implantação dos AttentionGates:** contagens por imagem coincidem com a simulação INT8. A diferença para FP32 é anterior à execução na placa, nas referências de quantização e avaliação por patches.

## 9. Fontes locais

Cada pasta abaixo contém os CSVs globais, por imagem, de desempenho/energia e config.txt usados neste comparativo.

- **AttentionGates easy + remaining — test (342):** [attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009](attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/metricas_globais.csv); [config](attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/config.txt); [desempenho](attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/benchmark_geral.csv); [energia](attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/benchmark_power_rails.csv).

  [Métricas oficiais](attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/validacao_oficial/metricas_globais.csv) e [resultados oficiais por imagem](attentiongates_dpu_easy_remaining_STARCOP_test_manual_1637344786009/validacao_oficial/metricas_por_imagem.csv).

- **AttentionGates only remaining — test (342):** [attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607](attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/metricas_globais.csv); [config](attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/config.txt); [desempenho](attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/benchmark_geral.csv); [energia](attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/benchmark_power_rails.csv).

  [Métricas oficiais](attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/validacao_oficial/metricas_globais.csv) e [resultados oficiais por imagem](attentiongates_dpu_only_remaining_STARCOP_test_manual_1637344899607/validacao_oficial/metricas_por_imagem.csv).

- **Baseline — test (342):** [methane_baseline_manual_1637342997940](methane_baseline_manual_1637342997940/metricas_globais.csv); [config](methane_baseline_manual_1637342997940/config.txt); [desempenho](methane_baseline_manual_1637342997940/benchmark_geral.csv); [energia](methane_baseline_manual_1637342997940/benchmark_power_rails.csv).

- **Depth Reduced — test (342):** [methane_depth_reduced_manual_1637343647493](methane_depth_reduced_manual_1637343647493/metricas_globais.csv); [config](methane_depth_reduced_manual_1637343647493/config.txt); [desempenho](methane_depth_reduced_manual_1637343647493/benchmark_geral.csv); [energia](methane_depth_reduced_manual_1637343647493/benchmark_power_rails.csv).

- **HyperSTARCOP (2 runners) — test (342):** [methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285](methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/metricas_globais.csv); [config](methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/config.txt); [desempenho](methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/benchmark_geral.csv); [energia](methane_hyperstarcop_STARCOP_com_AUPRC_test_manual_1637346267285/benchmark_power_rails.csv).

- **HyperSTARCOP (1 runner) — test (342):** [methane_hyperstarcop_STARCOP_test_manual_1runner_1637343748386](methane_hyperstarcop_STARCOP_test_manual_1runner_1637343748386/metricas_globais.csv); [config](methane_hyperstarcop_STARCOP_test_manual_1runner_1637343748386/config.txt); [desempenho](methane_hyperstarcop_STARCOP_test_manual_1runner_1637343748386/benchmark_geral.csv); [energia](methane_hyperstarcop_STARCOP_test_manual_1runner_1637343748386/benchmark_power_rails.csv).

- **MobileNetV2 — test (342):** [methane_mobilenet_v2_manual_1637344097551](methane_mobilenet_v2_manual_1637344097551/metricas_globais.csv); [config](methane_mobilenet_v2_manual_1637344097551/config.txt); [desempenho](methane_mobilenet_v2_manual_1637344097551/benchmark_geral.csv); [energia](methane_mobilenet_v2_manual_1637344097551/benchmark_power_rails.csv).

- **MobileNetV3 — test (342):** [methane_mobilenet_v3_manual_1637344528835](methane_mobilenet_v3_manual_1637344528835/metricas_globais.csv); [config](methane_mobilenet_v3_manual_1637344528835/config.txt); [desempenho](methane_mobilenet_v3_manual_1637344528835/benchmark_geral.csv); [energia](methane_mobilenet_v3_manual_1637344528835/benchmark_power_rails.csv).

- **Skip Connections — test (342):** [methane_skip_connections_manual_1637343910635](methane_skip_connections_manual_1637343910635/metricas_globais.csv); [config](methane_skip_connections_manual_1637343910635/config.txt); [desempenho](methane_skip_connections_manual_1637343910635/benchmark_geral.csv); [energia](methane_skip_connections_manual_1637343910635/benchmark_power_rails.csv).

- **Baseline — full (3425):** [methane_baseline_dataset_starcop_manual_1637345578273](methane_baseline_dataset_starcop_manual_1637345578273/metricas_globais.csv); [config](methane_baseline_dataset_starcop_manual_1637345578273/config.txt); [desempenho](methane_baseline_dataset_starcop_manual_1637345578273/benchmark_geral.csv); [energia](methane_baseline_dataset_starcop_manual_1637345578273/benchmark_power_rails.csv).


Referências de simulação: [resumo FP32/INT8](../VitisAI/build/vitis_ai/evaluation/summary.csv), [easy INT8 por imagem](../VitisAI/build/vitis_ai/evaluation/attentiongates_dpu_easy_remaining/attentiongates_dpu_easy_remaining_test_INT8_patches128.csv), [only INT8 por imagem](../VitisAI/build/vitis_ai/evaluation/attentiongates_dpu_only_remaining/attentiongates_dpu_only_remaining_test_INT8_patches128.csv).
