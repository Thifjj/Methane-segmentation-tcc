# Comparativo dos benchmarks STARCOP em CPU

Execuções do conjunto `test`, com **342 imagens por modelo**, realizadas em 22/09/2026. O comparativo usa os CSVs locais e cobre todos os campos gravados neles. Valores de proporção aparecem como fração (0–1), exceto FPR por tile que também está mostrado em percentual.

A comparação de throughput e métricas entre ZCU104, CPU e GPU está em
[`../comparativos/comparativo_test_zcu104_cpu_gpu.csv`](../comparativos/comparativo_test_zcu104_cpu_gpu.csv).
Este documento mantém o detalhamento das execuções CPU aqui descritas.

## Foco: desempenho e métricas do paper

FPS é `1000 / latência média`, para inferência sequencial de uma imagem por vez. `model_fps` mede só o modelo; `e2e_fps` inclui leitura, pré-processamento, inferência e pós-processamento.


### FPS e latências

| Métrica | baseline | depth_reduced | hyperstarcop | mobilenet_v2 | mobilenet_v3 | skip |
| --- | --- | --- | --- | --- | --- | --- |
| FPS modelo | 0.837 | 0.944 | 5.637 | 7.844 | 18.618 | 1.073 |
| Média (ms) | 1194.43 | 1059.75 | 177.39 | 127.48 | 53.71 | 932.06 |
| Mediana (ms) | 1190.63 | 1063.95 | 176.82 | 126.79 | 53.37 | 907.88 |
| P95 (ms) | 1240.04 | 1193.15 | 186.05 | 142.19 | 60.24 | 1078.26 |
| P99 (ms) | 1433.78 | 1244.24 | 211.43 | 156.56 | 64.29 | 1140.13 |
| FPS E2E | 0.820 | 0.922 | 5.010 | 6.618 | 13.221 | 1.043 |
| Média (ms) | 1218.96 | 1084.90 | 199.58 | 151.10 | 75.64 | 958.57 |
| Mediana (ms) | 1215.91 | 1091.50 | 198.74 | 149.98 | 74.89 | 934.06 |
| P95 (ms) | 1263.84 | 1218.96 | 209.77 | 167.80 | 85.30 | 1105.46 |
| P99 (ms) | 1456.39 | 1270.20 | 234.42 | 180.42 | 90.05 | 1168.06 |
### F1, FPR e AUPRC

| Métrica | baseline | depth_reduced | hyperstarcop | mobilenet_v2 | mobilenet_v3 | skip |
| --- | --- | --- | --- | --- | --- | --- |
| F1 global | 0.4458 | 0.3408 | 0.4510 | 0.6017 | 0.5261 | 0.2519 |
| F1 strong | 0.5451 | 0.4360 | 0.8308 | 0.7366 | 0.5322 | 0.3167 |
| F1 weak | 0.3572 | 0.1274 | 0.5372 | 0.5765 | 0.5180 | 0.0792 |
| FPR pixel | 0.0004 | 0.0003 | 0.0039 | 0.0008 | 0.0002 | 0.0001 |
| FPR tile (sem pluma) | 0.0455 | 0.0795 | 0.4148 | 0.0341 | 0.0170 | 0.0739 |
| FPR tile (todos/342) | 0.0234 | 0.0409 | 0.2135 | 0.0175 | 0.0088 | 0.0380 |
| AUPRC | 0.4524 | 0.3353 | 0.5061 | 0.5824 | 0.5083 | 0.3611 |

### Como ler as métricas

- **F1 global** usa pixels de todas as amostras com limiar de probabilidade `> 0,5`; `F1 strong` e `F1 weak` são subdivisões por intensidade de pluma (`qplume > 1000` e `<= 1000`).
- **FPR pixel** = `FP / (FP + TN)`. **FPR tile (sem pluma)** mede a fração de tiles sem pluma em que o modelo previu pluma (mais de 10 pixels previstos). Essa é a variante de falso positivo por tile mais diretamente interpretável como métrica do paper.
- **FPR tile (todos/342)** = `fp_tiles / total de imagens`; mantida porque também está gravada no CSV, mas usa outro denominador. Não comparar diretamente com FPR restrita aos tiles negativos.
- **AUPRC** é a área sob a curva precision-recall calculada sobre as probabilidades por pixel. F1 usa máscara binária no limiar fixo, portanto mede outro aspecto.

### Conferência com o paper: HyperSTARCOP mag1c + RGB

O paper define AUPRC sobre os mapas de segmentação, aplica limiar 0,5 para F1, separa F1 strong/weak por intensidade da pluma e classifica um tile como positivo quando há mais de 10 pixels previstos. O FPR é calculado nos tiles sem pluma. Essas definições correspondem ao pipeline deste benchmark. O conjunto local contém 342 imagens `test`; o paper descreve resultados no conjunto AVIRIS de teste. Portanto, a comparação do HyperSTARCOP é metodologicamente alinhada, não uma comparação entre tarefas ou métricas diferentes.

| Métrica | Este benchmark: `hyperstarcop` | Paper: HyperSTARCOP mag1c + RGB | Diferença local − paper |
|---|---:|---:|---:|
| F1 strong | 83,08% | 81,96% ± 3,71 p.p. | +1,12 p.p. |
| F1 weak | 53,72% | 43,42% ± 5,72 p.p. | +10,30 p.p. |
| FPR por tile sem pluma | 41,48% | 43,66% ± 7,36 p.p. (Tabela 2); 43,79% (Figura 7) | −2,18 p.p. vs. Tabela 2 |
| AUPRC | 50,61% | 51,99% ± 2,76 p.p. | −1,38 p.p. |

O paper apresenta a média e o desvio padrão de cinco treinos; o CSV local registra uma execução. Assim, a diferença é esperada e não invalida a comparabilidade do protocolo. O resultado local de F1 weak fica acima da média publicada por 10,30 pontos percentuais, ponto que merece ser registrado como variação entre execuções/checkpoints. FPS é uma medição adicional deste benchmark e não é reportado pelo paper.

`FPR tile (todos/342)` não é a mesma taxa: divide o número de falsos alarmes pelo total de tiles. Para comparar com o FPR do paper, use `FPR tile (sem pluma)`.


### Demais métricas de qualidade

| Métrica | baseline | depth_reduced | hyperstarcop | mobilenet_v2 | mobilenet_v3 | skip |
| --- | --- | --- | --- | --- | --- | --- |
| Precision | 0.6313 | 0.6516 | 0.3096 | 0.6183 | 0.7888 | 0.6960 |
| Recall | 0.3445 | 0.2308 | 0.8304 | 0.5860 | 0.3947 | 0.1538 |
| IoU | 0.2868 | 0.2054 | 0.2912 | 0.4303 | 0.3569 | 0.1441 |
### Tempos por etapa

| Métrica | baseline | depth_reduced | hyperstarcop | mobilenet_v2 | mobilenet_v3 | skip |
| --- | --- | --- | --- | --- | --- | --- |
| Carregamento (ms) | 22.96 | 23.42 | 20.99 | 22.30 | 20.74 | 24.75 |
| Pré-processamento (ms) | 1.10 | 1.35 | 0.96 | 1.04 | 0.93 | 1.42 |
| Pós-processamento (ms) | 0.46 | 0.36 | 0.23 | 0.27 | 0.25 | 0.33 |
### Contagens agregadas

| Campo | baseline | depth_reduced | hyperstarcop | mobilenet_v2 | mobilenet_v3 | skip |
| --- | --- | --- | --- | --- | --- | --- |
| TP | 65681 | 44001 | 158319 | 111713 | 75244 | 29320 |
| FP | 38352 | 23531 | 353050 | 68959 | 20146 | 12809 |
| FN | 124971 | 146651 | 32333 | 78939 | 115408 | 161332 |
| TN | 89424244 | 89439065 | 89109546 | 89393637 | 89442450 | 89449787 |
| Tiles FP | 8 | 14 | 73 | 6 | 3 | 13 |
| Tiles TN | 168 | 162 | 103 | 170 | 173 | 163 |
### Identificação das execuções

| Campo | baseline | depth_reduced | hyperstarcop | mobilenet_v2 | mobilenet_v3 | skip |
| --- | --- | --- | --- | --- | --- | --- |
| Data/hora | 2026-09-22 18:19:51 | 2026-09-22 18:30:19 | 2026-09-22 18:11:48 | 2026-09-22 18:39:27 | 2026-09-22 18:40:34 | 2026-09-22 18:37:30 |
| Dataset | test | test | test | test | test | test |
| Dispositivo | cpu | cpu | cpu | cpu | cpu | cpu |
| Imagens | 342 | 342 | 342 | 342 | 342 | 342 |

## Leitura rápida

- **Maior FPS:** `mobilenet_v3` (18,618 model-only; 13,221 E2E), com F1 global 0,5261 e AUPRC 0,5083.
- **Maior F1 global e AUPRC:** `mobilenet_v2` (0,6017 e 0,5824), com 7,844 FPS model-only e 6,618 FPS E2E.
- **Menor FPR por tile entre tiles sem pluma:** `mobilenet_v3` (1,705%); `mobilenet_v2` registra 3,409%.
- **Maior F1 strong:** `hyperstarcop` (0,8308), mas sua execução é bem mais lenta que as variantes MobileNet: 5,637 FPS model-only.

## Avaliação e escolha por caso de uso

Não há um modelo que lidere simultaneamente em qualidade, velocidade e falsos alarmes. A melhor escolha depende do custo de latência e do tipo de erro mais importante.

| Prioridade | Modelo indicado | Motivo e trade-off |
|---|---|---|
| Equilíbrio de qualidade geral | **MobileNetV2** | Maior F1 global (**0,6017**) e AUPRC (**0,5824**), além do maior F1 weak (**0,5765**). Entrega **6,618 FPS E2E**. |
| Velocidade e menos falsos alarmes em tiles sem pluma | **MobileNetV3** | Mais rápido (**13,221 FPS E2E**), menor FPR entre tiles sem pluma (**1,705%**) e maior precision (**0,7888**). F1 global e AUPRC ficam abaixo do MobileNetV2. |
| Detectar plumas fortes, aceitando mais falsos alarmes | **HyperSTARCOP** | Maior F1 strong (**0,8308**) e recall (**0,8304**), mas tem FPR por tile de **41,477%** e **5,010 FPS E2E**. Adequado quando perder uma pluma forte custa mais do que revisar alarmes falsos. |
| Boa qualidade com latência mais baixa que os modelos tradicionais | **MobileNetV2** | Entre os modelos rápidos, combina recall (**0,5860**), F1 weak (**0,5765**) e o melhor AUPRC. É mais lento que MobileNetV3. |

### Recomendação padrão

**MobileNetV2** é a escolha padrão mais equilibrada nestes resultados: lidera F1 global e AUPRC e mantém velocidade razoável. Escolha **MobileNetV3** quando o limite de latência ou o custo de falsos alarmes em tiles sem pluma for mais importante. Escolha **HyperSTARCOP** quando a prioridade for recuperar plumas fortes e houver capacidade para revisar mais alarmes falsos.

`baseline`, `depth_reduced` e `skip` não se destacam frente às duas MobileNet nos critérios principais: nestes resultados, são mais lentos e têm F1 global e AUPRC inferiores.

Esta recomendação descreve as execuções locais do conjunto `test` (342 imagens por modelo). Para `hyperstarcop`, os dados, o limiar e as definições das métricas seguem o protocolo do paper; a comparação numérica acima coloca uma execução local ao lado da média de cinco treinos publicados. Os demais modelos são comparados entre si neste benchmark, não contra valores publicados pelo paper. Os FPS são específicos do hardware e da execução local.

## Referências

- [STARCOP: Semantic Segmentation of Methane Plumes with Hyperspectral Machine Learning Models — paper (Seções 5.1 e 6.2, Tabela 2 e Figura 7)](https://assets-eu.researchsquare.com/files/rs-2899370/v1_covered_d22c1c67-b3e1-4617-8698-d579c4f53ff3.pdf?c=1700492853)
- [Repositório oficial STARCOP](https://github.com/spaceml-org/STARCOP)
