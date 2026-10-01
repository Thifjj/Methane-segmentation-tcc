# Comparação FP32 e INT8 — AttentionGates DPU

Calibração concluída com **um patch 128×128 de cada uma das 3.425 imagens**, em **531,3 segundos (8 min 51 s)**. Seleção aleatória com seed 12345, stride 64, lote de 16 patches e a mesma normalização e ordem de canais do treinamento.

## Resultados

Avaliação nas mesmas 342 imagens do TEST SET externo, usando os 49 patches por imagem e o protocolo registrado em `comparison.json`.

| Versão | F1 global | F1 strong | F1 weak | IoU | AUPRC |
|---|---:|---:|---:|---:|---:|
| FP32 | 0.6463 | 0.8155 | 0.5581 | 0.4775 | 0.5822 |
| INT8 — 1.000 patches | 0.5908 | 0.7482 | 0.4945 | 0.4193 | 0.5157 |
| INT8 — 3.425 imagens, 1 patch/imagem | 0.6213 | 0.7826 | 0.5138 | 0.4507 | 0.5250 |

Em relação ao INT8 anterior, o F1 global melhorou **3.05 pontos percentuais** e o F1 weak melhorou **1.93 pontos**. Em relação ao FP32, permanecem perdas de **2.50 pontos** no F1 global e **4.42 pontos** no F1 weak.

## Procedência e limites

- O conjunto completo inclui as 514 imagens da antiga validação. O TEST SET externo não entrou na calibração.
- A calibração anterior selecionou 1.000 patches de 840 imagens do subconjunto de treinamento; a nova cobre todas as 3.425 imagens. Quantidade, cobertura e seleção mudaram juntas, portanto a melhora não pode ser atribuída somente à quantidade.
- INT8 foi avaliado pela simulação PyTorch do Vitis AI em CPU. O XModel ainda não foi compilado nem executado na placa; estas latências não representam a vazão da DPU.
- A referência FP32 foi reaproveitada da avaliação anterior, com validação do checkpoint, CSV, canais e configuração do protocolo.
- O checkpoint original e a versão INT8 anterior de 1.000 patches foram preservados. O checkpoint congelado desta execução, a arquitetura, o XModel e a configuração de quantização tiveram seus hashes conferidos após a avaliação.

Entrada do XModel: `[1,128,128,4]` (NHWC), quantização de parâmetros e saídas em 8 bits.
