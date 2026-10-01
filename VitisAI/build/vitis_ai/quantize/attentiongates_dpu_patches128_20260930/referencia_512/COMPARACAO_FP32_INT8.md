# Comparação do AttentionGates DPU: FP32 e INT8 em CPU

## Protocolo

Avaliação das 342 imagens do `STARCOP_test/test.csv`, usando o mesmo checkpoint treinado da quantização. SHA256 do checkpoint, do XModel e da configuração INT8 conferidos após a avaliação; todos permaneceram intactos.

Os três modos usam canais `mag1c,460nm,550nm,640nm`, DataNormalizer, threshold de logits > 0, abertura morfológica com cruz 3×3, F1/IoU com contagens acumuladas e AUPRC média por imagem com pluma. Strong significa `difficulty == easy`; as demais imagens com pluma entram em Weak.

A variante de patches usa 49 recortes sobrepostos de 128×128, passo 64, por imagem. A variante de imagem inteira usa uma entrada de 512×512. As contagens de pixels entre esses protocolos diferem porque os patches sobrepostos repetem pixels.

## Resultados

| Modo | F1 Global | F1 Strong | F1 Weak | IoU | AUPRC | FPR sem pluma |
|---|---:|---:|---:|---:|---:|---:|
| fp32_full512 | 0.6249 | 0.7944 | 0.5287 | 0.4544 | 0.4683 | 0.001242 |
| int8_full512_cpu_simulation | 0.4650 | 0.5838 | 0.4144 | 0.3030 | 0.3867 | 0.001006 |
| fp32_patches128 | 0.6463 | 0.8155 | 0.5581 | 0.4775 | 0.5822 | 0.001523 |

## Efeito da quantização, mantendo entrada de 512×512

- F1 Global: diferença INT8 − FP32 de -15.99 pontos percentuais.
- F1 Strong: diferença INT8 − FP32 de -21.06 pontos percentuais.
- F1 Weak: diferença INT8 − FP32 de -11.43 pontos percentuais.
- IoU: diferença INT8 − FP32 de -15.15 pontos percentuais.
- AUPRC: diferença INT8 − FP32 de -8.16 pontos percentuais.

Recall global: 65.21% → 39.39%. Recall Weak: 48.05% → 30.42%. Precisão Weak: 58.76% → 65.01%.

A perda acompanha o aumento de falsos negativos e a diminuição de detecções. AUPRC também cai, portanto não se trata apenas de uma diferença no threshold binário. Esses resultados não identificam qual camada causou a perda.

## Efeito do tamanho de entrada em FP32

F1 Global: 0.6463 com patches → 0.6249 com imagem inteira. F1 Weak: 0.5581 → 0.5287. Essa diferença existe antes da quantização.

Para investigar uma quantização que preserve o protocolo original, o próximo experimento é calibrar/exportar especificamente para entrada de 128×128 e avaliar FP32 e INT8 nos mesmos patches. Os resultados atuais não antecipam a qualidade desse experimento.

## Limites da comparação

INT8 foi executado pelo simulador PyTorch do Vitis AI 3.5 em CPU, com a configuração usada na exportação. Não houve execução do XModel na placa. Os tempos no CSV medem apenas o forward em CPU; o custo do simulador INT8 não representa o desempenho da DPU.

Os CSVs `comparison.csv` e `per_image.csv` e o JSON `comparison.json` estão nesta pasta. O histórico original de testes foi preservado.
