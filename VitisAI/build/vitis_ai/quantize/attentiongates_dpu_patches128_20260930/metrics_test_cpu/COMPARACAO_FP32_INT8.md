# AttentionGates DPU: nova quantização com patches de 128

## Calibração e exportação

- Vitis AI 3.5, target `DPUCZDX8G_ISA1_B4096`.
- Entrada exportada: `[1, 4, 128, 128]`, confirmada pela biblioteca XIR.
- Mesmos patches do treinamento: 128×128, passo 64, 49 por imagem de 512×512.
- Mesma ordem de canais e DataNormalizer: `mag1c,460nm,550nm,640nm`.
- 1.000 patches sorteados entre os 142.639 patches de 2.911 imagens de treino, seed 12345, lotes de 16 patches.
- As 514 imagens de validação foram excluídas. São 1.000 patches, não 1.000 imagens inteiras.
- Calibração em modo eval, sem augmentações, máscaras, loss ou atualização de pesos.
- Manifesto dos recortes efetivamente usados: `../mobilenet_v3_attention_gates_dpu/calibration_patches.json`.

## Comparação nas 342 imagens do TEST SET

FP32 e INT8 usam os mesmos 49 patches por imagem, threshold de logits > 0 e abertura morfológica com cruz 3×3. As métricas acumulam TP, FP e FN, como o teste original. Strong corresponde a `difficulty == easy`; Weak reúne as demais imagens com pluma. AUPRC é a média de AP por imagem com pluma.

| Modo | F1 Global | Strong | Weak | IoU | AUPRC | FPR sem pluma |
|---|---:|---:|---:|---:|---:|---:|
| fp32_patches128 | 0.6463 | 0.8155 | 0.5581 | 0.4775 | 0.5822 | 0.001523 |
| int8_patches128_cpu_simulation | 0.5908 | 0.7482 | 0.4945 | 0.4193 | 0.5157 | 0.001376 |

## Diferenças INT8 − FP32 no mesmo protocolo

- F1 Global: -5.55 pontos percentuais.
- F1 Strong: -6.72 pontos percentuais.
- F1 Weak: -6.36 pontos percentuais.
- IoU: -5.82 pontos percentuais.
- AUPRC: -6.65 pontos percentuais.

Recall global: 68.95% → 57.55%. Recall Weak: 50.17% → 40.41%. A perda acompanha o aumento de falsos negativos. AUPRC também diminui; ajustar somente o threshold não explica toda a diferença.

## Substituição da quantização anterior

O XModel, a configuração e os arquivos gerados da quantização anterior de 512×512 foram excluídos. Somente os números e o relatório anterior foram preservados em `../referencia_512/`.

A versão anterior obteve F1 Global 0,4650 em INT8 com entrada 512×512. A nova obteve 0,5908 com patches. Isso descreve a melhoria do fluxo completo; não isola apenas a calibração, pois também mudou o tamanho de entrada e a unidade de amostragem. A comparação controlada da nova quantização é FP32 versus INT8 nos mesmos patches, na tabela acima.

## Limites

INT8 foi executado pelo simulador PyTorch do Vitis AI em CPU, não na placa. O custo do simulador não representa a vazão da DPU. Ainda é necessário compilar o XModel com o arch.json da placa e avaliar sua execução real.

O checkpoint FP32, o XModel novo e a configuração de quantização permaneceram intactos após a avaliação, confirmados por SHA256. O histórico de testes do usuário foi preservado.

Os resultados completos estão em `comparison.csv`, `comparison.json` e `per_image.csv` nesta pasta.
