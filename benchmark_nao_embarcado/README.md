# Benchmark PyTorch CPU / CUDA

Execute na raiz do projeto, usando a `.venv` Python 3.10.20 já instalada:

```bash
source .venv/bin/activate
python -m benchmark_nao_embarcado.benchmark_geral
```

O menu mostra os modelos cujos checkpoints existem em `Modelos_treinados/`.
Os caminhos dos pesos são relativos à raiz do projeto, sem depender do
computador usado anteriormente.

## Modelos com pesos disponíveis nesta atualização

| Nome | Checkpoint |
|---|---|
| baseline | UNET_mag1c_rgb.pth |
| depth_reduced | UNET_depth_reduced_mag1c_rgb.pth |
| mobilenet_v2 | Mobile_Net_v2_mag1c_rgb.pth |
| mobilenet_v3 | Mobile_Net_v3_mag1c_rgb.pth |
| skip | UNET_SkipConnections_mag1c_rgb.pth |
| hyperstarcop | HyperSTARCOP_oficial/final_checkpoint_model.ckpt |
| mobilenet_v3_focaldice | MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth |
| mobilenet_v3_attention_gates | UNetMobileNetV3AttentionGates_mag1c_rgb.pth |
| mobilenet_v3_attention_gates_63epoch | UNetMobileNetV3AttentionGates_mag1c_rgb_3245_63epoch.pth |
| resnet34 | UNet_ResNet34_mag1c_rgb.pth |
| segformer | UNet_SegFormer_mag1c_rgb.pth |

Attention Gates da U-Net e PSA também têm classes cadastradas, mas não
aparecem no menu enquanto faltar o checkpoint padrão. É possível fornecer
outros pesos com `--checkpoint`, sem substituir os existentes.

## Execução sem perguntas

```bash
python -m benchmark_nao_embarcado.benchmark_geral   --modelo mobilenet_v3_attention_gates --dataset test   --device cuda --quantidade 0
```

Para comparar os pesos de 63 épocas, use
`--modelo mobilenet_v3_attention_gates_63epoch`. Para CPU, use `--device cpu`.
`--quantidade 0` usa todas as imagens; outro número limita as primeiras linhas.

Datasets padrão:

- `full`: `/media/jacques/games/Datasets/STARCOP_train_remaining_all/train.csv`.
- `test`: `/media/jacques/games/Datasets/test/STARCOP_test/test.csv`.

`--data-root` e `--csv-name` permitem usar outro dataset. São respeitadas as
janelas do CSV e filtradas amostras sem bandas ou máscara obrigatórias.

## Entrada e canais

O padrão é `--input-mode patches`, igual ao `Teste_Unet.py` atual: recortes
128x128, passo 64. Uma imagem 512x512 produz 49 patches sobrepostos,
processados juntos. `--input-mode full` mede uma imagem inteira por chamada.

Os modelos antigos e o HyperSTARCOP mantêm a ordem do benchmark anterior:
`mag1c,640,550,460`. As novas entradas usam a ordem do notebook de treinamento:
`mag1c,460,550,640`, incluindo FocalDiceLoss, Attention Gates, ResNet34 e
SegFormer. A ordem é exibida e gravada no CSV. Ela precisa corresponder ao
experimento que produziu os pesos; o state_dict não registra essa informação.
Para comparar um modelo antigo com o notebook atual, informe a mesma ordem
explicitamente, se ela também foi usada no treinamento desses pesos:

```bash
--produtos mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm
```

Também é possível fornecer outra lista de bandas com o checkpoint compatível.
O HyperSTARCOP oficial mantém seus quatro canais e ordem obrigatória.
A normalização reutiliza `DataNormalizer`: mag1c/1750 e bandas visíveis/60,
limitadas a [0,2].

## Métricas alinhadas ao Teste_Unet.py

| Cálculo | Protocolo atual |
|---|---|
| Máscara prevista | `logit > 0`, seguido de abertura morfológica com cruz 3x3 |
| Presença de pluma | Pixels positivos na máscara real |
| F1 global e IoU | Contagens TP/FP/FN acumuladas em todas as imagens |
| F1 strong | Imagens com pluma e `difficulty == easy` |
| F1 weak | Demais imagens com pluma |
| AUPRC | Média de `average_precision_score` por imagem com pluma |
| fpr_pixel | FP/(FP+TN+1e-6), somente imagens sem pluma |

A AUPRC agrupa os pixels de todos os patches de uma imagem antes de calcular
AP, como no teste. Não calcula uma AP separada para cada patch.
F1 e IoU usam o mesmo epsilon de 1e-6 do teste. Precisão e recall também são
reportados. `fpr_pixel_global` conserva explicitamente o FPR de todos os
pixels; `fpr_tile` e `fpr_tile_tabela` são métricas adicionais do benchmark,
com o limiar anterior de pixels previstos por área. O teste não calcula
essas métricas extras.

O protocolo difere dos CSVs antigos: antes não havia abertura morfológica,
strong/weak usava qplume, a AUPRC era trapezoidal global e o FPR pixel incluía
todas as imagens. Não juntar esses resultados como se fossem o mesmo cálculo.
Também manter o mesmo checkpoint, bandas, dataset e modo de entrada ao
comparar com Teste_Unet ou com a placa. Patches sobrepostos contam pixels
repetidos; não equivalem à inferência de uma imagem inteira.

## Latência e FPS

- `model_*`: somente forward, com sincronização CUDA antes/depois.
- `e2e_*`: leitura das bandas, recortes/normalização/transferência, forward,
  sigmoid e abertura morfológica.
- Leitura da máscara, métricas (incluindo AUPRC) e escrita do CSV ficam fora
  dos tempos acima. A velocidade da barra inclui esse trabalho adicional.
- Dez inferências de aquecimento usam o mesmo modo de entrada da medição
  e também aquecem sigmoid e abertura morfológica.
- FPS = 1000 / latência média em ms: imagens originais por segundo. No modo
  patches, cada imagem representa o conjunto de recortes, não um único patch.
- É FPS sequencial baseado na latência, sem múltiplas inferências concorrentes;
  não é a vazão de um pipeline paralelo na ZCU104.

A instrumentação RAPL/NVML de energia foi mantida. Registra energia total,
energia por inferência e potência média por fonte, quando disponível. Essas
leituras não representam a potência total da tomada.

## Resultados

Os novos arquivos incluem modelo, dataset, device, modo de entrada e timestamp
com microssegundos. Os resultados anteriores são preservados. `--output`
permite escolher outro caminho de saída.

O CSV registra checkpoint, ordem das bandas, CSV do dataset, modo de entrada,
patches por imagem e `metricas_protocolo`, além das métricas e tempos.
