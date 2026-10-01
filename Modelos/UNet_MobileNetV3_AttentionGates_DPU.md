# Attention Gates: variante para a DPU

## Arquivos

- Modelo original: `UNet_MobileNetV3_AttentionGates.py`, classe `UNetMobileNetV3AttentionGates`.
- Variante separada: `UNet_MobileNetV3_AttentionGates_DPU.py`, classe `UNetMobileNetV3AttentionGatesDPU`.

O arquivo original e os checkpoints treinados foram preservados.

## Motivo da mudança

Os checkpoints originais `UNetMobileNetV3AttentionGates_mag1c_rgb.pth` e `UNetMobileNetV3AttentionGates_mag1c_rgb_3245_63epoch.pth` foram inspecionados separadamente no Vitis AI 3.5, para o target `DPUCZDX8G_ISA1_B4096`, com entrada `[1, 4, 512, 512]`.

Ambos apresentaram 165 operadores atribuídos à DPU e 8 à CPU: um sigmoid e uma multiplicação em cada gate (`ag1`, `ag2`, `ag3`, `ag4`). Os cinco upsamplings com `align_corners=False` foram aceitos na DPU.

O Inspector rejeitou as multiplicações originais sem detalhar a causa. O broadcast entre o mapa de atenção de um canal e as features de vários canais foi tratado como uma hipótese, não como uma causa confirmada.

## Alteração nos quatro gates

O trecho original gerava um mapa de atenção compartilhado por todos os canais:

```python
self.psi = nn.Sequential(
    nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
    nn.BatchNorm2d(1),
    nn.Sigmoid(),
)
```

A variante gera um mapa de atenção por canal e usa Hardsigmoid:

```python
self.psi = nn.Sequential(
    nn.Conv2d(F_int, F_l, kernel_size=1, bias=True),
    nn.BatchNorm2d(F_l),
    nn.Hardsigmoid(),
)
```

A operação final continua sendo `x * attn`. Agora ambos os tensores têm dimensão `[B, F_l, H, W]`, sem broadcast entre canais. Os gates produzem, respectivamente, 48, 24, 16 e 16 canais de atenção.

Hardsigmoid é uma aproximação do sigmoid; a troca altera o comportamento da ativação. Gerar atenção por canal também altera a arquitetura e aumenta os parâmetros dos gates. O encoder, o decoder e os upsamplings mantêm a estrutura do modelo original.

## Resultado da verificação

Inspeção executada em 30/09/2026, com pesos aleatórios na variante:

| Entrada | Operadores na DPU | Operadores na CPU |
|---|---:|---:|
| `[1, 4, 128, 128]` | 173 | 0 |
| `[1, 4, 512, 512]` | 173 | 0 |

O nó de entrada é classificado como `user` pelo Inspector e não está incluído na contagem de operadores acima.

Também foi verificada uma passagem de treinamento em CPU, com entrada `[2, 4, 128, 128]`, saída `[2, 1, 128, 128]`, BCEWithLogitsLoss e gradientes finitos.

Os relatórios estão em `../VitisAI/build/vitis_ai/inspect/attentiongates_dpu/`, nas pastas `128` e `512`.

Essa verificação confirma a atribuição dos operadores pelo Inspector para o target indicado. Não inclui calibração INT8, compilação completa, execução na placa ou avaliação da qualidade da segmentação.

## Treinamento e checkpoints

**A variante ainda precisa ser treinada.** Os pesos aleatórios usados na inspeção não constituem um modelo treinado.

Os checkpoints originais não podem ser carregados diretamente com `strict=True`, pois os pesos e os parâmetros de BatchNorm das camadas `psi` passaram de um canal para `F_l` canais. Um reaproveitamento parcial dos pesos exigiria um procedimento separado, que não foi implementado.

Para treinar pelo notebook, importe a nova classe e use um nome próprio, evitando carregar ou sobrescrever os checkpoints originais:

```python
from Modelos.UNet_MobileNetV3_AttentionGates_DPU import UNetMobileNetV3AttentionGatesDPU

MODELS = [
    (UNetMobileNetV3AttentionGatesDPU, "UNetMobileNetV3AttentionGatesDPU", 0.0),
]
```

A mudança de arquitetura não corrige a métrica de validação nem o balanceamento da loss no treinamento; esses pontos precisam ser tratados separadamente.
