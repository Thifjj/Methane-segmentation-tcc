# Log de treino quantizado (HGQ2) — UNet_SegFormer_DPU, loss FD

Modelo salvo: `Modelos_treinados/UNet_SegFormer_DPU_HGQ_hgq_FD_mag1c_rgb.keras`
Script: `Treinos/Treinamento_unet_quantizado.py` (venv `.venv_hgq`)

## Configuração usada nesta execução

| Item | Valor |
|---|---|
| Modo | `hgq` (bits e escalas treináveis, otimização de EBOPs) |
| Loss | FocalDice (alpha 0.25, gamma 2.0) |
| Otimizador | Adam, lr 1e-4 |
| Batch | 2 janelas por passo (`JANELAS_POR_PASSO = 2`) + acumulação de 2 passos = batch efetivo de 4 janelas |
| `BETA_EBOPS` | 1e-11 (sobe linearmente até a época 10) |
| Validação | 512×512 inteiras, F1 por época, paciência 10 |
| Tempo | ~560 s de treino + ~17 s de validação por época |

## Resultado (épocas 35 a 41)

- Melhor F1 de validação até aqui: **0.7103** (época 35), salvo no `.keras`.
- F1 do FP32 normal, citado pelo usuário: ~0.69. O HGQ ficou no mesmo patamar ou acima.
- EBOPs caiu de ~2.5e10 (bits cheios, início) para ~1.11e10 (época 41), cerca de 56% de redução.
- O F1 oscila bastante de uma época para outra (0.59 a 0.71). Isso é esperado enquanto o EBOPs ainda está caindo e os bits se ajustam. Vale olhar o melhor F1, não o último.
- Ainda não foi avaliado no conjunto de **teste** (342 imagens), nem convertido no hls4ml.

| Época | Loss | F1 val | EBOPs |
|---|---|---|---|
| 35 | 0.8001 | **0.7103** (recorde) | 1.1817e10 |
| 36 | 0.7585 | 0.5944 | 1.1615e10 |
| 37 | 0.7370 | 0.6966 | 1.1469e10 |
| 38 | 0.7636 | 0.6660 | 1.1346e10 |
| 39 | 0.7456 | 0.6533 | 1.1207e10 |
| 40 | 0.7760 | 0.6835 | 1.1121e10 |
| 41 | 0.7805 | 0.6392 | 1.1067e10 |

## Log original

```
Epoch 35/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 372ms/step - loss: 0.8001
Época 35 [Validação]: 100%|██████████| 129/129 [00:16<00:00,  7.67it/s]
Resumo da Época 35 -> Loss: 0.8001 | F1: 0.7103
 -> Novo recorde F1: 0.7103. Modelo salvo em Modelos_treinados/UNet_SegFormer_DPU_HGQ_hgq_FD_mag1c_rgb.keras
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 560s 384ms/step - loss: 0.8001 - beta: 1.0000e-11 - ebops: 11816855744.0000
Epoch 36/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 370ms/step - loss: 0.7585
Época 36 [Validação]: 100%|██████████| 129/129 [00:16<00:00,  7.70it/s]
Resumo da Época 36 -> Loss: 0.7585 | F1: 0.5944
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 556s 382ms/step - loss: 0.7585 - beta: 1.0000e-11 - ebops: 11614669008.0000
Epoch 37/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 370ms/step - loss: 0.7370
Época 37 [Validação]: 100%|██████████| 129/129 [00:16<00:00,  7.68it/s]
Resumo da Época 37 -> Loss: 0.7370 | F1: 0.6966
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 556s 382ms/step - loss: 0.7370 - beta: 1.0000e-11 - ebops: 11469041468.0000
Epoch 38/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 367ms/step - loss: 0.7636
Época 38 [Validação]: 100%|██████████| 129/129 [00:16<00:00,  7.63it/s]
Resumo da Época 38 -> Loss: 0.7636 | F1: 0.6660
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 552s 379ms/step - loss: 0.7636 - beta: 1.0000e-11 - ebops: 11346203784.0000
Epoch 39/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 372ms/step - loss: 0.7456
Época 39 [Validação]: 100%|██████████| 129/129 [00:16<00:00,  7.66it/s]
Resumo da Época 39 -> Loss: 0.7456 | F1: 0.6533
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 559s 383ms/step - loss: 0.7456 - beta: 1.0000e-11 - ebops: 11207363952.0000
Epoch 40/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 371ms/step - loss: 0.7760
Época 40 [Validação]: 100%|██████████| 129/129 [00:16<00:00,  7.67it/s]
Resumo da Época 40 -> Loss: 0.7760 | F1: 0.6835
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 558s 383ms/step - loss: 0.7760 - beta: 1.0000e-11 - ebops: 11120789520.0000
Epoch 41/200
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 0s 371ms/step - loss: 0.7805
Época 41 [Validação]: 100%|██████████| 129/129 [00:17<00:00,  7.57it/s]
Resumo da Época 41 -> Loss: 0.7805 | F1: 0.6392
1456/1456 ━━━━━━━━━━━━━━━━━━━━ 558s 382ms/step - loss: 0.7805 - beta: 1.0000e-11 - ebops: 11066845740.0000
Epoch 42/200
1222/1456 ━━━━━━━━━━━━━━━━━━━━ 1:26 369ms/step - loss: 0.7629
```
