# Treinamento dos modelos de segmentacao

O arquivo `Treinamento_Unet.py` usa a GPU do host, quando disponivel, e antecipa a leitura das imagens do dataset. Este ambiente de treinamento e separado do container CPU do Vitis AI usado para quantizacao e compilacao.

## Modificação realizada

O `DataLoader` original utilizava `batch_size=4` e carregava os dados somente pelo processo principal:

```python
dataloader = DataLoader(dataset_treino, batch_size=4, shuffle=True)
```

Ele passou a utilizar:

```python
dataloader = DataLoader(
    dataset_treino,
    batch_size=6,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
)
```

As transferências dos tensores para o dispositivo também passaram a usar `non_blocking=True`:

```python
inputs = normalizador(batch["input"].to(device, non_blocking=True))
targets = batch["output"].to(device, non_blocking=True)
pesos_loss = batch["weight_loss"].to(device, non_blocking=True)
```

## Finalidade das opções

- `batch_size=6`: processa seis amostras por iteração para aproveitar melhor a GPU. O valor anterior era 4.
- `num_workers=6`: utiliza seis processos para carregar os arquivos TIFF enquanto a GPU processa o batch atual.
- `pin_memory=True`: mantém os batches em uma área de RAM que permite transferências mais eficientes para a GPU.
- `persistent_workers=True`: mantém os processos de leitura ativos entre as épocas.
- `non_blocking=True`: permite transferências assíncronas da RAM para a VRAM quando usadas com memória fixada.

## Observações

O computador usado durante o ajuste possuía 12 GB de VRAM e aproximadamente 64 GB de RAM. Antes da alteração, o treinamento com batch 4 consumia aproximadamente 7,6 GB de VRAM.

Depois de iniciar o treinamento, acompanhe o consumo com:

```bash
watch -n 2 nvidia-smi
```

Se ocorrer `CUDA out of memory`, reduza o `batch_size` para 5 ou 4. Se a leitura ficar mais lenta por causa do disco rígido, teste `num_workers=2` ou `0`. Ao usar `num_workers=0`, altere também `persistent_workers=False`.

As mudanças entram em vigor somente depois que a execução atual do notebook for interrompida e a célula de treinamento for executada novamente.

Os checkpoints são gravados em `Modelos_treinados/`. Depois do treinamento,
siga o fluxo de quantização, exportação e compilação descrito em
[`VitisAI/README.md`](../VitisAI/README.md). A quantização não substitui o
treinamento e não precisa da GPU NVIDIA.
