import torch
from Utils.DataLoader import DataNormalizer
from .dataset import CANAIS_ENTRADA


def recortar_patches(tensor):
    # Mesmo tamanho, passo e ordem de STARCOPDataset(patching=True).
    height, width = tensor.shape[-2:]
    return torch.stack([
        tensor[..., y:y + 128, x:x + 128]
        for y in range(0, height - 128 + 1, 64)
        for x in range(0, width - 128 + 1, 64)
    ])


def preprocess(canais, produtos=None, input_mode="full"):
    produtos = tuple(produtos or CANAIS_ENTRADA)
    x = torch.stack(canais, dim=0)
    x = recortar_patches(x) if input_mode == "patches" else x.unsqueeze(0)
    return DataNormalizer(produtos)(x)
