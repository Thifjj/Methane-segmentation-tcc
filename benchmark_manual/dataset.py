import os
import rasterio
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

CANAIS_ENTRADA = [
    "mag1c",
    "TOA_AVIRIS_640nm",
    "TOA_AVIRIS_550nm",
    "TOA_AVIRIS_460nm",
]

def encontrar_sample(caminho_dataset):
    caminho_csv = os.path.join(caminho_dataset, "train.csv")

    tabela = pd.read_csv(caminho_csv)

    amostras = []

    for pasta_csv in tabela["folder"]:
        nome_pasta = os.path.basename(
            os.path.normpath(pasta_csv)
        )

        pasta = os.path.join(
            caminho_dataset,
            nome_pasta
        )

        if os.path.isdir(pasta):
            amostras.append(pasta)

    return amostras

def carregar_sample(caminho_amostra):
    canais = []

    for canal in CANAIS_ENTRADA:
        caminho = os.path.join(caminho_amostra, f"{canal}.tif")

        with rasterio.open(caminho) as arquivo:
            imagem = arquivo.read(1)

        canais.append(torch.from_numpy(imagem).float())

    return canais

def carregar_label(pasta_amostra):
    caminho = os.path.join(pasta_amostra, "labelbinary.tif")

    with rasterio.open(caminho) as arquivo:
        label = arquivo.read(1)

    return torch.from_numpy(label).float()