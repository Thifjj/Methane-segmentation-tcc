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

def encontrar_sample(caminho_dataset, nome_csv="test.csv"):
    caminho_csv = os.path.join(caminho_dataset, nome_csv)

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

def carregar_classificacao_por_pasta(caminho_dataset, nome_csv="test.csv"):
    tabela = pd.read_csv(os.path.join(caminho_dataset, nome_csv))
    colunas_ausentes = {"folder", "has_plume", "qplume"} - set(tabela.columns)
    if colunas_ausentes:
        raise ValueError(f"Colunas ausentes no train.csv: {sorted(colunas_ausentes)}")

    resultado = {}
    for _, linha in tabela.iterrows():
        nome = os.path.basename(os.path.normpath(linha["folder"]))
        valor_has_plume = linha["has_plume"]
        if pd.isna(valor_has_plume):
            raise ValueError(f"has_plume ausente para {nome}")
        if isinstance(valor_has_plume, str):
            texto = valor_has_plume.strip().lower()
            if texto not in {"true", "false"}:
                raise ValueError(f"has_plume inválido para {nome}: {valor_has_plume}")
            has_plume = texto == "true"
        elif valor_has_plume in (True, False, 0, 1):
            has_plume = bool(valor_has_plume)
        else:
            raise ValueError(f"has_plume inválido para {nome}: {valor_has_plume}")

        valor_qplume = linha["qplume"]
        if has_plume and pd.isna(valor_qplume):
            raise ValueError(f"qplume ausente para amostra com pluma: {nome}")
        qplume = 0.0 if pd.isna(valor_qplume) else float(valor_qplume)
        resultado[nome] = (has_plume, qplume)
    return resultado

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
