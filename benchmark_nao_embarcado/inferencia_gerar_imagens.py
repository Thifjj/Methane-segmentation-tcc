"""Gera um painel visual do AttentionGates FocalDice no conjunto de teste.

Uso: python -m benchmark_nao_embarcado.inferencia_gerar_imagens [id_amostra ...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from .dataset import carregar_label, carregar_sample
from .model_loader import load_model
from .postprocess import postprocess
from .preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amostras", nargs="*", help="IDs das pastas em STARCOP_test")
    parser.add_argument("--saida", type=Path, default=Path("benchmark_nao_embarcado/attgate_focaldice.png"))
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parents[1] / "STARCOP_test"
    tabela = pd.read_csv(raiz / "test.csv")
    ids = args.amostras or tabela[tabela.has_plume].nlargest(2, "qplume")["id"].tolist()
    conhecidos = set(tabela["id"])
    for nome in ids:
        if nome not in conhecidos or not (raiz / nome).is_dir():
            parser.error(f"amostra ausente do conjunto de teste: {nome}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("attentiongates_focaldice_artigo", device)
    cores = ListedColormap(["white", "lime", "blue", "red"])
    fig, axes = plt.subplots(len(ids), 4, figsize=(16, 4 * len(ids)), squeeze=False)
    for linha, nome in enumerate(ids):
        canais = carregar_sample(raiz / nome)
        label = carregar_label(raiz / nome).numpy() > 0
        with torch.inference_mode():
            pred = postprocess(model(preprocess(canais).to(device))).squeeze().cpu().numpy() > 0
        assert pred.shape == label.shape == tuple(canais[0].shape)

        rgb = np.stack([c.numpy() for c in canais[1:]], axis=-1)
        rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
        mag1c = canais[0].numpy()
        erros = np.zeros(label.shape, dtype=np.uint8)
        erros[pred & label] = 1
        erros[pred & ~label] = 2
        erros[~pred & label] = 3
        for coluna, (imagem, titulo, kwargs) in enumerate((
            (rgb, "RGB", {}),
            (mag1c, "mag1c", {"cmap": "magma", "vmin": 0, "vmax": 1750}),
            (label, "Referência", {"cmap": "gray_r", "vmin": 0, "vmax": 1}),
            (erros, "AttGate FocalDice", {"cmap": cores, "vmin": 0, "vmax": 3}),
        )):
            ax = axes[linha, coluna]
            ax.imshow(imagem, **kwargs)
            ax.set_title(titulo)
            ax.axis("off")
        axes[linha, 0].set_ylabel(nome, rotation=0, labelpad=8)

    fig.legend(handles=[Patch(color=c, label=l) for c, l in
                        (("lime", "TP"), ("blue", "FP"), ("red", "FN"), ("white", "TN"))],
               loc="center right")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.saida, dpi=160)
    plt.close(fig)
    print(args.saida)


if __name__ == "__main__":
    main()
