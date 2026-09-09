#!/usr/bin/env python3
"""Inspeciona quais operadores do modelo podem ser executados na DPU."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import DEFAULT_PRODUCTS, MODEL_REGISTRY, build_model, parse_products


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_REGISTRY, default="depth_reduced")
    parser.add_argument("--checkpoint", help="Sobrescreve o checkpoint padrao do modelo.")
    parser.add_argument("--products", default=",".join(DEFAULT_PRODUCTS))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--target", required=True, help="Fingerprint/nome da DPU aceito pelo Inspector.")
    parser.add_argument("--output-dir", default="build/vitis_ai/inspect")
    return parser.parse_args()


def inspect_model(args):
    try:
        from pytorch_nndct.apis import Inspector
    except ImportError as exc:
        raise SystemExit("pytorch_nndct nao encontrado. Execute dentro do container PyTorch do Vitis AI.") from exc

    products = parse_products(args.products)
    model, checkpoint = build_model(args.model, args.checkpoint, len(products))
    dummy = torch.randn(1, len(products), args.height, args.width)

    # O trace detecta cedo construcoes PyTorch que o quantizador nao consegue capturar.
    torch.jit.trace(model, dummy, strict=True)
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modelo: {args.model}\nCheckpoint: {checkpoint}\nSaida: {output_dir}")
    Inspector(args.target).inspect(model, (dummy,), device=torch.device("cpu"), output_dir=str(output_dir))


if __name__ == "__main__":
    inspect_model(parse_args())

