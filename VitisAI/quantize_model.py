#!/usr/bin/env python3
"""Calibra, testa e exporta um modelo INT8 com vai_q_pytorch."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import (
    DEFAULT_PRODUCTS,
    MODEL_REGISTRY,
    build_calibration_loader,
    build_model,
    normalized_inputs,
    parse_products,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_REGISTRY, default="depth_reduced")
    parser.add_argument("--checkpoint")
    parser.add_argument("--quant-mode", choices=("calib", "test"), required=True)
    parser.add_argument("--csv", required=True, help="CSV STARCOP usado na calibracao/validacao.")
    parser.add_argument("--data-root", required=True, help="Diretorio que contem as pastas das imagens.")
    parser.add_argument("--products", default=",".join(DEFAULT_PRODUCTS))
    parser.add_argument("--subset-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--target", help="Opcional: habilita quantizacao consciente do hardware.")
    parser.add_argument("--output-dir", default="build/vitis_ai/quantize")
    parser.add_argument("--deploy", action="store_true", help="No modo test, exporta o xmodel INT8.")
    parser.add_argument("--deploy-check", action="store_true")
    return parser.parse_args()


def quantize(args):
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError as exc:
        raise SystemExit("pytorch_nndct nao encontrado. Execute dentro do container PyTorch do Vitis AI.") from exc

    if args.deploy and args.quant_mode != "test":
        raise SystemExit("--deploy somente pode ser usado com --quant-mode test.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size deve ser maior que zero.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers nao pode ser negativo.")
    if args.deploy:
        # Exigencia do export_xmodel: batch 1 e uma unica inferencia.
        args.batch_size = 1
        args.subset_len = 1

    products = parse_products(args.products)
    model, checkpoint = build_model(args.model, args.checkpoint, len(products))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = build_calibration_loader(
        args.csv,
        args.data_root,
        products,
        args.subset_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(args.batch_size, len(products), args.height, args.width, device=device)

    kwargs = dict(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(example,),
        output_dir=str(output_dir),
        device=device,
    )
    if args.target:
        kwargs["target"] = args.target
    quantizer = torch_quantizer(**kwargs)
    quant_model = quantizer.quant_model.eval()

    count = 0
    total = len(loader.dataset)
    started_at = time.perf_counter()
    with torch.inference_mode():
        for inputs in normalized_inputs(loader, products, device):
            quant_model(inputs)
            count += inputs.shape[0]
            if count == total or (
                args.progress_every > 0 and count % args.progress_every < inputs.shape[0]
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    f"{args.quant_mode}: {count}/{total} amostras "
                    f"({elapsed:.1f}s, {elapsed / count:.2f}s/amostra)",
                    flush=True,
                )
    elapsed = time.perf_counter() - started_at
    print(
        f"{args.quant_mode}: {count} amostras processadas em {elapsed:.1f}s; "
        f"checkpoint={checkpoint}"
    )

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
    elif args.deploy:
        # A exportacao deve ocorrer com batch 1 e apenas uma execucao e usa a
        # configuracao gerada anteriormente no mesmo output_dir.
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=args.deploy_check)
        print(f"XModel quantizado exportado em: {output_dir}")


if __name__ == "__main__":
    quantize(parse_args())
