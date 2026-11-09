#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from pytorch_nndct.apis import torch_quantizer

from common import (
    DEFAULT_PRODUCTS,
    build_model,
    build_calibration_loader,
    normalized_inputs,
    resolve_model,
    calibration_contract,
    validate_calibration,
    file_sha256,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantização PTQ nativa AMD Vitis AI 3.5")

    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--architecture")
    parser.add_argument("--quant-mode", choices=["calib", "test"], required=True)
    parser.add_argument("--csv", default="/dataset_STARCOP/train.csv")
    parser.add_argument("--data-root", default="/dataset_STARCOP")
    parser.add_argument("--subset-len", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
    parser.add_argument("--output-dir", default="build/vitis_ai/quantize_amd")
    parser.add_argument("--deploy", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.subset_len < 1 or args.num_workers < 0:
        raise ValueError("Parâmetros de calibração inválidos.")

    if args.deploy and args.quant_mode != "test":
        raise ValueError("--deploy requer --quant-mode test.")

    model_name, architecture = resolve_model(
        args.model, args.checkpoint, args.architecture
    )

    model, checkpoint = build_model(
        model_name, args.checkpoint, len(DEFAULT_PRODUCTS), architecture
    )

    device = torch.device("cpu")
    model = model.to(device).eval()

    output_dir = Path(args.output_dir) / model_name
    manifest_path = output_dir / "calibration_manifest.json"
    quant_config_path = output_dir / "quant_info.json"

    if args.quant_mode == "calib" and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Diretório já contém arquivos: {output_dir}")

    if args.quant_mode == "test":
        validate_calibration(
            output_dir, model_name, checkpoint,
            list(DEFAULT_PRODUCTS), 512, 512,
            args.target, False, architecture
        )

    loader = build_calibration_loader(
        csv_path=args.csv,
        data_root=args.data_root,
        products=list(DEFAULT_PRODUCTS),
        subset_len=1 if args.deploy else args.subset_len,
        batch_size=1,
        num_workers=args.num_workers,
        seed=12345,
        patching=False,
        balanced=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(1, len(DEFAULT_PRODUCTS), 512, 512)

    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(example,),
        output_dir=str(output_dir),
        device=device,
        target=args.target,
    )

    quant_model = quantizer.quant_model.eval()

    count = 0

    with torch.no_grad():
        for inputs in normalized_inputs(loader, list(DEFAULT_PRODUCTS), device):
            if tuple(inputs.shape) != (1, len(DEFAULT_PRODUCTS), 512, 512):
                raise ValueError(f"Dimensões inválidas: {tuple(inputs.shape)}")

            if not torch.isfinite(inputs).all():
                raise ValueError("Entrada contém NaN ou infinito.")

            quant_model(inputs)
            count += 1

            print(f"{args.quant_mode}: {count} amostras", flush=True)

            if args.deploy:
                break

    if args.quant_mode == "calib":
        quantizer.export_quant_config()

        if not quant_config_path.is_file():
            raise RuntimeError("Quantização não gerou quant_info.json.")

        manifest = calibration_contract(
            model_name, checkpoint, list(DEFAULT_PRODUCTS),
            512, 512, args.target, False, architecture
        )

        manifest.update({
            "checkpoint": str(checkpoint),
            "quantization_method": "AMD_NATIVE_PTQ",
            "subset_len": count,
            "seed": 12345,
            "csv": str(Path(args.csv).resolve()),
            "csv_sha256": file_sha256(args.csv),
            "images": loader.dataset.dataframe.id.tolist(),
            "artifacts": {
                filename: file_sha256(output_dir / filename)
                for filename in ("quant_info.json", "bias_corr.pth")
                if (output_dir / filename).is_file()
            },
        })

        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Calibração AMD concluída: {output_dir}")

    elif args.deploy:
        quantizer.export_xmodel(output_dir=str(output_dir))
        print(f"XModel exportado: {output_dir}")

    else:
        print("Inferência INT8 concluída.")


if __name__ == "__main__":
    main()