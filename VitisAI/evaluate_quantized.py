#!/usr/bin/env python3
"""Compara FP32 e simulacao INT8 em CPU, sem executar a DPU da placa."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import average_precision_score

from common import MODEL_REGISTRY, build_model, parse_products
from Utils.DataLoader import DataNormalizer, STARCOPDataset, carregar_dataframe_starcop


def opening(mask):
    # Equivale ao binary_opening do Teste_Unet, inclusive nas bordas:
    # a erosao geodesica ignora valores fora da imagem; a dilatacao usa zero.
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)[None, None]
    return binary_dilation(
        binary_erosion(mask, structure=kernel, border_value=1),
        structure=kernel,
        border_value=0,
    )


def patches(tensor):
    _, height, width = tensor.shape
    return torch.stack([
        tensor[:, y:y + 128, x:x + 128]
        for y in range(0, height - 128 + 1, 64)
        for x in range(0, width - 128 + 1, 64)
    ])


def counts(prediction, target):
    return np.array([
        np.count_nonzero(prediction & target),
        np.count_nonzero(prediction & ~target),
        np.count_nonzero(~prediction & target),
        np.count_nonzero(~prediction & ~target),
    ], dtype=np.int64)


def scores(values):
    tp, fp, fn, tn = (int(v) for v in values)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "F1": 2 * tp / (2 * tp + fp + fn + 1e-6),
        "IoU": tp / (tp + fp + fn + 1e-6),
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def evaluate(args):
    from pytorch_nndct.apis import torch_quantizer

    torch.set_num_threads(args.threads)
    products = parse_products(args.products)
    fp32, checkpoint = build_model(args.model, args.checkpoint, len(products))
    reference_result = None
    if args.fp32_reference:
        reference = json.loads(Path(args.fp32_reference).read_text())
        if (
            reference["checkpoint_sha256"] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            or reference["dataset_csv_sha256"] != hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
            or reference["products"] != products
            or reference.get("input_mode", "full") != args.input_mode
            or reference["device"] != "cpu"
            or reference["torch"] != torch.__version__
            or args.limit != 0
        ):
            raise ValueError("Referencia FP32 nao corresponde ao checkpoint, dataset ou protocolo atual.")
        reference_mode = "fp32_patches128" if args.input_mode == "patches" else "fp32_full512"
        reference_result = next(r for r in reference["results"] if r["mode"] == reference_mode)
    config_path = Path(args.quant_dir) / "quant_info.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_before = config_path.read_bytes()
    input_size = 128 if args.input_mode == "patches" else 512
    # O quantizador pode transformar/equalizar o modelo recebido: mantenha
    # uma instancia independente para a referencia FP32.
    quantizer = torch_quantizer(
        quant_mode="test", module=copy.deepcopy(fp32),
        input_args=(torch.zeros(1, len(products), input_size, input_size),),
        output_dir=str(args.quant_dir), device=torch.device("cpu"),
        target=args.target,
    )
    int8 = quantizer.quant_model.eval()
    modes = {
        "fp32_full512": (fp32, False),
        "int8_full512_cpu_simulation": (int8, False),
        "fp32_patches128": (fp32, True),
    }
    if args.input_mode == "patches":
        modes = {
            "fp32_patches128": (fp32, True),
            "int8_patches128_cpu_simulation": (int8, True),
        }
    if reference_result is not None:
        modes = {name: value for name, value in modes.items() if name.startswith("int8_")}
    df = carregar_dataframe_starcop(
        args.csv, args.data_root,
        produtos_obrigatorios=products + ["labelbinary"],
        limite_amostras=args.limit or None,
    )
    if df.empty:
        raise RuntimeError("Dataset vazio.")
    dataset = STARCOPDataset(df, products, ["labelbinary"], patching=False)
    normalizer = DataNormalizer(products).eval()
    accumulators = {
        name: {"all": np.zeros(4, dtype=np.int64),
               "strong": np.zeros(4, dtype=np.int64),
               "weak": np.zeros(4, dtype=np.int64),
               "no_plume": np.zeros(4, dtype=np.int64),
               "ap": [], "latencies_ms": []}
        for name in modes
    }
    records = []
    started = time.perf_counter()
    with torch.no_grad():
        # Aquecimento fora dos tempos de inferencia.
        for name, (model, use_patches) in modes.items():
            shape = (49, len(products), 128, 128) if use_patches else (1, len(products), 512, 512)
            for _ in range(3):
                model(torch.zeros(shape))
        for index in range(len(dataset)):
            sample = dataset[index]
            if tuple(sample["input"].shape[-2:]) != (512, 512):
                raise ValueError("Esta comparacao requer imagens de 512x512.")
            target_full = sample["output"].unsqueeze(0)
            if not bool(((target_full == 0) | (target_full == 1)).all()):
                raise ValueError("Mascara deve conter somente 0 e 1.")
            inputs_full = normalizer(sample["input"].unsqueeze(0))
            inputs_patches = patches(inputs_full[0])
            target_patches = patches(sample["output"])
            has_plume = bool(target_full.sum() > 0)
            difficulty = df.iloc[index].get("difficulty", "")
            group = ("strong" if difficulty == "easy" else "weak") if has_plume else "no_plume"
            for name, (model, use_patches) in modes.items():
                inputs = inputs_patches if use_patches else inputs_full
                target = (target_patches if use_patches else target_full).numpy().astype(bool)
                start = time.perf_counter()
                logits = model(inputs)
                latency_ms = (time.perf_counter() - start) * 1000
                if logits.shape != torch.Size(target.shape) or not bool(logits.isfinite().all()):
                    raise RuntimeError("Saida invalida em " + name)
                prediction = opening((logits > 0).numpy())
                values = counts(prediction, target)
                acc = accumulators[name]
                acc["all"] += values
                acc[group] += values
                acc["latencies_ms"].append(latency_ms)
                ap = None
                if has_plume:
                    ap = float(average_precision_score(
                        target.reshape(-1), torch.sigmoid(logits).numpy().reshape(-1),
                    ))
                    acc["ap"].append(ap)
                records.append({
                    "mode": name, "row": index,
                    "folder": df.iloc[index]["folder"], "difficulty": difficulty,
                    "group": group, **scores(values), "AP": ap,
                    "inference_ms": latency_ms,
                })
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print(f"evaluation: {index + 1}/{len(dataset)} images; {time.perf_counter() - started:.1f}s", flush=True)

    assert config_path.read_bytes() == config_before, "Configuracao de quantizacao mudou."
    results = [reference_result] if reference_result is not None else []
    if reference_result is not None and reference_result["images"] != len(dataset):
        raise ValueError("Quantidade de imagens difere da referencia FP32.")
    for name, acc in accumulators.items():
        result = {"mode": name, "images": len(dataset), **scores(acc["all"])}
        for group in ["strong", "weak"]:
            result.update({group + "_" + k: v for k, v in scores(acc[group]).items()})
        fp, tn = int(acc["no_plume"][1]), int(acc["no_plume"][3])
        result.update({
            "AUPRC": float(np.mean(acc["ap"])) if acc["ap"] else 0.0,
            "FPR_no_plume": fp / (fp + tn + 1e-6),
            "inference_ms_mean": float(np.mean(acc["latencies_ms"])),
        })
        results.append(result)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(records).to_csv(output / "per_image.csv", index=False)
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "quant_config_sha256": hashlib.sha256(config_before).hexdigest(),
        "dataset_csv": str(args.csv), "dataset_csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        "data_root": str(args.data_root), "products": products,
        "device": "cpu", "threads": args.threads, "torch": torch.__version__,
        "target": args.target,
        "input_mode": args.input_mode,
        "fp32_reference": args.fp32_reference,
        "protocol": "logits>0; cross3x3 opening; pooled counts; strong=difficulty easy; mean AP over plume images",
        "int8_execution": "Vitis AI PyTorch simulation in test mode, not execution of XModel on the board",
        "latency_scope": "model forward only; CPU INT8 simulation is not DPU throughput",
        "results": results,
    }
    (output / "comparison.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(results)[["mode", "F1", "strong_F1", "weak_F1", "IoU", "AUPRC", "FPR_no_plume"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_REGISTRY, default="mobilenet_v3_attention_gates_dpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--quant-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--products", default="mag1c,TOA_AVIRIS_460nm,TOA_AVIRIS_550nm,TOA_AVIRIS_640nm")
    parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--input-mode", choices=("full", "patches"), default="full")
    parser.add_argument("--fp32-reference", help="JSON de uma avaliacao FP32 identica; evita repetir a referencia.")
    evaluate(parser.parse_args())
