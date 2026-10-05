"""Compara FP32 e simulacao INT8 nas mesmas imagens e metricas do historico."""
import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from common import PROJECT_ROOT, DEFAULT_PRODUCTS, build_model, validate_calibration, add_model_arguments, resolve_model
from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer
from Testes.Teste_Unet import binary_opening

PRODUCTS = list(DEFAULT_PRODUCTS)


def tiled_logits(model, inputs):
    b, c, h, w = inputs.shape
    if b != 1 or h % 128 or w % 128:
        raise ValueError(f"Avaliacao por patches requer batch 1 e dimensoes multiplas de 128: {inputs.shape}")
    patches = inputs.unfold(2, 128, 128).unfold(3, 128, 128)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, 128, 128)
    logits = model(patches)
    return logits.reshape(h // 128, w // 128, 1, 128, 128).permute(2, 0, 3, 1, 4).reshape(1, 1, h, w)


def image_metrics(logits, target, kernel):
    pred = binary_opening(logits > 0, kernel).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    positive = target.sum().item() > 0
    ap = average_precision_score(target.reshape(-1).numpy(), torch.sigmoid(logits).reshape(-1).numpy()) if positive else None
    return tp, fp, fn, tn, ap


def summarize(rows):
    d = pd.DataFrame(rows)
    tp, fp, fn, tn = d[["TP", "FP", "FN", "TN"]].sum()
    result = {"Imagens": len(d), "F1-Global": 2 * tp / (2 * tp + fp + fn + 1e-6),
              "IoU": tp / (tp + fp + fn + 1e-6), "AUPRC": d.AUPRC.mean(),
              "Acuracia_pixel": (tp + tn) / (tp + fp + fn + tn)}
    for name, selected in [("Strong", d[d.positive & (d.difficulty == "easy")]),
                           ("Weak", d[d.positive & (d.difficulty != "easy")])]:
        t, f, n = selected[["TP", "FP", "FN"]].sum()
        result["F1-" + name] = 2 * t / (2 * t + f + n + 1e-6)
    background = d[~d.positive]
    result["FPR (No-Plume)"] = background.FP.sum() / (background.FP.sum() + background.TN.sum() + 1e-6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    add_model_arguments(parser)
    parser.add_argument("--dataset", choices=["test", "full_remaining_easy"])
    parser.add_argument("--csv", help="CSV do conjunto selecionado por --dataset.")
    parser.add_argument("--data-root", help="Diretorio das imagens do conjunto selecionado.")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", default="build/vitis_ai/evaluation")
    parser.add_argument("--quant-dir", default="build/vitis_ai/quantize",
                        help="Diretorio com as configuracoes calibradas de cada modelo.")
    args = parser.parse_args()
    if (args.csv or args.data_root) and not (args.csv and args.data_root and args.dataset):
        parser.error("--csv e --data-root devem ser usados juntos com --dataset.")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Indice de particao invalido.")
    from pytorch_nndct.apis import torch_quantizer
    torch.set_num_threads(args.num_threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalizer = DataNormalizer(PRODUCTS).eval()
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    summaries = []
    selected = [resolve_model(args.model, args.checkpoint, args.architecture)] if (args.model or args.checkpoint or args.architecture) else [
        resolve_model("attentiongates_dpu_easy_remaining"), resolve_model("attentiongates_dpu_only_remaining")]
    for name, architecture in selected:
        model, checkpoint = build_model(name, args.checkpoint, 4, architecture)
        source_dir = Path(args.quant_dir) / name
        contract = json.loads((source_dir / "calibration_manifest.json").read_text())
        height, width = contract["height"], contract["width"]
        patching, target_name = contract["patching"], contract["target"]
        if (height, width, patching) not in ((128, 128, True), (512, 512, False)):
            raise ValueError("A avaliacao suporta patches 128x128 ou imagens inteiras 512x512.")
        validate_calibration(source_dir, name, checkpoint, PRODUCTS,
                             height, width, target_name, patching, architecture)
        model_output = output / name
        model_output.mkdir(exist_ok=True)
        quant_dir = model_output / "quantize"
        quant_dir.mkdir(exist_ok=True)
        for filename in ["quant_info.json", "bias_corr.pth", "calibration_manifest.json"]:
            shutil.copyfile(Path(args.quant_dir) / name / filename, quant_dir / filename)
        quant_source, _ = build_model(name, args.checkpoint, 4, architecture)
        quantizer = torch_quantizer("test", quant_source, (torch.zeros(1, 4, height, width),),
                                   output_dir=str(quant_dir), device=torch.device("cpu"), target=target_name)
        quant_model = quantizer.quant_model.eval()
        for dataset_name, csv, root in [
            ("test", PROJECT_ROOT / "STARCOP_test/test.csv", PROJECT_ROOT / "STARCOP_test"),
            ("full_remaining_easy", Path("/dataset_STARCOP/train.csv"), Path("/dataset_STARCOP")),
        ]:
            if args.dataset and dataset_name != args.dataset:
                continue
            if args.csv:
                csv, root = Path(args.csv), Path(args.data_root)
            df = carregar_dataframe_starcop(str(csv), str(root), produtos_obrigatorios=PRODUCTS + ["labelbinary"])
            if args.limit:
                df = df.iloc[:args.limit].reset_index(drop=True)
            df = df.iloc[args.shard_index::args.shard_count].reset_index(drop=True)
            loader = DataLoader(STARCOPDataset(df, PRODUCTS, ["labelbinary"]), batch_size=1, num_workers=0)
            modes = ["FP32_512", "FP32_patches128", "INT8_patches128"] if patching else ["FP32_512", "INT8_512"]
            rows = {mode: [] for mode in modes}
            started = time.perf_counter()
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    inputs = normalizer(batch["input"])
                    target = batch["output"]
                    for mode in rows:
                        if mode == "FP32_512":
                            logits = model(inputs)
                        elif mode == "INT8_512":
                            if tuple(inputs.shape[2:]) != (height, width):
                                raise ValueError("Imagem diferente da resolucao calibrada.")
                            logits = quant_model(inputs)
                        else:
                            logits = tiled_logits(quant_model if mode == "INT8_patches128" else model, inputs)
                        if logits.shape != target.shape or not torch.isfinite(logits).all():
                            raise ValueError(f"Saida invalida: {mode}, {df.iloc[i]['id']}")
                        tp, fp, fn, tn, ap = image_metrics(logits, target, kernel)
                        rows[mode].append(dict(id=df.iloc[i]["id"], difficulty=df.iloc[i].get("difficulty", ""),
                                               positive=bool(target.sum() > 0), TP=tp, FP=fp, FN=fn, TN=tn, AUPRC=ap))
                    if (i + 1) % 10 == 0 or i + 1 == len(df):
                        print(f"{name} {dataset_name}: {i+1}/{len(df)} imagens, {time.perf_counter()-started:.1f}s", flush=True)
            for mode, values in rows.items():
                pd.DataFrame(values).to_csv(model_output / f"{name}_{dataset_name}_{mode}.csv", index=False)
                summaries.append(dict(model=name, dataset=dataset_name, mode=mode, checkpoint=str(checkpoint), **summarize(values)))
            pd.DataFrame([row for row in summaries if row["model"] == name]).to_csv(model_output / "summary.csv", index=False)
            print(pd.DataFrame(summaries).tail(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
