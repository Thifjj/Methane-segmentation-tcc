import time
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
import csv
from datetime import datetime

from .model_loader import load_model
from .dataset import (
    carregar_sample, encontrar_sample, carregar_label,
    carregar_classificacao_por_pasta,
)
from .preprocess import preprocess
from .postprocess import postprocess
from .metricas import calcular_metricas, calcular_f1_contagens, classificar_pluma

#MODEL_PATH = "/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/Mobile_Net_v3_mag1c_rgb.pth"
DATASET_PATH = "/media/jacques/games/Datasets/STARCOP_train_remaining_all"

WARMUP = 10

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modelo",
    required=True,
    choices=[
        "baseline",
        "depth_reduced",
        "mobilenet_v2",
        "mobilenet_v3",
        "skip"
    ]
)

args = parser.parse_args()

device = torch.device("cuda")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA não disponível")

print("GPU:", torch.cuda.get_device_name(0))

model = load_model(args.modelo, device)

amostras = encontrar_sample(DATASET_PATH)
classificacao_por_pasta = carregar_classificacao_por_pasta(DATASET_PATH)

print("Amostras encontradas:", len(amostras))
print("Primeira amostra:", amostras[0])

canais = carregar_sample(amostras[0])
label = carregar_label(amostras[0])

print("Shapes dos canais:")
for canal in canais:
    print(canal.shape)

print("Shape label:", label.shape)

quantidade = -1

while quantidade < 0 or quantidade > len(amostras):
    quantidade = int(
        input(f"Quantas imagens deseja usar? (0 = todas, máximo {len(amostras)}): ")
    )

if quantidade > 0:
    amostras = amostras[:quantidade]

print("Amostras usadas:", len(amostras))

# WARMUP
canais = carregar_sample(amostras[0])
entrada = preprocess(canais)
entrada = entrada.to(device)

with torch.inference_mode():
    for _ in range(WARMUP):
        model(entrada)

torch.cuda.synchronize()

tempos_model = []
tempos_e2e = []
tempos_preprocess = []
tempos_posprocess = []
tempo_carregamento = []

tp_total = 0
fp_total = 0
fn_total = 0
tn_total = 0
contagens_grupo = {
    "strong_plume": [0, 0, 0],
    "weak_plume": [0, 0, 0],
}

with torch.inference_mode():
    for pasta in tqdm(amostras, desc="Benchmark"):
        inicio_e2e = time.perf_counter()

        inicio_carregamento = time.perf_counter()
        canais = carregar_sample(pasta)
        fim_carregamento = time.perf_counter()

        inicio_preprocess = time.perf_counter()
        entrada = preprocess(canais)
        entrada = entrada.to(device)
        torch.cuda.synchronize()
        fim_preprocess = time.perf_counter()

        torch.cuda.synchronize()
        inicio_model = time.perf_counter()
        saida = model(entrada)
        torch.cuda.synchronize()
        fim_model = time.perf_counter()

        inicio_posprocess = time.perf_counter()
        mascara = postprocess(saida)
        torch.cuda.synchronize()
        fim_posprocess = time.perf_counter()

        fim_e2e = time.perf_counter()

        label = carregar_label(pasta)

        mascara = mascara.squeeze().cpu()

        tp, fp, fn, tn, _, _, _, _ = calcular_metricas(
            mascara,
            label
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        has_plume, qplume = classificacao_por_pasta[
            os.path.basename(os.path.normpath(pasta))
        ]
        grupo = classificar_pluma(has_plume, qplume)
        if grupo is not None:
            contagens_grupo[grupo][0] += tp
            contagens_grupo[grupo][1] += fp
            contagens_grupo[grupo][2] += fn

        tempos_model.append((fim_model-inicio_model)*1000)
        tempos_e2e.append((fim_e2e-inicio_e2e)*1000)
        tempos_preprocess.append((fim_preprocess-inicio_preprocess)*1000)
        tempos_posprocess.append((fim_posprocess-inicio_posprocess)*1000)
        tempo_carregamento.append((fim_carregamento-inicio_carregamento)*1000)


precision = tp_total / (tp_total + fp_total)

recall = tp_total / (tp_total + fn_total)

f1_global = calcular_f1_contagens(tp_total, fp_total, fn_total)
f1_strong_plume = calcular_f1_contagens(*contagens_grupo["strong_plume"])
f1_weak_plume = calcular_f1_contagens(*contagens_grupo["weak_plume"])

iou = tp_total / (tp_total + fp_total + fn_total)

fpr = fp_total / (fp_total + tn_total)

media_e2e = np.mean(tempos_e2e)

print("\n===== MODEL ONLY =====")
print(f"Média:   {np.mean(tempos_model):.3f} ms")
print(f"Mediana: {np.median(tempos_model):.3f} ms")
print(f"Mínimo:  {np.min(tempos_model):.3f} ms")
print(f"Máximo:  {np.max(tempos_model):.3f} ms")
print(f"P95:     {np.percentile(tempos_model, 95):.3f} ms")
print(f"P99:     {np.percentile(tempos_model, 99):.3f} ms")
print(f"FPS:     {1000 / np.mean(tempos_model):.2f}")

print("\n===== E2E =====")
print(f"Média:   {np.mean(tempos_e2e):.3f} ms")
print(f"Mediana: {np.median(tempos_e2e):.3f} ms")
print(f"Mínimo:  {np.min(tempos_e2e):.3f} ms")
print(f"Máximo:  {np.max(tempos_e2e):.3f} ms")
print(f"P95:     {np.percentile(tempos_e2e, 95):.3f} ms")
print(f"P99:     {np.percentile(tempos_e2e, 99):.3f} ms")
print(f"FPS:     {1000 / np.mean(tempos_e2e):.2f}")

print("\n===== MÉTRICAS =====")
print("TP:", tp_total)
print("FP:", fp_total)
print("FN:", fn_total)
print("TN:", tn_total)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 global:       {f1_global:.4f}")
print(f"F1 strong plume: {f1_strong_plume:.4f}")
print(f"F1 weak plume:   {f1_weak_plume:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"FPR:       {fpr:.6f}")

print("\n===== CARREGAMENTO =====")
print(f"Média:   {np.mean(tempo_carregamento):.3f} ms")
print(f"Mediana: {np.median(tempo_carregamento):.3f} ms")
print(f"Mínimo:  {np.min(tempo_carregamento):.3f} ms")
print(f"Máximo:  {np.max(tempo_carregamento):.3f} ms")
print(f"P95:     {np.percentile(tempo_carregamento, 95):.3f} ms")
print(f"P99:     {np.percentile(tempo_carregamento, 99):.3f} ms")

print("\n===== PREPROCESS =====")
print(f"Média:   {np.mean(tempos_preprocess):.3f} ms")
print(f"Mediana: {np.median(tempos_preprocess):.3f} ms")
print(f"Mínimo:  {np.min(tempos_preprocess):.3f} ms")
print(f"Máximo:  {np.max(tempos_preprocess):.3f} ms")
print(f"P95:     {np.percentile(tempos_preprocess, 95):.3f} ms")
print(f"P99:     {np.percentile(tempos_preprocess, 99):.3f} ms")

print("\n===== POSPROCESS =====")
print(f"Média:   {np.mean(tempos_posprocess):.3f} ms")
print(f"Mediana: {np.median(tempos_posprocess):.3f} ms")
print(f"Mínimo:  {np.min(tempos_posprocess):.3f} ms")
print(f"Máximo:  {np.max(tempos_posprocess):.3f} ms")
print(f"P95:     {np.percentile(tempos_posprocess, 95):.3f} ms")
print(f"P99:     {np.percentile(tempos_posprocess, 99):.3f} ms")

print("\n===== DISTRIBUIÇÃO DO E2E =====")
print(f"Carregamento: {np.mean(tempo_carregamento):.3f} ms "
      f"({np.mean(tempo_carregamento) / media_e2e * 100:.1f}%)")

print(f"Preprocess:   {np.mean(tempos_preprocess):.3f} ms "
      f"({np.mean(tempos_preprocess) / media_e2e * 100:.1f}%)")

print(f"Modelo:       {np.mean(tempos_model):.3f} ms "
      f"({np.mean(tempos_model) / media_e2e * 100:.1f}%)")

print(f"Posprocess:   {np.mean(tempos_posprocess):.3f} ms "
      f"({np.mean(tempos_posprocess) / media_e2e * 100:.1f}%)")

data_hora = datetime.now().strftime("%Y%m%d_%H%M")

ARQUIVO_RESULTADOS = f"benchmark_manual/resultado_{args.modelo}_{data_hora}.csv"

resultado = {
    "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": args.modelo,
    "imagens": len(amostras),

    "model_media_ms": np.mean(tempos_model),
    "model_mediana_ms": np.median(tempos_model),
    "model_p95_ms": np.percentile(tempos_model, 95),
    "model_p99_ms": np.percentile(tempos_model, 99),
    "model_fps": 1000 / np.mean(tempos_model),

    "e2e_media_ms": np.mean(tempos_e2e),
    "e2e_mediana_ms": np.median(tempos_e2e),
    "e2e_p95_ms": np.percentile(tempos_e2e, 95),
    "e2e_p99_ms": np.percentile(tempos_e2e, 99),
    "e2e_fps": 1000 / np.mean(tempos_e2e),

    "carregamento_ms": np.mean(tempo_carregamento),
    "preprocess_ms": np.mean(tempos_preprocess),
    "posprocess_ms": np.mean(tempos_posprocess),

    "precision": precision,
    "recall": recall,
    "f1_global": f1_global,
    "f1_strong_plume": f1_strong_plume,
    "f1_weak_plume": f1_weak_plume,
    "iou": iou,
    "fpr": fpr,

    "tp": tp_total,
    "fp": fp_total,
    "fn": fn_total,
    "tn": tn_total
}
with open(ARQUIVO_RESULTADOS, "w", newline="") as arquivo:
    escritor = csv.DictWriter(
        arquivo,
        fieldnames=resultado.keys()
    )

    escritor.writeheader()
    escritor.writerow(resultado)

print(f"\nResultado salvo em: {ARQUIVO_RESULTADOS}")
