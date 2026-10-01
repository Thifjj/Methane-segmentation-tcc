import time
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
import csv
import ctypes
import re
from datetime import datetime
from pathlib import Path
from Utils.DataLoader import carregar_dataframe_starcop

from .model_loader import load_model, MODEL_REGISTRY, modelos_disponiveis, produtos_modelo, caminho_checkpoint
from .dataset import (
    carregar_sample, carregar_label,
)
from .preprocess import preprocess, recortar_patches
from .postprocess import postprocess
from .metricas import calcular_metricas, calcular_f1_contagens, classificar_pluma, calcular_auprc_imagem

#MODEL_PATH = "/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/Mobile_Net_v3_mag1c_rgb.pth"
#DATASET_PATH = "/media/jacques/games/Datasets/STARCOP_train_remaining_all"
DATASETS = {
    "full": ("/media/jacques/games/Datasets/STARCOP_train_remaining_all", "train.csv"),
    "test": (
        "/media/jacques/games/Datasets/test/STARCOP_test",
        "test.csv",
    ),
}
WARMUP = 10


class MedidorEnergia:
    """Energia do pacote CPU (RAPL) e, em CUDA, da GPU (NVML)."""
    def __init__(self, dispositivo):
        self.fontes = {}
        self.nvml = None
        self.fds = []
        for zona in Path("/sys/class/powercap").glob("*-rapl:*"):
            if zona.name.count(":") != 1:
                continue  # Subzonas de núcleos já estão incluídas no pacote.
            try:
                if not (zona / "name").read_text().strip().startswith("package-"):
                    continue
                limite = int((zona / "max_energy_range_uj").read_text())
                int((zona / "energy_uj").read_text())
                self.fontes[f"cpu_{zona.name}"] = (
                    lambda z=zona: int((z / "energy_uj").read_text()), 1e-6, limite
                )
            except (OSError, ValueError):
                pass

        if not any(nome.startswith("cpu_") for nome in self.fontes):
            self._abrir_rapl_perf()
        if dispositivo.type == "cuda":
            self._abrir_nvml()

    def _abrir_rapl_perf(self):
        """Alternativa quando o RAPL do sysfs não é legível."""
        try:
            evento = Path("/sys/bus/event_source/devices/power/events/energy-pkg")
            codigo = re.search(r"event=(0x[0-9a-fA-F]+|[0-9]+)", evento.read_text())
            if codigo is None:
                return
            tipo = int(Path("/sys/bus/event_source/devices/power/type").read_text())
            escala = float(Path(f"{evento}.scale").read_text())
            attr = (ctypes.c_ubyte * 128)()
            ctypes.c_uint32.from_buffer(attr, 0).value = tipo
            ctypes.c_uint32.from_buffer(attr, 4).value = 128
            ctypes.c_uint64.from_buffer(attr, 8).value = int(codigo.group(1), 0)
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.syscall(298, ctypes.byref(attr), -1, 0, -1, 0)  # x86_64
            if fd < 0:
                return
            self.fds.append(fd)
            self.fontes["cpu_rapl_pkg"] = (
                lambda f=fd: self._ler_perf(f), escala, None
            )
        except (OSError, ValueError, AttributeError):
            pass

    @staticmethod
    def _ler_perf(fd):
        dados = os.read(fd, 8)
        if len(dados) != 8:
            raise OSError("Leitura incompleta do contador RAPL")
        return int.from_bytes(dados, "little")

    def _abrir_nvml(self):
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            nvml.nvmlInit_v2.restype = ctypes.c_int
            if nvml.nvmlInit_v2() != 0:
                return
            self.nvml = nvml
            nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
            ]
            nvml.nvmlDeviceGetTotalEnergyConsumption.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)
            ]
            indice = torch.cuda._get_nvml_device_index(0)
            gpu = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(indice, ctypes.byref(gpu)) != 0:
                return

            def ler_gpu():
                energia = ctypes.c_ulonglong()
                if nvml.nvmlDeviceGetTotalEnergyConsumption(gpu, ctypes.byref(energia)) != 0:
                    raise OSError("Contador de energia NVML indisponível")
                return energia.value

            ler_gpu()
            self.fontes["gpu_nvml"] = (ler_gpu, 1e-3, None)
        except (OSError, AttributeError, RuntimeError):
            pass

    def ler(self):
        valores = {}
        for nome, (ler, _, _) in self.fontes.items():
            try:
                valores[nome] = ler()
            except (OSError, ValueError):
                pass
        return time.perf_counter(), valores

    def fechar(self):
        for fd in self.fds:
            os.close(fd)
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


def medir_energia(medidor, antes, depois):
    medidas = {}
    duracao = depois[0] - antes[0]
    for nome in antes[1].keys() & depois[1].keys():
        _, escala, limite = medidor.fontes[nome]
        diferenca = depois[1][nome] - antes[1][nome]
        if diferenca < 0 and limite is not None:
            diferenca += limite  # RAPL volta a zero após max_energy_range_uj.
        if diferenca >= 0 and duracao > 0:
            joules = diferenca * escala
            medidas[nome] = (joules, joules / duracao, duracao)
    return medidas

parser = argparse.ArgumentParser(description="Benchmark manual geral de inferência")

parser.add_argument("--modelo", choices=tuple(MODEL_REGISTRY))
parser.add_argument("--dataset", choices=("full", "test"))
parser.add_argument("--device", choices=("cpu", "cuda", "gpu"))
parser.add_argument("--quantidade", type=int)
parser.add_argument("--input-mode", choices=("patches", "full"), default="patches")
parser.add_argument("--checkpoint", help="Checkpoint alternativo, sem substituir os pesos padrão")
parser.add_argument("--produtos", help="Produtos separados por vírgula, na ordem usada no treinamento")
parser.add_argument("--data-root", help="Diretório alternativo do dataset")
parser.add_argument("--csv-name", help="Nome do CSV dentro do diretório do dataset")
parser.add_argument("--output", help="Caminho alternativo do CSV de resultados")
args = parser.parse_args()

if args.modelo is None:
    disponiveis = modelos_disponiveis()
    print("Modelos disponíveis com checkpoint:")
    for indice, nome in enumerate(disponiveis, start=1):
        print(f"{indice} - {nome}")
    while True:
        escolha = input("Escolha o modelo: ").strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(disponiveis):
            args.modelo = disponiveis[int(escolha) - 1]
            break
        print("Opção inválida.")
if args.dataset is None:
    print("Datasets: 1 - full; 2 - test")
    while True:
        escolha = input("Escolha o dataset [1/2]: ").strip()
        if escolha in ("1", "2"):
            args.dataset = "full" if escolha == "1" else "test"
            break
DATASET_PATH, CSV_DATASET = DATASETS[args.dataset]
DATASET_PATH = args.data_root or DATASET_PATH
CSV_DATASET = args.csv_name or CSV_DATASET
if args.device is None:
    while True:
        escolha = input("Dispositivo: 1 - CPU; 2 - CUDA [1/2]: ").strip()
        if escolha in ("1", "2"):
            args.device = "cpu" if escolha == "1" else "gpu"
            break
if args.device == "cuda":
    args.device = "gpu"
if args.device == "gpu" and not torch.cuda.is_available():
    raise RuntimeError("GPU solicitada, mas CUDA não está disponível")
device = torch.device("cuda" if args.device == "gpu" else "cpu")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
produtos = tuple(p.strip() for p in args.produtos.split(",")) if args.produtos else produtos_modelo(args.modelo)
print("Produtos na ordem:", produtos)
print("Modo de entrada:", args.input_mode)
print("Checkpoint:", caminho_checkpoint(args.modelo, args.checkpoint))
model = load_model(args.modelo, device, checkpoint=args.checkpoint, produtos=produtos)

df = carregar_dataframe_starcop(
    str(Path(DATASET_PATH) / CSV_DATASET), DATASET_PATH,
    produtos_obrigatorios=list(produtos) + ["labelbinary"],
)
if df.empty:
    raise RuntimeError("Nenhuma amostra válida encontrada")
quantidade = args.quantidade
if quantidade is None:
    while True:
        escolha = input(f"Quantas imagens? (0=todas, máximo {len(df)}): ").strip()
        if escolha.isdigit() and int(escolha) <= len(df):
            quantidade = int(escolha)
            break
if quantidade < 0 or quantidade > len(df):
    raise ValueError(f"Quantidade deve estar entre 0 e {len(df)}")
if quantidade:
    df = df.iloc[:quantidade]
amostras = list(df["folder"])

print("Amostras usadas:", len(amostras))

# WARMUP
canais = carregar_sample(amostras[0], produtos, window=df.iloc[0]["window"])
entrada = preprocess(canais, produtos, args.input_mode).to(device)

with torch.inference_mode():
    for _ in range(WARMUP):
        saida_aquecimento = model(entrada)
        torch.sigmoid(saida_aquecimento)
        postprocess(saida_aquecimento)
    del saida_aquecimento
if device.type == "cuda":
    torch.cuda.synchronize()

medidor_energia = MedidorEnergia(device)
energia_por_modo = {
    modo: {fonte: [] for fonte in medidor_energia.fontes}
    for modo in ("model_only", "end_to_end")
}
if not medidor_energia.fontes:
    print("Energia indisponível: sem acesso ao RAPL ou ao contador NVML.")
tempos_model = []
tempos_e2e = []
tempos_preprocess = []
tempos_posprocess = []
tempo_carregamento = []

tp_total = 0
fp_total = 0
fn_total = 0
tn_total = 0
fp_tiles = 0
tn_tiles = 0

auprc_por_imagem = []
fp_no_plume = 0
tn_no_plume = 0
patches_por_imagem = entrada.shape[0]
contagens_grupo = {
    "strong_plume": [0, 0, 0],
    "weak_plume": [0, 0, 0],
}

with torch.inference_mode():
    for indice, pasta in enumerate(tqdm(amostras, desc="Benchmark")):
        row = df.iloc[indice]
        window = row["window"]
        energia_e2e_antes = medidor_energia.ler()
        inicio_e2e = time.perf_counter()

        inicio_carregamento = time.perf_counter()
        canais = carregar_sample(pasta, produtos, window=window)
        fim_carregamento = time.perf_counter()

        inicio_preprocess = time.perf_counter()
        entrada = preprocess(canais, produtos, args.input_mode).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fim_preprocess = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.synchronize()
        energia_model_antes = medidor_energia.ler()
        inicio_model = time.perf_counter()

        saida = model(entrada)
        if device.type == "cuda":
            torch.cuda.synchronize()

        fim_model = time.perf_counter()
        energia_model_depois = medidor_energia.ler()
        for evento, medida in medir_energia(
            medidor_energia, energia_model_antes, energia_model_depois
        ).items():
            energia_por_modo["model_only"][evento].append(medida)

        inicio_posprocess = time.perf_counter()
        
        probs = torch.sigmoid(saida)
        mascara = postprocess(saida)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        fim_posprocess = time.perf_counter()

        fim_e2e = time.perf_counter()
        energia_e2e_depois = medidor_energia.ler()
        for evento, medida in medir_energia(
            medidor_energia, energia_e2e_antes, energia_e2e_depois
        ).items():
            energia_por_modo["end_to_end"][evento].append(medida)
        
        label = carregar_label(pasta, window=window)
        label = recortar_patches(label.unsqueeze(0)) if args.input_mode == "patches" else label[None, None]
        mascara = mascara.cpu()
        probs_cpu = probs.cpu()
        auprc_img = calcular_auprc_imagem(label.numpy(), probs_cpu.numpy())
        if auprc_img is not None:
            auprc_por_imagem.append(auprc_img)

        tp, fp, fn, tn, _, _, _, _ = calcular_metricas(
            mascara,
            label
        )

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        # A máscara real determina presença de pluma, como em Teste_Unet.
        has_plume = (tp + fn) > 0
        if not has_plume:
            fp_no_plume += fp
            tn_no_plume += tn
        # FPR BY TILE do artigo
        if not has_plume:

            pixels_preditos = mascara.sum().item()

            pred_tile_tem_pluma = pixels_preditos > 10 * mascara.numel() / (64 ** 2)

            if pred_tile_tem_pluma:
                fp_tiles += 1
            else:
                tn_tiles += 1
        grupo = classificar_pluma(has_plume, row.get("difficulty", ""))
        if grupo is not None:
            contagens_grupo[grupo][0] += tp
            contagens_grupo[grupo][1] += fp
            contagens_grupo[grupo][2] += fn

        tempos_model.append((fim_model-inicio_model)*1000)
        tempos_e2e.append((fim_e2e-inicio_e2e)*1000)
        tempos_preprocess.append((fim_preprocess-inicio_preprocess)*1000)
        tempos_posprocess.append((fim_posprocess-inicio_posprocess)*1000)
        tempo_carregamento.append((fim_carregamento-inicio_carregamento)*1000)
        
fpr_tile = (
    fp_tiles / (fp_tiles + tn_tiles)
    if (fp_tiles + tn_tiles) > 0
    else 0.0
)
fpr_tile_tabela = fp_tiles / len(amostras) if amostras else 0.0

auprc = float(np.mean(auprc_por_imagem)) if auprc_por_imagem else 0.0
precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0

f1_global = calcular_f1_contagens(tp_total, fp_total, fn_total)
f1_strong_plume = calcular_f1_contagens(*contagens_grupo["strong_plume"])
f1_weak_plume = calcular_f1_contagens(*contagens_grupo["weak_plume"])

iou = tp_total / (tp_total + fp_total + fn_total + 1e-6)

fpr = fp_no_plume / (fp_no_plume + tn_no_plume + 1e-6)
fpr_global = fp_total / (fp_total + tn_total + 1e-6)

media_e2e = np.mean(tempos_e2e)

medidor_energia.fechar()
for modo, fontes in energia_por_modo.items():
    for fonte, medidas in fontes.items():
        if medidas:
            energia_total = sum(medida[0] for medida in medidas)
            tempo_medido = sum(medida[2] for medida in medidas)
            print(f"\n===== ENERGIA {modo} / {fonte} =====")
            print(f"Energia total: {energia_total:.6f} J")
            print(f"Energia por inferência: {energia_total / len(medidas):.6f} J")
            print(f"Potência média: {energia_total / tempo_medido:.3f} W")
        else:
            print(f"Energia {modo} / {fonte}: contador indisponível.")

print("\n===== MÉTRICAS COMPATÍVEIS COM TESTE_UNET =====")
print(f"F1 strong: {f1_strong_plume:.4f}")
print(f"F1 weak:   {f1_weak_plume:.4f}")
print(f"FPR tile:  {fpr_tile:.4f} ({fpr_tile * 100:.2f}%)")
print(f"FP tiles / total de imagens: {fpr_tile_tabela:.4f} ({fpr_tile_tabela * 100:.2f}%)")
print(f"AUPRC:     {auprc:.4f} ({auprc * 100:.2f}%)")
print(f"FP tiles:  {fp_tiles}")
print(f"TN tiles:  {tn_tiles}")

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
print(f"FPR pixel:       {fpr:.6f}")

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

data_hora = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

ARQUIVO_RESULTADOS = Path(args.output) if args.output else Path(__file__).resolve().parent / f"resultado_{args.dataset}_{args.device}_{args.modelo}_{args.input_mode}_{data_hora}.csv"

resultado = {
    "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": args.modelo,
    "dataset": args.dataset,
    "device": args.device,
    "imagens": len(amostras),
    "input_mode": args.input_mode,
    "patches_por_imagem": patches_por_imagem,
    "produtos": ",".join(produtos),
    "checkpoint": str(caminho_checkpoint(args.modelo, args.checkpoint)),
    "metricas_protocolo": "teste_unet_difficulty_opening_ap_mean_plume_v1",
    "csv_dataset": str(Path(DATASET_PATH) / CSV_DATASET),

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
    "fpr_pixel": fpr,
    "fpr_pixel_global": fpr_global,
    "imagens_auprc": len(auprc_por_imagem),
    "fpr_tile": fpr_tile,
    "fpr_tile_tabela": fpr_tile_tabela,
    "auprc": auprc,
    "fp_tiles": fp_tiles,
    "tn_tiles": tn_tiles,

    "tp": tp_total,
    "fp": fp_total,
    "fn": fn_total,
    "tn": tn_total
}
resultado["energia_fontes"] = ",".join(medidor_energia.fontes) or "indisponivel"
for modo, fontes in energia_por_modo.items():
    for fonte, medidas in fontes.items():
        if not medidas:
            continue
        energia_total = sum(medida[0] for medida in medidas)
        tempo_medido = sum(medida[2] for medida in medidas)
        resultado.update({
            f"{modo}_{fonte}_amostras": len(medidas),
            f"{modo}_{fonte}_energia_total_j": energia_total,
            f"{modo}_{fonte}_energia_por_inferencia_j": energia_total / len(medidas),
            f"{modo}_{fonte}_potencia_media_w": energia_total / tempo_medido,
            f"{modo}_{fonte}_potencia_min_w": min(medida[1] for medida in medidas),
            f"{modo}_{fonte}_potencia_max_w": max(medida[1] for medida in medidas),
        })
with open(ARQUIVO_RESULTADOS, "w", newline="") as arquivo:
    escritor = csv.DictWriter(
        arquivo,
        fieldnames=resultado.keys()
    )

    escritor.writeheader()
    escritor.writerow(resultado)

print(f"\nResultado salvo em: {ARQUIVO_RESULTADOS}")
