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
from sklearn.metrics import precision_recall_curve, auc

from .model_loader import load_model
from .dataset import (
    carregar_sample, encontrar_sample, carregar_label,
    carregar_classificacao_por_pasta,
)
from .preprocess import preprocess
from .postprocess import postprocess
from .metricas import calcular_metricas, calcular_f1_contagens, classificar_pluma

#MODEL_PATH = "/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/Mobile_Net_v3_mag1c_rgb.pth"
#DATASET_PATH = "/media/jacques/games/Datasets/STARCOP_train_remaining_all"
DATASETS = {
    "full": ("/home/thiago/Documents/STARCOP_DATASET", "train.csv"),
    "test": (
        "/home/thiago/Documents/Laboratorio_LEDS/Projetos_aceleradores/Segmentacao_de_metano/Joao/projeto/Methane-segmentation-tcc/STARCOP_test",
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

# Extensao DPU: as opcoes e medicoes dos modelos antigos seguem no fluxo abaixo.
import sys
if any(a == "--attention-dpu" or a.startswith("--attention-dpu=") for a in sys.argv[1:]):
    from .benchmark_dpu import main
    main(MedidorEnergia, medir_energia)
    raise SystemExit(0)

args = parser.parse_args()

modelos_disponiveis = (
    "baseline", "depth_reduced", "mobilenet_v2", "mobilenet_v3", "skip",
    "hyperstarcop",
    "attentiongates_dpu_easy_remaining", "attentiongates_dpu_only_remaining",
)
print("Modelos disponíveis:")
for indice, nome_modelo in enumerate(modelos_disponiveis, start=1):
    print(f"{indice} - {nome_modelo}")
while True:
    escolha_modelo = input("Escolha o modelo: ").strip()
    if escolha_modelo.isdigit() and 1 <= int(escolha_modelo) <= len(modelos_disponiveis):
        break
    print(f"Opção inválida. Digite um número de 1 a {len(modelos_disponiveis)}.")
args.modelo = modelos_disponiveis[int(escolha_modelo) - 1]

if args.modelo.startswith("attentiongates_dpu_"):
    from .benchmark_dpu import interativo
    interativo(args.modelo, DATASETS, MedidorEnergia, medir_energia)
    raise SystemExit(0)

print("Datasets disponíveis:")
print("1 - full")
print("2 - test")
while True:
    escolha_dataset = input("Escolha o dataset [1/2]: ").strip()
    if escolha_dataset in ("1", "2"):
        break
    print("Opção inválida. Digite 1 ou 2.")
args.dataset = "full" if escolha_dataset == "1" else "test"
DATASET_PATH, CSV_DATASET = DATASETS[args.dataset]

print("Dispositivos disponíveis:")
print("1 - CPU")
print("2 - GPU (CUDA)")
while True:
    escolha_device = input("Escolha o dispositivo [1/2]: ").strip()
    if escolha_device in ("1", "2"):
        break
    print("Opção inválida. Digite 1 ou 2.")
args.device = "cpu" if escolha_device == "1" else "gpu"

if args.device == "gpu" and not torch.cuda.is_available():
    raise RuntimeError("GPU solicitada, mas CUDA não está disponível")

device = torch.device("cuda" if args.device == "gpu" else "cpu")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

model = load_model(args.modelo, device)

amostras = encontrar_sample(DATASET_PATH, CSV_DATASET)
classificacao_por_pasta = carregar_classificacao_por_pasta(DATASET_PATH, CSV_DATASET)

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
entrada = preprocess(canais).to(device)

with torch.inference_mode():
    for _ in range(WARMUP):
        model(entrada)
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

probabilidades_auprc = []
labels_auprc = []
contagens_grupo = {
    "strong_plume": [0, 0, 0],
    "weak_plume": [0, 0, 0],
}

with torch.inference_mode():
    for pasta in tqdm(amostras, desc="Benchmark"):
        energia_e2e_antes = medidor_energia.ler()
        inicio_e2e = time.perf_counter()

        inicio_carregamento = time.perf_counter()
        canais = carregar_sample(pasta)
        fim_carregamento = time.perf_counter()

        inicio_preprocess = time.perf_counter()
        entrada = preprocess(canais).to(device)
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
        mascara = (probs > 0.5).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        fim_posprocess = time.perf_counter()

        fim_e2e = time.perf_counter()
        energia_e2e_depois = medidor_energia.ler()
        for evento, medida in medir_energia(
            medidor_energia, energia_e2e_antes, energia_e2e_depois
        ).items():
            energia_por_modo["end_to_end"][evento].append(medida)
        
        label = carregar_label(pasta)

        mascara = mascara.squeeze().cpu()
        
        probs_2d = probs.squeeze().cpu()

        probabilidades_auprc.append(
            probs_2d.flatten().numpy().astype(np.float32)
        )

        labels_auprc.append(
            label.flatten().numpy().astype(np.uint8)
        )

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
        # FPR BY TILE do artigo
        if not has_plume:

            pixels_preditos = mascara.sum().item()

            pred_tile_tem_pluma = pixels_preditos > 10 * mascara.numel() / (64 ** 2)

            if pred_tile_tem_pluma:
                fp_tiles += 1
            else:
                tn_tiles += 1
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
        
fpr_tile = (
    fp_tiles / (fp_tiles + tn_tiles)
    if (fp_tiles + tn_tiles) > 0
    else 0.0
)
fpr_tile_tabela = fp_tiles / len(amostras) if amostras else 0.0

y_true = np.concatenate(labels_auprc)
y_score = np.concatenate(probabilidades_auprc)

precision_curve, recall_curve, _ = precision_recall_curve(
    y_true,
    y_score
)

auprc = auc(
    recall_curve,
    precision_curve
)

precision = tp_total / (tp_total + fp_total)

recall = tp_total / (tp_total + fn_total)

f1_global = calcular_f1_contagens(tp_total, fp_total, fn_total)
f1_strong_plume = calcular_f1_contagens(*contagens_grupo["strong_plume"])
f1_weak_plume = calcular_f1_contagens(*contagens_grupo["weak_plume"])

iou = tp_total / (tp_total + fp_total + fn_total)

fpr = fp_total / (fp_total + tn_total)

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

print("\n===== MÉTRICAS ARTIGO STARCOP =====")
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

data_hora = datetime.now().strftime("%Y%m%d_%H%M")

ARQUIVO_RESULTADOS = Path(__file__).resolve().parent / f"resultado_{args.dataset}_{args.device}_{args.modelo}_{data_hora}.csv"

resultado = {
    "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": args.modelo,
    "dataset": args.dataset,
    "device": args.device,
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
    "fpr_pixel": fpr,
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
