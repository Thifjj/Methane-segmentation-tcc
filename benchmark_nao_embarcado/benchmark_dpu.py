"""Extensao FP32 CPU/CUDA para comparar os AttentionGates com a ZCU104."""
import argparse
import csv
import json
import os
import platform
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from .dataset import carregar_classificacao_por_pasta
from .metricas import calcular_metricas
from .model_loader import load_model, NOVOS_MODELOS
from .energia import resumir_energia_cpu
from Testes.Teste_Unet import binary_opening
from Utils.DataLoader import DataNormalizer, STARCOPDataset, carregar_dataframe_starcop
from VitisAI.common import DEFAULT_PRODUCTS, MODEL_REGISTRY, PROJECT_ROOT, file_sha256

MODEL_REGISTRY = {**MODEL_REGISTRY, **NOVOS_MODELOS}
MODELOS = tuple(MODEL_REGISTRY)
ESTAGIOS = ("leitura", "preprocess", "sync_entrada", "inferencia", "sync_saida",
            "postprocess", "latencia_total")


def preparar_entrada(inputs, normalizer, patch_size):
    if inputs.shape != (1, 4, 512, 512) or not torch.isfinite(inputs).all():
        raise ValueError("Entrada deve ser finita, com geometria [1,4,512,512].")
    inputs = normalizer(inputs)
    patches = inputs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 4, patch_size, patch_size).contiguous()


def inferir(model, patches, batch_size):
    # Batch 1 por padrao: as mesmas 16 chamadas por imagem da placa.
    return torch.cat([model(batch) for batch in patches.split(batch_size)], dim=0)


def reconstruir(outputs, patch_size):
    grid = 512 // patch_size
    if outputs.shape != (grid * grid, 1, patch_size, patch_size):
        raise ValueError(f"Geometria de saida invalida: {tuple(outputs.shape)}")
    return outputs.reshape(grid, grid, 1, patch_size, patch_size).permute(
        2, 0, 3, 1, 4).reshape(1, 1, 512, 512)


def estatisticas(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(amostras=len(a), media_ms=float(a.mean()), mediana_ms=float(np.median(a)),
                minimo_ms=float(a.min()), maximo_ms=float(a.max()),
                p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
                desvio_padrao_ms=float(a.std()))


def escrever_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Sem dados para {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def iniciar_energia(medidor):
    readings = [medidor.ler()]
    stop = threading.Event()
    def collect():
        while not stop.wait(0.2):
            readings.append(medidor.ler())
    thread = threading.Thread(target=collect, daemon=True) if medidor.fontes else None
    if thread:
        thread.start()
    return stop, thread, readings


def finalizar_energia(medidor, sampling, measure):
    stop, thread, readings = sampling
    stop.set()
    if thread:
        thread.join()
    readings.append(medidor.ler())
    intervals = [measure(medidor, a, b) for a, b in zip(readings, readings[1:])]
    result = {}
    for source in medidor.fontes:
        values = [r[source] for r in intervals if source in r]
        if not values:
            continue
        joules = sum(v[0] for v in values)
        duration = sum(v[2] for v in values)
        result[source] = dict(energia_j=joules, duracao_s=duration, media_w=joules/duration,
                              minima_w=min(v[1] for v in values), maxima_w=max(v[1] for v in values),
                              amostras=len(values), status="ok" if len(values) == len(intervals) else "parcial")
    return result


def metricas_contagens(counts, official=False):
    tp, fp, fn, tn = map(int, counts)
    def div(a, b):
        return a / b if b else 0.0
    eps = 1e-6 if official else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=div(tp, tp + fp),
                recall=div(tp, tp + fn), f1=div(2 * tp, 2 * tp + fp + fn + eps),
                iou=div(tp, tp + fp + fn + eps), fpr=div(fp, fp + tn),
                acuracia=div(tp + tn, tp + fp + fn + tn))


def metricas_imagem(logits, target, record, official):
    if logits.shape != target.shape or not torch.isfinite(logits).all():
        raise ValueError(f"Saida invalida em {record['id']}")
    mask = logits > 0
    if official:
        kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
        mask = binary_opening(mask, kernel)
    counts = calcular_metricas(mask, target)[:4]
    positive = bool(target.any())
    background = not positive if official else not record["has_plume"]
    strong = record["difficulty"] == "easy" if official else record["qplume"] > 1000
    group = "sem_pluma" if background else "forte" if strong else "fraca"
    return dict(id=record["id"], difficulty=group, positive=positive,
                difficulty_csv=record["difficulty"], pixels_preditos=int(mask.sum()),
                **metricas_contagens(counts))


def resumir_metricas(rows, official, global_auprc=None):
    def counts(selected):
        return [sum(r[k] for r in selected) for k in ("tp", "fp", "fn", "tn")]
    groups = {g: metricas_contagens(counts([r for r in rows if r["difficulty"] == g]), official)
              for g in ("forte", "fraca", "sem_pluma")}
    metrics = metricas_contagens(counts(rows), official)
    background = [r for r in rows if r["difficulty"] == "sem_pluma"]
    fp_tiles = sum(r["pixels_preditos"] > 640 for r in background)
    positive = [r for r in rows if r["positive"]]
    ap = float(np.mean([r["average_precision"] for r in positive])) if positive else float("nan")
    bg = groups["sem_pluma"]
    bg_den = bg["fp"] + bg["tn"] + (1e-6 if official else 0.0)
    result = dict(imagens=len(rows), **metrics,
                  f1_global=metrics["f1"], f1_strong_plume=groups["forte"]["f1"],
                  f1_weak_plume=groups["fraca"]["f1"], auprc=ap if official else global_auprc,
                  fpr_sem_pluma=bg["fp"] / bg_den if bg_den else 0.0,
                  fpr_tile=fp_tiles / len(background) if background else 0.0,
                  fpr_tile_tabela=fp_tiles / len(rows), fp_tiles=fp_tiles,
                  tn_tiles=len(background) - fp_tiles, imagens_positivas=len(positive),
                  auprc_metodo="media_average_precision_imagens_positivas" if official else "auc_pr_global_pixels",
                  posprocessamento="abertura_cruz_3x3" if official else "logit_maior_zero",
                  grupo_forte="label_positivo_e_difficulty_easy" if official else "has_plume_e_qplume_maior_1000",
                  protocolo="vitis_ai_evaluate_quantized" if official else "benchmark_zcu104_bruto",
                  limiar_pixels_tile=640)
    return result, [dict(grupo=g, **m) for g, m in groups.items()]


def executar(args, medidor_class, medir_energia):
    if args.limit < 0 or args.warmup < 0 or args.num_threads < 1:
        raise ValueError("Limite/warmup devem ser >=0; threads deve ser >=1.")
    patches_per_image = (512 // args.patch_size) ** 2
    if not 1 <= args.patch_batch_size <= patches_per_image:
        raise ValueError("Batch de patches fora do intervalo valido.")
    device = torch.device("cuda" if args.device in ("cuda", "gpu") else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponivel.")
    torch.set_num_threads(args.num_threads)
    root = Path(args.data_root or (PROJECT_ROOT / "STARCOP_test" if args.dataset == "test"
                                  else Path("/media/jacques/games/Datasets/test/STARCOP_test/"))).resolve()
    dataset_csv = Path(args.csv).resolve() if args.csv else root / ("test.csv" if args.dataset == "test" else "train.csv")
    df = carregar_dataframe_starcop(str(dataset_csv), str(root),
                                   produtos_obrigatorios=list(DEFAULT_PRODUCTS) + ["labelbinary"])
    classifications = carregar_classificacao_por_pasta(root, str(dataset_csv))
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    if df.empty:
        raise ValueError("Nenhuma imagem valida encontrada.")
    inputs_dataset = STARCOPDataset(df, list(DEFAULT_PRODUCTS), [], patching=False)
    targets_dataset = STARCOPDataset(df, [], ["labelbinary"], patching=False)
    records = []
    for _, row in df.iterrows():
        folder = Path(row.folder).name
        has_plume, qplume = classifications[folder]
        if has_plume and not np.isfinite(qplume):
            raise ValueError(f"qplume nao finito: {folder}")
        records.append(dict(id=str(row.get("id", folder)), folder=row.folder,
                            difficulty=str(row.get("difficulty", "")), has_plume=has_plume,
                            qplume=qplume))
    if "difficulty" not in df:
        raise ValueError("O protocolo oficial exige a coluna difficulty.")
    checkpoint = PROJECT_ROOT / "Modelos_treinados" / MODEL_REGISTRY[args.attention_dpu][1]
    checkpoint_hash = file_sha256(checkpoint)
    calibration = PROJECT_ROOT / "VitisAI/build/vitis_ai/quantize" / args.attention_dpu / "calibration_manifest.json"
    if calibration.exists() and json.loads(calibration.read_text()).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint diferente do usado na calibracao oficial da DPU.")
    model = load_model(args.attention_dpu, device)
    normalizer = DataNormalizer(list(DEFAULT_PRODUCTS)).eval()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir) / f"{args.attention_dpu}_{args.dataset}_{device.type}_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(modelo=args.attention_dpu, run_id=run_id, csv_dataset=str(dataset_csv),
                    dataset=args.dataset, device=device.type, precisao="FP32", checkpoint=str(checkpoint),
                    checkpoint_sha256=checkpoint_hash, csv_sha256=file_sha256(dataset_csv),
                    imagens=len(df), imagem_altura=512, imagem_largura=512,
                    tamanho_patch=args.patch_size, patches_por_imagem=patches_per_image,
                    patch_batch_size=args.patch_batch_size, ordem_canais="mag1c,460,550,640",
                    unidade="imagem_512x512", protocolo="pytorch_sequencial", runners=0,
                    instancias_modelo=1, threads_pytorch=args.num_threads,
                    cpus_permitidas=len(os.sched_getaffinity(0)),
                    workers_pre=1, workers_pos=1, slots_por_runner=0,
                    warmup=args.warmup, parametros=sum(p.numel() for p in model.parameters()),
                    checkpoint_tamanho_mb=checkpoint.stat().st_size/2**20,
                    linux_kernel=platform.release(), arquitetura=platform.machine(),
                    cpu=next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                              if l.startswith("model name")), platform.processor()),
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "n/a",
                    pytorch_versao=str(torch.__version__), cuda_versao=torch.version.cuda,
                    normalizacao={p: dict(fator=1750 if p == "mag1c" else 60, clip=[0, 2]) for p in DEFAULT_PRODUCTS},
                    model_only_entradas="primeiras_4_preparadas_reutilizadas",
                    e2e_exclui="label,metricas,escrita_csv", status="em_execucao")
    (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    escrever_csv(output / "amostras.csv", records)
    print(f"{args.attention_dpu}: {len(df)} imagens; {device}; FP32; patch {args.patch_size}; batch {args.patch_batch_size}", flush=True)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    prepared = [preparar_entrada(inputs_dataset[i]["input"].unsqueeze(0), normalizer,
                                args.patch_size).to(device) for i in range(min(4, len(df)))]
    medidor = medidor_class(device)
    sampling = None
    perf_rows, stage_rows, sample_rows, energy_rows = [], [], [], []
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                inferir(model, prepared[0], args.patch_batch_size)
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for mode in ("model_only", "end_to_end"):
                print(f"Medindo {mode}...", flush=True)
                timings = []
                sampling = iniciar_energia(medidor)
                start = time.perf_counter()
                for i in range(len(df)):
                    t0 = time.perf_counter()
                    if mode == "model_only":
                        inferir(model, prepared[i % len(prepared)], args.patch_batch_size)
                        sync()
                        t1 = time.perf_counter()
                        durations = dict.fromkeys(ESTAGIOS, 0.0)
                        durations.update(inferencia=(t1-t0)*1000, latencia_total=(t1-t0)*1000)
                        sample_index = i % len(prepared)
                    else:
                        image = inputs_dataset[i]["input"].unsqueeze(0)
                        t1 = time.perf_counter()
                        patches = preparar_entrada(image, normalizer, args.patch_size)
                        t2 = time.perf_counter()
                        patches = patches.to(device)
                        sync()
                        t3 = time.perf_counter()
                        logits_patches = inferir(model, patches, args.patch_batch_size)
                        sync()
                        t4 = time.perf_counter()
                        logits = reconstruir(logits_patches.cpu(), args.patch_size)
                        t5 = time.perf_counter()
                        mask = logits > 0
                        t6 = time.perf_counter()
                        durations = dict(zip(ESTAGIOS, [(t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000,
                                                      (t4-t3)*1000, (t5-t4)*1000, (t6-t5)*1000, (t6-t0)*1000]))
                        sample_index = i
                    timings.append(durations)
                    sample_rows.append(dict(modelo=args.attention_dpu, run_id=run_id, modo=mode,
                                            trabalho=i, indice_amostra=sample_index, id=records[sample_index]["id"],
                                            **{k+"_ms": v for k, v in durations.items()}))
                duration = time.perf_counter() - start
                measures = finalizar_energia(medidor, sampling, medir_energia)
                sampling = None
                stats = {s: estatisticas([t[s] for t in timings]) for s in ESTAGIOS}
                latency = stats["latencia_total"]
                perf = dict(metadata, modo=mode, inferencias=len(df),
                            **resumir_energia_cpu(measures, len(df)),
                            entradas_preparadas=len(prepared) if mode == "model_only" else 0,
                            duracao_s=duration, throughput_fps=len(df)/duration,
                            fps_latencia=1000/latency["media_ms"],
                            execucoes_modelo=len(df)*((patches_per_image+args.patch_batch_size-1)//args.patch_batch_size),
                            vram_pico_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
                            **{"latencia_"+{"minimo_ms": "min_ms", "maximo_ms": "max_ms",
                                           "desvio_padrao_ms": "desvio_ms"}.get(k, k): v
                               for k, v in latency.items() if k != "amostras"},
                            **{s+"_media_ms": stats[s]["media_ms"] for s in ESTAGIOS if s != "latencia_total"})
                for key in ("normalizacao", "status"):
                    perf.pop(key)
                perf_rows.append(perf)
                stage_rows.extend(dict(modelo=args.attention_dpu, run_id=run_id, modo=mode,
                                       estagio=s, **st) for s, st in stats.items())
                for source in medidor.fontes.keys() | measures.keys():
                    measure = measures.get(source)
                    energy_rows.append(dict(modelo=args.attention_dpu, run_id=run_id, modo=mode,
                                            trilho=source, status=measure["status"] if measure else "indisponivel",
                                            amostras=measure["amostras"] if measure else 0,
                                            energia_j=measure["energia_j"] if measure else "",
                                            energia_por_inferencia_j=measure["energia_j"]/len(df) if measure else "",
                                            media_w=measure["media_w"] if measure else "",
                                            minima_w=measure["minima_w"] if measure else "",
                                            maxima_w=measure["maxima_w"] if measure else "",
                                            duracao_s=measure["duracao_s"] if measure else "",
                                            metodo="diferenca_contadores_intervalos_200ms", unidade="imagem_512x512"))
                if not medidor.fontes:
                    energy_rows.append(dict(modelo=args.attention_dpu, run_id=run_id, modo=mode,
                                            trilho="", status="indisponivel", amostras=0, energia_j="",
                                            energia_por_inferencia_j="", media_w="", minima_w="", maxima_w="", duracao_s="",
                                            metodo="diferenca_contadores_intervalos_200ms", unidade="imagem_512x512"))
                print(f"{mode}: {perf['throughput_fps']:.3f} imagens/s; {latency['media_ms']:.3f} ms/imagem", flush=True)
                print(f"CPU {mode}: potencia media = {perf['cpu_potencia_media_w']:.3f} W; "
                      f"energia/inferencia = {perf['cpu_energia_por_inferencia_j']:.6f} J; "
                      f"status = {perf['cpu_energia_status']}", flush=True)
            escrever_csv(output / "benchmark_geral.csv", perf_rows)
            escrever_csv(output / "benchmark_estagios.csv", stage_rows)
            escrever_csv(output / "benchmark_samples.csv", sample_rows)
            escrever_csv(output / "benchmark_power_rails.csv", energy_rows)

            print("Validando separadamente: relatorio bruto e validacao_oficial...", flush=True)
            rows = {False: [], True: []}
            # Dados temporarios em disco; a ordenacao da AUPRC global ainda usa RAM.
            with tempfile.TemporaryDirectory(prefix="auprc_", dir=output) as tmp:
                scores = np.memmap(Path(tmp)/"scores.bin", dtype="float32", mode="w+", shape=(len(df), 512*512))
                labels = np.memmap(Path(tmp)/"labels.bin", dtype="uint8", mode="w+", shape=scores.shape)
                for i, record in enumerate(records):
                    patches = preparar_entrada(inputs_dataset[i]["input"].unsqueeze(0), normalizer,
                                                args.patch_size).to(device)
                    logits = reconstruir(inferir(model, patches, args.patch_batch_size).cpu(), args.patch_size)
                    target = targets_dataset[i]["output"].unsqueeze(0)
                    if target.shape != (1, 1, 512, 512) or not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
                        raise ValueError(f"Label deve ser binario e 512x512: {record['id']}")
                    if not torch.isfinite(logits).all():
                        raise ValueError(f"Logits nao finitos: {record['id']}")
                    scores[i] = torch.sigmoid(logits).reshape(-1).numpy()
                    labels[i] = target.reshape(-1).numpy().astype(np.uint8)
                    ap = average_precision_score(labels[i], scores[i]) if target.any() else float("nan")
                    for official in rows:
                        row = metricas_imagem(logits, target, record, official)
                        row["average_precision"] = float(ap)
                        rows[official].append(row)
                    if (i+1) % 50 == 0 or i+1 == len(df):
                        print(f"Validacao: {i+1}/{len(df)}", flush=True)
                print("Calculando AUPRC global do relatorio bruto...", flush=True)
                if labels.any():
                    precision, recall, _ = precision_recall_curve(labels.reshape(-1), scores.reshape(-1))
                    global_ap = float(auc(recall, precision))
                else:
                    global_ap = 0.5  # Mesma convencao do relatorio bruto C++ sem pixels positivos.
                del scores, labels
            for official, values in rows.items():
                folder = output / "validacao_oficial" if official else output
                summary, groups = resumir_metricas(values, official, global_ap)
                escrever_csv(folder / "metricas_por_imagem.csv", values)
                escrever_csv(folder / "metricas_grupos.csv", groups)
                escrever_csv(folder / "metricas_globais.csv", [dict(modelo=args.attention_dpu, run_id=run_id,
                                                                      csv_dataset=str(dataset_csv), **summary)])
                print(f"{'Oficial' if official else 'Bruto'}: F1={summary['f1_global']:.6f}; IoU={summary['iou']:.6f}; AUPRC={summary['auprc']:.6f}", flush=True)
        metadata["status"] = "concluido"
        metadata["energia_fontes"] = list(medidor.fontes)
        (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    finally:
        if sampling is not None:
            sampling[0].set()
            if sampling[1]:
                sampling[1].join()
        medidor.fechar()
    print(f"Resultados: {output.resolve()}", flush=True)
    return output


def argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dpu", required=True, choices=MODELOS)
    parser.add_argument("--device", choices=("cpu", "cuda", "gpu"), default="cpu")
    parser.add_argument("--dataset", choices=("test", "full"), default="test")
    parser.add_argument("--data-root")
    parser.add_argument("--csv", help="Caminho completo do CSV; padrao test.csv/train.csv no dataset.")
    parser.add_argument("--limit", type=int, default=0, help="0 = todas as imagens validas.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, choices=(128, 512), default=128)
    parser.add_argument("--patch-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "resultados_dpu"))
    return parser.parse_args(argv)


def main(medidor_class, medir_energia):
    return executar(argumentos(), medidor_class, medir_energia)


def interativo(model_name, datasets, medidor_class, medir_energia):
    def escolher(prompt, options):
        while True:
            value = input(prompt).strip()
            if value in options:
                return options[value]
            print("Opcao invalida.")
    dataset = escolher("Dataset: 1-full / 2-test: ", {"1": "full", "2": "test"})
    device = escolher("Dispositivo: 1-CPU / 2-GPU CUDA: ", {"1": "cpu", "2": "cuda"})
    while True:
        value = input("Quantas imagens? 0=todas: ").strip()
        if value.isdigit():
            break
        print("Digite um inteiro >=0.")
    args = argumentos(["--attention-dpu", model_name, "--device", device, "--dataset", dataset,
                       "--data-root", datasets[dataset][0], "--limit", value])
    return executar(args, medidor_class, medir_energia)
