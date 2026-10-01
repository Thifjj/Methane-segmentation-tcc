import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from kornia.morphology import erosion, dilation
import kornia.augmentation as K
from Utils.FocalDiceLoss import FocalDiceLoss
from sklearn.model_selection import GroupShuffleSplit

from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer

def binary_opening(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    eroded = torch.clamp(erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(dilation(eroded.float(), kernel), 0, 1) > 0

def calcular_f1_score(previsao_logits, gabarito, threshold=0.0):
    device = previsao_logits.device
    previsao_binaria = (previsao_logits > threshold).float()
    
    kernel_cruz = torch.tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]).float().to(device)
    
    previsao_limpa = binary_opening(previsao_binaria, kernel_cruz).float()
    
    intersecao = (previsao_limpa * gabarito).sum(dim=(2, 3))
    soma_areas = previsao_limpa.sum(dim=(2, 3)) + gabarito.sum(dim=(2, 3))
    f1 = (2 * intersecao + 1e-6) / (soma_areas + 1e-6)
    
    return f1.sum().item(), f1.numel() 


from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import rasterio


def criar_dataframe_patches(dataframe, patch_size=128, stride=64, num_workers=4):
    """Uma linha por patch, com has_plume calculado como no HyperSTARCOP."""
    def recortar(row):
        with rasterio.open(Path(row["folder"]) / "labelbinary.tif") as src:
            mask = src.read(1, window=row["window"])
        rows = []
        for y in range(0, mask.shape[0] - patch_size + 1, stride):
            for x in range(0, mask.shape[1] - patch_size + 1, stride):
                item = dict(row)
                fraction = float(mask[y:y + patch_size, x:x + patch_size].mean())
                item["frac_positives"] = fraction
                item["has_plume"] = fraction > (10 / 64**2)
                item["window"] = rasterio.windows.Window(
                    row["window"].col_off + x, row["window"].row_off + y,
                    patch_size, patch_size,
                )
                rows.append(item)
        return rows

    records = dataframe.to_dict("records")
    tiled = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for rows in tqdm(pool.map(recortar, records), total=len(records), desc="Estatísticas dos patches"):
            tiled.extend(rows)
    if not tiled:
        raise ValueError("Nenhum patch completo foi encontrado.")
    return pd.DataFrame(tiled)


def criar_sampler_balanceado(dataframe):
    fraction = float(dataframe["has_plume"].mean())
    if not 0 < fraction < 1:
        raise ValueError("A amostragem balanceada requer patches com e sem pluma.")
    weights = np.where(dataframe["has_plume"], 1 / fraction, 1 / (1 - fraction))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(dataframe), replacement=True)


def calcular_loss(criterion, logits, targets, weights, loss_name):
    if loss_name == "focal_dice":
        return criterion(logits, targets, weight_map=weights)
    return (criterion(logits, targets) * weights).mean()


def validar_modelo(modelo, dataloader, normalizador, criterion, loss_name, device, amp):
    """Loss por imagem e F1 acumulado, sem abertura morfológica na validação."""
    modelo.eval()
    total_loss = 0.0
    images = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validação", leave=False):
            inputs = normalizador(batch["input"].to(device, non_blocking=True))
            targets = batch["output"].to(device, non_blocking=True)
            weights = batch["weight_loss"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = modelo(inputs)
                loss = calcular_loss(criterion, logits, targets, weights, loss_name)
            total_loss += float(loss) * len(inputs)
            images += len(inputs)
            prediction = logits >= 0
            truth = targets > 0
            tp += int((prediction & truth).sum())
            fp += int((prediction & ~truth).sum())
            fn += int((~prediction & truth).sum())
    if not images:
        raise ValueError("Dataset de validação vazio.")
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return total_loss / images, f1


def treinar_modelo(
    modelo_escolhido, nome_modelo_salvar, starting_point, produtos_entrada,
    loss_name="focal_dice", resume=True, *, batch_size=32, num_workers=4,
    val_batch_size=None, epocas=15, lr=1e-4, pos_weight=1.0,
    weight_sampling=True, validation_mode="official", val_check_interval=0.5,
    early_stopping=False, paciencia_maxima=8, lr_decay=0.5, lr_patience=4,
    device="auto", amp=False,
    prefetch_factor=4, persistent_workers=True, cache_max_gb=2.0, val_cache_max_gb=0.25,
    caminho_csv="/media/jacques/games/Datasets/STARCOP_train_remaining_all/train.csv",
    diretorio_dados="/media/jacques/games/Datasets/STARCOP_train_remaining_all",
    caminho_csv_val="/media/jacques/games/Datasets/test/STARCOP_test/test.csv",
    diretorio_dados_val="/media/jacques/games/Datasets/test/STARCOP_test",
    focal_alpha=0.25, focal_gamma=2.0, weight_focal=1.0, weight_dice=1.0,
):
    """Protocolo HyperSTARCOP com escolha de arquitetura e loss.

    batch_size conta patches independentes, não imagens de 49 patches.
    starting_point é mantido para compatibilidade; checkpoint usa val_loss.
    resume carrega os pesos; não restaura o estado do otimizador.
    validation_mode='official' usa test.csv; 'split' mantém o split de 15%.
    """
    if loss_name not in ("focal_dice", "bce"):
        raise ValueError("loss_name deve ser focal_dice ou bce")
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("device deve ser auto, cuda ou cpu")
    if validation_mode not in ("official", "split"):
        raise ValueError("validation_mode deve ser official ou split")
    if not (0 < val_check_interval <= 1) or batch_size < 1 or epocas < 1 or pos_weight <= 0:
        raise ValueError("Batch, épocas, pos_weight e intervalo de validação inválidos.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if num_workers < 0 or prefetch_factor < 1 or cache_max_gb < 0 or val_cache_max_gb < 0:
        raise ValueError("Workers/cache devem ser não negativos e prefetch_factor >= 1.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível.")
    amp = amp and device.type == "cuda"
    print(f"\n--- Iniciando Treinamento: {nome_modelo_salvar} ---")
    required = produtos_entrada + ["labelbinary", "weight_mag1c"]
    df_train = carregar_dataframe_starcop(caminho_csv, diretorio_dados, produtos_obrigatorios=required)
    if validation_mode == "split":
        train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(df_train, groups=df_train["folder"]))
        df_val = df_train.iloc[val_idx].reset_index(drop=True)
        df_train = df_train.iloc[train_idx].reset_index(drop=True)
    else:
        df_val = carregar_dataframe_starcop(caminho_csv_val, diretorio_dados_val, produtos_obrigatorios=required)
        print("Protocolo oficial: test.csv participa da seleção do checkpoint.")
    if df_train.empty or df_val.empty:
        raise ValueError("Treino e validação devem conter amostras válidas.")
    df_patches = criar_dataframe_patches(df_train, num_workers=num_workers)
    dataset_treino = STARCOPDataset(df_patches, produtos_entrada, ["labelbinary"], weight_loss="weight_mag1c", patching=False, cache_max_bytes=int(cache_max_gb * 1024**3))
    dataset_val = STARCOPDataset(df_val, produtos_entrada, ["labelbinary"], weight_loss="weight_mag1c", patching=False, cache_max_bytes=int(val_cache_max_gb * 1024**3))
    sampler = criar_sampler_balanceado(df_patches) if weight_sampling else None
    dl_options = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        dl_options.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    dataloader_treino = DataLoader(dataset_treino, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, **dl_options)
    dataloader_val = DataLoader(dataset_val, batch_size=val_batch_size or batch_size, shuffle=False, **dl_options)
    print(f"Treino: {len(df_train)} imagens / {len(df_patches)} patches | Validação: {len(df_val)} imagens completas")
    print(f"Batch: {batch_size} patches | Amostragem balanceada: {weight_sampling} | Loss: {loss_name} | Device: {device}")
    print(f"Leitura: prefetch={prefetch_factor if num_workers else 0}, workers persistentes={persistent_workers and num_workers > 0}, cache treino={cache_max_gb} GiB/worker, validação={val_cache_max_gb} GiB/worker")
    normalizador = DataNormalizer(produtos_entrada).to(device)
    modelo = modelo_escolhido(in_channels=len(produtos_entrada), out_channels=1).to(device)
    caminho_salvamento = Path("Modelos_treinados") / f"{nome_modelo_salvar}.pth"
    caminho_salvamento.parent.mkdir(exist_ok=True)
    loaded = resume and caminho_salvamento.exists()
    if loaded:
        print("Carregando pesos existentes...")
        modelo.load_state_dict(torch.load(caminho_salvamento, map_location=device, weights_only=True))
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_decay, patience=lr_patience)
    criterion = (FocalDiceLoss(alpha=focal_alpha, gamma=focal_gamma, weight_focal=weight_focal, weight_dice=weight_dice)
                 if loss_name == "focal_dice" else nn.BCEWithLogitsLoss(
                     pos_weight=torch.tensor(pos_weight, device=device), reduction="none"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    augmentacoes = K.AugmentationSequential(
        K.RandomRotation(p=0.5, degrees=90),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        data_keys=["input", "mask", "input"], keepdim=True, same_on_batch=False,
    ).to(device)
    melhor_loss = float("inf")
    if loaded:
        melhor_loss, _ = validar_modelo(modelo, dataloader_val, normalizador, criterion, loss_name, device, amp)
    paciencia_atual = 0
    history = []
    settings = dict(loss_name=loss_name, batch_size=batch_size, num_workers=num_workers,
                    val_batch_size=val_batch_size or batch_size, epocas=epocas, lr=lr,
                    pos_weight=pos_weight, weight_sampling=weight_sampling,
                    validation_mode=validation_mode, val_check_interval=val_check_interval,
                    early_stopping=early_stopping, paciencia_maxima=paciencia_maxima,
                    lr_decay=lr_decay, lr_patience=lr_patience, device=str(device), amp=amp,
                    resume=resume, produtos_entrada=produtos_entrada,
                    prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                    cache_max_gb=cache_max_gb, val_cache_max_gb=val_cache_max_gb,
                    caminho_csv=caminho_csv, caminho_csv_val=caminho_csv_val,
                    checkpoint_metric="val_loss", starting_point_ignored=starting_point,
                    focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                    weight_focal=weight_focal, weight_dice=weight_dice)
    history_path = caminho_salvamento.with_suffix(".training.json")
    stop = False
    for epoca in range(epocas):
        modelo.train()
        loss_acumulada = 0.0
        amostras = 0
        # Lightning valida a cada int(len(loader) * intervalo), inclusive no fim.
        interval = max(1, int(len(dataloader_treino) * val_check_interval))
        validation_steps = set(range(interval, len(dataloader_treino) + 1, interval))
        validation_steps.add(len(dataloader_treino))
        loop = tqdm(dataloader_treino, desc=f"Época {epoca + 1}/{epocas} [Treino]")
        for step, batch in enumerate(loop, 1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["output"].to(device, non_blocking=True)
            weights = batch["weight_loss"].to(device, non_blocking=True)
            inputs, targets, weights = augmentacoes(inputs, targets, weights)
            inputs = normalizador(inputs)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                previsoes = modelo(inputs)
                loss = calcular_loss(criterion, previsoes, targets, weights, loss_name)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss não finita; treinamento interrompido.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_acumulada += float(loss) * len(inputs)
            amostras += len(inputs)
            loop.set_postfix(Loss=f"{float(loss):.4f}")
            if step in validation_steps:
                val_loss, f1 = validar_modelo(modelo, dataloader_val, normalizador, criterion, loss_name, device, amp)
                if not math.isfinite(val_loss):
                    raise RuntimeError("Loss de validação não finita.")
                improved = val_loss < melhor_loss
                if improved:
                    melhor_loss = val_loss
                    paciencia_atual = 0
                    torch.save(modelo.state_dict(), caminho_salvamento)
                    print(f" -> Menor val_loss: {melhor_loss:.6f}. Modelo salvo!")
                else:
                    paciencia_atual += 1
                record = dict(epoch=epoca + 1, step=step, train_loss=loss_acumulada / amostras,
                              val_loss=val_loss, val_f1_global=f1,
                              lr=optimizer.param_groups[0]["lr"], checkpoint_saved=improved)
                history.append(record)
                history_path.write_text(json.dumps(dict(settings=settings, history=history, best_val_loss=melhor_loss), indent=2, ensure_ascii=False) + "\n")
                print(f"Época {epoca + 1}, passo {step}: val_loss={val_loss:.6f} | F1 global={f1:.4f}")
                modelo.train()
                if early_stopping and paciencia_atual >= paciencia_maxima:
                    print(f"=== Early Stopping! Menor val_loss: {melhor_loss:.6f} ===")
                    stop = True
                    break
        # ReduceLROnPlateau é atualizado uma vez por época, como no Lightning.
        scheduler.step(history[-1]["val_loss"])
        print(f"Resumo da Época {epoca + 1}: loss={loss_acumulada / amostras:.6f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        if stop:
            break
    return dict(checkpoint=str(caminho_salvamento), best_val_loss=melhor_loss, history=history)
