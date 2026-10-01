from __future__ import annotations

import sys
import bisect
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modelos.UNet_MobileNet_v2 import UNetMobileNetV2
from Modelos.UNet_MobileNet_v3 import UNetMobileNetV3
from Modelos.UNet_MobileNetV3_AttentionGates_DPU import UNetMobileNetV3AttentionGatesDPU
from Modelos.UNet_SkipConnections import UNetElementWise
from Modelos.UNet_baseline import UNetBaseline
from Modelos.UNet_depth_reduced import UNetDepthReduced
from Utils.DataLoader import DataNormalizer, STARCOPDataset, carregar_dataframe_starcop


DEFAULT_PRODUCTS = (
    "mag1c",
    "TOA_AVIRIS_640nm",
    "TOA_AVIRIS_550nm",
    "TOA_AVIRIS_460nm",
)

MODEL_REGISTRY = {
    "baseline": (UNetBaseline, "UNET_mag1c_rgb.pth"),
    "depth_reduced": (UNetDepthReduced, "UNET_depth_reduced_mag1c_rgb.pth"),
    "skip_connections": (UNetElementWise, "UNET_SkipConnections_mag1c_rgb.pth"),
    "mobilenet_v2": (UNetMobileNetV2, "Mobile_Net_v2_mag1c_rgb.pth"),
    "mobilenet_v3": (UNetMobileNetV3, "Mobile_Net_v3_mag1c_rgb.pth"),
    "mobilenet_v3_attention_gates_dpu": (UNetMobileNetV3AttentionGatesDPU, "UNetMobileNetV3AttentionGatesDPU_mag1c_rgb.pth"),
    "hyperstarcop": (None, "HyperSTARCOP_oficial/final_checkpoint_model.ckpt"),
}


def parse_products(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        products = [item.strip() for item in value.split(",") if item.strip()]
    else:
        products = list(value)
    if not products:
        raise ValueError("Informe ao menos um produto de entrada.")
    return products


def build_model(model_name: str, checkpoint: str | Path | None, in_channels: int):
    model_class, default_checkpoint = MODEL_REGISTRY[model_name]
    checkpoint_path = Path(checkpoint) if checkpoint else PROJECT_ROOT / "Modelos_treinados" / default_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    if model_name == "hyperstarcop":
        if in_channels != 4:
            raise ValueError("HyperSTARCOP requer os quatro canais mag1c e RGB.")
        from Modelos.HyperStarcop_oficial import carregar_hyperstarcop

        return carregar_hyperstarcop(checkpoint_path, torch.device("cpu")), checkpoint_path

    model = model_class(in_channels=in_channels, out_channels=1)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibilidade com o PyTorch do container Vitis AI.
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint_path


def build_calibration_loader(
    csv_path: str | Path,
    data_root: str | Path,
    products: list[str],
    subset_len: int,
    batch_size: int = 1,
    num_workers: int = 2,
    seed: int = 12345,
    pin_memory: bool = False,
    patching: bool = False,
    patch_count: int = 1000,
    patches_per_image: int = 0,
) -> DataLoader:
    dataframe = carregar_dataframe_starcop(
        str(csv_path),
        str(data_root),
        produtos_obrigatorios=products,
        limite_amostras=subset_len,
        seed=seed,
    )
    if dataframe.empty:
        raise RuntimeError("Nenhuma amostra valida foi encontrada para calibracao.")
    # Rotulos nao participam da calibracao nem da exportacao.
    dataset = STARCOPDataset(dataframe, products, [])
    if patching:
        dataset = CalibrationPatches(dataset)
        generator = torch.Generator().manual_seed(seed)
        if patches_per_image > 0:
            selected = []
            for start, end in zip(dataset.offsets[:-1], dataset.offsets[1:]):
                local = torch.randperm(end - start, generator=generator)[:patches_per_image]
                selected.extend((local + start).tolist())
            order = torch.randperm(len(selected), generator=generator)
            indices = torch.tensor(selected)[order]
        else:
            indices = torch.randperm(len(dataset), generator=generator)
            if patch_count > 0:
                indices = indices[:patch_count]
        dataset = Subset(dataset, indices.tolist())
        print(f"Calibracao: {len(dataset)} patches de 128x128, passo 64; {len(dataframe)} imagens elegiveis.")
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **loader_kwargs)


class CalibrationPatches(Dataset):
    """Apresenta os mesmos recortes do treino como amostras individuais."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.offsets = [0]
        for window in dataset.dataframe["window"]:
            count = max(0, (int(window.height) - 128) // 64 + 1) * max(0, (int(window.width) - 128) // 64 + 1)
            self.offsets.append(self.offsets[-1] + count)
        if self.offsets[-1] == 0:
            raise ValueError("Nenhum patch de 128x128 disponivel.")

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        image = bisect.bisect_right(self.offsets, index) - 1
        patch = index - self.offsets[image]
        tensor = self.dataset[image]["input"]
        columns = (tensor.shape[-1] - 128) // 64 + 1
        y, x = (patch // columns) * 64, (patch % columns) * 64
        return {"input": tensor[:, y:y + 128, x:x + 128]}


def normalized_inputs(loader: DataLoader, products: list[str], device: torch.device):
    normalizer = DataNormalizer(products).to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            yield normalizer(
                batch["input"].to(device, non_blocking=device.type == "cuda")
            )
