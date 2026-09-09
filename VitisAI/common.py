from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modelos.UNet_MobileNet_v2 import UNetMobileNetV2
from Modelos.UNet_MobileNet_v3 import UNetMobileNetV3
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
) -> DataLoader:
    dataframe = carregar_dataframe_starcop(str(csv_path), str(data_root))
    if dataframe.empty:
        raise RuntimeError("Nenhuma amostra valida foi encontrada para calibracao.")
    dataset = STARCOPDataset(dataframe, products, ["labelbinary"])
    if subset_len > 0:
        dataset = Subset(dataset, range(min(subset_len, len(dataset))))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def normalized_inputs(loader: DataLoader, products: list[str], device: torch.device):
    normalizer = DataNormalizer(products).to(device).eval()
    with torch.no_grad():
        for batch in loader:
            yield normalizer(batch["input"].to(device))
