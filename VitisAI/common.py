from __future__ import annotations

import sys
import random
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Modelos
from Utils.DataLoader import BAND_NORMALIZATION, DataNormalizer, STARCOPDataset, carregar_dataframe_starcop


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_contract(model, checkpoint, products, height, width, target, patching, architecture=None):
    return dict(manifest_version=2, model=model, architecture=resolve_model(model, checkpoint, architecture)[1], checkpoint_sha256=file_sha256(checkpoint),
                products=list(products), height=height, width=width, target=target,
                patching=patching, patch_stride=64 if patching else None,
                normalization=json.loads(json.dumps({p: BAND_NORMALIZATION[p] for p in products})))


def validate_calibration(directory, model, checkpoint, products, height, width, target, patching, architecture=None):
    directory = Path(directory)
    path = directory / "calibration_manifest.json"
    if not path.is_file():
        raise ValueError(f"Manifesto de calibracao ausente: {path}")
    saved = json.loads(path.read_text())
    if saved.get("manifest_version") != 2:
        raise ValueError(f"Manifesto legado sem hash/normalizacao: {path}. Recalibre em outro diretorio; preserve o resultado antigo.")
    expected = calibration_contract(model, checkpoint, products, height, width, target, patching, architecture)
    # Manifestos oficiais anteriores usam as arquiteturas fixas do registro.
    if "architecture" not in saved and model in MODEL_REGISTRY and expected["architecture"] == MODEL_REGISTRY[model][0]:
        expected.pop("architecture")
    different = [key for key, value in expected.items() if saved.get(key) != value]
    if different:
        raise ValueError(f"Calibracao incompativel ({', '.join(different)}): {path}")
    artifacts = saved.get("artifacts", {})
    if "quant_info.json" not in artifacts:
        raise ValueError(f"Manifesto sem hash da quantizacao: {path}")
    for filename in ("quant_info.json", "bias_corr.pth", "adapted_parameters.pth"):
        artifact = directory / filename
        if artifact.is_file() != (filename in artifacts) or (artifact.is_file() and file_sha256(artifact) != artifacts[filename]):
            raise ValueError(f"Artefato ausente ou alterado: {artifact}")
    return saved


def load_quantized_parameters(quantizer, directory):
    """Carrega reconstrucao PTQ registrada separadamente do checkpoint FP32."""
    path = Path(directory) / "adapted_parameters.pth"
    if not path.exists():
        return
    parameters = torch.load(path, map_location="cpu", weights_only=True)
    modules = {m.node.name: m for m in quantizer.quant_model.modules() if getattr(m, "node", None) is not None}
    with torch.no_grad():
        for name, values in parameters.items():
            if name not in modules:
                raise ValueError(f"Camada adaptada ausente no grafo: {name}")
            module = modules[name]
            for key, value in values.items():
                original = getattr(module, key, None)
                if key not in ("weight", "bias") or original is None or original.shape != value.shape or not torch.isfinite(value).all():
                    raise ValueError(f"Parametro adaptado invalido: {name}/{key}")
                original.copy_(value.to(original.device))
            module.param_saved = True  # Bias ja reconstruido; nao aplicar a correcao antiga outra vez.
            module.param_quantized = False


DEFAULT_PRODUCTS = (
    "mag1c",
    "TOA_AVIRIS_460nm",
    "TOA_AVIRIS_550nm",
    "TOA_AVIRIS_640nm",
)

MODEL_REGISTRY = {
    "attentiongates_dpu_easy_remaining": ("UNetMobileNetV3AttentionGatesDPU", "UnetMobilenetV3AttentionGates_dpu_easy_remaining_mag1c_rgb.pth"),
    "attentiongates_dpu_only_remaining": ("UNetMobileNetV3AttentionGatesDPU", "UnetMobilenetV3AttentionGates_dpu_only_remaining_mag1c_rgb.pth"),
    "baseline": ("UNetBaseline", "UNET_mag1c_rgb.pth"),
    "depth_reduced": ("UNetDepthReduced", "UNET_depth_reduced_mag1c_rgb.pth"),
    "skip_connections": ("UNetElementWise", "UNET_SkipConnections_mag1c_rgb.pth"),
    "mobilenet_v2": ("UNetMobileNetV2", "Mobile_Net_v2_mag1c_rgb.pth"),
    "mobilenet_v3": ("UNetMobileNetV3", "Mobile_Net_v3_mag1c_rgb.pth"),
    "mobilenet_v3_dpu": ("UNetMobileNetV3_dpu", "Mobile_Net_v3_dpu_mag1c_rgb.pth"),
    "mobilenet_v3_dpu_512": ("UNetMobileNetV3_dpu", "Mobile_Net_v3_dpu_mag1c_rgb.pth"),
    "mobilenet_v3_attentiongates": ("UNetMobileNetV3AttentionGates", "MobileNetV3_AttentionGates_mag1c_rgb.pth"),
    "resnet34": ("UNetResNet34", "UNet_ResNet34_mag1c_rgb.pth"),
    "segformer": ("SegFormerB0", "UNet_SegFormer_mag1c_rgb.pth"),
    "attentiongates": ("UNetAttentionGates", None),
    "psa": ("UNetPSA", None),
    "hyperstarcop": ("HyperSTARCOPOficial", "HyperSTARCOP_oficial/final_checkpoint_model.ckpt"),
}

CHECKPOINT_ALIASES = {
    "UNetMobileNetV3AttentionGates_mag1c_rgb.pth": "mobilenet_v3_attentiongates",
    "Mobile_Net_v3_mag1c_rgb_dpu.pth": "mobilenet_v3_dpu",
}

OFFICIAL_CALIBRATION = {
    "attentiongates_dpu_easy_remaining": dict(subset_len=300, patches_per_group=900, range_samples=2048, refine_layers=12),
    "attentiongates_dpu_only_remaining": dict(subset_len=100, patches_per_group=300, range_samples=512, refine_layers=0),
}


def parse_products(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        products = [item.strip() for item in value.split(",") if item.strip()]
    else:
        products = list(value)
    if not products:
        raise ValueError("Informe ao menos um produto de entrada.")
    return products


def add_model_arguments(parser):
    parser.add_argument("--model", help="Alias registrado ou classe exportada por Modelos.")
    parser.add_argument("--checkpoint", help="Caminho dos pesos; nomes conhecidos identificam a arquitetura.")
    parser.add_argument("--architecture", choices=[n for n in Modelos.__all__ if n != "carregar_hyperstarcop"],
                        help="Classe em Modelos para checkpoints com nome personalizado.")


def resolve_model(model_name=None, checkpoint=None, architecture=None):
    if model_name is None:
        if checkpoint:
            filename = Path(checkpoint).name
            model_name = CHECKPOINT_ALIASES.get(filename)
            if model_name is None:
                model_name = next((name for name, (_, path) in MODEL_REGISTRY.items()
                                   if path and Path(path).name == filename), None)
            if model_name is None:
                if architecture is None:
                    raise ValueError("Nome de checkpoint desconhecido: informe --architecture (classe de Modelos).")
                model_name = Path(checkpoint).stem
        else:
            model_name = architecture or "attentiongates_dpu_easy_remaining"
    if architecture is None:
        architecture = MODEL_REGISTRY[model_name][0] if model_name in MODEL_REGISTRY else model_name
    if architecture not in Modelos.__all__ or architecture == "carregar_hyperstarcop":
        raise ValueError(f"Arquitetura nao exportada por Modelos: {architecture}")
    if Path(model_name).name != model_name or model_name in (".", ".."):
        raise ValueError("--model deve ser um nome simples, sem diretorios.")
    return model_name, architecture


def calibration_defaults(model_name):
    return OFFICIAL_CALIBRATION.get(model_name, OFFICIAL_CALIBRATION["attentiongates_dpu_easy_remaining"])


def build_model(model_name: str, checkpoint: str | Path | None, in_channels: int, architecture=None):
    model_name, architecture = resolve_model(model_name, checkpoint, architecture)
    default_checkpoint = MODEL_REGISTRY.get(model_name, (None, None))[1]
    if not checkpoint and not default_checkpoint:
        raise ValueError(f"Informe --checkpoint para {model_name}.")
    checkpoint_path = Path(checkpoint) if checkpoint else PROJECT_ROOT / "Modelos_treinados" / default_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    if architecture == "HyperSTARCOPOficial":
        if in_channels != 4:
            raise ValueError("HyperSTARCOPOficial requer quatro canais.")
        return Modelos.carregar_hyperstarcop(checkpoint_path, torch.device("cpu")), checkpoint_path
    model_class = getattr(Modelos, architecture)
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


class BalancedCalibrationPatches(Dataset):
    """Patches selecionados; reutiliza a leitura e o recorte do treinamento."""
    def __init__(self, dataframe, products, patches_per_group, seed):
        self.dataframe = dataframe
        self.inputs = STARCOPDataset(dataframe, products, [], patching=True)
        labels = STARCOPDataset(dataframe, [], ["labelbinary"], patching=True)
        groups = {name: [] for name in ("strong", "weak", "background")}
        for image_index in range(len(dataframe)):
            positive = labels[image_index]["output"].flatten(1).sum(1) > 0
            plume_group = "strong" if dataframe.iloc[image_index].difficulty == "easy" else "weak"
            for patch_index, has_plume in enumerate(positive.tolist()):
                groups[plume_group if has_plume else "background"].append((image_index, patch_index))
        count = min(patches_per_group, *(len(values) for values in groups.values()))
        if count == 0:
            raise ValueError(f"Calibracao equilibrada requer patches fortes, fracos e fundo: { {k: len(v) for k, v in groups.items()} }")
        rng = random.Random(seed)
        self.records = sorted((image, patch, name) for name, values in groups.items()
                              for image, patch in rng.sample(values, count))
        self.group_counts = {name: count for name in groups}
        self._image_index = None
        self._patches = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_index, patch_index, _ = self.records[index]
        if image_index != self._image_index:
            self._patches = self.inputs[image_index]["input"]
            self._image_index = image_index
        return {"input": self._patches[patch_index]}


def select_balanced_images(dataframe, subset_len, seed):
    if subset_len < 3:
        raise ValueError("Calibracao equilibrada requer --subset-len >= 3.")
    if not {"has_plume", "difficulty"}.issubset(dataframe.columns):
        raise ValueError("CSV deve conter has_plume e difficulty para equilibrar a calibracao.")
    if not dataframe.has_plume.isin([True, False]).all():
        raise ValueError("has_plume deve conter valores booleanos.")
    groups = [dataframe[dataframe.has_plume & (dataframe.difficulty == "easy")],
              dataframe[dataframe.has_plume & (dataframe.difficulty != "easy")],
              dataframe[~dataframe.has_plume]]
    selected = []
    for i, group in enumerate(groups):
        count = subset_len // 3 + (i < subset_len % 3)
        if len(group) < count:
            raise ValueError(f"Grupo {('strong', 'weak', 'background')[i]} tem {len(group)} imagens; necessita {count}.")
        selected.extend(group.sample(n=count, random_state=seed).index.tolist())
    return dataframe.loc[selected].reset_index(drop=True)


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
    balanced: bool = False,
    patches_per_group: int = 300,
) -> DataLoader:
    if balanced and (not patching or patches_per_group < 1):
        raise ValueError("Amostragem equilibrada requer patching e patches_per_group > 0.")
    dataframe = carregar_dataframe_starcop(
        str(csv_path),
        str(data_root),
        produtos_obrigatorios=products + (["labelbinary"] if balanced else []),
        limite_amostras=None if balanced else subset_len,
        seed=seed,
    )
    if dataframe.empty:
        raise RuntimeError("Nenhuma amostra valida foi encontrada para calibracao.")
    # Rotulos selecionam patches, mas nunca entram no modelo quantizado.
    if balanced:
        dataframe = select_balanced_images(dataframe, subset_len, seed)
        dataset = BalancedCalibrationPatches(dataframe, products, patches_per_group, seed)
    else:
        dataset = STARCOPDataset(dataframe, products, [], patching=patching)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **loader_kwargs)


def normalized_inputs(loader: DataLoader, products: list[str], device: torch.device):
    normalizer = DataNormalizer(products).to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"]
            if inputs.ndim == 5:
                inputs = inputs.flatten(0, 1)
            # Limita o batch de patches para nao multiplicar o uso de memoria.
            for inputs_batch in inputs.split(loader.batch_size):
                yield normalizer(inputs_batch.to(device, non_blocking=device.type == "cuda"))
