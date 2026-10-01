from pathlib import Path
from importlib import import_module
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = PROJECT_ROOT / "Modelos_treinados"
RGB = ("mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm")
BGR = ("mag1c", "TOA_AVIRIS_460nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_640nm")
# Nome: módulo, classe, checkpoint, ordem dos produtos.
MODEL_REGISTRY = {
    "baseline": ("UNet_baseline", "UNetBaseline", "UNET_mag1c_rgb.pth", RGB),
    "depth_reduced": ("UNet_depth_reduced", "UNetDepthReduced", "UNET_depth_reduced_mag1c_rgb.pth", RGB),
    "mobilenet_v2": ("UNet_MobileNet_v2", "UNetMobileNetV2", "Mobile_Net_v2_mag1c_rgb.pth", RGB),
    "mobilenet_v3": ("UNet_MobileNet_v3", "UNetMobileNetV3", "Mobile_Net_v3_mag1c_rgb.pth", RGB),
    "skip": ("UNet_SkipConnections", "UNetElementWise", "UNET_SkipConnections_mag1c_rgb.pth", RGB),
    "hyperstarcop": ("HyperStarcop_oficial", "HyperSTARCOPOficial", "HyperSTARCOP_oficial/final_checkpoint_model.ckpt", RGB),
    "mobilenet_v3_focaldice": ("UNet_MobileNet_v3", "UNetMobileNetV3", "MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth", BGR),
    "mobilenet_v3_attention_gates": ("UNet_MobileNetV3_AttentionGates", "UNetMobileNetV3AttentionGates", "UNetMobileNetV3AttentionGates_mag1c_rgb.pth", BGR),
    "mobilenet_v3_attention_gates_63epoch": ("UNet_MobileNetV3_AttentionGates", "UNetMobileNetV3AttentionGates", "UNetMobileNetV3AttentionGates_mag1c_rgb_3245_63epoch.pth", BGR),
    "resnet34": ("UNet_ResNet34", "UNetResNet34", "UNet_ResNet34_mag1c_rgb.pth", BGR),
    "segformer": ("UNet_SegFormer", "SegFormerB0", "UNet_SegFormer_mag1c_rgb.pth", BGR),
    "attention_gates": ("UNet_AttentionGates", "UNetAttentionGates", "UNet_AttentionGates_mag1c_rgb.pth", BGR),
    "psa": ("UNet_PSA", "UNetPSA", "UNet_PSA_mag1c_rgb.pth", BGR),
}


def modelos_disponiveis():
    return tuple(name for name, info in MODEL_REGISTRY.items() if (WEIGHTS_ROOT / info[2]).is_file())


def produtos_modelo(nome_modelo):
    return MODEL_REGISTRY[nome_modelo][3]


def caminho_checkpoint(nome_modelo, checkpoint=None):
    return Path(checkpoint).resolve() if checkpoint else WEIGHTS_ROOT / MODEL_REGISTRY[nome_modelo][2]


def load_model(nome_modelo, device, checkpoint=None, produtos=None):
    modulo, classe, _, padrao = MODEL_REGISTRY[nome_modelo]
    produtos = tuple(produtos or padrao)
    caminho = caminho_checkpoint(nome_modelo, checkpoint)
    if not caminho.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {caminho}")
    if nome_modelo == "hyperstarcop":
        if produtos != RGB:
            raise ValueError("HyperSTARCOP oficial exige mag1c,640,550,460 nm nessa ordem.")
        from Modelos.HyperStarcop_oficial import carregar_hyperstarcop
        return carregar_hyperstarcop(caminho, device)
    model = getattr(import_module("Modelos." + modulo), classe)(in_channels=len(produtos), out_channels=1)
    pesos = torch.load(caminho, map_location=device, weights_only=True)
    model.load_state_dict(pesos, strict=True)
    return model.to(device).eval()
