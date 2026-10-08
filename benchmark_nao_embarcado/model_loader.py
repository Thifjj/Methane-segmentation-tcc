import torch
from pathlib import Path
import Modelos

NOVOS_MODELOS = {
    "mobilenet_v3_focaldice": ("UNetMobileNetV3_dpu", "MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth"),
    "attentiongates_bce": ("UNetMobileNetV3AttentionGates", "MobileNetV3_AttentionGates_BCE_mag1c_rgb.pth"),
    "attentiongates_checkpoint_unet": ("UNetMobileNetV3AttentionGates", "UNetMobileNetV3AttentionGates_mag1c_rgb.pth"),
    "attentiongates_dpu_focaldice_artigo": ("UNetMobileNetV3AttentionGatesDPU", "UnetMobilenetV3AttentionGates_dpu_FocalDiceLoss_mag1c_rgb.pth"),
    "attentiongates_dpu_bce_artigo": ("UNetMobileNetV3AttentionGatesDPU", "UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth"),
    "attentiongates_dpu_bce_remaining_all": ("UNetMobileNetV3AttentionGatesDPU", "remaining_all_dpu_retrain_original/UnetMobilenetV3AttentionGates_dpu_BCEloss_mag1c_rgb.pth"),
    "attentiongates_dpu_focaldice_remaining_all": ("UNetMobileNetV3AttentionGatesDPU", "remaining_all_dpu_retrain_original/UnetMobilenetV3AttentionGates_dpu_FocalDiceLoss_mag1c_rgb.pth"),
    "attentiongates_focaldice_artigo": ("UNetMobileNetV3AttentionGates", "MobileNetV3_AttentionGates_FocalDiceLossmag1c_rgb.pth"),
    "mobilenet_v3_bce_artigo": ("UNetMobileNetV3", "Mobile_Net_v3_BCELoss_mag1c_rgb.pth"),
    "mobilenet_v3_focaldice_artigo": ("UNetMobileNetV3", "MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth"),
    "mobilenet_v3_dpu_bce_artigo": ("UNetMobileNetV3_dpu", "Mobile_Net_v3_BCELoss_mag1c_rgb.pth"),
    "mobilenet_v3_dpu_focaldice_artigo": ("UNetMobileNetV3_dpu", "MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth"),
}

from Modelos import UNetBaseline, UNetDepthReduced, UNetMobileNetV2, UNetMobileNetV3, UNetElementWise, carregar_hyperstarcop

def load_model(nome_modelo, device):
    if nome_modelo in NOVOS_MODELOS:
        arquitetura, checkpoint = NOVOS_MODELOS[nome_modelo]
        model = getattr(Modelos, arquitetura)(in_channels=4, out_channels=1)
        caminho = Path(__file__).resolve().parents[1] / "Modelos_treinados" / checkpoint
        model.load_state_dict(torch.load(caminho, map_location=device, weights_only=True))
        return model.to(device).eval()
    if nome_modelo in ("attentiongates_dpu_easy_remaining", "attentiongates_dpu_only_remaining",
                       "mobilenet_v3_attentiongates"):
        from VitisAI.common import build_model
        model, _ = build_model(nome_modelo, None, 4)
        return model.to(device).eval()

    if nome_modelo =="hyperstarcop":
        caminho = Path(__file__).resolve().parents[1] / "Modelos_treinados/HyperSTARCOP_oficial/final_checkpoint_model.ckpt"
        return carregar_hyperstarcop(caminho,device)

    if nome_modelo in ("skip_connections", "mobilenet_v3_dpu", "mobilenet_v3_dpu_512",
                       "resnet34", "segformer"):
        from VitisAI.common import build_model
        model, _ = build_model(nome_modelo, None, 4)
        return model.to(device).eval()
    
    modelos = {
        "baseline": (UNetBaseline, "UNET_mag1c_rgb.pth"),
        "depth_reduced": (UNetDepthReduced, "UNET_depth_reduced_mag1c_rgb.pth"),
        "mobilenet_v2": (UNetMobileNetV2, "Mobile_Net_v2_mag1c_rgb.pth"),
        "mobilenet_v3": (UNetMobileNetV3, "Mobile_Net_v3_mag1c_rgb.pth"),
        "skip": (UNetElementWise, "UNET_SkipConnections_mag1c_rgb.pth"),
    }

    classe_modelo, checkpoint = modelos[nome_modelo]
    caminho = Path(__file__).resolve().parents[1] / "Modelos_treinados" / checkpoint

    model = classe_modelo(in_channels =4, out_channels=1)

    pesos = torch.load(caminho,map_location=device,weights_only=True)

    model.load_state_dict(pesos)
    model.to(device)
    model.eval()
    
    return model
