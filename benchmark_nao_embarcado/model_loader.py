import torch
from pathlib import Path
import Modelos

NOVOS_MODELOS = {
    "mobilenet_v3_focaldice": ("UNetMobileNetV3_dpu", "unet_mobilenet_v3_focaldice_mag1c_rgb.pth"),
    "attentiongates_bce": ("UNetMobileNetV3AttentionGates", "unet_mobilenet_v3_attentiongates_bce_mag1c_rgb.pth"),
    "attentiongates_checkpoint_unet": ("UNetMobileNetV3AttentionGates", "unet_mobilenet_v3_attentiongates_focaldice_remaining_all_mag1c_rgb.pth"),
    "attentiongates_dpu_focaldice_artigo": ("UNetMobileNetV3AttentionGatesDPU", "unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth"),
    "attentiongates_dpu_bce_artigo": ("UNetMobileNetV3AttentionGatesDPU", "unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth"),
    "attentiongates_dpu_bce_remaining_all": ("UNetMobileNetV3AttentionGatesDPU", "unet_mobilenet_v3_attentiongates_dpu_bce_remaining_all_mag1c_rgb.pth"),
    "attentiongates_dpu_focaldice_remaining_all": ("UNetMobileNetV3AttentionGatesDPU", "unet_mobilenet_v3_attentiongates_dpu_focaldice_remaining_all_mag1c_rgb.pth"),
    "attentiongates_focaldice_artigo": ("UNetMobileNetV3AttentionGates", "unet_mobilenet_v3_attentiongates_focaldice_mag1c_rgb.pth"),
    "mobilenet_v3_bce_artigo": ("UNetMobileNetV3", "unet_mobilenet_v3_bce_mag1c_rgb.pth"),
    "mobilenet_v3_focaldice_artigo": ("UNetMobileNetV3", "unet_mobilenet_v3_focaldice_mag1c_rgb.pth"),
    "mobilenet_v3_dpu_bce_artigo": ("UNetMobileNetV3_dpu", "unet_mobilenet_v3_bce_mag1c_rgb.pth"),
    "mobilenet_v3_dpu_focaldice_artigo": ("UNetMobileNetV3_dpu", "unet_mobilenet_v3_focaldice_mag1c_rgb.pth"),
}

# Checkpoint do artigo treinado com align_corners=True. Os pesos sao compativeis
# com a variante DPU, mas essa diferenca afeta a interpolacao durante inferencia.
ARTIGO_BCE_MODELO = "unet_mobilenet_v3_attentiongates_dpu_bce_retrained_mag1c_rgb.pth"

from Modelos import UNetBaseline, UNetDepthReduced, UNetMobileNetV2, UNetMobileNetV3, UNetElementWise, carregar_hyperstarcop

def load_model(nome_modelo, device, align_corners=True):
    if nome_modelo == "attentiongates_dpu_bce_artigo":
        model = Modelos.UNetMobileNetV3AttentionGatesDPU(in_channels=4, out_channels=1)
        for nome in ("up1", "up2", "up3", "up4", "up_final"):
            getattr(model, nome).align_corners = align_corners
        caminho = (
            Path(__file__).resolve().parents[1]
            / "Modelos_treinados" / "Modelo_artigo" / ARTIGO_BCE_MODELO
        )
        if not caminho.is_file():
            raise FileNotFoundError(f"Checkpoint do modelo de artigo nao encontrado: {caminho}")
        model.load_state_dict(torch.load(caminho, map_location=device, weights_only=True))
        return model.to(device).eval()

    if nome_modelo in NOVOS_MODELOS:
        arquitetura, checkpoint = NOVOS_MODELOS[nome_modelo]
        model = getattr(Modelos, arquitetura)(in_channels=4, out_channels=1)
        caminho = Path(__file__).resolve().parents[1] / "Modelos_treinados" / checkpoint
        # Alguns pesos publicados ficam em Modelos_treinados/Modelo_artigo.
        if not caminho.is_file():
            caminho_artigo = caminho.parent / "Modelo_artigo" / caminho.name
            if caminho_artigo.is_file():
                caminho = caminho_artigo
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
        "baseline": (UNetBaseline, "unet_baseline_mag1c_rgb.pth"),
        "depth_reduced": (UNetDepthReduced, "unet_depth_reduced_mag1c_rgb.pth"),
        "mobilenet_v2": (UNetMobileNetV2, "unet_mobilenet_v2_mag1c_rgb.pth"),
        "mobilenet_v3": (UNetMobileNetV3, "unet_mobilenet_v3_bce_mag1c_rgb.pth"),
        "skip": (UNetElementWise, "unet_skip_connections_mag1c_rgb.pth"),
    }

    classe_modelo, checkpoint = modelos[nome_modelo]
    caminho = Path(__file__).resolve().parents[1] / "Modelos_treinados" / checkpoint

    model = classe_modelo(in_channels =4, out_channels=1)

    pesos = torch.load(caminho,map_location=device,weights_only=True)

    model.load_state_dict(pesos)
    model.to(device)
    model.eval()
    
    return model
