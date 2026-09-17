import torch

from Modelos import UNetBaseline, UNetDepthReduced, UNetMobileNetV2, UNetMobileNetV3, UNetElementWise

def load_model(nome_modelo, device):
    device = torch.device("cpu")

    modelos = {
        "baseline": (
            UNetBaseline,
            #"/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/UNET_mag1c_rgb.pth"
            ""
        ),

        "depth_reduced": (
            UNetDepthReduced,
            #"/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/UNET_depth_reduced_mag1c_rgb.pth"
        ),

        "mobilenet_v2": (
            UNetMobileNetV2,
            #"/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/Mobile_Net_v2_mag1c_rgb.pth"
            "/home/thiago/Documents/Laboratorio_LEDS/Projetos_aceleradores/Segmentacao_de_metano/Joao/projeto/Methane-segmentation-tcc/Modelos_treinados/Mobile_Net_v2_mag1c_rgb.pth"
        ),

        "mobilenet_v3": (
            UNetMobileNetV3,
            #"/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/Mobile_Net_v3_mag1c_rgb.pth"
        ),
        "skip": (
            UNetElementWise,
            #"/media/jacques/hdd/Laboratorio/Projeto_joao/Methane_segmentation/Modelos_treinados/UNET_SkipConnections_mag1c_rgb.pth"
            "/home/thiago/Documents/Laboratorio_LEDS/Projetos_aceleradores/Segmentacao_de_metano/Joao/projeto/Methane-segmentation-tcc/Modelos_treinados/UNET_SkipConnections_mag1c_rgb.pth"
        )
    }

    classe_modelo, caminho = modelos[nome_modelo]

    model = classe_modelo(in_channels =4, out_channels=1)

    pesos = torch.load(caminho,map_location=device,weights_only=True)

    model.load_state_dict(pesos)
    model.to(device)
    model.eval()
    
    return model