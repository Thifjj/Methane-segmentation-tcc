import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class HyperSTARCOPOficial(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=4,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.network(x)


def carregar_hyperstarcop(caminho_checkpoint, device):

    checkpoint = torch.load(
        caminho_checkpoint,
        map_location=device,
        weights_only=False
    )

    state_dict = checkpoint["state_dict"]

    # O checkpoint Lightning usa:
    # network.encoder...
    # network.decoder...
    # etc.
    #
    # Removemos "network." para carregar diretamente na U-Net.
    pesos_network = {}

    for nome, peso in state_dict.items():
        if nome.startswith("network."):
            novo_nome = nome.removeprefix("network.")
            pesos_network[novo_nome] = peso

    model = HyperSTARCOPOficial()

    model.network.load_state_dict(
        pesos_network,
        strict=True
    )

    model.to(device)
    model.eval()

    return model