import torch
from kornia.morphology import erosion, dilation


def postprocess(logits):
    # Mesmo limiar estrito e abertura com cruz 3x3 de Testes/Teste_Unet.py.
    mask = (logits > 0.0).float()
    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], device=logits.device, dtype=torch.float32)
    eroded = torch.clamp(erosion(mask, kernel), 0, 1) > 0
    return (torch.clamp(dilation(eroded.float(), kernel), 0, 1) > 0).float()
