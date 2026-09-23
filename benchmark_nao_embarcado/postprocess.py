import torch


def postprocess(logits):
    probs = torch.sigmoid(logits)

    mask = (probs >= 0.5).float()

    return mask