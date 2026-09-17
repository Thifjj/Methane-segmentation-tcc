import torch


def preprocess(canais):
    mag1c = canais[0] / 1750.0
    r = canais[1] / 60.0
    g = canais[2] / 60.0
    b = canais[3] / 60.0

    mag1c = torch.clamp(mag1c, 0, 2)
    r = torch.clamp(r, 0, 2)
    g = torch.clamp(g, 0, 2)
    b = torch.clamp(b, 0, 2)

    x = torch.stack([
        mag1c,
        r,
        g,
        b
    ], dim=0)

    x = x.unsqueeze(0)

    return x