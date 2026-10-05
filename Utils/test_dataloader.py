"""Execute na raiz: venv/bin/python -m Utils.test_dataloader."""
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.windows import Window
from torch.utils.data import DataLoader

from Utils.DataLoader import STARCOPDataset


def main():
    with TemporaryDirectory() as directory:
        values = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
        for product in ("mag1c", "labelbinary", "weight_mag1c"):
            with rasterio.open(Path(directory) / f"{product}.tif", "w",
                               driver="GTiff", width=512, height=512, count=1,
                               dtype="float32", transform=from_origin(1, 1, 1, 1)) as tif:
                tif.write(values, 1)
        frame = pd.DataFrame([dict(folder=directory, window=Window(0, 0, 512, 512))])
        patches = STARCOPDataset(frame, ["mag1c"], ["labelbinary"], "weight_mag1c")
        batch = next(iter(DataLoader(patches, batch_size=1)))
        expected = torch.stack([
            torch.from_numpy(values[y:y + 128, x:x + 128]).unsqueeze(0)
            for y in range(0, 385, 64) for x in range(0, 385, 64)
        ]).unsqueeze(0)
        for key in ("input", "output", "weight_loss"):
            assert batch[key].shape == (1, 49, 1, 128, 128)
            assert torch.equal(batch[key], expected), key
        full = STARCOPDataset(frame, ["mag1c"], ["labelbinary"], patching=False)[0]
        assert full["input"].shape == (1, 512, 512)
        assert torch.equal(full["input"][0], torch.from_numpy(values))
    print("Patches 128/passo 64, alinhamento input/label/pesos e imagem inteira OK")


if __name__ == "__main__":
    main()
