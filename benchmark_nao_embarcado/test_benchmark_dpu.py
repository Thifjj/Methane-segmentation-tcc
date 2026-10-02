"""Verificacao curta: patches, protocolo oficial/bruto e estatisticas."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .benchmark_dpu import (
    inferir, metricas_imagem, preparar_entrada, reconstruir, resumir_metricas, estatisticas,
    finalizar_energia,
)
from Utils.DataLoader import DataNormalizer
from VitisAI.common import DEFAULT_PRODUCTS


def main():
    torch.set_num_threads(2)
    image = torch.arange(512*512, dtype=torch.float32).reshape(1, 1, 512, 512) / (512*512)
    inputs = torch.cat([image*1750, image*30, image*45, image*60], dim=1)
    normalizer = DataNormalizer(list(DEFAULT_PRODUCTS)).eval()
    model = torch.nn.Conv2d(4, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1))
        expected = model(normalizer(inputs))
        for size in (128, 512):
            patches = preparar_entrada(inputs, normalizer, size)
            for batch in (1, len(patches)):
                result = reconstruir(inferir(model, patches, batch), size)
                torch.testing.assert_close(result, expected)
        try:
            preparar_entrada(torch.full_like(inputs, float("nan")), normalizer, 128)
        except ValueError:
            pass
        else:
            raise AssertionError("Entrada nao finita aceita")

        target = torch.zeros(1, 1, 512, 512)
        target[..., 0:6, 0:6] = 1  # Inclui a borda: abertura geodesica.
        logits = torch.full_like(target, -2.)
        logits[..., 0:6, 0:6] = 2.
        logits[..., 30, 30] = 2.  # Falso positivo isolado deve desaparecer na abertura.
        record = dict(id="pluma", has_plume=False, difficulty="easy", qplume=500)
        raw = metricas_imagem(logits, target, record, False)
        official = metricas_imagem(logits, target, record, True)
        assert raw["fp"] == 1 and official["fp"] == 0
        assert raw["difficulty"] == "sem_pluma" and official["difficulty"] == "forte"
        assert official["tp"] == 35  # Abertura remove apenas o canto interior do quadrado.
        ap = average_precision_score(target.reshape(-1), torch.sigmoid(logits).reshape(-1))
        for row in (raw, official):
            row["average_precision"] = ap
        background_logits = torch.full_like(target, -1.)
        background_logits[..., 100:125, 100:127] = 1.  # >640 mesmo apos abertura.
        bg = metricas_imagem(background_logits, torch.zeros_like(target),
                             dict(id="fundo", has_plume=False, difficulty="", qplume=0), True)
        bg["average_precision"] = float("nan")
        summary, groups = resumir_metricas([official, bg], True)
        assert summary["imagens_positivas"] == 1 and summary["auprc"] == ap
        assert summary["fp_tiles"] == 1 and summary["fpr_tile"] == 1
        assert summary["fpr_tile_tabela"] == .5
        assert len(groups) == 3
        raw_summary, _ = resumir_metricas([raw], False, .75)
        assert raw_summary["auprc"] == .75
        empty, _ = resumir_metricas([bg], True)
        assert np.isnan(empty["auprc"]) and empty["f1_global"] == 0
        stats = estatisticas([1, 2, 3])
        assert stats["media_ms"] == 2 and stats["p95_ms"] == 2.9
    # Contador com wrap: integra intervalos e pondera potencia pelo tempo.
    import threading
    class Counter:
        fontes = {"cpu": None}
        def ler(self):
            return (3., {"cpu": 10})
    def measure(_, a, b):
        joules = (b[1]["cpu"] - a[1]["cpu"]) % 100
        duration = b[0] - a[0]
        return {"cpu": (joules, joules/duration, duration)}
    energy = finalizar_energia(Counter(), (threading.Event(), None,
                              [(0., {"cpu": 90}), (1., {"cpu": 0})]), measure)["cpu"]
    assert energy["energia_j"] == 20 and energy["duracao_s"] == 3
    assert energy["media_w"] == 20/3 and energy["minima_w"] == 5 and energy["maxima_w"] == 10
    print("Self-test DPU OK: patches, bordas, grupos, AP, FPR e estatisticas.")


if __name__ == "__main__":
    main()
