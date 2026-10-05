"""Verifica conversao RAPL, wrap, potencia ponderada e ausencia de sensor."""
import math
import threading

from .energia import medir_energia, resumir_energia_cpu
from .benchmark_dpu import finalizar_energia


def main():
    class Contador:
        fontes = {"cpu_teste": (None, 1e-6, 100_000_000)}

        def ler(self):
            return 3., {"cpu_teste": 10_000_000}

    contador = Contador()
    energia = finalizar_energia(contador, (threading.Event(), None,
                               [(0., {"cpu_teste": 90_000_000}),
                                (1., {"cpu_teste": 0})]), medir_energia)
    cpu = resumir_energia_cpu(energia, 4)
    assert cpu["cpu_energia_j"] == 20
    assert cpu["cpu_energia_por_inferencia_j"] == 5
    assert math.isclose(cpu["cpu_potencia_media_w"], 20/3)  # Nao media simples de 10 e 5 W.
    assert cpu["cpu_energia_status"] == "ok"
    energia["gpu_nvml"] = dict(energia_j=999, media_w=999, duracao_s=3, status="ok")
    assert resumir_energia_cpu(energia, 4) == cpu  # GPU nao conta como CPU.
    energia["cpu_teste"]["status"] = "parcial"
    assert math.isnan(resumir_energia_cpu(energia, 4)["cpu_potencia_media_w"])
    sem_sensor = resumir_energia_cpu({}, 4)
    assert sem_sensor["cpu_energia_status"] == "indisponivel"
    assert math.isnan(sem_sensor["cpu_potencia_media_w"])
    print("Energia OK: RAPL uJ -> J, wrap, W ponderados, J/inferencia e NaN sem sensor.")


if __name__ == "__main__":
    main()
