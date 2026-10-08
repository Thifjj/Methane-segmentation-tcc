"""Leitura de energia real do pacote CPU (RAPL) e GPU (NVML)."""
import ctypes
import os
import re
import time
from pathlib import Path

import torch


class MedidorEnergia:
    """Energia do pacote CPU (RAPL) e, em CUDA, da GPU (NVML)."""
    def __init__(self, dispositivo):
        self.fontes = {}
        self.nvml = None
        self.fds = []
        bloqueados = []
        for zona in Path("/sys/class/powercap").glob("*-rapl:*"):
            if zona.name.count(":") != 1:
                continue  # Subzonas de núcleos já estão incluídas no pacote.
            try:
                if not (zona / "name").read_text().strip().startswith("package-"):
                    continue
                limite = int((zona / "max_energy_range_uj").read_text())
                int((zona / "energy_uj").read_text())
                self.fontes[f"cpu_{zona.name}"] = (
                    lambda z=zona: int((z / "energy_uj").read_text()), 1e-6, limite
                )
            except PermissionError:
                bloqueados.append(str((zona / "energy_uj").resolve()))
            except (OSError, ValueError):
                pass

        if not any(nome.startswith("cpu_") for nome in self.fontes):
            self._abrir_rapl_perf()
        if not any(nome.startswith("cpu_") for nome in self.fontes):
            print("Potencia CPU indisponivel: sem acesso ao contador RAPL; sera registrada como NaN.", flush=True)
            for caminho in bloqueados:
                print(f"Para liberar a leitura nesta sessao, execute no terminal: sudo chmod a+r {caminho}", flush=True)
        if dispositivo.type == "cuda":
            self._abrir_nvml()

    def _abrir_rapl_perf(self):
        """Alternativa quando o RAPL do sysfs não é legível."""
        try:
            evento = Path("/sys/bus/event_source/devices/power/events/energy-pkg")
            codigo = re.search(r"event=(0x[0-9a-fA-F]+|[0-9]+)", evento.read_text())
            if codigo is None:
                return
            tipo = int(Path("/sys/bus/event_source/devices/power/type").read_text())
            escala = float(Path(f"{evento}.scale").read_text())
            attr = (ctypes.c_ubyte * 128)()
            ctypes.c_uint32.from_buffer(attr, 0).value = tipo
            ctypes.c_uint32.from_buffer(attr, 4).value = 128
            ctypes.c_uint64.from_buffer(attr, 8).value = int(codigo.group(1), 0)
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.syscall(298, ctypes.byref(attr), -1, 0, -1, 0)  # x86_64
            if fd < 0:
                return
            self.fds.append(fd)
            self.fontes["cpu_rapl_pkg"] = (
                lambda f=fd: self._ler_perf(f), escala, None
            )
        except (OSError, ValueError, AttributeError):
            pass

    @staticmethod
    def _ler_perf(fd):
        dados = os.read(fd, 8)
        if len(dados) != 8:
            raise OSError("Leitura incompleta do contador RAPL")
        return int.from_bytes(dados, "little")

    def _abrir_nvml(self):
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            nvml.nvmlInit_v2.restype = ctypes.c_int
            if nvml.nvmlInit_v2() != 0:
                return
            self.nvml = nvml
            nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
            ]
            nvml.nvmlDeviceGetTotalEnergyConsumption.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)
            ]
            indice = torch.cuda._get_nvml_device_index(0)
            gpu = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(indice, ctypes.byref(gpu)) != 0:
                return

            def ler_gpu():
                energia = ctypes.c_ulonglong()
                if nvml.nvmlDeviceGetTotalEnergyConsumption(gpu, ctypes.byref(energia)) != 0:
                    raise OSError("Contador de energia NVML indisponível")
                return energia.value

            ler_gpu()
            self.fontes["gpu_nvml"] = (ler_gpu, 1e-3, None)
        except (OSError, AttributeError, RuntimeError):
            pass

    def ler(self):
        valores = {}
        for nome, (ler, _, _) in self.fontes.items():
            try:
                valores[nome] = ler()
            except (OSError, ValueError):
                pass
        return time.perf_counter(), valores

    def fechar(self):
        for fd in self.fds:
            os.close(fd)
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


def medir_energia(medidor, antes, depois):
    medidas = {}
    duracao = depois[0] - antes[0]
    for nome in antes[1].keys() & depois[1].keys():
        _, escala, limite = medidor.fontes[nome]
        diferenca = depois[1][nome] - antes[1][nome]
        if diferenca < 0 and limite is not None:
            diferenca += limite  # RAPL volta a zero após max_energy_range_uj.
        if diferenca >= 0 and duracao > 0:
            joules = diferenca * escala
            medidas[nome] = (joules, joules / duracao, duracao)
    return medidas


def resumir_energia_cpu(medidas, inferencias):
    """Soma apenas pacotes CPU; potencia = joules / tempo observado, sem estimativa por TDP."""
    pacotes = [m for nome, m in medidas.items() if nome.startswith("cpu_")]
    valido = pacotes and all(m["status"] == "ok" and m["duracao_s"] > 0 for m in pacotes)
    if not valido:
        return dict(cpu_potencia_media_w=float("nan"), cpu_energia_j=float("nan"),
                    cpu_energia_por_inferencia_j=float("nan"),
                    cpu_energia_status="parcial" if pacotes else "indisponivel")
    energia = sum(m["energia_j"] for m in pacotes)
    return dict(cpu_potencia_media_w=sum(m["media_w"] for m in pacotes),
                cpu_energia_j=energia, cpu_energia_por_inferencia_j=energia/inferencias,
                cpu_energia_status="ok")


def resumir_energia_cpu_gpu(medidas, inferencias):
    """Soma os domínios monitorados CPU RAPL e GPU NVML em execuções CUDA."""
    cpu = resumir_energia_cpu(medidas, inferencias)
    gpu = medidas.get("gpu_nvml")
    gpu_valido = gpu is not None and gpu["status"] == "ok" and gpu["duracao_s"] > 0
    gpu_energia = gpu["energia_j"] if gpu_valido else float("nan")
    gpu_potencia = gpu["media_w"] if gpu_valido else float("nan")
    completo = cpu["cpu_energia_status"] == "ok" and gpu_valido
    total = cpu["cpu_energia_j"] + gpu_energia if completo else float("nan")
    return dict(gpu_energia_j=gpu_energia,
                gpu_potencia_media_w=gpu_potencia,
                gpu_energia_por_inferencia_j=gpu_energia/inferencias if gpu_valido else float("nan"),
                gpu_energia_status="ok" if gpu_valido else "parcial" if gpu else "indisponivel",
                cpu_gpu_energia_j=total,
                cpu_gpu_potencia_media_w=cpu["cpu_potencia_media_w"] + gpu_potencia if completo else float("nan"),
                cpu_gpu_energia_por_inferencia_j=total/inferencias if completo else float("nan"),
                cpu_gpu_energia_status="ok" if completo else "parcial" if gpu or cpu["cpu_energia_status"] != "indisponivel" else "indisponivel")
