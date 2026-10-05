"""Exports dos modelos, carregados sob demanda para evitar dependências extras."""

from importlib import import_module


_EXPORTS = {
    "UNetBaseline": (".UNet_baseline", "UNetBaseline"),
    "UNetDepthReduced": (".UNet_depth_reduced", "UNetDepthReduced"),
    "UNetMobileNetV2": (".UNet_MobileNet_v2", "UNetMobileNetV2"),
    "UNetMobileNetV3": (".UNet_MobileNet_v3", "UNetMobileNetV3"),
    "UNetElementWise": (".UNet_SkipConnections", "UNetElementWise"),
    "HyperSTARCOPOficial": (".HyperStarcop_oficial", "HyperSTARCOPOficial"),
    "carregar_hyperstarcop": (".HyperStarcop_oficial", "carregar_hyperstarcop"),
    "UNetAttentionGates": (".UNet_AttentionGates", "UNetAttentionGates"),
    "UNetPSA": (".UNet_PSA", "UNetPSA"),
    "UNetResNet34": (".UNet_ResNet34", "UNetResNet34"),
    "SegFormerB0": (".UNet_SegFormer", "SegFormerB0"),
    "UNetMobileNetV3AttentionGatesDPU" : (".UNet_MobileNetV3_AttentionGates_DPU", "UNetMobileNetV3AttentionGatesDPU"),
    "UNetMobileNetV3AttentionGates": (".UNet_MobileNetV3_AttentionGates", "UNetMobileNetV3AttentionGates"),
    "UNetMobileNetV3_dpu" : (".UNet_MobileNet_v3_dpu", "UNetMobileNetV3_dpu")
}

__all__ = list(_EXPORTS)


def __getattr__(nome):
    if nome not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")
    modulo, atributo = _EXPORTS[nome]
    valor = getattr(import_module(modulo, __name__), atributo)
    globals()[nome] = valor
    return valor
