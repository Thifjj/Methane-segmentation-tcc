"""Exporta checkpoints PyTorch do projeto para ONNX usado no Cortex-A53."""

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

MODELOS = {
    "baseline": ("Modelos.UNet_baseline:UNetBaseline", "UNET_mag1c_rgb.pth"),
    "depth_reduced": ("Modelos.UNet_depth_reduced:UNetDepthReduced", "UNET_depth_reduced_mag1c_rgb.pth"),
    "mobilenet_v2": ("Modelos.UNet_MobileNet_v2:UNetMobileNetV2", "Mobile_Net_v2_mag1c_rgb.pth"),
    "mobilenet_v3": ("Modelos.UNet_MobileNet_v3:UNetMobileNetV3", "Mobile_Net_v3_mag1c_rgb.pth"),
    "skip": ("Modelos.UNet_SkipConnections:UNetElementWise", "UNET_SkipConnections_mag1c_rgb.pth"),
    "resnet34": ("Modelos.UNet_ResNet34:UNetResNet34", "UNet_ResNet34_mag1c_rgb.pth"),
    "segformer": ("Modelos.UNet_SegFormer:SegFormerB0", "UNet_SegFormer_mag1c_rgb.pth"),
    "hyperstarcop": ("Modelos.HyperStarcop_oficial:carregar_hyperstarcop", "HyperSTARCOP_oficial/final_checkpoint_model.ckpt"),
    "attention_gates": ("Modelos.UNet_AttentionGates:UNetAttentionGates", None),
    "psa": ("Modelos.UNet_PSA:UNetPSA", None),
}


def exportar(nome, especificacao, checkpoint, destino):
    modulo, classe = especificacao.split(":", 1)
    construtor = getattr(importlib.import_module(modulo), classe)
    if nome == "hyperstarcop":
        modelo = construtor(checkpoint, "cpu")
    else:
        modelo = construtor(in_channels=4, out_channels=1)
        pesos = torch.load(checkpoint, map_location="cpu", weights_only=True)
        modelo.load_state_dict(pesos)
    modelo.eval()
    destino.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            modelo, torch.zeros(1, 4, 512, 512), str(destino),
            input_names=["input"], output_names=["logits"], opset_version=16,
            dynamo=False,
        )
    print(destino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Nome conhecido, all, ou modulo:Classe")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint .pth/.ckpt (obrigatorio para modelo novo)")
    parser.add_argument("--output", type=Path, help="Arquivo ONNX; somente com um modelo")
    args = parser.parse_args()
    if args.model == "all" and (args.checkpoint or args.output):
        parser.error("--checkpoint/--output nao se aplicam a --model all")
    nomes = [nome for nome, (_, pesos) in MODELOS.items() if pesos] if args.model == "all" else [args.model]
    if args.model not in MODELOS and args.model != "all" and (":" not in args.model or not args.checkpoint):
        parser.error("Modelo novo exige --model modulo:Classe --checkpoint ARQUIVO")
    for nome in nomes:
        especificacao, padrao = MODELOS.get(nome, (nome, None))
        if not args.checkpoint and padrao is None:
            parser.error(f"{nome} exige --checkpoint ARQUIVO")
        checkpoint = args.checkpoint or ROOT / "Modelos_treinados" / padrao
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        destino = args.output or ROOT / "benchmark_arm/exportar_to_onnx/modelos_convertidos_onnx" / f"{nome}.onnx"
        exportar(nome, especificacao, checkpoint, destino)


if __name__ == "__main__":
    main()
