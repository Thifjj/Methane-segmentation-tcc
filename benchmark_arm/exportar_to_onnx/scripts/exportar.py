"""Exporta checkpoints PyTorch do projeto para ONNX usado no Cortex-A53."""

import argparse
import importlib
import hashlib
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
    "mobilenet_v3_bce": ("Modelos.UNet_MobileNet_v3:UNetMobileNetV3", "Mobile_Net_v3_mag1c_rgb.pth"),
    "mobilenet_v3_focaldice": ("Modelos.UNet_MobileNet_v3_dpu:UNetMobileNetV3_dpu", "MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth"),
    "mobilenet_v3_dpu": ("Modelos.UNet_MobileNet_v3_dpu:UNetMobileNetV3_dpu", "Mobile_Net_v3_dpu_mag1c_rgb.pth"),
    "mobilenet_v3_attentiongates": ("Modelos.UNet_MobileNetV3_AttentionGates:UNetMobileNetV3AttentionGates", "UNetMobileNetV3AttentionGates_mag1c_rgb.pth"),
    "attentiongates_bce": ("Modelos.UNet_MobileNetV3_AttentionGates:UNetMobileNetV3AttentionGates", "MobileNetV3_AttentionGates_BCE_mag1c_rgb.pth"),
    "attentiongates_focaldice_artigo": ("Modelos.UNet_MobileNetV3_AttentionGates:UNetMobileNetV3AttentionGates", "MobileNetV3_AttentionGates_FocalDiceLossmag1c_rgb.pth"),
    "attentiongates_dpu_bce_artigo": ("Modelos.UNet_MobileNetV3_AttentionGates_DPU:UNetMobileNetV3AttentionGatesDPU", "UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth"),
    "attentiongates_dpu_bce_retreinado": ("Modelos.UNet_MobileNetV3_AttentionGates_DPU:UNetMobileNetV3AttentionGatesDPU", "UNetMobileNetV3AttentionGatesDPU_BCE_mag1c_rgb.pth"),
    "attentiongates_dpu_focaldice_artigo": ("Modelos.UNet_MobileNetV3_AttentionGates_DPU:UNetMobileNetV3AttentionGatesDPU", "UnetMobilenetV3AttentionGates_dpu_FocalDiceLossmag1c_rgb.pth"),
    "mobilenet_v3_bce_artigo": ("Modelos.UNet_MobileNet_v3:UNetMobileNetV3", "Mobile_Net_v3_BCELoss_mag1c_rgb.pth"),
    "mobilenet_v3_focaldice_artigo": ("Modelos.UNet_MobileNet_v3:UNetMobileNetV3", "MobileNet_v3_FocalDiceLoss_mag1c_rgb.pth"),
    "skip": ("Modelos.UNet_SkipConnections:UNetElementWise", "UNET_SkipConnections_mag1c_rgb.pth"),
    "resnet34": ("Modelos.UNet_ResNet34:UNetResNet34", "UNet_ResNet34_mag1c_rgb.pth"),
    "segformer": ("Modelos.UNet_SegFormer:SegFormerB0", "UNet_SegFormer_mag1c_rgb.pth"),
    "hyperstarcop": ("Modelos.HyperStarcop_oficial:carregar_hyperstarcop", "HyperSTARCOP_oficial/final_checkpoint_model.ckpt"),
    "attention_gates": ("Modelos.UNet_AttentionGates:UNetAttentionGates", None),
    "attentiongates_dpu_easy_remaining": ("Modelos.UNet_MobileNetV3_AttentionGates_DPU:UNetMobileNetV3AttentionGatesDPU", "UnetMobilenetV3AttentionGates_dpu_easy_remaining_mag1c_rgb.pth"),
    "attentiongates_dpu_only_remaining": ("Modelos.UNet_MobileNetV3_AttentionGates_DPU:UNetMobileNetV3AttentionGatesDPU", "UnetMobilenetV3AttentionGates_dpu_only_remaining_mag1c_rgb.pth"),
    "psa": ("Modelos.UNet_PSA:UNetPSA", None),
}


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(f)


def _hash_stream(f):
    digest = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def exportar(nome, especificacao, checkpoint, destino, patch_size=512, dynamic_batch=False, verify=False, verify_atol=1e-4):
    import onnx

    modulo, classe = especificacao.split(":", 1)
    construtor = getattr(importlib.import_module(modulo), classe)
    if nome == "hyperstarcop":
        modelo = construtor(checkpoint, "cpu")
    else:
        modelo = construtor(in_channels=4, out_channels=1)
        pesos = torch.load(checkpoint, map_location="cpu", weights_only=True)
        modelo.load_state_dict(pesos)
    modelo.eval()
    torch.set_num_threads(4)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            modelo, torch.zeros(1, 4, patch_size, patch_size), str(destino),
            input_names=["input"], output_names=["logits"], opset_version=16,
            dynamo=False,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None,
        )
    graph = onnx.load(str(destino))
    onnx.checker.check_model(graph, full_check=True)
    rgb = classe in {"UNetMobileNetV3", "UNetMobileNetV3_dpu", "UNetMobileNetV3AttentionGates", "UNetMobileNetV3AttentionGatesDPU"}
    metadata = dict(modelo=nome, arquitetura=especificacao, checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=sha256(checkpoint), parametros=sum(p.numel() for p in modelo.parameters()),
                    ordem_canais="mag1c,460,550,640" if rgb else "mag1c,640,550,460",
                    tamanho_patch=patch_size, batch_dinamico=dynamic_batch, precisao="FP32",
                    normalizacao="mag1c/1750;bandas/60;clip[0,2]", saida="logits", opset=16,
                    ir_version=graph.ir_version, pytorch_versao=str(torch.__version__))
    if graph.ir_version > 8:
        raise ValueError("IR acima de 8: incompatível com ONNX Runtime 1.14 da placa")
    onnx.helper.set_model_props(graph, {k: str(v) for k, v in metadata.items()})
    onnx.save(graph, str(destino))
    if verify:
        import numpy as np
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(destino), options, providers=["CPUExecutionProvider"])
        checks = []
        generator = torch.Generator().manual_seed(12345)
        for batch in ([1, (512 // patch_size) ** 2] if dynamic_batch and patch_size < 512 else [1]):
            inputs = torch.rand(batch, 4, patch_size, patch_size, generator=generator) * 2
            with torch.inference_mode():
                expected = modelo(inputs).numpy()
            actual = session.run(["logits"], {"input": inputs.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=verify_atol)
            checks.append(dict(batch=batch, max_abs_error=float(np.abs(actual-expected).max()),
                               mean_abs_error=float(np.abs(actual-expected).mean()),
                               pixels_mascara_divergentes=int(np.count_nonzero((actual > 0) != (expected > 0)))))
        print("Verificacao PyTorch/ONNX:", checks)
    print(destino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Nome conhecido, all, ou modulo:Classe")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint .pth/.ckpt (obrigatorio para modelo novo)")
    parser.add_argument("--output", type=Path, help="Arquivo ONNX; somente com um modelo")
    parser.add_argument("--patch-size", type=int, choices=(128, 512), help="Padrão: 512 para todos os modelos")
    parser.add_argument("--dynamic-batch", action="store_true", help="Permite batch de patches 1–16; nome recebe _batch_dynamic")
    parser.add_argument("--verify", action="store_true", help="Confere ONNX Runtime contra PyTorch (requer onnxruntime)")
    parser.add_argument("--verify-atol", type=float, default=1e-4, help="Tolerancia absoluta da verificacao (padrao: 1e-4)")
    args = parser.parse_args()
    if not 0 < args.verify_atol < float("inf"):
        parser.error("--verify-atol deve ser finito e positivo")
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
        patch_size = args.patch_size or 512
        suffix = ("_patches128" if patch_size == 128 and "AttentionGatesDPU" not in especificacao else
                  "_512" if patch_size == 512 and "AttentionGatesDPU" in especificacao else "")
        suffix += "_batch_dynamic" if args.dynamic_batch else ""
        destino = args.output or ROOT / "benchmark_arm/exportar_to_onnx/modelos_convertidos_onnx" / f"{nome}{suffix}.onnx"
        exportar(nome, especificacao, checkpoint, destino, patch_size, args.dynamic_batch, args.verify, args.verify_atol)


if __name__ == "__main__":
    main()
