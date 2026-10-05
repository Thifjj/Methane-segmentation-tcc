#!/usr/bin/env python3
"""Calibra, testa e exporta um modelo INT8 com vai_q_pytorch."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import (
    DEFAULT_PRODUCTS,
    calibration_defaults,
    add_model_arguments,
    resolve_model,
    build_calibration_loader,
    build_model,
    normalized_inputs,
    parse_products,
    calibration_contract,
    validate_calibration,
    file_sha256,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--config", help="JSON com padroes de calibracao; argumentos CLI tem prioridade.")
    parser.add_argument("--quant-mode", choices=("calib", "test"), required=True)
    parser.add_argument("--csv", required=True, help="CSV STARCOP usado na calibracao/validacao.")
    parser.add_argument("--data-root", required=True, help="Diretorio que contem as pastas das imagens.")
    parser.add_argument("--products", help="Ordem dos canais usada no treinamento.")
    patches = parser.add_mutually_exclusive_group()
    patches.add_argument("--patching", dest="patching", action="store_true")
    patches.add_argument("--no-patching", dest="patching", action="store_false")
    parser.set_defaults(patching=None)
    balance = parser.add_mutually_exclusive_group()
    balance.add_argument("--balanced-patches", dest="balanced", action="store_true")
    balance.add_argument("--unbalanced-patches", dest="balanced", action="store_false")
    parser.set_defaults(balanced=None)
    parser.add_argument("--patches-per-group", type=int,
                        help="Limite por grupo forte/fraco/fundo; usa igual quantidade nos tres.")
    parser.add_argument("--range-policy", choices=("layer_mse",), default="layer_mse",
                        help="Politica oficial: faixas por camada.")
    parser.add_argument("--max-clipping-percent", type=float, default=0.1)
    parser.add_argument("--range-samples", type=int)
    parser.add_argument("--refine-layers", type=int,
                        help="Camadas candidatas ao refinamento apos ajustes DPU; 0 desabilita.")
    parser.add_argument("--subset-len", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--target", help="Opcional: habilita quantizacao consciente do hardware.")
    parser.add_argument("--output-dir", default="build/vitis_ai/quantize")
    parser.add_argument("--deploy", action="store_true", help="No modo test, exporta o xmodel INT8.")
    parser.add_argument("--deploy-check", action="store_true")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        try:
            config = json.loads(Path(config_args.config).read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"Configuracao invalida: {exc}")
        allowed = {action.dest for action in parser._actions} - {"help", "config", "deploy", "deploy_check", "quant_mode", "csv", "data_root"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("JSON deve conter apenas parametros conhecidos de calibracao.")
        parser.set_defaults(**config)
    return parser.parse_args(argv)


class WidestRangeHistory(list):
    """Menor posicao de ponto fixo corresponde a maior faixa representavel."""
    def append(self, position):
        self[:] = [min([position] + self)]


def preserve_widest_ranges(quantizer):
    # ponytail: historico interno do Vitis AI 3.5; rever ao atualizar o quantizador.
    for kind in ("output", "input"):
        for name, histories in quantizer.config_history[kind].items():
            quantizer.config_history[kind][name] = [WidestRangeHistory(values) for values in histories]


def quantize(args):
    args.model, args.architecture = resolve_model(args.model, args.checkpoint, args.architecture)
    for key, value in calibration_defaults(args.model).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError as exc:
        raise SystemExit("pytorch_nndct nao encontrado. Execute dentro do container PyTorch do Vitis AI.") from exc

    if args.deploy and args.quant_mode != "test":
        raise SystemExit("--deploy somente pode ser usado com --quant-mode test.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size deve ser maior que zero.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers nao pode ser negativo.")
    if not 0 <= args.max_clipping_percent <= 100:
        raise SystemExit("--max-clipping-percent deve estar entre 0 e 100.")
    if args.range_samples < 1 or args.refine_layers < 0:
        raise SystemExit("--range-samples deve ser positivo e --refine-layers nao negativo.")
    if args.patching is None:
        args.patching = args.model != "mobilenet_v3_dpu_512"
    if args.balanced is None:
        args.balanced = args.patching and args.quant_mode == "calib"
    if args.target is None:
        args.target = "DPUCZDX8G_ISA1_B4096"
    if args.balanced and (args.quant_mode != "calib" or not args.patching):
        raise SystemExit("--balanced-patches requer --quant-mode calib e --patching.")
    args.height = args.height if args.height is not None else (128 if args.patching else 512)
    args.width = args.width if args.width is not None else (128 if args.patching else 512)
    if args.height < 1 or args.width < 1:
        raise SystemExit("Altura e largura devem ser positivas.")
    if args.patching and (args.height, args.width) != (128, 128):
        raise SystemExit("--patching requer --height 128 --width 128; use --no-patching para imagens inteiras.")
    if not args.products:
        args.products = ",".join(DEFAULT_PRODUCTS)
    if args.deploy:
        # Exigencia do export_xmodel: batch 1 e uma unica inferencia.
        args.batch_size = 1
        args.subset_len = 1

    products = parse_products(args.products)
    model, checkpoint = build_model(args.model, args.checkpoint, len(products), args.architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = build_calibration_loader(
        args.csv,
        args.data_root,
        products,
        args.subset_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        patching=args.patching,
        balanced=args.balanced,
        patches_per_group=args.patches_per_group,
    )
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.quant_mode == "test" and not (output_dir / "quant_info.json").is_file():
        raise SystemExit(f"Calibre primeiro: configuracao ausente em {output_dir / 'quant_info.json'}")
    if args.quant_mode == "test":
        validate_calibration(output_dir, args.model, checkpoint, products,
                             args.height, args.width, args.target, args.patching, args.architecture)
    elif (output_dir / "quant_info.json").exists():
        raise SystemExit(f"Diretorio ja calibrado: {output_dir}. Use outro --output-dir.")
    example = torch.zeros(args.batch_size, len(products), args.height, args.width, device=device)

    kwargs = dict(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(example,),
        output_dir=str(output_dir),
        device=device,
    )
    if args.target:
        kwargs["target"] = args.target
    from nndct_shared.utils import NndctOption
    NndctOption.nndct_diffs_mode.value = "maxmin"
    quantizer = torch_quantizer(**kwargs)
    if args.quant_mode == "calib":
        preserve_widest_ranges(quantizer.processor.quantizer)
    quant_model = quantizer.quant_model.eval()
    ranges = None
    if args.quant_mode == "calib":
        from calibration_ranges import LayerRanges
        ranges = LayerRanges(quantizer.processor.quantizer.quant_config, args.max_clipping_percent, args.range_samples)
        hooks = [module.register_forward_hook(ranges.capture) for module in quant_model.modules()]
        NndctOption.nndct_quant_off.value = True
        try:
            with torch.inference_mode():
                for i, inputs in enumerate(normalized_inputs(loader, products, device)):
                    if tuple(inputs.shape[1:]) != (len(products), args.height, args.width):
                        raise ValueError("Dimensoes dos patches diferem da configuracao.")
                    quant_model(inputs)
                    if (i + 1) % 100 == 0:
                        print(f"Observacao FP32 por camada: {i + 1} batches", flush=True)
        finally:
            for hook in hooks:
                hook.remove()
            NndctOption.nndct_quant_off.value = False
        ranges.apply(quantizer.processor.quantizer)
        print(f"Faixas selecionadas individualmente: {len(ranges.stats)} camadas", flush=True)

    count = 0
    print(f"Modelo: {args.model}; produtos: {products}; patching: {args.patching}; imagens: {len(loader.dataset.dataframe)}; range-policy: {args.range_policy}", flush=True)
    if args.balanced:
        print(f"Patches selecionados: {loader.dataset.group_counts}", flush=True)
    started_at = time.perf_counter()
    with torch.inference_mode():
        for inputs in normalized_inputs(loader, products, device):
            if tuple(inputs.shape[1:]) != (len(products), args.height, args.width):
                raise ValueError(f"Entrada {tuple(inputs.shape)} diferente da forma configurada; verifique --patching/--height/--width.")
            if not torch.isfinite(inputs).all():
                raise ValueError("Entrada normalizada contem NaN ou infinito.")
            quant_model(inputs)
            count += inputs.shape[0]
            if (
                args.progress_every > 0 and count % args.progress_every < inputs.shape[0]
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    f"{args.quant_mode}: {count} entradas "
                    f"({elapsed:.1f}s, {elapsed / count:.2f}s/amostra)",
                    flush=True,
                )
            if args.deploy:
                break
    elapsed = time.perf_counter() - started_at
    print(
        f"{args.quant_mode}: {count} amostras processadas em {elapsed:.1f}s; "
        f"checkpoint={checkpoint}"
    )

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        refinement = None
        if ranges is not None and args.refine_layers:
            reference, _ = build_model(args.model, args.checkpoint, len(products), args.architecture)
            source, _ = build_model(args.model, args.checkpoint, len(products), args.architecture)
            kwargs.update(quant_mode="test", module=source.to(device))
            refined = torch_quantizer(**kwargs)
            refinement = ranges.refine(refined.processor.quantizer, refined.quant_model.eval(),
                                       reference.to(device).eval(), loader, products, device, args.refine_layers)
            # A API nativa exporta sem refazer bias correction no modo test.
            refined.processor.quantizer.export_quant_config()
            quantizer = refined
            (output_dir / "refinement.json").write_text(json.dumps(refinement, indent=2))
        if ranges is not None:
            (output_dir / "layer_ranges.json").write_text(json.dumps(ranges.report(quantizer.processor.quantizer), indent=2))
        metadata = dict(calibration_contract(args.model, checkpoint, products, args.height,
                                             args.width, args.target, args.patching, args.architecture),
                        checkpoint=str(checkpoint), max_clipping_percent=args.max_clipping_percent,
                        range_samples=args.range_samples, refine_layers=args.refine_layers,
                        csv=str(Path(args.csv).resolve()), csv_sha256=file_sha256(args.csv),
                        artifacts={f: file_sha256(output_dir / f) for f in ("quant_info.json", "bias_corr.pth") if (output_dir / f).is_file()},
                        range_policy=args.range_policy, balanced=args.balanced, seed=args.seed,
                        images=loader.dataset.dataframe.id.tolist(), entries=count,
                        group_counts=getattr(loader.dataset, "group_counts", None),
                        patches=getattr(loader.dataset, "records", None))
        (output_dir / "calibration_manifest.json").write_text(json.dumps(metadata, indent=2))
    elif args.deploy:
        # A exportacao deve ocorrer com batch 1 e apenas uma execucao e usa a
        # configuracao gerada anteriormente no mesmo output_dir.
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=args.deploy_check)
        print(f"XModel quantizado exportado em: {output_dir}")


if __name__ == "__main__":
    quantize(parse_args())
