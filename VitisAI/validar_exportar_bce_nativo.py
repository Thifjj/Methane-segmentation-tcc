#!/usr/bin/env python3
"""Valida a calibração PTQ nativa e exporta XModel para ZCU104 (Vitis AI 3.5).

Requer o resultado de quantize_bce_native.py em
build/vitis_ai/ptq_nativa_bce_comparacao. Não recalibra nem altera o checkpoint.

cd /workspace/VitisAI
PYTHONPATH=/workspace:$PYTHONPATH python validar_exportar_bce_nativo.py --mode validate
PYTHONPATH=/workspace:$PYTHONPATH python validar_exportar_bce_nativo.py --mode export
"""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from common import PROJECT_ROOT, build_model, file_sha256, load_quantized_parameters
from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer
from Testes.Teste_Unet import binary_opening
from pytorch_nndct.apis import torch_quantizer


def create_context(path, csv, dataset_root, limit, threads):
    torch.set_num_threads(threads)
    base = Path(path).resolve()
    meta_file = base / 'manifest_comparacao.json'
    if not meta_file.is_file():
        raise FileNotFoundError(f'Manifesto da comparacao ausente: {meta_file}')
    comparison = json.loads(meta_file.read_text())
    original = Path(comparison['original_dir']).resolve()
    native = Path(comparison['native_dir']).resolve()
    original_manifest = json.loads((original / 'calibration_manifest.json').read_text())
    checkpoint = Path(comparison['checkpoint']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint nao encontrado: {checkpoint}')
    ck_hash = file_sha256(checkpoint)
    for value in (comparison.get('checkpoint_sha256'), original_manifest.get('checkpoint_sha256')):
        if value and value != ck_hash:
            raise RuntimeError('Checkpoint nao corresponde ao usado na calibracao!')
    if not (native / 'quant_info.json').is_file():
        raise FileNotFoundError(f'quant_info.json nativo nao encontrado em {native}')
    products = original_manifest['products']
    height, width = int(original_manifest['height']), int(original_manifest['width'])
    if (height, width) != (512, 512) or original_manifest.get('patching'):
        raise RuntimeError('Este teste requer imagens 512x512 sem patching')
    if comparison.get('target') != original_manifest.get('target'):
        raise RuntimeError('Target do manifesto da comparacao difere do original')
    name = original_manifest['model']
    arch = original_manifest['architecture']
    source, _ = build_model(name, checkpoint, len(products), arch)
    quant_source, _ = build_model(name, checkpoint, len(products), arch)
    target = original_manifest['target']
    kw = dict(quant_mode='test', module=quant_source.cpu().eval(),
              input_args=(torch.zeros(1, len(products), height, width),),
              output_dir=str(native), device=torch.device('cpu'))
    if target:
        kw['target'] = target
    quantizer = torch_quantizer(**kw)
    load_quantized_parameters(quantizer, native)
    qmodel = quantizer.quant_model.eval()
    source.cpu().eval()
    normalizer = DataNormalizer(products).cpu().eval()
    test_df = carregar_dataframe_starcop(
        str(csv), str(dataset_root), produtos_obrigatorios=products + ['labelbinary'])
    if limit:
        test_df = test_df.iloc[:limit].reset_index(drop=True)
    if len(test_df) == 0:
        raise RuntimeError('Nenhuma imagem de teste valida encontrada')
    ds = STARCOPDataset(test_df, products, ['labelbinary'], patching=False)
    dl = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
    return base, original, native, checkpoint, comparison, original_manifest, source, quantizer, qmodel, normalizer, test_df, dl


def validate(base, source, qmodel, normalizer, df, loader, checkpoint, native):
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    per_image = []
    acc = {label: dict(tp=0, fp=0, fn=0, aps=[]) for label in ('FP32', 'INT8_NATIVE')}
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            x = normalizer(batch['input'].float()).contiguous()
            if tuple(x.shape) != (1, len(normalizer.factors), 512, 512):
                raise ValueError(f'Entrada {i+1}: shape invalido {tuple(x.shape)}')
            y = (batch['output'][0, 0].cpu().numpy() > 0.5)
            for label, module in [('FP32', source), ('INT8_NATIVE', qmodel)]:
                z = module(x).detach().cpu().float()
                if z.shape != (1, 1, 512, 512) or not torch.isfinite(z).all():
                    raise RuntimeError(f'Saida invalida: {label}, imagem {i+1}')
                pred = binary_opening((z > 0).float(), kernel).numpy()[0,0].astype(bool)
                tp = int(np.count_nonzero(pred & y))
                fp = int(np.count_nonzero(pred & ~y))
                fn = int(np.count_nonzero(~pred & y))
                acc[label]['tp'] += tp
                acc[label]['fp'] += fp
                acc[label]['fn'] += fn
                ap = np.nan
                if y.any():
                    ap = float(average_precision_score(y.ravel(), z.numpy().astype(np.float64).ravel()))
                    acc[label]['aps'].append(ap)
                per_image.append(dict(modelo=label, id=str(df.iloc[i]['id']), has_plume=bool(y.any()),
                                      TP=tp, FP=fp, FN=fn, AP_logits=ap,
                                      min_logit=float(z.min()), max_logit=float(z.max()),
                                      frac_logit_positive=float((z > 0).float().mean())))
            if (i + 1) % 10 == 0 or i + 1 == len(loader):
                print(f'Avaliadas {i+1}/{len(loader)} imagens', flush=True)
    rows = []
    for label in acc:
        a = acc[label]
        tp, fp, fn = a['tp'], a['fp'], a['fn']
        f1 = 2*tp/(2*tp+fp+fn+1e-6)
        iou = tp/(tp+fp+fn+1e-6)
        auprc = float(np.mean(a['aps'])) if a['aps'] else np.nan
        rows.append(dict(modelo=label, imagens=len(df), imagens_com_pluma=len(a['aps']),
                         F1_global_abertura=f1, IoU_abertura=iou, AUPRC_media_logits_brutos=auprc))
        print(f'{label:>12} F1={f1:.6f} IoU={iou:.6f} AUPRC={auprc:.6f}', flush=True)
    output = base / 'validacao_completa'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'resumo.csv', index=False)
    pd.DataFrame(per_image).to_csv(output / 'por_imagem.csv', index=False)
    (output / 'metadados.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        quant_info_sha256=file_sha256(native/'quant_info.json'),
        images=df['id'].tolist()), indent=2))
    print('Resultados:', output)


def export_xmodel(base, native, quantizer, qmodel, normalizer, loader):
    output = base / 'xmodel_native_bce'
    if output.exists() and any(output.glob('*.xmodel')):
        raise RuntimeError(f'XModel ja existe em {output}; nao sobrescrever.')
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        batch = next(iter(loader))
        x = normalizer(batch['input'].float()).contiguous()
        if tuple(x.shape) != (1, len(normalizer.factors), 512, 512):
            raise RuntimeError(f'Shape invalido para exportacao: {tuple(x.shape)}')
        _ = qmodel(x)
        quantizer.export_xmodel(output_dir=str(output), deploy_check=False)
    generated = list(output.glob('*.xmodel'))
    if not generated:
        raise RuntimeError('export_xmodel terminou mas nenhum .xmodel foi encontrado')
    print('XModels exportados:')
    for f in generated:
        print(f'  {f} ({f.stat().st_size} bytes)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['validate','export','both'], default='validate')
    parser.add_argument('--comparison-dir', default='build/vitis_ai/ptq_nativa_bce_comparacao')
    parser.add_argument('--test-csv', default=str(PROJECT_ROOT / 'STARCOP_test/test.csv'))
    parser.add_argument('--test-root', default=str(PROJECT_ROOT / 'STARCOP_test'))
    parser.add_argument('--limit', type=int, default=0, help='0 = todas as imagens')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1:
        parser.error('--limit >= 0 e --threads >= 1')
    base, old, native, ckpt, comparison, manifest, source, quantizer, qmodel, normalizer, df, loader = create_context(
        args.comparison_dir, args.test_csv, args.test_root, args.limit, args.threads)
    print('Checkpoint:', ckpt)
    print('Calibracao nativa:', native)
    print('Quantizador target:', manifest['target'])
    print('Imagens:', len(df))
    print('Nenhum checkpoint/quant_info antigo sera sobrescrito.')
    if args.mode in ('validate','both'):
        validate(base, source, qmodel, normalizer, df, loader, ckpt, native)
    if args.mode in ('export','both'):
        export_xmodel(base, native, quantizer, qmodel, normalizer, loader)

if __name__ == '__main__':
    main()
