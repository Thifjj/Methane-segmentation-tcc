#!/usr/bin/env python3
"""Compara PTQ original do BCE com PTQ nativa do Vitis AI 3.5.

Execute em /workspace/VitisAI, com diagnostico_bce_vitisai35.py ao lado.
Não altera pesos ou quant_info.json originais.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import PROJECT_ROOT, build_model, file_sha256
from Utils.DataLoader import DataNormalizer, STARCOPDataset, carregar_dataframe_starcop
from diagnostico_bce_vitisai35 import (
    locate_quant_dir, find_checkpoint, get_outputs, calc_metrics,
)


def select_calibration_rows(csv_path, data_root, products, images, limit):
    raw = pd.read_csv(csv_path)
    if 'id' not in raw.columns:
        raise ValueError('CSV de calibracao precisa da coluna id.')
    if images:
        # Tenta reproduzir exatamente o conjunto e a ordem da calibracao atual.
        if len(images) != len(set(images)):
            raise ValueError('Manifesto contem IDs repetidos; selecao ambigua.')
        if raw['id'].duplicated().any():
            raise ValueError('CSV de treino tem IDs repetidos; selecao ambigua.')
        lookup = raw.set_index('id', drop=False)
        missing = [name for name in images if name not in lookup.index]
        if missing:
            raise ValueError(f'IDs do manifesto ausentes no CSV de treino: {missing[:5]}')
        raw = lookup.loc[images].reset_index(drop=True)
    else:
        print('AVISO: manifesto sem lista de imagens; usando amostragem reproduzivel.', flush=True)
        raw = raw.sample(n=min(limit, len(raw)), random_state=12345).reset_index(drop=True)
    if limit and len(raw) != limit:
        print(f'AVISO: selecionadas {len(raw)} imagens; solicitado {limit}.', flush=True)
    # Evita varrer milhares de arquivos quando somente 100 serão calibrados.
    with tempfile.TemporaryDirectory(prefix='calib_csv_') as tmp:
        tmp_csv = Path(tmp) / 'amostras.csv'
        raw.to_csv(tmp_csv, index=False)
        df = carregar_dataframe_starcop(
            str(tmp_csv), str(data_root), produtos_obrigatorios=products
        )
    if len(df) != len(raw):
        raise RuntimeError(f'Faltam imagens de calibracao: CSV={len(raw)} validas={len(df)}')
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-quant-dir', required=True, help='Pasta da quantizacao atual com quant_info.json')
    parser.add_argument('--checkpoint', help='Override do caminho do checkpoint, validado por hash')
    parser.add_argument('--train-csv', default='/dataset_STARCOP/train.csv')
    parser.add_argument('--train-root', default='/dataset_STARCOP')
    parser.add_argument('--eval-csv', default='/tmp/diagnostico_bce_amostras.csv')
    parser.add_argument('--eval-root', default=str(PROJECT_ROOT / 'STARCOP_test'))
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output-dir', default='build/vitis_ai/ptq_nativa_bce_comparacao')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    source = locate_quant_dir(args.source_quant_dir)
    manifest = json.loads((source / 'calibration_manifest.json').read_text())
    checkpoint = find_checkpoint(args.checkpoint or manifest['checkpoint'], source)
    if manifest.get('checkpoint_sha256') and file_sha256(checkpoint) != manifest['checkpoint_sha256']:
        raise RuntimeError('Checkpoint não corresponde ao manifesto da calibracao existente!')
    for filename, digest in manifest.get('artifacts', {}).items():
        p = source / filename
        if not p.is_file() or file_sha256(p) != digest:
            raise RuntimeError(f'Arquivo original ausente/alterado: {p}')

    products = manifest['products']
    name = manifest['model']
    architecture = manifest['architecture']
    height, width = int(manifest['height']), int(manifest['width'])
    target = manifest.get('target')
    if (height, width) != (512, 512) or manifest.get('patching', False):
        raise RuntimeError('Este teste foi implementado para a calibracao 512x512 sem patches.')

    dst = Path(args.output_dir).expanduser().resolve()
    calib_dir = dst / 'quantize_native'
    if dst.resolve() == source.resolve() or dst.resolve() in source.parents:
        raise RuntimeError('Diretorio de saída coincide com os artefatos originais.')
    if (calib_dir / 'quant_info.json').exists():
        raise RuntimeError(f'Calibracao nativa ja existe: {calib_dir}; use outro --output-dir.')
    calib_dir.mkdir(parents=True, exist_ok=True)

    ids = manifest.get('images', [])
    if ids and len(ids) != args.samples:
        print(f'AVISO: manifesto tem {len(ids)} IDs; usarei o conjunto original completo.', flush=True)
    df_calib = select_calibration_rows(args.train_csv, args.train_root, products, ids, args.samples)
    ds_calib = STARCOPDataset(df_calib, products, [], patching=False)
    dl_calib = DataLoader(ds_calib, batch_size=1, shuffle=False, num_workers=0)
    normalizer = DataNormalizer(products).eval()

    print(f'Checkpoint: {checkpoint}\nCalibracao original: {source}')
    print(f'Calibracao nativa nova: {calib_dir}\nImagens: {len(df_calib)}')
    print('Iniciando quantizador PADRAO AMD (sem LayerRanges / preserve_widest_ranges).')
    from pytorch_nndct.apis import torch_quantizer
    model, _ = build_model(name, checkpoint, len(products), architecture)
    kw = dict(quant_mode='calib', module=model.eval().cpu(),
              input_args=(torch.zeros(1, len(products), height, width),),
              output_dir=str(calib_dir), device=torch.device('cpu'))
    if target:
        kw['target'] = target
    quantizer = torch_quantizer(**kw)
    quant_model = quantizer.quant_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl_calib, 1):
            x = normalizer(batch['input'].float())
            if tuple(x.shape) != (1, len(products), height, width):
                raise RuntimeError(f'Imagem {i}: shape {tuple(x.shape)}; esperada 1x{len(products)}x512x512.')
            if not torch.isfinite(x).all():
                raise RuntimeError(f'Imagem {i}: input contem NaN ou infinito.')
            quant_model(x)
            print(f'CALIB AMD: {i}/{len(ds_calib)}', flush=True)
    quantizer.export_quant_config()
    if not (calib_dir / 'quant_info.json').is_file():
        raise RuntimeError('Quantizador nao gerou quant_info.json')

    df_eval = carregar_dataframe_starcop(args.eval_csv, args.eval_root,
                                         produtos_obrigatorios=products + ['labelbinary'])
    ds_eval = STARCOPDataset(df_eval, products, ['labelbinary'], patching=False)
    dl_eval = DataLoader(ds_eval, batch_size=1, shuffle=False, num_workers=0)
    xs, ys, eval_ids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dl_eval):
            x = normalizer(batch['input'].float())
            if tuple(x.shape) != (1, len(products), height, width):
                raise RuntimeError(f'Imagem de teste {i}: shape inesperado {tuple(x.shape)}')
            xs.append(x.contiguous())
            ys.append(batch['output'].numpy()[0, 0].copy())
            eval_ids.append(str(df_eval.iloc[i]['id']))
    if not xs or not any((y > 0.5).any() for y in ys):
        raise RuntimeError('Precisa usar CSV de avaliacao que contenha imagens positivas.')
    common = set(eval_ids).intersection(ids)
    if common:
        raise RuntimeError(f'AVISO: {len(common)} imagens da avaliacao foram usadas na calibracao.')

    outputs = {}
    with tempfile.TemporaryDirectory(prefix='comparacao_native_ptq_') as tempdir:
        root = Path(tempdir)
        outputs['FP32'] = get_outputs('FP32_PyTorch', name, architecture, checkpoint,
                                       products, height, width, target, xs, root / 'fp32')
        for label, quant_path in [('INT8_ANTIGO', source), ('INT8_NATIVO_AMD', calib_dir)]:
            temp = root / label
            dest = temp / 'int8_nndct'
            dest.mkdir(parents=True)
            for filename in ('quant_info.json', 'bias_corr.pth', 'adapted_parameters.pth'):
                src_file = quant_path / filename
                if src_file.exists():
                    shutil.copy2(src_file, dest / filename)
            outputs[label] = get_outputs('INT8_NNDCT', name, architecture, checkpoint,
                                         products, height, width, target, xs, temp)
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
    summaries, rows = [], []
    for label, values in outputs.items():
        summary, detail = calc_metrics(label, values, ys, outputs['FP32'], eval_ids, kernel)
        summaries.append(summary)
        rows.extend(detail)
        print(f"{label:>16} F1={summary['F1_global_abertura']:.6f} "
              f"AUPRC={summary['AUPRC_media_logits_brutos']:.6f} "
              f"max={summary['logits_max']:.6f} "
              f"pos={100 * summary['frac_logit_gt_zero']:.4f}%", flush=True)
    pd.DataFrame(summaries).to_csv(dst / 'comparacao_resumo.csv', index=False)
    pd.DataFrame(rows).to_csv(dst / 'comparacao_imagens.csv', index=False)
    (dst / 'manifest_comparacao.json').write_text(json.dumps({
        'original_dir': str(source), 'native_dir': str(calib_dir),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
        'calibration_images': df_calib['id'].tolist(), 'evaluation_images': eval_ids,
        'train_csv': str(args.train_csv), 'eval_csv': str(args.eval_csv), 'target': target
    }, indent=2, ensure_ascii=False))
    print('\nRESULTADO:', dst / 'comparacao_resumo.csv')
    print('Esta comparacao ainda e em simulacao NNDCT: exige compilacao e execucao na DPU para validacao final.')


if __name__ == '__main__':
    main()
