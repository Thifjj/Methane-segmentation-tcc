"""Teste do executável real com TIFFs/ONNX e referência independente PyTorch/sklearn.

Uso: python benchmark_arm/tests/test_integracao.py --binary /caminho/benchmark_arm
"""
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import rasterio
import torch
from onnx import TensorProto, helper, numpy_helper
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Testes.Teste_Unet import binary_opening


def rows(path):
    with path.open() as f:
        reader = csv.DictReader(f)
        result = list(reader)
    assert all(None not in r and None not in r.values() for r in result), path
    return result


def graph(path, size, dynamic):
    batch = 'batch' if dynamic else 1
    inputs = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch, 4, size, size])
    outputs = helper.make_tensor_value_info('logits', TensorProto.FLOAT, [batch, 1, size, size])
    # Seleciona a banda de índice 1; detecta ordem RGB/legacy incorreta.
    nodes = [helper.make_node('Gather', ['input', 'index'], ['band'], axis=1),
             helper.make_node('Sub', ['band', 'threshold'], ['logits'])]
    g = helper.make_graph(nodes, 'fixture', [inputs], [outputs],
                          [numpy_helper.from_array(np.array([1], np.int64), 'index'),
                           numpy_helper.from_array(np.array(0.5, np.float32), 'threshold')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 16)], ir_version=8)
    helper.set_model_props(m, {'ordem_canais': 'mag1c,460,550,640', 'modelo': 'fixture'})
    onnx.save(m, path)


def summary(expected, official):
    groups = {g: np.zeros(4, dtype=np.int64) for g in ['forte', 'fraca', 'sem_pluma']}
    counts = np.zeros(4, dtype=np.int64)
    fp_tiles = tn_tiles = 0
    aps, scores, targets = [], [], []
    for z, truth, plume, difficulty in expected:
        mask = torch.tensor(z).reshape(1, 1, 512, 512) > 0
        if official:
            mask = binary_opening(mask, torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]))
        mask = mask.numpy().reshape(512, 512).astype(bool)
        positive = bool(truth.any())
        bg = not positive if official else not plume
        strong = difficulty == 'easy' if official else plume == 2
        group = 'sem_pluma' if bg else 'forte' if strong else 'fraca'
        c = np.array([(mask & truth).sum(), (mask & ~truth).sum(), (~mask & truth).sum(), (~mask & ~truth).sum()])
        groups[group] += c
        counts += c
        if bg:
            fp_tiles += mask.sum() > 640
            tn_tiles += mask.sum() <= 640
        score = torch.sigmoid(torch.tensor(z)).numpy().reshape(-1)
        target = truth.reshape(-1)
        scores.append(score)
        targets.append(target)
        if positive:
            aps.append(average_precision_score(target, score))
    def f1(c):
        return 2 * c[0] / (2 * c[0] + c[1] + c[2] + (1e-6 if official else 0)) if c[:3].any() else 0
    if official:
        area = np.mean(aps) if aps else np.nan
    else:
        target = np.concatenate(targets)
        if target.any():
            precision, recall, _ = precision_recall_curve(target, np.concatenate(scores))
            area = auc(recall, precision)
        else:
            area = 0.5
    return dict(zip(['tp', 'fp', 'fn', 'tn'], counts)), dict(
        f1_global=f1(counts), f1_strong_plume=f1(groups['forte']), f1_weak_plume=f1(groups['fraca']), auprc=area,
        fpr_tile=fp_tiles / (fp_tiles + tn_tiles) if fp_tiles + tn_tiles else 0,
        fp_tiles=fp_tiles, tn_tiles=tn_tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix='arm_integracao_') as tmp:
        root = Path(tmp)
        expected = []
        dataset = root / 'dataset'
        dataset.mkdir()
        with (dataset / 'test.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id','folder','has_plume','qplume','difficulty',
                'window_col_off','window_row_off','window_width','window_height'])
            writer.writeheader()
            # 35 imagens exercitam o merge com fan-in 32; inclui labels que discordam de has_plume.
            for i in range(35):
                folder = dataset / f'img_{i}'
                folder.mkdir()
                values = np.zeros((514, 516), np.float32)
                values[1:513, 2:514] = 12  # logit -0.3
                values[40:80, 60:100] = 60  # falso/verdadeiro blob, >640 pixels
                values[130, 130] = 60      # removido pela abertura
                values[200:210, 200:210] = 30  # empate logit zero
                truth = np.zeros(values.shape, np.uint8)
                if i % 3:
                    truth[42:78, 62:98] = 1
                if i == 0:  # label positivo com has_plume=false
                    truth[300:302,300:302] = 1
                plume = 0 if i % 3 == 0 else 2 if i % 3 == 1 else 1
                diff = 'easy' if i % 2 else 'hard'
                for name, array in [('mag1c', np.zeros_like(values)), ('TOA_AVIRIS_460nm', values),
                    ('TOA_AVIRIS_550nm', np.zeros_like(values)), ('TOA_AVIRIS_640nm', np.zeros_like(values)),
                    ('labelbinary', truth)]:
                    with rasterio.open(folder / (name+'.tif'), 'w', driver='GTiff', height=514, width=516,
                        count=1, dtype=array.dtype) as out:
                        out.write(array, 1)
                writer.writerow(dict(id=f'id_{i}', folder=str(folder)+'/', has_plume=bool(plume),
                    qplume=1500 if plume == 2 else 500 if plume else '', difficulty=diff,
                    window_col_off=2, window_row_off=1, window_width=512, window_height=512))
                expected.append((np.clip(values[1:513,2:514]/60,0,2)-0.5,
                                 truth[1:513,2:514].astype(bool), plume, diff))
        for size,batch,dynamic in [(128,1,False),(128,16,True),(128,3,True),(512,1,False)]:
            model = root / f'model_{size}_{batch}.onnx'
            graph(model,size,dynamic)
            output = root / f'output_{size}_{batch}'
            command = [str(binary),'--model',str(model),'--dataset',str(dataset),'--output',str(output),
                       '--warmup','1','--threads','2','--patch-batch-size',str(batch),'--no-power']
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            config = json.loads((output/'config.json').read_text())
            assert config['status']=='concluido'
            assert config['onnx_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
            assert config['csv_sha256']==hashlib.sha256((dataset/'test.csv').read_bytes()).hexdigest()
            assert config['ordem_canais']=='mag1c,460,550,640'
            perf=rows(output/'benchmark_geral.csv')
            assert len(perf)==2
            assert int(perf[0]['entradas_preparadas'])==4
            assert int(perf[0]['execucoes_modelo']) == 35 * int(np.ceil((512//size)**2/batch))
            for path in output.rglob('*.csv'):
                assert rows(path), path
            for official in [False,True]:
                folder=output/'validacao_oficial' if official else output
                actual=rows(folder/'metricas_globais.csv')[0]
                c,m=summary(expected,official)
                for key,value in {**c,**m}.items():
                    np.testing.assert_allclose(float(actual[key]),value,rtol=1e-9,atol=1e-10,err_msg=key)
                images=rows(folder/'metricas_por_imagem.csv')
                assert images[0]['id']=='id_0'
            assert all(p['status']=='desativada' for p in rows(output/'benchmark_power_rails.csv'))
            assert not (output/'.scores_pr').exists()
            # Saídas anteriores não podem ser sobrescritas.
            assert subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        bad=root/'bad.onnx';graph(bad,128,False)
        assert subprocess.run([str(binary),'--model',str(bad),'--dataset',str(dataset),'--patch-batch-size','16'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        # Sem positivos: AUPRC bruta 0.5, oficial NaN; potência ausente sinalizada.
        empty=dataset/'negative.csv'
        with empty.open('w',newline='') as f:
            orig=rows(dataset/'test.csv')[3];orig['id']='negative';orig['has_plume']='False';orig['qplume']=''
            writer=csv.DictWriter(f,fieldnames=orig.keys());writer.writeheader();writer.writerow(orig)
        subprocess.run([str(binary),'--model',str(bad),'--dataset',str(dataset),'--csv','negative.csv',
            '--output',str(root/'negative'),'--warmup','0','--inferencias','2'],check=True,stdout=subprocess.DEVNULL)
        raw=rows(root/'negative'/'metricas_globais.csv')[0]
        official=rows(root/'negative'/'validacao_oficial'/'metricas_globais.csv')[0]
        assert float(raw['auprc'])==0.5 and np.isnan(float(official['auprc']))
        print('Integração OK: patches/batches, recortes, canais, abertura, PR exata, grupos, CSVs, hashes e proteção das saídas.')


if __name__=='__main__':
    main()
