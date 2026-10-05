"""Reconstrucao PTQ da convolucao final, usando somente o professor FP32 na calibracao."""
import argparse
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

import torch

from common import (build_model, build_calibration_loader, DEFAULT_PRODUCTS, DataNormalizer,
                    validate_calibration, file_sha256)
from Testes.Teste_Unet import binary_opening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', default='mobilenet_v3_focaldice_dpu_512')
    parser.add_argument('--model', default='mobilenet_v3_focaldice_reconstructed_512')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--quant-dir', default='build/vitis_ai/quantize')
    args = parser.parse_args()
    if Path(args.model).name != args.model or args.model in ('.', '..'):
        parser.error('--model deve ser um nome simples.')
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    source = Path(args.quant_dir) / args.source_model
    out = Path(args.quant_dir) / args.model
    manifest = json.loads((source/'calibration_manifest.json').read_text())
    validate_calibration(source, args.source_model, args.checkpoint, DEFAULT_PRODUCTS,
                         512, 512, manifest['target'], False, manifest['architecture'])
    assert file_sha256(args.csv) == manifest['csv_sha256']
    loader = build_calibration_loader(args.csv, args.data_root, list(DEFAULT_PRODUCTS),
                                     len(manifest['images']), num_workers=0, seed=manifest['seed'])
    assert loader.dataset.dataframe.id.tolist() == manifest['images']
    out.mkdir(parents=True, exist_ok=False)
    for filename in ('quant_info.json', 'bias_corr.pth'):
        shutil.copyfile(source/filename, out/filename)
    model, _ = build_model(args.model, args.checkpoint, 4, manifest['architecture'])
    reference, _ = build_model(args.model, args.checkpoint, 4, manifest['architecture'])
    from pytorch_nndct.apis import torch_quantizer
    from nndct_shared.utils import NndctOption
    NndctOption.nndct_diffs_mode.value = 'maxmin'
    quant = torch_quantizer('test', model, (torch.zeros(1,4,512,512),),
                           output_dir=str(out), device=torch.device('cpu'), target=manifest['target'])
    qm = quant.quant_model.eval()
    q = quant.processor.quantizer
    last = [m for m in qm.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    assert last.out_channels == 1 and last.kernel_size == (1,1) and last.bias is not None
    prior = torch.cat([last.weight.detach().flatten(), last.bias.detach()]).double()
    size = len(prior)
    gram = torch.zeros(size,size,dtype=torch.double)
    rhs = torch.zeros(size,dtype=torch.double)
    captured = {}
    hook = last.register_forward_pre_hook(lambda _, x: captured.update(input=x[0].detach()))
    norm = DataNormalizer(list(DEFAULT_PRODUCTS))
    kernel = torch.tensor([[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]])
    indices = torch.linspace(0,len(loader.dataset)-1,24).long().tolist()
    bank = []
    with torch.no_grad():
        for i in range(len(loader.dataset)):
            x = norm(loader.dataset[i]['input'].unsqueeze(0))
            y = reference(x)
            z = qm(x)
            features = captured['input'][0].flatten(1).T
            target = y.flatten()
            uniform = torch.linspace(0,len(target)-1,2048).long()
            boundary = torch.where(target.abs() < 3)[0]
            if len(boundary):
                boundary = boundary[torch.linspace(0,len(boundary)-1,min(2048,len(boundary))).long()]
            selected = torch.unique(torch.cat([uniform,boundary]))
            matrix = torch.cat([features[selected],torch.ones(len(selected),1)],1).double()
            prob = torch.sigmoid(target[selected]).double()
            weights = (prob*(1-prob)).square() + 1e-6
            gram += matrix.T @ (matrix*weights[:,None])
            rhs += matrix.T @ (target[selected].double()*weights)
            if i in indices:
                bank.append((x,torch.sigmoid(y),binary_opening(y>0,kernel)))
            if (i+1)%50 == 0:
                print(f'Reconstrucao: {i+1}/{len(loader.dataset)} imagens de calibracao', flush=True)
    hook.remove()
    original = deepcopy(q.quant_config)

    def score():
        tp=fp=fn=0
        mse=0.
        with torch.no_grad():
            for x,prob,mask in bank:
                logits=qm(x)
                pred=binary_opening(logits>0,kernel)
                tp+=int((pred & mask).sum()); fp+=int((pred & ~mask).sum()); fn+=int((~pred & mask).sum())
                mse+=float((torch.sigmoid(logits)-prob).square().mean())
        dice=2*tp/(2*tp+fp+fn+1e-6)
        return dict(loss=1-dice+.1*mse/len(bank),teacher_f1=dice,probability_mse=mse/len(bank))

    baseline = score()
    best = None
    trials = []
    # ponytail: somente a cabeca linear binaria e 24 imagens para escolher entre
    # as reconstrucoes; ampliar a adaptacao se essa camada nao for suficiente.
    for ridge in (1e-5,1e-3,.1):
        penalty = gram.trace()/size*ridge
        coefficients = torch.linalg.solve(gram + penalty*torch.eye(size,dtype=torch.double),rhs + penalty*prior).float()
        assert torch.isfinite(coefficients).all()
        for position in (2,3,4):
            q.quant_config.clear(); q.quant_config.update(deepcopy(original))
            for key,value in zip(last.params_name,(coefficients[:-1],coefficients[-1:])):
                scale = min(12,math.floor(math.log2(127/max(float(value.abs().max()),1e-8))))
                q.set_quant_config(key,[8,scale],'param',0)
            q.set_quant_config(last.node.name,[8,position],'output',0)
            for _ in range(4):
                before=deepcopy(q.quant_config); q.organize_quant_pos()
                if before==q.quant_config: break
            if before!=q.quant_config: continue
            with torch.no_grad():
                last.weight.copy_(coefficients[:-1].reshape_as(last.weight))
                last.bias.copy_(coefficients[-1:])
            last.param_saved=True; last.param_quantized=False
            result=score()
            trials.append(dict(ridge=ridge,output_position=position,**result))
            print('Reconstrucao candidata:',trials[-1],flush=True)
            if best is None or result['loss']<best['result']['loss']:
                best=dict(result=result,config=deepcopy(q.quant_config),coefficients=coefficients.clone())
    assert best is not None and best['result']['loss'] < baseline['loss']
    q.quant_config.clear(); q.quant_config.update(best['config'])
    parameters={last.node.name:dict(weight=best['coefficients'][:-1].reshape_as(last.weight),bias=best['coefficients'][-1:])}
    torch.save(parameters,out/'adapted_parameters.pth')
    q.export_quant_config()
    manifest.update(model=args.model,source_model=args.source_model,
                    reconstruction='FP32_teacher_boundary_weighted_ridge_final_conv',
                    artifacts={f:file_sha256(out/f) for f in ('quant_info.json','bias_corr.pth','adapted_parameters.pth')})
    (out/'calibration_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'reconstruction.json').write_text(json.dumps(dict(baseline=baseline,final=best['result'],
        calibration_check_indices=indices,weight='[sigmoid(fp32)*(1-sigmoid(fp32))]^2 + 1e-6',
        pixel_sampling='2048 uniformes + ate 2048 com abs(logit_fp32)<3 por imagem',trials=trials),indent=2))
    print('RECONSTRUCAO FINAL:',baseline,'->',best['result'],flush=True)


if __name__=='__main__':
    main()
