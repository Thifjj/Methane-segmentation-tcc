"""
Conversor de modelos PyTorch (pasta `Modelos/`) para Keras 3 + HGQ2 (compatível com hls4ml).

Como funciona
-------------
1. O modelo PyTorch é rastreado com `torch.fx` (grafo de operações) e as formas de
   cada tensor são obtidas com `ShapeProp` usando uma entrada de exemplo.
2. Cada nó do grafo vira a camada equivalente HGQ2/Keras, em NHWC
   (Conv2d -> QConv2D, BatchNorm2d -> QBatchNormalization, MaxPool2d -> QMaxPool2D,
   torch.cat -> Concatenate, soma -> QAdd, produto -> QMultiply, ...).
3. Os pesos treinados são copiados (com a transposição NCHW -> NHWC).

As camadas são criadas dentro dos escopos de quantização passados (`escopos`), então a
configuração INT8/HGQ definida no script de treino vale para todas elas.

Limitações (aparecem em `relatorio` e são impressas):
- `ConvTranspose2d`: o HGQ2 não tem versão quantizada. Padrão: UpSampling2D(nearest) +
  QConv2D (quantizado, pesos NÃO copiados, precisa de treino). Opção `"keras"`:
  Conv2DTranspose do Keras com pesos exatos, mas sem quantizador HGQ.
- `Upsample`/`interpolate` bilinear com `align_corners=True` vira bilinear "half-pixel"
  (diferença numérica pequena). hls4ml suporta melhor o modo `nearest`.
- Operações de formato (view, reshape, flatten, permute, indexação de tensores) e
  operações com constantes (x + 3) não são suportadas: geram erro com o nome do nó.
- Funções de ativação (ReLU, Hardswish, Hardsigmoid, Sigmoid, GELU) viram
  `keras.layers.Activation`. O hls4ml implementa ReLU/sigmoid/hard_sigmoid; confira as
  demais na versão instalada.
- Convoluções com `groups > 1` (depthwise da MobileNet) usam QConv2D(groups=...). Confira
  o suporte da sua versão do hls4ml antes de sintetizar.
"""

import operator
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import contextlib

import keras
import numpy as np
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from hgq.layers import (
    QAdd,
    QAvgPool2D,
    QBatchNormalization,
    QConv2D,
    QDense,
    QGlobalAvgPool2D,
    QMaxPool2D,
    QMultiply,
)

# torch dim (tensor 4D NCHW) -> eixo Keras (NHWC)
_EIXO_NCHW_PARA_NHWC = {0: 0, 1: -1, 2: 1, 3: 2, -1: 2, -2: 1, -3: -1}

_ATIVACOES_MODULO = {
    nn.ReLU: "relu",
    nn.ReLU6: "relu6",
    nn.Hardswish: "hard_silu",
    nn.Hardsigmoid: "hard_sigmoid",  # Keras 3: relu6(x + 3) / 6, igual ao PyTorch
    nn.Sigmoid: "sigmoid",
    nn.GELU: "gelu",
    nn.Tanh: "tanh",
}
_ATIVACOES_FUNCAO = {
    F.relu: "relu",
    torch.relu: "relu",
    F.relu6: "relu6",
    F.hardswish: "hard_silu",
    F.hardsigmoid: "hard_sigmoid",
    torch.sigmoid: "sigmoid",
    F.sigmoid: "sigmoid",
    F.gelu: "gelu",
    torch.tanh: "tanh",
}
_IDENTIDADES = (nn.Identity, nn.Dropout, nn.Dropout2d)


def _par(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _peso(camada, nome):
    """Variável Keras `nome` da própria camada, pelo caminho exato.

    Não usa só `w.name`: no HGQ2 existe a variável `beta` (peso do custo EBOPs) em toda
    camada, e o QBatchNormalization guarda gamma/beta como `bn_gamma`/`bn_beta`."""
    candidatos = (f"{camada.name}/{nome}", f"{camada.name}/bn_{nome}")
    for w in camada.weights:
        if w.path in candidatos:
            return w
    raise KeyError(f"Peso '{nome}' não encontrado em {camada.name}: "
                   f"{[w.path for w in camada.weights]}")


def _atribuir(camada, nome, valor):
    _peso(camada, nome).assign(np.ascontiguousarray(valor.detach().cpu().numpy()))


def _nome(no):
    return no.name.replace(".", "_")


class _Conversor:
    def __init__(self, modelo_torch, modo_convtranspose):
        self.mt = modelo_torch
        self.modulos = dict(modelo_torch.named_modules())
        self.modo_convtranspose = modo_convtranspose
        self.copias = []      # (camada_keras, func_copia) executadas após o build
        self.relatorio = []

    # ---------------- módulos ----------------
    def modulo(self, no, m, x, meta_in, meta_out):
        nome = _nome(no)
        if isinstance(m, _IDENTIDADES):
            return x
        if type(m) in _ATIVACOES_MODULO:
            return keras.layers.Activation(_ATIVACOES_MODULO[type(m)], name=nome)(x)

        if isinstance(m, nn.Conv2d):
            return self._conv(nome, m, x)

        if isinstance(m, nn.BatchNorm2d):
            c = QBatchNormalization(axis=-1, epsilon=m.eps, momentum=1 - (m.momentum or 0.1), name=nome)
            self.copias.append(lambda c=c, m=m: (
                _atribuir(c, "gamma", m.weight), _atribuir(c, "beta", m.bias),
                _atribuir(c, "moving_mean", m.running_mean),
                _atribuir(c, "moving_variance", m.running_var)))
            return c(x)

        if isinstance(m, nn.MaxPool2d):
            return self._pool(QMaxPool2D, nome, m, x)
        if isinstance(m, nn.AvgPool2d):
            return self._pool(QAvgPool2D, nome, m, x)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            if _par(m.output_size) != (1, 1):
                raise NotImplementedError(f"{no.target}: AdaptiveAvgPool2d só com saída 1x1")
            return QGlobalAvgPool2D(keepdims=True, name=nome)(x)

        if isinstance(m, nn.Upsample):
            return self._upsample(nome, m.mode, m.align_corners, x, meta_in, meta_out)

        if isinstance(m, nn.ConvTranspose2d):
            return self._conv_transposta(nome, m, x)

        if isinstance(m, nn.Linear):
            c = QDense(m.out_features, use_bias=m.bias is not None, name=nome)
            self.copias.append(lambda c=c, m=m: (
                _atribuir(c, "kernel", m.weight.T),
                m.bias is not None and _atribuir(c, "bias", m.bias)))
            return c(x)

        raise NotImplementedError(f"Módulo sem equivalente HGQ: {no.target} ({type(m).__name__})")

    def _conv(self, nome, m, x):
        k, s, d = _par(m.kernel_size), _par(m.stride), _par(m.dilation)
        p = _par(m.padding) if not isinstance(m.padding, str) else m.padding
        if p == "same" or (s == (1, 1) and p == tuple((ki - 1) * di // 2 for ki, di in zip(k, d))
                           and all(ki % 2 == 1 for ki in k)):
            padding = "same"
        else:
            if p == "valid":
                p = (0, 0)
            if any(p):  # padding simétrico do PyTorch (Keras 'same' com stride 2 é assimétrico)
                x = keras.layers.ZeroPadding2D(p, name=f"{nome}_pad")(x)
            padding = "valid"
        c = QConv2D(m.out_channels, k, strides=s, padding=padding, dilation_rate=d,
                    groups=m.groups, use_bias=m.bias is not None, name=nome)
        if m.groups > 1:
            self.relatorio.append(f"{nome}: convolução com groups={m.groups} (verificar suporte no hls4ml)")
        self.copias.append(lambda c=c, m=m: (
            _atribuir(c, "kernel", m.weight.permute(2, 3, 1, 0)),
            m.bias is not None and _atribuir(c, "bias", m.bias)))
        return c(x)

    def _pool(self, cls, nome, m, x):
        k = _par(m.kernel_size)
        s = _par(m.stride if m.stride is not None else m.kernel_size)
        p = _par(m.padding)
        if any(p):
            x = keras.layers.ZeroPadding2D(p, name=f"{nome}_pad")(x)
            self.relatorio.append(f"{nome}: pooling com padding (PyTorch usa -inf; aqui zero)")
        return cls(pool_size=k, strides=s, padding="valid", name=nome)(x)

    def _upsample(self, nome, modo, align_corners, x, meta_in, meta_out):
        hi, wi = meta_in.shape[2:]
        ho, wo = meta_out.shape[2:]
        if ho % hi or wo % wi:
            raise NotImplementedError(f"{nome}: upsample com fator não inteiro ({hi}x{wi} -> {ho}x{wo})")
        interp = "nearest" if modo == "nearest" else "bilinear"
        if modo not in ("nearest", "bilinear"):
            raise NotImplementedError(f"{nome}: modo de upsample '{modo}'")
        if interp == "bilinear" and align_corners:
            self.relatorio.append(f"{nome}: bilinear align_corners=True -> half-pixel (aproximado)")
        return keras.layers.UpSampling2D((ho // hi, wo // wi), interpolation=interp, name=nome)(x)

    def _conv_transposta(self, nome, m, x):
        k, s = _par(m.kernel_size), _par(m.stride)
        if _par(m.padding) != (0, 0) or _par(m.output_padding) != (0, 0) or m.groups != 1:
            raise NotImplementedError(f"{nome}: ConvTranspose2d só com padding=0 e groups=1")
        if self.modo_convtranspose == "keras":
            c = keras.layers.Conv2DTranspose(m.out_channels, k, strides=s, padding="valid",
                                             use_bias=m.bias is not None, name=nome)
            self.copias.append(lambda c=c, m=m: (
                _atribuir(c, "kernel", m.weight.permute(2, 3, 1, 0)),
                m.bias is not None and _atribuir(c, "bias", m.bias)))
            self.relatorio.append(f"{nome}: Conv2DTranspose Keras (pesos copiados, SEM quantização HGQ)")
            return c(x)
        x = keras.layers.UpSampling2D(s, interpolation="nearest", name=f"{nome}_up")(x)
        c = QConv2D(m.out_channels, k, padding="same", use_bias=m.bias is not None, name=nome)
        self.relatorio.append(f"{nome}: ConvTranspose2d -> UpSampling2D + QConv2D (pesos NÃO copiados)")
        return c(x)

    # ---------------- funções / métodos ----------------
    def funcao(self, no, alvo, args, kwargs, meta_in, meta_out):
        nome = _nome(no)
        if alvo in _ATIVACOES_FUNCAO:
            return keras.layers.Activation(_ATIVACOES_FUNCAO[alvo], name=nome)(args[0])
        if alvo in (torch.cat, torch.concat):
            tensores = args[0]
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            return keras.layers.Concatenate(axis=_EIXO_NCHW_PARA_NHWC[dim], name=nome)(list(tensores))
        if alvo in (operator.add, torch.add, operator.iadd):
            self._exigir_tensores(nome, args)
            return QAdd(name=nome)([args[0], args[1]])
        if alvo in (operator.mul, torch.mul, operator.imul):
            self._exigir_tensores(nome, args)
            return QMultiply(name=nome)([args[0], args[1]])
        if alvo is F.interpolate:
            modo = kwargs.get("mode", args[3] if len(args) > 3 else "nearest")
            return self._upsample(nome, modo, kwargs.get("align_corners"), args[0], meta_in, meta_out)
        if alvo is F.adaptive_avg_pool2d:
            return QGlobalAvgPool2D(keepdims=True, name=nome)(args[0])
        if alvo in (F.dropout, F.dropout2d):
            return args[0]
        raise NotImplementedError(f"Função sem equivalente HGQ: {no.name} ({getattr(alvo, '__name__', alvo)})")

    def metodo(self, no, alvo, args, meta_in, meta_out):
        nome = _nome(no)
        if alvo in ("relu", "sigmoid", "tanh"):
            return keras.layers.Activation(alvo, name=nome)(args[0])
        if alvo in ("contiguous", "clone"):
            return args[0]
        if alvo in ("add", "add_"):
            return QAdd(name=nome)([args[0], args[1]])
        if alvo in ("mul", "mul_"):
            return QMultiply(name=nome)([args[0], args[1]])
        if alvo in ("size", "dim"):
            return None  # usado só para calcular formas (resolvidas via ShapeProp)
        raise NotImplementedError(f"Método de tensor sem equivalente HGQ: {no.name} (.{alvo})")

    @staticmethod
    def _exigir_tensores(nome, args):
        if not all(isinstance(a, keras.KerasTensor) for a in args[:2]):
            raise NotImplementedError(f"{nome}: operação com constante não suportada (use uma camada)")


def converter_modelo_para_hgq(
    modelo_torch: nn.Module,
    in_channels: int = 4,
    escopos=(),
    tamanho_exemplo: int = 128,
    copiar_pesos: bool = True,
    modo_convtranspose: str = "upsample_conv",
    entrada_fixa: tuple | None = None,
    verificar: bool = True,
):
    """Converte um `nn.Module` em `keras.Model` com camadas HGQ2.

    Args:
        modelo_torch: instância do modelo (já com pesos carregados, se quiser copiá-los).
        in_channels: canais de entrada.
        escopos: context managers do HGQ2 (QuantizerConfigScope/LayerConfigScope) aplicados
            na criação das camadas (ex.: configuração INT8).
        tamanho_exemplo: H=W da entrada usada para rastrear formas (múltiplo de 32).
        copiar_pesos: copia pesos/estatísticas do PyTorch para o modelo Keras.
        modo_convtranspose: "upsample_conv" (quantizado, sem cópia) ou "keras" (exato, não quantizado).
        entrada_fixa: (H, W) da entrada Keras. O HGQ2 exige H e W fixos (usa o tamanho para
            estimar o custo de hardware). None = (tamanho_exemplo, tamanho_exemplo).
        verificar: compara a saída Keras com a PyTorch (antes da quantização ter efeito
            relevante, a diferença indica erros de conversão).

    Returns:
        (modelo_keras, relatorio) — relatorio é uma lista de avisos sobre aproximações.
    """
    modelo_torch = modelo_torch.eval().cpu()
    grafo = torch.fx.symbolic_trace(modelo_torch)
    exemplo = torch.rand(1, in_channels, tamanho_exemplo, tamanho_exemplo) * 2
    ShapeProp(grafo).propagate(exemplo)

    conv = _Conversor(modelo_torch, modo_convtranspose)
    valores = {}
    meta = {}

    def resolver(a):
        return torch.fx.node.map_arg(a, lambda n: valores[n.name])

    h, w = entrada_fixa if entrada_fixa else (tamanho_exemplo, tamanho_exemplo)
    with contextlib.ExitStack() as pilha:
        for e in escopos:
            pilha.enter_context(e)

        entrada_keras = None
        saida_keras = None
        for no in grafo.graph.nodes:
            meta[no.name] = no.meta.get("tensor_meta")
            primeiro = next((a for a in no.all_input_nodes), None)
            meta_in = meta.get(primeiro.name) if primeiro is not None else None
            meta_out = meta[no.name]

            if no.op == "placeholder":
                entrada_keras = keras.Input(shape=(h, w, in_channels), name=_nome(no))
                valores[no.name] = entrada_keras
            elif no.op == "call_module":
                x = resolver(no.args)[0]
                valores[no.name] = conv.modulo(no, conv.modulos[no.target], x, meta_in, meta_out)
            elif no.op == "call_function":
                if no.target is operator.getitem or no.target is getattr:
                    valores[no.name] = None  # índices de formas; não geram camadas
                    continue
                valores[no.name] = conv.funcao(no, no.target, resolver(no.args), resolver(no.kwargs),
                                               meta_in, meta_out)
            elif no.op == "call_method":
                valores[no.name] = conv.metodo(no, no.target, resolver(no.args), meta_in, meta_out)
            elif no.op == "get_attr":
                raise NotImplementedError(f"Parâmetro solto no forward não suportado: {no.target}")
            elif no.op == "output":
                saida_keras = resolver(no.args)[0]
                if isinstance(saida_keras, (tuple, list)):
                    saida_keras = saida_keras[0]

        nome = f"{type(modelo_torch).__name__}_HGQ"
        modelo_keras = keras.Model(entrada_keras, saida_keras, name=nome)

    if copiar_pesos:
        for copia in conv.copias:
            copia()

    if verificar and copiar_pesos:
        with torch.no_grad():
            ref = modelo_torch(exemplo).numpy()
            x = exemplo.permute(0, 2, 3, 1).contiguous()
            out = keras.ops.convert_to_numpy(modelo_keras(x, training=False))
        out = np.transpose(out, (0, 3, 1, 2))
        erro = float(np.abs(out - ref).max())
        concord = float(((out > 0) == (ref > 0)).mean())
        conv.relatorio.append(f"Verificação: erro máx. |keras - torch| = {erro:.4g}; "
                              f"concordância da máscara (logit>0) = {concord:.2%} "
                              f"(já inclui o efeito da quantização)")

    for linha in conv.relatorio:
        print(f"[ConversorHGQ] {linha}")
    return modelo_keras, conv.relatorio


def _calibrar_um_passe(modelo, lotes_nhwc, margem=0, piso_ativ=2, piso_peso=0):
    """Ajusta, a partir de dados reais, a posição do ponto (`i`) de todos os quantizadores
    de ponto fixo (pesos, bias e ativações) de um modelo HGQ2 com largura de bits fixa.

    Sem isso, `i` começa em um valor genérico (ex.: ±4 nas ativações, ±1 nos pesos) e só
    se move ~lr por passo, então a rede passa as primeiras épocas saturando/zerando
    valores. Aqui `i` vira o mínimo que cobre o maior |valor| visto por canal.

    lotes_nhwc: iterável de tensores/arrays NHWC (alguns lotes bastam).
    margem: bits inteiros extras de folga (o treino faz as ativações crescerem).
    piso_ativ/piso_peso: menor `i` permitido (ativações ±2^piso_ativ; pesos ±2^piso_peso)."""
    import numpy as np
    from hgq.quantizer.internal.fixed_point_quantizer import FixedPointQuantizerBase

    quantizadores = [c for c in modelo._flatten_layers(include_self=False, recursive=True)
                     if isinstance(c, FixedPointQuantizerBase)]
    maximos = {}
    originais = {}
    for q in quantizadores:
        originais[q] = q.call

        def _gravar(inputs, training=None, _q=q, _orig=q.call):
            i_min = keras.ops.convert_to_numpy(_q.get_minimal_i(inputs))
            anterior = maximos.get(_q)
            maximos[_q] = i_min if anterior is None else np.maximum(anterior, i_min)
            return _orig(inputs, training=training)

        q.call = _gravar
    try:
        with torch.no_grad():
            for x in lotes_nhwc:
                modelo(x, training=False)
    finally:
        for q in quantizadores:
            q.call = originais[q]
    n = 0
    for q, i_min in maximos.items():
        peso = q.name.endswith(("_kq", "_bq"))
        # piso: canal sem sinal na calibração (i=-23) ou com faixa minúscula ficaria preso,
        # porque quantizador saturado tem gradiente zero e o valor não consegue crescer.
        i_novo = np.clip(i_min + margem, piso_peso if peso else piso_ativ, 23).astype("float32")
        q._i.assign(np.broadcast_to(i_novo, q._i.shape))
        n += 1
    return n


def calibrar_escalas_hgq(modelo, lotes_nhwc, margem=1, max_passes=20, piso_ativ=2, piso_peso=0):
    """Repete a calibração até as escalas pararem de mudar. Um passe só não basta: cada
    quantizador vê a saída dos anteriores, que ainda estavam mal calibrados (saturados).
    Em teste (SegFormer, INT8, sem treino) o erro médio caiu de 25,7 para 1,6 em ~8 passes.
    `lotes_nhwc` deve ser uma lista (é percorrida várias vezes). Retorna o nº de passes."""
    from hgq.quantizer.internal.fixed_point_quantizer import FixedPointQuantizerBase
    qs = [c for c in modelo._flatten_layers(include_self=False, recursive=True)
          if isinstance(c, FixedPointQuantizerBase)]
    lotes_nhwc = list(lotes_nhwc)
    for passe in range(1, max_passes + 1):
        antes = [keras.ops.convert_to_numpy(q._i).copy() for q in qs]
        _calibrar_um_passe(modelo, lotes_nhwc, margem, piso_ativ, piso_peso)
        mudou = any(not np.array_equal(a, keras.ops.convert_to_numpy(q._i)) for a, q in zip(antes, qs))
        if not mudou:
            break
    return passe
