"""
Treinamento quantizado (QAT) dos modelos de `Modelos/` com HGQ2, compatível com hls4ml.

Por que o modelo é convertido
-----------------------------
HGQ2 não quantiza um modelo PyTorch existente: ele é uma biblioteca de camadas
Keras 3 (QConv2D, QMaxPool2D, ...). A quantização mora DENTRO da estrutura do
modelo. Por isso a classe PyTorch escolhida é convertida por
`Utils/ConversorHGQ.py`, que troca cada camada pela equivalente HGQ2 (e copia os
pesos). O hls4ml lê esse modelo Keras diretamente e herda as precisões (ap_fixed).

O resto do fluxo é o mesmo de `Treinamento_Unet.py`: mesmo CSV/split
(GroupShuffleSplit 15%, seed 42), patches 128x128 stride 64, 4 imagens por
batch, augmentations kornia, normalização (mag1c/1750, RGB/60, clip [0,2]),
validação em imagens 512 inteiras, F1 com abertura morfológica (cruz 3x3),
Adam lr 1e-4, até 200 épocas, paciência 10. Loss "BCE" (ponderada por
weight_mag1c) ou "FD" (FocalDice alpha 0.25, gamma 2.0).

Modo INT8 (padrão): pesos e ativações com 8 bits fixos (1 sinal + 7), escala
potência de 2 treinável por canal -> ap_fixed<8, I> no hls4ml. Com
`MODO_QUANT = "hgq"` os bits passam a ser treináveis (otimização de EBOPs).

Dependências (NÃO estão no .venv atual; sugiro um venv separado, pois keras 3
pode conflitar com outras libs):
    pip install "keras>=3.15" HGQ2 hls4ml
O backend do Keras é forçado para torch, então o mesmo torch/kornia/rasterio
do projeto continua sendo usado (inclusive GPU).

Uso (da raiz do repositório):
    python -m Treinos.Treinamento_unet_quantizado
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # precisa vir antes de importar keras
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # menos fragmentação da GPU

import keras
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia.augmentation as K
from sklearn.model_selection import GroupShuffleSplit

from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import MinMax

from Utils.ConversorHGQ import calibrar_escalas_hgq, converter_modelo_para_hgq
from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer
from Treinos.Treinamento_Unet import calcular_f1_score

assert keras.backend.backend() == "torch", "Defina KERAS_BACKEND=torch antes de importar keras."

CAMINHO_CSV = "/media/jacques/games/Datasets/STARCOP_train_remaining_all/train.csv"
DIRETORIO_DADOS = "/media/jacques/games/Datasets/STARCOP_train_remaining_all"
PRODUTOS_ENTRADA = ["mag1c", "TOA_AVIRIS_460nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_640nm"]

MODO_QUANT = "hgq"  # "int8" (bits fixos) ou "hgq" (bits treináveis + EBOPs)
# Memória: o QAT guarda mais ativações que o FP32 (e sem AMP). Em vez de 4 janelas (196 patches)
# por passo, usa 1 janela (49 patches) e acumula 4 passos => mesmo batch efetivo (4 janelas).
JANELAS_POR_PASSO = 2
ACUMULACAO = 4 // JANELAS_POR_PASSO
# Peso do custo de hardware (EBOPs) na loss, só no modo "hgq". EBOPs é contado por imagem de
# 128x128 e vale ~2.5e10 no SegFormerB0 com bits cheios; com 1e-6 a loss chegava a 25000 e
# engolia o sinal do F1. 1e-11 deixa o termo em ~0.25. Sobe linearmente nas primeiras épocas.
BETA_EBOPS = 1e-11
EPOCAS_AQUECIMENTO_BETA = 10


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def _escopo_quantizacao(modo):
    """Contextos HGQ2 que definem os quantizadores padrão das camadas criadas dentro deles."""
    if modo == "int8":
        # b = bits sem o sinal -> 7 + 1 sinal = 8 bits. bc=MinMax(7,7) congela a largura;
        # só a posição do ponto (escala 2^i) é aprendida, por canal de saída.
        pesos = QuantizerConfigScope(
            place=("weight", "bias"), default_q_type="kbi",
            k0=1, b0=7, i0=0, bc=MinMax(7, 7),
            overflow_mode="SAT_SYM", heterogeneous_axis=(-1,),
        )
        ativacoes = QuantizerConfigScope(
            place="datalane", default_q_type="kbi",
            k0=1, b0=7, i0=2, bc=MinMax(7, 7),
            overflow_mode="SAT", heterogeneous_axis=(-1,),  # uma precisão por canal (io_stream)
        )
        camadas = LayerConfigScope(enable_ebops=False)
    elif modo == "hgq":
        pesos = QuantizerConfigScope(place=("weight", "bias"), default_q_type="kbi",
                                     overflow_mode="SAT_SYM", bc=MinMax(0, 8))
        ativacoes = QuantizerConfigScope(place="datalane", default_q_type="kif",
                                         overflow_mode="SAT", heterogeneous_axis=(-1,))
        camadas = LayerConfigScope(enable_ebops=True, beta0=BETA_EBOPS)
    else:
        raise ValueError(f"MODO_QUANT inválido: {modo}")
    return pesos, ativacoes, camadas


def construir_modelo_hgq(classe_modelo, in_channels=4, modo=MODO_QUANT,
                         modo_convtranspose="upsample_conv", tamanho=128):
    """Instancia um modelo de `Modelos/` (PyTorch, pesos iniciais aleatórios, sem .pth) e troca
    suas camadas pelas do HGQ2 (Utils/ConversorHGQ.py), dentro dos escopos de quantização de
    `modo`. O treino é feito do zero, já quantizado.

    O HGQ2 exige entrada de tamanho fixo (`tamanho` x `tamanho`). Modelos do mesmo
    `classe_modelo`/`modo` com tamanhos diferentes têm os mesmos pesos, na mesma ordem,
    então dá para passar pesos entre eles com get_weights/set_weights."""
    modelo_torch = classe_modelo(in_channels=in_channels, out_channels=1)
    modelo, _ = converter_modelo_para_hgq(
        modelo_torch, in_channels=in_channels, escopos=_escopo_quantizacao(modo),
        copiar_pesos=True, modo_convtranspose=modo_convtranspose,
        verificar=False, tamanho_exemplo=tamanho,
    )
    return modelo


# ---------------------------------------------------------------------------
# Losses (y_true chega empacotado: canal 0 = máscara, canal 1 = weight_mag1c)
# ---------------------------------------------------------------------------
def bce_ponderada(y_pack, logits):
    alvo, peso = y_pack[..., :1], y_pack[..., 1:2]
    bce = keras.ops.maximum(logits, 0) - logits * alvo + keras.ops.log1p(keras.ops.exp(-keras.ops.abs(logits)))
    return keras.ops.mean(bce * peso)


def focal_dice(y_pack, logits, alpha=0.25, gamma=2.0, smooth=1e-6):
    alvo, peso = y_pack[..., :1], y_pack[..., 1:2]
    probs = keras.ops.sigmoid(logits)
    bce = keras.ops.maximum(logits, 0) - logits * alvo + keras.ops.log1p(keras.ops.exp(-keras.ops.abs(logits)))
    p_t = probs * alvo + (1 - probs) * (1 - alvo)
    alpha_f = alpha * alvo + (1 - alpha) * (1 - alvo)
    focal = keras.ops.mean(alpha_f * (1 - p_t) ** gamma * bce * peso)
    inter = keras.ops.sum(probs * alvo)
    dice = (2 * inter + smooth) / (keras.ops.sum(probs) + keras.ops.sum(alvo) + smooth)
    return focal + (1 - dice)


# ---------------------------------------------------------------------------
# Dados: mesma lógica do treino PyTorch, convertida para NHWC no collate
# ---------------------------------------------------------------------------
_normalizador_cpu = DataNormalizer(PRODUTOS_ENTRADA)
_augmentacoes = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=90.0, p=0.5),
    data_keys=["input", "mask", "mask"],
    same_on_batch=False,
)


def collate_treino(amostras):
    inputs = torch.cat([a["input"] for a in amostras])            # (B*P, C, 128, 128)
    alvos = torch.cat([a["output"] for a in amostras])
    pesos = torch.cat([a["weight_loss"] for a in amostras])
    idx = torch.randperm(inputs.shape[0])
    inputs, alvos, pesos = inputs[idx], alvos[idx], pesos[idx]
    with torch.no_grad():
        inputs, alvos, pesos = _augmentacoes(inputs, alvos, pesos)
        inputs = _normalizador_cpu(inputs)
    y_pack = torch.cat([alvos, pesos], dim=1)
    return inputs.permute(0, 2, 3, 1).contiguous(), y_pack.permute(0, 2, 3, 1).contiguous()


def collate_val(amostras):
    inputs = torch.stack([a["input"] for a in amostras])
    alvos = torch.stack([a["output"] for a in amostras])
    with torch.no_grad():
        inputs = _normalizador_cpu(inputs)
    return inputs.permute(0, 2, 3, 1).contiguous(), alvos  # alvo fica NCHW para o F1


class RecalibrarEscalas(keras.callbacks.Callback):
    """INT8 de largura fixa: só a escala (i) de cada canal é aprendida, e com lr 1e-4 ela quase
    não acompanha o crescimento das ativações/pesos (saturam -> gradiente zero -> F1 estagna).
    No início de cada época, recalibra `i` com dados reais (máximo por canal + folga)."""

    def __init__(self, lotes, cada=1):
        super().__init__()
        self.lotes, self.cada = lotes, cada

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.cada == 0:
            passes = calibrar_escalas_hgq(self.model, self.lotes)
            print(f"[RecalibrarEscalas] época {epoch + 1}: escalas INT8 recalibradas ({passes} passes)")


class ValidacaoF1(keras.callbacks.Callback):
    """F1 em imagens 512 inteiras, salvamento no recorde e early stopping (paciência 10).

    O modelo de treino tem entrada 128x128 (fixa no HGQ2); a validação usa um segundo
    modelo 512x512 que recebe os pesos atuais a cada época."""

    def __init__(self, dataloader_val, caminho, melhor_f1, modelo_val, paciencia=10):
        super().__init__()
        self.dl = dataloader_val
        self.modelo_val = modelo_val
        self.caminho = caminho
        self.melhor = melhor_f1
        self.paciencia = paciencia
        self.sem_melhora = 0

    def on_epoch_end(self, epoca, logs=None):
        soma, n = 0.0, 0
        self.modelo_val.set_weights(self.model.get_weights())
        with torch.no_grad():
            for x, y in tqdm(self.dl, desc=f"Época {epoca + 1} [Validação]"):
                logits = self.modelo_val(x, training=False)             # NHWC (tensor torch)
                logits = logits.permute(0, 3, 1, 2)
                s, q = calcular_f1_score(logits, y.to(logits.device))
                soma += s
                n += q
        f1 = soma / n
        print(f"Resumo da Época {epoca + 1} -> Loss: {logs.get('loss', float('nan')):.4f} | F1: {f1:.4f}")
        if f1 > self.melhor:
            self.melhor, self.sem_melhora = f1, 0
            self.model.save(self.caminho)
            print(f" -> Novo recorde F1: {f1:.4f}. Modelo salvo em {self.caminho}")
        else:
            self.sem_melhora += 1
            if self.sem_melhora >= self.paciencia:
                print(f"=== Early Stopping! Melhor F1-Score: {self.melhor:.4f} ===")
                self.model.stop_training = True


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
def treinar_modelo_quantizado(classe_modelo, nome_modelo_salvar, starting_point=0.0, selected_loss="BCE",
                              produtos_entrada=PRODUTOS_ENTRADA, modo=MODO_QUANT, epocas=200):
    print(f"\n--- Iniciando Treinamento Quantizado ({modo}): {nome_modelo_salvar} ---")
    print(f"Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    df_train = carregar_dataframe_starcop(CAMINHO_CSV, DIRETORIO_DADOS)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df_train, groups=df_train["folder"]))
    df_treino = df_train.iloc[train_idx].reset_index(drop=True)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    print(f"Amostras Treino: {len(df_treino)} | Amostras Validação: {len(df_val)}")

    ds_treino = STARCOPDataset(df_treino, produtos_entrada, ["labelbinary"], weight_loss="weight_mag1c", modo_treino=True)
    ds_val = STARCOPDataset(df_val, produtos_entrada, ["labelbinary"], weight_loss="weight_mag1c", modo_treino=False)
    dl_treino = DataLoader(ds_treino, batch_size=JANELAS_POR_PASSO, shuffle=True, num_workers=4,
                           pin_memory=True, collate_fn=collate_treino, persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=collate_val)

    os.makedirs("Modelos_treinados", exist_ok=True)
    caminho = f"Modelos_treinados/{nome_modelo_salvar}.keras"

    if os.path.exists(caminho):
        print("Carregando modelo salvo...")
        modelo = keras.models.load_model(caminho, compile=False)
    else:
        modelo = construir_modelo_hgq(classe_modelo, in_channels=len(produtos_entrada),
                                      modo=modo, tamanho=128)
    modelo_val = construir_modelo_hgq(classe_modelo, in_channels=len(produtos_entrada),
                                      modo=modo, tamanho=512)
    modelo.summary()

    loss = focal_dice if selected_loss == "FD" else bce_ponderada
    # Sem AMP: em QAT a simulação de ponto fixo deve rodar em float32.
    modelo.compile(optimizer=keras.optimizers.Adam(1e-4, gradient_accumulation_steps=ACUMULACAO if ACUMULACAO > 1 else None), loss=loss)

    callbacks = [ValidacaoF1(dl_val, caminho, starting_point, modelo_val, paciencia=10)]
    if modo == "hgq":
        from hgq.utils.sugar import BetaScheduler
        callbacks.append(BetaScheduler(lambda e: BETA_EBOPS * min(1.0, e / EPOCAS_AQUECIMENTO_BETA)))
    if modo == "int8":
        lotes = [next(iter(dl_treino))[0][:64] for _ in range(2)]
        callbacks.insert(0, RecalibrarEscalas(lotes))
    if modo == "hgq":
        from hgq.utils.sugar import FreeEBOPs
        callbacks.append(FreeEBOPs())  # registra EBOPs (custo estimado de LUTs) nos logs

    modelo.fit(dl_treino, epochs=epocas, callbacks=callbacks, verbose=1)
    return caminho


# ---------------------------------------------------------------------------
# Conversão para hls4ml (depois do treino)
# ---------------------------------------------------------------------------
def converter_hls4ml(caminho_keras, classe_modelo, modo=MODO_QUANT, saida="hls4ml_unet_hgq",
                     tamanho=128, part="xczu7ev-ffvc1156-2-e"):  # ZCU104
    """Gera o projeto HLS. Precisões vêm do próprio HGQ2 (não defina default_precision).
    io_stream é obrigatório para CNN deste tamanho. Para 512x512 o consumo de BRAM/LUT
    tende a ser muito grande: comece testando com tamanho menor (ex.: 128)."""
    import hls4ml

    modelo = keras.models.load_model(caminho_keras, compile=False)
    # Reconstrói o mesmo modelo com a entrada desejada e copia os pesos treinados
    modelo_fixo = construir_modelo_hgq(classe_modelo, in_channels=modelo.input_shape[-1],
                                       modo=modo, tamanho=tamanho)
    modelo_fixo.set_weights(modelo.get_weights())

    cfg = hls4ml.utils.config_from_keras_model(modelo_fixo, granularity="name")
    hls_model = hls4ml.converters.convert_from_keras_model(
        modelo_fixo, hls_config=cfg, output_dir=saida, backend="Vitis",
        io_type="io_stream", part=part,
    )
    hls_model.compile()
    return hls_model


if __name__ == "__main__":
    from Modelos import UNetMobileNetV3AttentionGatesDPU  # qualquer classe de Modelos/

    selected_loss = "BCE"  # "BCE" ou "FD"
    classe = UNetMobileNetV3AttentionGatesDPU
    nome = f"{classe.__name__}_HGQ_{MODO_QUANT}_{selected_loss}_mag1c_rgb"
    treinar_modelo_quantizado(classe, nome, starting_point=0.0, selected_loss=selected_loss)
