"""Average precision global exata com ordenação externa e RAM limitada."""
import os
import tempfile
import warnings
import numpy as np


class AUPRCDisco:
    # Preserva os valores float32 produzidos pelo teste, sem discretização.
    dtype = np.dtype([("score", "<f4"), ("label", "u1")])
    bloco = 262144

    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix=".auprc_", dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.arquivos = []
        self.total = self.positivos = 0
        self.serial = 0

    def _caminho(self):
        self.serial += 1
        return os.path.join(self.temp.name, f"{self.serial}.bin")

    def adicionar(self, labels, scores):
        scores = np.asarray(scores).ravel()
        labels = np.asarray(labels).ravel()
        if scores.dtype != np.float32:
            raise ValueError("AUPRCDisco espera probabilidades float32, sem conversão de precisão.")
        if scores.size != labels.size or not np.isfinite(scores).all():
            raise ValueError("Probabilidades inválidas ou tamanhos diferentes.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("AUPRCDisco espera rótulos binários 0/1.")
        for inicio in range(0, scores.size, self.bloco):
            fim = inicio + self.bloco
            dados = np.empty(scores[inicio:fim].size, dtype=self.dtype)
            dados["score"] = scores[inicio:fim]
            dados["label"] = labels[inicio:fim]
            dados.sort(order="score")
            caminho = self._caminho()
            dados.tofile(caminho)
            self.arquivos.append(caminho)
        self.total += scores.size
        self.positivos += int(np.count_nonzero(labels))

    def _mesclar(self, esquerda, direita):
        saida = self._caminho()
        with open(esquerda, "rb") as fa, open(direita, "rb") as fb, open(saida, "wb") as fo:
            a = np.fromfile(fa, self.dtype, self.bloco)
            b = np.fromfile(fb, self.dtype, self.bloco)
            while a.size and b.size:
                limite = min(a["score"][-1], b["score"][-1])
                na = np.searchsorted(a["score"], limite, side="right")
                nb = np.searchsorted(b["score"], limite, side="right")
                juntos = np.concatenate((a[:na], b[:nb]))
                juntos.sort(order="score")
                juntos.tofile(fo)
                a = a[na:]
                b = b[nb:]
                if not a.size:
                    a = np.fromfile(fa, self.dtype, self.bloco)
                if not b.size:
                    b = np.fromfile(fb, self.dtype, self.bloco)
            for resto, arquivo in ((a, fa), (b, fb)):
                resto.tofile(fo)
                while True:
                    dados = np.fromfile(arquivo, self.dtype, self.bloco)
                    if not dados.size:
                        break
                    dados.tofile(fo)
        os.remove(esquerda)
        os.remove(direita)
        return saida

    def calcular(self):
        try:
            if not self.total:
                raise ValueError("Nenhum pixel para calcular AUPRC.")
            if not self.positivos:
                warnings.warn("No positive class found in y_true, recall is set to one for all thresholds.")
                return 0.0
            arquivos = self.arquivos[:]
            while len(arquivos) > 1:
                proximos = []
                for i in range(0, len(arquivos), 2):
                    proximos.append(self._mesclar(arquivos[i], arquivos[i+1]) if i+1 < len(arquivos) else arquivos[i])
                arquivos = proximos
            # AP = soma(delta recall * precision). Empates são um único limiar,
            # inclusive quando atravessam a fronteira entre blocos de leitura.
            anteriores = positivos_anteriores = 0
            score_pendente = None
            n_pendente = p_pendente = 0
            soma = 0.0
            with open(arquivos[0], "rb") as arquivo:
                while True:
                    dados = np.fromfile(arquivo, self.dtype, self.bloco)
                    if not dados.size:
                        break
                    inicios = np.r_[0, np.flatnonzero(np.diff(dados["score"])) + 1]
                    scores = dados["score"][inicios]
                    contagens = np.diff(np.r_[inicios, dados.size]).astype(np.int64)
                    positivos = np.add.reduceat(dados["label"].astype(np.int64), inicios)
                    if score_pendente is not None:
                        if scores[0] == score_pendente:
                            contagens[0] += n_pendente
                            positivos[0] += p_pendente
                        else:
                            scores = np.r_[score_pendente, scores]
                            contagens = np.r_[n_pendente, contagens]
                            positivos = np.r_[p_pendente, positivos]
                    score_pendente = scores[-1]
                    n_pendente = int(contagens[-1])
                    p_pendente = int(positivos[-1])
                    n, p = contagens[:-1], positivos[:-1]
                    antes_n = anteriores + np.r_[0, np.cumsum(n)[:-1]] if n.size else np.empty(0)
                    antes_p = positivos_anteriores + np.r_[0, np.cumsum(p)[:-1]] if p.size else np.empty(0)
                    precisao = (self.positivos - antes_p) / (self.total - antes_n)
                    soma += float(np.sum((p / self.positivos) * precisao))
                    anteriores += int(n.sum())
                    positivos_anteriores += int(p.sum())
            soma += (p_pendente / self.positivos) * ((self.positivos - positivos_anteriores) / (self.total - anteriores))
            return float(soma)
        finally:
            self.temp.cleanup()
