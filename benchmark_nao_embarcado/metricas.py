from sklearn.metrics import average_precision_score

EPS = 1e-6


def calcular_metricas(predicao, label):
    predicao, label = predicao.bool(), label.bool()
    tp = (predicao & label).sum().item()
    fp = (predicao & ~label).sum().item()
    fn = (~predicao & label).sum().item()
    tn = (~predicao & ~label).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = calcular_f1_contagens(tp, fp, fn)
    iou = tp / (tp + fp + fn + EPS)
    return tp, fp, fn, tn, precision, recall, f1, iou


def classificar_pluma(has_plume, difficulty):
    if not has_plume:
        return None
    return "strong_plume" if difficulty == "easy" else "weak_plume"


def calcular_f1_contagens(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn + EPS)


def calcular_auprc_imagem(label, probabilidades):
    if label.sum() <= 0:
        return None
    return average_precision_score(label.reshape(-1), probabilidades.reshape(-1))


if __name__ == "__main__":
    assert classificar_pluma(False, "easy") is None
    assert classificar_pluma(True, "easy") == "strong_plume"
    assert classificar_pluma(True, "hard") == "weak_plume"
    assert calcular_f1_contagens(0, 0, 0) == 0.0
    print("Self-test OK")
