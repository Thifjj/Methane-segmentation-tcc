def calcular_metricas(predicao, label):

    predicao = predicao.bool()
    label = label.bool()

    tp = (predicao & label).sum().item()
    fp = (predicao & ~label).sum().item()
    fn = (~predicao & label).sum().item()
    tn = (~predicao & ~label).sum().item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

    return tp, fp, fn, tn, precision, recall, f1, iou

def classificar_pluma(has_plume, qplume):
    if not has_plume:
        return None
    return "strong_plume" if qplume > 1000 else "weak_plume"

def calcular_f1_contagens(tp, fp, fn):
    denominador = 2 * tp + fp + fn
    return 2 * tp / denominador if denominador > 0 else 0.0

if __name__ == "__main__":
    assert classificar_pluma(False, 2000) is None
    assert classificar_pluma(True, 999) == "weak_plume"
    assert classificar_pluma(True, 1000) == "weak_plume"
    assert classificar_pluma(True, 1000.1) == "strong_plume"
    assert calcular_f1_contagens(2, 1, 1) == 2 / 3
    print("Self-test OK")
