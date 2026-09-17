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