import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             brier_score_loss)


def scores(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not len(y):
        return {'n': 0}
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': len(y), 'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'pr_auc': float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else None,
            'accuracy': float(accuracy_score(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'f1': float(f1_score(y, pred, zero_division=0)), 'brier': float(brier_score_loss(y, p)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
