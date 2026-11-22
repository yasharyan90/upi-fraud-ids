"""Evaluation metrics and mathematical formulations (Section VI).

Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * P * R / (P + R)
FPR       = FP / (FP + TN)   (friction imposed on genuine customers)
FNR       = FN / (FN + TP)   (exposure to direct financial loss)
"""
from __future__ import annotations

import numpy as np


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **c,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def best_threshold_by_f1(y_true, scores, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Pick the decision threshold that maximises F1 on a *validation* split."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = classification_metrics(y_true, (np.asarray(scores) >= t).astype(int))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1
