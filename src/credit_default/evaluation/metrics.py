"""Calculo centralizado de metricas de classificacao."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas padrao de classificacao binaria.

    Parameters
    ----------
    y_true  : labels reais (0/1)
    y_pred  : predicoes binarias (0/1)
    y_proba : probabilidades da classe positiva (shape (n,))

    Returns
    -------
    dict com chaves: roc_auc, f1_macro, precision_macro, recall_macro, accuracy
    Todos os valores sao float.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
