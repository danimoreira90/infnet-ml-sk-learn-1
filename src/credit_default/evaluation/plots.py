"""Geracao de graficos de avaliacao de modelos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    output_path: Path,
) -> Path:
    """Salva confusion matrix como PNG em output_path. Retorna output_path.

    Fecha a figura apos salvar (plt.close()).
    """
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def roc_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    output_path: Path,
) -> Path:
    """Salva ROC curve como PNG em output_path. Retorna output_path.

    Fecha a figura apos salvar (plt.close()).
    """
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def pr_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    output_path: Path,
) -> Path:
    """Salva Precision-Recall curve como PNG em output_path. Retorna output_path.

    Fecha a figura apos salvar (plt.close()).
    """
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
