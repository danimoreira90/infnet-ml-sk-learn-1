"""Consolidacao de resultados de experimentos MLflow."""

from __future__ import annotations

import pandas as pd
from mlflow.tracking import MlflowClient


def consolidated_results_table(
    experiment_name: str = "infnet-ml-sistema",
    *,
    stage: str | None = None,
) -> pd.DataFrame:
    """Query MLflow via MlflowClient, retorna DataFrame com todos os runs.

    Parameters
    ----------
    experiment_name : nome do experiment MLflow
    stage           : se fornecido, filtra runs pela tag 'stage'

    Returns
    -------
    pd.DataFrame com colunas:
    - run_name, model_family, stage, search_type
    - roc_auc, f1_macro, precision_macro, recall_macro, accuracy
    - cv_roc_auc_mean, cv_roc_auc_std
    - cv_f1_mean, cv_f1_std
    - training_time_s, inference_latency_ms
    Ordenado por roc_auc descendente (metrica primaria).

    Raises
    ------
    ValueError : se experiment nao existir.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' nao encontrado no MLflow.")

    filter_string = f"tags.stage = '{stage}'" if stage else ""
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=200,
    )

    rows = []
    for run in runs:
        m = run.data.metrics
        t = run.data.tags
        rows.append(
            {
                "run_name": run.info.run_name,
                "model_family": t.get("model_family", ""),
                "stage": t.get("stage", ""),
                "search_type": t.get("search_type", ""),
                "roc_auc": m.get("roc_auc", float("nan")),
                "f1_macro": m.get("f1_macro", float("nan")),
                "precision_macro": m.get("precision_macro", float("nan")),
                "recall_macro": m.get("recall_macro", float("nan")),
                "accuracy": m.get("accuracy", float("nan")),
                "cv_roc_auc_mean": m.get("cv_roc_auc_mean", float("nan")),
                "cv_roc_auc_std": m.get("cv_roc_auc_std", float("nan")),
                "cv_f1_mean": m.get("cv_f1_mean", float("nan")),
                "cv_f1_std": m.get("cv_f1_std", float("nan")),
                "training_time_s": m.get("training_time_s", float("nan")),
                "inference_latency_ms": m.get("inference_latency_ms", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
