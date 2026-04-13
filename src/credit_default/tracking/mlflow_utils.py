"""Utilitarios padronizados para logging no MLflow."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(name: str = "infnet-ml-sistema") -> str:
    """Retorna experiment_id. Cria o experiment se nao existir.

    Returns
    -------
    str : experiment_id
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(name)


def log_standard_tags(
    run: mlflow.ActiveRun,
    *,
    model_family: str,
    git_commit: str,
    dataset_fingerprint: str,
    compute_profile_s: float,
    dimred_method: str = "none",
    dimred_n_components: int = 0,
    dimred_explained_variance: float | str = "na",
    baseline_run_id: str = "",
    project_part: str = "parte_3",
) -> None:
    """Loga tags padronizadas no run ativo.

    Tags logadas (10 base + 4 dimred opcionais):
    1. model_family
    2. git_commit
    3. dataset_fingerprint
    4. compute_profile_s (str do float)
    5. project_part (default "parte_3"; "parte_4" para runs dimred)
    6. framework = "scikit-learn"
    7. python_version (sys.version)
    8. os_platform (platform.platform())
    9. stage (extraido do run_name: primeira parte antes de '__')
    10. search_type (extraido do run_name: quinta parte)
    11. dimred_method (default "none")
    12. dimred_n_components (default "0")
    13. dimred_explained_variance (default "na"; float para PCA)
    14. baseline_run_id (default ""; run_id do run P3 equivalente)

    Nota: mlflow.runName e atributo nativo do run (setado em start_run),
    NAO duplicado como tag.
    """
    run_name = run.info.run_name or ""
    parts = run_name.split("__")
    stage = parts[0] if len(parts) > 0 else ""
    search_type = parts[4] if len(parts) > 4 else "none"

    mlflow.set_tags(
        {
            "model_family": model_family,
            "git_commit": git_commit,
            "dataset_fingerprint": dataset_fingerprint,
            "compute_profile_s": str(compute_profile_s),
            "project_part": project_part,
            "framework": "scikit-learn",
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "stage": stage,
            "search_type": search_type,
            "dimred_method": dimred_method,
            "dimred_n_components": str(dimred_n_components),
            "dimred_explained_variance": str(dimred_explained_variance),
            "baseline_run_id": baseline_run_id,
        }
    )


def log_standard_metrics(
    run: mlflow.ActiveRun,
    metrics: dict[str, float],
    *,
    cv_roc_auc_mean: float,
    cv_roc_auc_std: float,
    cv_f1_mean: float,
    cv_f1_std: float,
    training_time_s: float,
    inference_latency_ms: float,
) -> None:
    """Loga metricas padronizadas no run ativo.

    Metricas logadas (11 total):
    1-5. roc_auc, f1_macro, precision_macro, recall_macro, accuracy (do dict metrics)
    6.   cv_roc_auc_mean  (metrica PRIMARIA de CV — scoring='roc_auc')
    7.   cv_roc_auc_std
    8.   cv_f1_mean       (metrica secundaria de CV — scoring='f1_macro')
    9.   cv_f1_std
    10.  training_time_s
    11.  inference_latency_ms
    """
    mlflow.log_metrics(
        {
            **metrics,
            "cv_roc_auc_mean": cv_roc_auc_mean,
            "cv_roc_auc_std": cv_roc_auc_std,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "training_time_s": training_time_s,
            "inference_latency_ms": inference_latency_ms,
        }
    )


def log_standard_params(
    run: mlflow.ActiveRun,
    *,
    model_name: str,
    seed: int,
    cv_folds: int,
    n_train: int,
    n_val: int,
    search_type: str,
    clf_params: dict[str, Any],
    dimred_method: str = "none",
    dimred_n_components: int = 0,
) -> None:
    """Loga parametros padronizados no run ativo.

    Params logados:
    Meta (10): model_name, seed, cv_folds, scoring_primary, split_strategy,
               search_type, n_train, n_val, dimred_method, dimred_n_components
    Classifier (clf__*): clf_params — defaults para baseline, best_params_ para tuned

    Total minimo: 10 meta + len(clf_params) params. DoD exige >= 8 total
    (minimo 8 meta + 3 clf__ hyperparams do estimador).
    """
    meta: dict[str, str] = {
        "model_name": model_name,
        "seed": str(seed),
        "cv_folds": str(cv_folds),
        "scoring_primary": "roc_auc",
        "split_strategy": "stratified_70_15_15_from_part2",
        "search_type": search_type,
        "n_train": str(n_train),
        "n_val": str(n_val),
        "dimred_method": dimred_method,
        "dimred_n_components": str(dimred_n_components),
    }
    clf_params_str = {k: str(v) for k, v in clf_params.items()}
    mlflow.log_params({**meta, **clf_params_str})


def log_standard_artifacts(
    run: mlflow.ActiveRun,
    tmp_dir: Path,
    *,
    has_feature_importances: bool = False,
) -> None:
    """Loga artefatos padronizados do tmp_dir no run ativo.

    Artefatos esperados no tmp_dir:
    - confusion_matrix.png
    - roc_curve.png
    - pr_curve.png
    - run_summary.json
    - split_fingerprint.txt
    - feature_importances.csv (se has_feature_importances=True)

    Usa mlflow.log_artifacts(str(tmp_dir)).
    """
    mlflow.log_artifacts(str(tmp_dir))
