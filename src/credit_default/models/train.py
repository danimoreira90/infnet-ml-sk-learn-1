"""Orquestracao de treino, avaliacao e logging MLflow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)

from credit_default.evaluation.metrics import compute_all_metrics
from credit_default.evaluation.plots import confusion_matrix_plot, pr_plot, roc_plot
from credit_default.models.pipeline import build_pipeline
from credit_default.models.registry import get_model_spec
from credit_default.tracking.mlflow_utils import (
    get_or_create_experiment,
    log_standard_artifacts,
    log_standard_metrics,
    log_standard_tags,
)
from credit_default.tracking.run_naming import compose_run_name


def load_split_data(
    parquet_path: Path,
    split_indices_path: Path,
    target_col: str = "default payment next month",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Carrega dataset e retorna splits usando indices de split_indices.json.

    Retorna X_train, X_val, X_test, y_train, y_val, y_test.
    Usa SOMENTE train_idx e val_idx para treino/validacao.
    test_idx e carregado apenas para completude da tupla mas NUNCA usado em treino.

    IMPORTANTE: Esta funcao le os indices do JSON — nunca regenera splits.

    Parameters
    ----------
    parquet_path       : caminho para credit_card_cleaned.parquet
    split_indices_path : caminho para split_indices.json
    target_col         : nome da coluna target

    Returns
    -------
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    with open(split_indices_path) as f:
        split_info = json.load(f)

    train_idx = split_info["train_idx"]
    val_idx = split_info["val_idx"]
    test_idx = split_info["test_idx"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate(
    model_name: str,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
    cv_folds: int = 5,
    tune: bool = False,
    experiment_name: str = "infnet-ml-sistema",
    datahash8: str,
    githash7: str,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Pipeline completo: train -> CV -> [tune] -> evaluate -> MLflow log.

    Fluxo:
    1. build_pipeline(model_name, seed=seed)
    2. Cross-validation com cross_validate() usando
       scoring={'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'},
       cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
       no X_train. Extrai cv_roc_auc_mean, cv_roc_auc_std (PRIMARIA) e
       cv_f1_mean, cv_f1_std (secundaria).
    3. Se tune=True E search_type != 'none':
       - GridSearchCV ou RandomizedSearchCV conforme registry.search_type
       - Para RandomizedSearchCV: n_iter=30, random_state=seed
       - scoring='roc_auc' (metrica PRIMARIA)
       - cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
       - Refit=True (default) — ja refita no X_train completo
       - pipeline = search.best_estimator_
    4. Se tune=False: fit simples no X_train
    5. Predict no X_val: y_pred e y_proba
    6. compute_all_metrics(y_val, y_pred, y_proba)
    7. Medir inference_latency_ms: tempo medio por amostra no X_val
    8. Gerar plots (confusion_matrix, roc, pr) no tmp_dir
    9. Gerar run_summary.json no tmp_dir com todas as metricas, params e primary_metric="roc_auc"
    10. Gerar split_fingerprint.txt no tmp_dir com datahash8
    11. Se modelo tree/ensemble com feature_importances_: gerar feature_importances.csv
    12. compose_run_name() para gerar run_name
    13. MLflow: start_run(run_name=run_name, experiment_id=...)
    14. log_standard_tags, log_standard_metrics, log_standard_artifacts
    15. end_run
    16. Retorna dict com best_pipeline, metrics, run_id, run_name,
        cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std

    Parameters
    ----------
    model_name       : chave do MODEL_REGISTRY
    X_train, y_train : dados de treino (indices de split_indices.json train_idx)
    X_val, y_val     : dados de validacao (indices de split_indices.json val_idx)
    seed             : semente para reprodutibilidade
    cv_folds         : numero de folds para cross-validation
    tune             : se True, executa hyperparameter search
    experiment_name  : nome do experiment MLflow
    datahash8        : primeiros 8 chars do SHA-256 do dataset
    githash7         : primeiros 7 chars do git commit hash
    tmp_dir          : diretorio temporario para artefatos (caller gerencia lifecycle)

    Returns
    -------
    dict com chaves: best_pipeline, metrics, run_id, run_name,
                     cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std
    """
    spec = get_model_spec(model_name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # 1. Build pipeline
    pipeline = build_pipeline(model_name, seed=seed)

    # 2. Cross-validation com duas metricas (PRIMARY: roc_auc, secondary: f1_macro)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    cv_roc_auc_mean = float(np.mean(cv_results["test_roc_auc"]))
    cv_roc_auc_std = float(np.std(cv_results["test_roc_auc"]))
    cv_f1_mean = float(np.mean(cv_results["test_f1_macro"]))
    cv_f1_std = float(np.std(cv_results["test_f1_macro"]))

    # 3/4. Tune ou fit simples
    search_type = spec["search_type"]
    search_name = "none"
    t0 = time.perf_counter()

    if tune and search_type != "none":
        pipeline = build_pipeline(model_name, seed=seed)  # fresh pipeline para search
        if search_type == "grid":
            search = GridSearchCV(
                pipeline,
                param_grid=spec["param_grid"],
                scoring="roc_auc",
                cv=cv,
                refit=True,
                n_jobs=-1,
            )
            search_name = "grid"
        else:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=spec["param_grid"],
                n_iter=30,
                scoring="roc_auc",
                cv=cv,
                refit=True,
                random_state=seed,
                n_jobs=-1,
            )
            search_name = "random"
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)

    training_time_s = time.perf_counter() - t0

    # 5. Predict no X_val
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    # 6. Compute metrics
    metrics = compute_all_metrics(np.array(y_val), y_pred, y_proba)

    # 7. Inference latency
    n_samples = len(X_val)
    t_inf = time.perf_counter()
    pipeline.predict_proba(X_val)
    inference_latency_ms = (time.perf_counter() - t_inf) / n_samples * 1000

    # 8. Plots
    confusion_matrix_plot(np.array(y_val), y_pred, output_path=tmp_dir / "confusion_matrix.png")
    roc_plot(np.array(y_val), y_proba, output_path=tmp_dir / "roc_curve.png")
    pr_plot(np.array(y_val), y_proba, output_path=tmp_dir / "pr_curve.png")

    # 9. run_summary.json
    summary = {
        "model_name": model_name,
        "primary_metric": "roc_auc",
        "metrics": metrics,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "tune": tune,
        "search_type": search_name,
        "seed": seed,
        "training_time_s": training_time_s,
        "inference_latency_ms": inference_latency_ms,
    }
    with open(tmp_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 10. split_fingerprint.txt
    (tmp_dir / "split_fingerprint.txt").write_text(f"datahash8={datahash8}\ngithash7={githash7}\n")

    # 11. feature_importances.csv (tree/ensemble)
    has_fi = False
    clf_step = pipeline.named_steps.get("clf")
    estimator = getattr(clf_step, "estimator", clf_step)  # unwrap Calibrated if needed
    if hasattr(estimator, "feature_importances_"):
        has_fi = True
        fi = pd.DataFrame({"importance": estimator.feature_importances_})
        fi.to_csv(tmp_dir / "feature_importances.csv", index=False)

    # 12. Run name
    run_name = compose_run_name(
        stage="tune" if tune else "baseline",
        model=model_name,
        search=search_name,
        seed=seed,
        datahash8=datahash8,
        githash7=githash7,
    )

    # 13-15. MLflow logging
    experiment_id = get_or_create_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        log_standard_tags(
            run,
            model_family=spec["model_family"],
            git_commit=githash7,
            dataset_fingerprint=datahash8,
            compute_profile_s=training_time_s,
        )
        log_standard_metrics(
            run,
            metrics,
            cv_roc_auc_mean=cv_roc_auc_mean,
            cv_roc_auc_std=cv_roc_auc_std,
            cv_f1_mean=cv_f1_mean,
            cv_f1_std=cv_f1_std,
            training_time_s=training_time_s,
            inference_latency_ms=inference_latency_ms,
        )
        log_standard_artifacts(run, tmp_dir, has_feature_importances=has_fi)
        run_id = run.info.run_id

    return {
        "best_pipeline": pipeline,
        "metrics": metrics,
        "run_id": run_id,
        "run_name": run_name,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
    }
