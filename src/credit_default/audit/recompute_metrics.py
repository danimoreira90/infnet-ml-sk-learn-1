"""Auditor de integridade de métricas MLflow."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.tracking import MlflowClient

from credit_default.evaluation.metrics import compute_all_metrics
from credit_default.features.dimred import build_dimred_pipeline
from credit_default.models.pipeline import build_pipeline

TOLERANCE = 1e-4
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PARQUET = _REPO_ROOT / "data" / "credit_card_cleaned.parquet"
DEFAULT_SPLIT   = _REPO_ROOT / "artifacts" / "splits" / "split_indices.json"
TARGET_COL = "default payment next month"
VAL_METRICS = ["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"]


@dataclass
class RecomputeResult:
    run_id: str
    ok: bool
    mismatches: dict[str, tuple[float, float]] = field(default_factory=dict)


def recompute_run_metrics(
    run_id: str,
    *,
    parquet_path: Path = DEFAULT_PARQUET,
    split_path: Path = DEFAULT_SPLIT,
    tolerance: float = TOLERANCE,
) -> RecomputeResult:
    """Reconstrói pipeline do run_id, treina em X_train, avalia em X_val.

    Compara métricas recomputadas com as logadas (tolerância = 1e-4).
    Chama sys.exit(1) se qualquer métrica exceder tolerância.
    Nunca lê test_idx.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {parquet_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices não encontrado: {split_path}")

    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    logged_metrics = run.data.metrics

    X_train, X_val, y_train, y_val = _load_splits(parquet_path, split_path)

    model_name    = params["model_name"]
    seed          = int(params.get("seed", "42"))
    dimred_method = params.get("dimred_method", "none")
    dimred_n      = int(params.get("dimred_n_components", "0"))

    if dimred_method and dimred_method != "none" and dimred_n > 0:
        pipeline = build_dimred_pipeline(model_name, dimred_method, dimred_n, seed=seed)
    else:
        pipeline = build_pipeline(model_name, seed=seed)

    clf_params = {k: _cast_param(v) for k, v in params.items() if k.startswith("clf__")}
    if clf_params:
        pipeline.set_params(**clf_params)

    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    recomputed = compute_all_metrics(y_val.to_numpy(), y_pred, y_proba)

    mismatches: dict[str, tuple[float, float]] = {}
    for metric in VAL_METRICS:
        logged_val = logged_metrics.get(metric)
        if logged_val is None:
            continue
        recomp_val = recomputed.get(metric, float("nan"))
        if abs(logged_val - recomp_val) > tolerance:
            mismatches[metric] = (logged_val, recomp_val)

    ok = len(mismatches) == 0
    result = RecomputeResult(run_id=run_id, ok=ok, mismatches=mismatches)

    if not ok:
        print(f"[MISMATCH] run_id={run_id}", flush=True)
        for m, (log_v, rec_v) in mismatches.items():
            diff = abs(log_v - rec_v)
            print(f"  {m}: logged={log_v:.6f} recomputed={rec_v:.6f} diff={diff:.2e}", flush=True)
        sys.exit(1)

    print(f"[OK] run_id={run_id} — todas métricas dentro da tolerância {tolerance}", flush=True)
    return result


def _load_splits(
    parquet_path: Path,
    split_path: Path,
    *,
    include_test: bool = False,
) -> tuple:
    """Carrega splits do dataset a partir de split_indices.json.

    Quando include_test=False (padrão): retorna (X_train, X_val, y_train, y_val).
    Quando include_test=True: retorna (X_train, X_val, X_test, y_train, y_val, y_test).

    CONTRATO ANTI-LEAKAGE: test_idx só é lido quando include_test=True.
    O auditor (recompute_run_metrics) NUNCA passa include_test=True.
    Somente evaluate_final.py passa include_test=True.
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    with open(split_path) as f:
        split_info = json.load(f)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train = X.iloc[split_info["train_idx"]].reset_index(drop=True)
    y_train = y.iloc[split_info["train_idx"]].reset_index(drop=True)
    X_val   = X.iloc[split_info["val_idx"]].reset_index(drop=True)
    y_val   = y.iloc[split_info["val_idx"]].reset_index(drop=True)
    if not include_test:
        return X_train, X_val, y_train, y_val
    X_test = X.iloc[split_info["test_idx"]].reset_index(drop=True)
    y_test = y.iloc[split_info["test_idx"]].reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _cast_param(v: str) -> Any:
    """Converte string de param MLflow para int, float, bool ou None."""
    if v == "None":
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    for fn in (int, float):
        try:
            return fn(v)
        except (ValueError, TypeError):
            pass
    return v
