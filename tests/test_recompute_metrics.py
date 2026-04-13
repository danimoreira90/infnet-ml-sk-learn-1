"""Testes TDD para o auditor de integridade de métricas MLflow."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from credit_default.audit.recompute_metrics import (
    RecomputeResult,
    _cast_param,
    _load_splits,
    recompute_run_metrics,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def split_indices_file(tmp_path, minimal_df):
    """Cria parquet + split_indices.json sintéticos no tmp_path."""
    parquet_path = tmp_path / "credit_card_cleaned.parquet"
    target_col = "default payment next month"
    minimal_df.to_parquet(parquet_path, engine="pyarrow", index=False)

    n = len(minimal_df)
    train_idx = list(range(0, 350))
    val_idx   = list(range(350, 425))
    test_idx  = list(range(425, n))

    split_path = tmp_path / "split_indices.json"
    split_path.write_text(json.dumps({
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "test_idx":  test_idx,
    }))
    return parquet_path, split_path


def _make_mock_run(params: dict, metrics: dict):
    """Constrói mock de mlflow Run com params e metrics."""
    run = MagicMock()
    run.data.params   = params
    run.data.metrics  = metrics
    run.data.tags     = {"model_family": "linear"}
    run.info.run_name = "baseline__logreg__numstd_catoh__none__none__seed42__data30c6be3a__codeabc1234"
    return run


# ─── Testes ────────────────────────────────────────────────────────────────────

def test_recompute_result_is_dataclass():
    result = RecomputeResult(run_id="abc", ok=True)
    assert result.run_id == "abc"
    assert result.ok is True
    assert result.mismatches == {}


def test_recompute_matching_run_returns_ok(split_indices_file):
    parquet_path, split_path = split_indices_file
    params = {
        "model_name": "logreg",
        "seed": "42",
        "dimred_method": "none",
        "dimred_n_components": "0",
    }
    # Treina localmente para obter métricas reais
    from credit_default.models.pipeline import build_pipeline
    import pandas as pd

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    target_col = "default payment next month"
    with open(split_path) as f:
        import json
        split_info = json.load(f)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train = X.iloc[split_info["train_idx"]].reset_index(drop=True)
    y_train = y.iloc[split_info["train_idx"]].reset_index(drop=True)
    X_val   = X.iloc[split_info["val_idx"]].reset_index(drop=True)
    y_val   = y.iloc[split_info["val_idx"]].reset_index(drop=True)

    pipeline = build_pipeline("logreg", seed=42)
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    from credit_default.evaluation.metrics import compute_all_metrics
    real_metrics = compute_all_metrics(y_val.to_numpy(), y_pred, y_proba)

    mock_run = _make_mock_run(params, real_metrics)
    with patch("credit_default.audit.recompute_metrics.MlflowClient") as MockClient:
        MockClient.return_value.get_run.return_value = mock_run
        result = recompute_run_metrics(
            "fake-run-id",
            parquet_path=parquet_path,
            split_path=split_path,
        )
    assert result.ok is True
    assert result.mismatches == {}


def test_recompute_metric_mismatch_exits_nonzero(split_indices_file):
    parquet_path, split_path = split_indices_file
    params = {
        "model_name": "logreg",
        "seed": "42",
        "dimred_method": "none",
        "dimred_n_components": "0",
    }
    # Métricas deliberadamente erradas (diff > 1e-4)
    wrong_metrics = {"roc_auc": 0.9999, "f1_macro": 0.9999}
    mock_run = _make_mock_run(params, wrong_metrics)
    with patch("credit_default.audit.recompute_metrics.MlflowClient") as MockClient:
        MockClient.return_value.get_run.return_value = mock_run
        with pytest.raises(SystemExit) as exc_info:
            recompute_run_metrics(
                "fake-run-id",
                parquet_path=parquet_path,
                split_path=split_path,
            )
    assert exc_info.value.code != 0


def test_recompute_tolerates_float_rounding(split_indices_file):
    parquet_path, split_path = split_indices_file
    params = {
        "model_name": "logreg",
        "seed": "42",
        "dimred_method": "none",
        "dimred_n_components": "0",
    }
    import pandas as pd, json
    from credit_default.models.pipeline import build_pipeline
    from credit_default.evaluation.metrics import compute_all_metrics

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    target_col = "default payment next month"
    with open(split_path) as f:
        split_info = json.load(f)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train = X.iloc[split_info["train_idx"]].reset_index(drop=True)
    y_train = y.iloc[split_info["train_idx"]].reset_index(drop=True)
    X_val   = X.iloc[split_info["val_idx"]].reset_index(drop=True)
    y_val   = y.iloc[split_info["val_idx"]].reset_index(drop=True)

    pipeline = build_pipeline("logreg", seed=42)
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    real_metrics = compute_all_metrics(y_val.to_numpy(), y_pred, y_proba)

    # Adiciona ruído menor que 1e-4
    noisy_metrics = {k: v + 5e-5 for k, v in real_metrics.items()}
    mock_run = _make_mock_run(params, noisy_metrics)
    with patch("credit_default.audit.recompute_metrics.MlflowClient") as MockClient:
        MockClient.return_value.get_run.return_value = mock_run
        result = recompute_run_metrics(
            "fake-run-id",
            parquet_path=parquet_path,
            split_path=split_path,
        )
    assert result.ok is True


def test_recompute_unknown_model_raises_keyerror(split_indices_file):
    parquet_path, split_path = split_indices_file
    params = {
        "model_name": "modelo_inexistente",
        "seed": "42",
        "dimred_method": "none",
        "dimred_n_components": "0",
    }
    mock_run = _make_mock_run(params, {"roc_auc": 0.5})
    with patch("credit_default.audit.recompute_metrics.MlflowClient") as MockClient:
        MockClient.return_value.get_run.return_value = mock_run
        with pytest.raises(KeyError):
            recompute_run_metrics(
                "fake-run-id",
                parquet_path=parquet_path,
                split_path=split_path,
            )


def test_recompute_missing_split_artifact_raises(tmp_path):
    parquet_path = tmp_path / "nonexistent.parquet"
    split_path   = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        recompute_run_metrics(
            "fake-run-id",
            parquet_path=parquet_path,
            split_path=split_path,
        )


# ─── Testes para _cast_param ───────────────────────────────────────────────────

def test_cast_param_int():
    assert _cast_param("42") == 42

def test_cast_param_float():
    assert _cast_param("0.1") == pytest.approx(0.1)

def test_cast_param_none():
    assert _cast_param("None") is None

def test_cast_param_bool_true():
    assert _cast_param("True") is True
