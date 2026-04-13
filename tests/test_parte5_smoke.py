"""Smoke tests para a Parte 5 — modulo audit e artefatos gerados."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─── Modulo audit ──────────────────────────────────────────────────────────────

def test_audit_module_importable():
    from credit_default.audit import RecomputeResult, recompute_run_metrics  # noqa: F401

    assert callable(recompute_run_metrics)
    r = RecomputeResult(run_id="abc", ok=True)
    assert r.run_id == "abc"
    assert r.ok is True
    assert r.mismatches == {}


# ─── Artefatos gerados ─────────────────────────────────────────────────────────

def test_consolidated_results_file_exists():
    path = REPO_ROOT / "reports" / "parte_5" / "consolidated_results.md"
    assert path.exists(), f"Nao encontrado: {path}"
    content = path.read_text(encoding="utf-8")
    # Deve ter pelo menos 25 linhas de dados (header + 25 runs)
    data_lines = [l for l in content.splitlines() if l.startswith("|") and "rank" not in l and "---" not in l]
    assert len(data_lines) >= 25, f"Esperado >= 25 linhas de dados, encontrado {len(data_lines)}"


def test_rationale_has_winner_run_id():
    path = REPO_ROOT / "reports" / "parte_5" / "final_selection_rationale.md"
    assert path.exists(), f"Nao encontrado: {path}"
    content = path.read_text(encoding="utf-8")
    assert "winner_run_id:" in content
    assert "decision_step:" in content


def test_test_metrics_json_exists_and_valid():
    path = REPO_ROOT / "reports" / "parte_5" / "test_metrics.json"
    assert path.exists(), f"Nao encontrado: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "roc_auc" in data, "roc_auc ausente em test_metrics.json"
    assert 0.0 < data["roc_auc"] < 1.0, f"roc_auc fora do intervalo: {data['roc_auc']}"
    assert "training_time_s" in data
    assert "inference_latency_ms" in data


# ─── _cast_param ───────────────────────────────────────────────────────────────

def test_cast_param_int():
    from credit_default.audit.recompute_metrics import _cast_param
    assert _cast_param("42") == 42
    assert isinstance(_cast_param("42"), int)


def test_cast_param_float():
    from credit_default.audit.recompute_metrics import _cast_param
    assert _cast_param("0.1") == pytest.approx(0.1)


def test_cast_param_none():
    from credit_default.audit.recompute_metrics import _cast_param
    assert _cast_param("None") is None


def test_cast_param_bool_true():
    from credit_default.audit.recompute_metrics import _cast_param
    assert _cast_param("True") is True


# ─── _load_splits ──────────────────────────────────────────────────────────────

def test_load_splits_include_test_false_omits_test(tmp_path, minimal_df):
    import json
    from credit_default.audit.recompute_metrics import _load_splits

    parquet_path = tmp_path / "credit_card_cleaned.parquet"
    minimal_df.to_parquet(parquet_path, engine="pyarrow", index=False)
    n = len(minimal_df)
    split_path = tmp_path / "split_indices.json"
    split_path.write_text(json.dumps({
        "train_idx": list(range(0, 350)),
        "val_idx":   list(range(350, 425)),
        "test_idx":  list(range(425, n)),
    }))

    result = _load_splits(parquet_path, split_path, include_test=False)
    assert len(result) == 4  # X_train, X_val, y_train, y_val


def test_load_splits_include_test_true_returns_test(tmp_path, minimal_df):
    import json
    from credit_default.audit.recompute_metrics import _load_splits

    parquet_path = tmp_path / "credit_card_cleaned.parquet"
    minimal_df.to_parquet(parquet_path, engine="pyarrow", index=False)
    n = len(minimal_df)
    split_path = tmp_path / "split_indices.json"
    split_path.write_text(json.dumps({
        "train_idx": list(range(0, 350)),
        "val_idx":   list(range(350, 425)),
        "test_idx":  list(range(425, n)),
    }))

    result = _load_splits(parquet_path, split_path, include_test=True)
    assert len(result) == 6  # X_train, X_val, X_test, y_train, y_val, y_test
    X_train, X_val, X_test, y_train, y_val, y_test = result
    assert len(X_test) == n - 425
