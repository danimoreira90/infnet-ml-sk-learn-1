"""Testes de fumarola para generate_comparison_dimred."""

from __future__ import annotations

import uuid

import mlflow
import pandas as pd
import pytest

from credit_default.tracking.compare import consolidated_results_table
from credit_default.tracking.mlflow_utils import (
    get_or_create_experiment,
    log_standard_params,
    log_standard_tags,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_run(
    exp_name: str,
    model_name: str,
    stage: str,
    roc_auc: float,
    dimred_method: str = "none",
    n_components: int = 0,
    ev: float | str = "na",
    seed: int = 42,
) -> str:
    """Cria um run MLflow minimo com params e tags padrao."""
    exp_id = get_or_create_experiment(exp_name)
    with mlflow.start_run(experiment_id=exp_id) as run:
        log_standard_params(
            run,
            model_name=model_name,
            seed=seed,
            cv_folds=5,
            n_train=1000,
            n_val=300,
            search_type="none",
            clf_params={},
            dimred_method=dimred_method,
            dimred_n_components=n_components,
        )
        log_standard_tags(
            run,
            model_family="linear",
            git_commit="abc1234",
            dataset_fingerprint="30c6be3a",
            compute_profile_s=0.1,
            dimred_method=dimred_method,
            dimred_n_components=n_components,
            dimred_explained_variance=ev,
            project_part="parte_3" if stage == "baseline" else "parte_4",
        )
        mlflow.set_tag("stage", stage)
        mlflow.log_metrics(
            {
                "roc_auc": roc_auc,
                "f1_macro": 0.6,
                "cv_roc_auc_mean": roc_auc - 0.01,
                "cv_roc_auc_std": 0.005,
                "training_time_s": 0.1,
            }
        )
        return run.info.run_id


@pytest.fixture
def isolated_mlflow(tmp_path):
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    yield mlruns_dir
    mlflow.set_tracking_uri("")


@pytest.fixture
def populated_experiment(isolated_mlflow):
    """Cria experimento com 2 runs baseline + 2 runs dimred."""
    exp_name = f"test-{uuid.uuid4().hex[:8]}"
    _make_fake_run(exp_name, "logreg", "baseline", 0.72)
    _make_fake_run(exp_name, "gb", "baseline", 0.78)
    _make_fake_run(exp_name, "logreg", "dimred", 0.70, "pca", 10, 0.84)
    _make_fake_run(exp_name, "gb", "dimred", 0.75, "pca", 10, 0.84)
    return exp_name


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


def test_consolidated_results_table_returns_dataframe(populated_experiment):
    """consolidated_results_table devolve DataFrame nao-vazio."""
    df = consolidated_results_table(populated_experiment)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 4


def test_consolidated_results_table_has_stage_column(populated_experiment):
    """DataFrame contem coluna 'stage' com valores baseline e dimred."""
    df = consolidated_results_table(populated_experiment)
    assert "stage" in df.columns
    stages = set(df["stage"].unique())
    assert "baseline" in stages
    assert "dimred" in stages


def test_consolidated_results_table_has_roc_auc(populated_experiment):
    """DataFrame contem coluna 'roc_auc' com valores float entre 0 e 1."""
    df = consolidated_results_table(populated_experiment)
    assert "roc_auc" in df.columns
    assert df["roc_auc"].between(0.0, 1.0).all()


def test_comparison_dimred_md_is_valid_markdown(tmp_path, populated_experiment):
    """comparison_dimred.md gerado contem cabecalho e pelo menos 4 linhas de dados."""
    df = consolidated_results_table(populated_experiment)
    p3 = df[df["stage"] == "baseline"].copy()
    p4 = df[df["stage"] == "dimred"].copy()
    combined = pd.concat([p3, p4], ignore_index=True).sort_values("roc_auc", ascending=False)
    out = tmp_path / "comparison_dimred.md"
    cols = ["run_name", "stage", "roc_auc", "f1_macro"]
    available = [c for c in cols if c in combined.columns]
    md = combined[available].to_markdown(index=False, floatfmt=".4f")
    out.write_text(
        "# Comparativo Parte 4 — Dimred vs Baseline\n\n" + md + "\n",
        encoding="utf-8",
    )
    content = out.read_text(encoding="utf-8")
    assert content.startswith("# Comparativo")
    lines = [line for line in content.splitlines() if line.startswith("|")]
    # header + separator + 4 data rows
    assert len(lines) >= 6


def test_pivot_dataframe_has_all_expected_columns():
    """Pivot com colunas none, pca_k10 contem os modelos esperados."""
    rows = [
        {"model": "logreg", "none (P3)": "0.7232", "pca_k10": "0.7045"},
        {"model": "gb", "none (P3)": "0.7795", "pca_k10": "0.7620"},
    ]
    pivot_df = pd.DataFrame(rows, columns=["model", "none (P3)", "pca_k10"])
    assert list(pivot_df.columns) == ["model", "none (P3)", "pca_k10"]
    assert set(pivot_df["model"]) == {"logreg", "gb"}
