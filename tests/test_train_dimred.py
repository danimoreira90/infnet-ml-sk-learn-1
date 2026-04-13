"""Testes de integracao para train_dimred_and_evaluate."""
from __future__ import annotations

import uuid

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from credit_default.features.preprocessing import CATEGORICAL, NUMERIC_CONTINUOUS, NUMERIC_ORDINAL

# Import vai FALHAR ate C6 adicionar a funcao:
from credit_default.models.train import train_dimred_and_evaluate


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    """Garante tracking URI isolado por teste."""
    mlflow.set_tracking_uri(tmp_path.as_uri())
    yield
    mlflow.set_tracking_uri("")


@pytest.fixture
def minimal_df_dimred() -> pd.DataFrame:
    """DataFrame sintetico 50 linhas com todas as colunas do dataset real."""
    rng = np.random.default_rng(7)
    n = 50
    data: dict = {}
    for col in NUMERIC_CONTINUOUS:
        data[col] = rng.integers(1000, 100000, n).astype(float)
    for col in NUMERIC_ORDINAL:
        data[col] = rng.integers(-2, 9, n)
    for col in CATEGORICAL:
        data[col] = rng.integers(1, 4, n)
    return pd.DataFrame(data)


@pytest.fixture
def minimal_y() -> pd.Series:
    """Series binaria 50 elementos com 25 de cada classe."""
    y = np.array([0] * 25 + [1] * 25)
    return pd.Series(y)


def _make_exp_id() -> str:
    return mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")


def test_train_dimred_pca_returns_dict_with_required_keys(
    minimal_df_dimred, minimal_y, tmp_path
):
    """train_dimred_and_evaluate com PCA retorna dict com chaves obrigatorias."""
    result = train_dimred_and_evaluate(
        "logreg",
        "pca",
        n_components=2,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=_make_exp_id(),
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    required_keys = {
        "best_pipeline",
        "metrics",
        "run_id",
        "run_name",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_f1_mean",
        "cv_f1_std",
        "dimred_explained_variance",
    }
    assert required_keys <= set(result.keys())


def test_train_dimred_lda_n_components_is_1(minimal_df_dimred, minimal_y, tmp_path):
    """LDA pipeline: step dimred tem n_components == 1."""
    result = train_dimred_and_evaluate(
        "logreg",
        "lda",
        n_components=1,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=_make_exp_id(),
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    pipeline = result["best_pipeline"]
    assert isinstance(pipeline.named_steps["dimred"], LinearDiscriminantAnalysis)
    assert pipeline.named_steps["dimred"].n_components == 1


def test_train_dimred_params_not_empty(minimal_df_dimred, minimal_y, tmp_path):
    """Apos run PCA, params MLflow contem dimred_method e dimred_n_components."""
    result = train_dimred_and_evaluate(
        "logreg",
        "pca",
        n_components=2,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=_make_exp_id(),
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(result["run_id"]).data.params
    assert params.get("dimred_method") == "pca"
    assert params.get("dimred_n_components") == "2"
    assert params.get("scoring_primary") == "roc_auc"


def test_train_dimred_pca_explained_variance_is_float(
    minimal_df_dimred, minimal_y, tmp_path
):
    """result['dimred_explained_variance'] e float entre 0 e 1 para PCA."""
    result = train_dimred_and_evaluate(
        "logreg",
        "pca",
        n_components=2,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=_make_exp_id(),
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    ev = result["dimred_explained_variance"]
    assert isinstance(ev, float)
    assert 0.0 < ev <= 1.0
