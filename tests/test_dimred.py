"""Testes RED para build_dimred_pipeline e compute_pca_n_components."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

# Esses imports vão FALHAR até C1 criar o módulo:
from credit_default.features.dimred import build_dimred_pipeline, compute_pca_n_components


@pytest.fixture
def small_X():
    rng = np.random.default_rng(42)
    return rng.standard_normal((30, 10))


def test_build_dimred_pipeline_pca_returns_3step_pipeline():
    """Pipeline tem 3 steps: pre, dimred, clf; dimred é PCA."""
    pipe = build_dimred_pipeline("logreg", "pca", n_components=5)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps.keys()) == ["pre", "dimred", "clf"]
    assert isinstance(pipe.named_steps["dimred"], PCA)
    assert pipe.named_steps["dimred"].n_components == 5


def test_build_dimred_pipeline_lda_returns_3step_pipeline():
    """Pipeline tem 3 steps: pre, dimred, clf; dimred é LDA com n_components=1."""
    pipe = build_dimred_pipeline("logreg", "lda", n_components=1)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps.keys()) == ["pre", "dimred", "clf"]
    assert isinstance(pipe.named_steps["dimred"], LinearDiscriminantAnalysis)
    assert pipe.named_steps["dimred"].n_components == 1


def test_build_dimred_pipeline_invalid_method_raises():
    """dimred_method inválido levanta ValueError."""
    with pytest.raises(ValueError, match="pca.*lda"):
        build_dimred_pipeline("logreg", "tsne", n_components=2)


def test_compute_pca_n_components_returns_valid_int(small_X):
    """compute_pca_n_components retorna int >= 1 e <= n_features."""
    k = compute_pca_n_components(small_X, threshold=0.85)
    assert isinstance(k, int)
    assert 1 <= k <= small_X.shape[1]


def test_compute_pca_n_components_higher_threshold_gives_more_components(small_X):
    """threshold maior → mais componentes."""
    k85 = compute_pca_n_components(small_X, threshold=0.85)
    k95 = compute_pca_n_components(small_X, threshold=0.95)
    assert k95 >= k85


def test_dimred_pipeline_pca_fit_predict_smoke():
    """Pipeline PCA fit+predict_proba não levanta exceção em dados mínimos."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 40
    X_simple = pd.DataFrame(
        rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)]
    )
    y = np.array([0] * 20 + [1] * 20)
    pipe = SKPipeline(
        [
            ("pre", StandardScaler()),
            ("dimred", PCA(n_components=2, random_state=42)),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipe.fit(X_simple, y)
    proba = pipe.predict_proba(X_simple)
    assert proba.shape == (n, 2)
