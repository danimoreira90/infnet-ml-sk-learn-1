"""Reducao de dimensionalidade: PCA e LDA integrados ao pipeline sklearn."""
from __future__ import annotations

import copy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from credit_default.features.preprocessing import build_preprocessor
from credit_default.models.registry import get_model_spec


def compute_pca_n_components(
    X_transformed: np.ndarray,
    threshold: float = 0.85,
) -> int:
    """Retorna menor k tal que sum(explained_variance_ratio_[:k]) >= threshold.

    Parameters
    ----------
    X_transformed : array numpy ja escalado (saida de pre.transform()).
    threshold     : variancia acumulada minima (default 0.85).

    Returns
    -------
    int : numero de componentes PCA, entre 1 e n_features.
    """
    pca_full = PCA(random_state=42)
    pca_full.fit(X_transformed)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, X_transformed.shape[1])


def build_dimred_pipeline(
    model_name: str,
    dimred_method: str,
    n_components: int,
    *,
    seed: int = 42,
) -> Pipeline:
    """Constroi Pipeline de 3 etapas: pre -> dimred -> clf.

    Parameters
    ----------
    model_name    : chave do MODEL_REGISTRY (perceptron, logreg, dtree, rf, gb)
    dimred_method : "pca" ou "lda"
    n_components  : numero de componentes (PCA: k>=1; LDA binario: sempre 1)
    seed          : random_state para PCA e estimador

    Returns
    -------
    sklearn.pipeline.Pipeline com steps ["pre", "dimred", "clf"]

    Raises
    ------
    ValueError : se dimred_method nao for "pca" ou "lda"
    """
    spec = get_model_spec(model_name)
    estimator = copy.deepcopy(spec["estimator"])

    pre = build_preprocessor()

    if dimred_method == "pca":
        dimred_step: PCA | LinearDiscriminantAnalysis = PCA(
            n_components=n_components, random_state=seed
        )
    elif dimred_method == "lda":
        dimred_step = LinearDiscriminantAnalysis(n_components=n_components)
    else:
        raise ValueError(
            f"dimred_method deve ser 'pca' ou 'lda', recebido: {dimred_method!r}"
        )

    return Pipeline(
        steps=[
            ("pre", pre),
            ("dimred", dimred_step),
            ("clf", estimator),
        ]
    )
