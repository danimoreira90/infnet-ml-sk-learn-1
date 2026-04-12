"""Construcao de pipelines sklearn com preprocessor + modelo."""

from __future__ import annotations

import sklearn
from sklearn.pipeline import Pipeline

from credit_default.features.preprocessing import build_preprocessor
from credit_default.models.registry import get_model_spec

# sklearn >= 1.2 suporta set_output; verificamos suporte a set_params com random_state
_SKLEARN_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def build_pipeline(model_name: str, *, seed: int = 42) -> Pipeline:
    """Retorna Pipeline([('pre', build_preprocessor()), ('clf', estimator)]).

    O estimator e obtido de MODEL_REGISTRY[model_name].
    Se o estimator aceitar random_state, ele e setado com seed via set_params.

    Parameters
    ----------
    model_name : chave do MODEL_REGISTRY (perceptron, logreg, dtree, rf, gb)
    seed       : semente para reprodutibilidade

    Returns
    -------
    Pipeline com steps 'pre' e 'clf'

    Raises
    ------
    KeyError : se model_name nao existir no MODEL_REGISTRY.
    """
    spec = get_model_spec(model_name)
    estimator = spec["estimator"]

    # Tenta setar random_state se o estimator suportar
    try:
        estimator.set_params(random_state=seed)
    except ValueError:
        pass  # estimator nao suporta random_state (ex: Perceptron encapsulado)

    return Pipeline(steps=[("pre", build_preprocessor()), ("clf", estimator)])
