"""Registro central de modelos e seus hyperparameter grids."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier


class ModelSpec(TypedDict):
    estimator: Any
    param_grid: dict[str, list]
    search_type: Literal["none", "grid", "random"]
    model_family: Literal["linear", "tree", "ensemble"]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "perceptron": {
        "estimator": CalibratedClassifierCV(
            Perceptron(random_state=42), cv=3, method="sigmoid"
        ),
        "param_grid": {
            "clf__estimator__max_iter": [300, 1000],
            "clf__estimator__eta0": [0.01, 0.1, 1.0],
            "clf__estimator__penalty": ["l2", "elasticnet"],
        },
        "search_type": "grid",
        "model_family": "linear",
    },
    "logreg": {
        "estimator": LogisticRegression(random_state=42),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "saga"],
            "clf__max_iter": [500],
        },
        "search_type": "grid",
        "model_family": "linear",
    },
    "dtree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "clf__max_depth": [3, 5, 10, None],
            "clf__min_samples_split": [2, 10, 50],
            "clf__criterion": ["gini", "entropy"],
        },
        "search_type": "grid",
        "model_family": "tree",
    },
    "rf": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 10],
            "clf__max_features": ["sqrt", "log2"],
        },
        "search_type": "random",
        "model_family": "ensemble",
    },
    "gb": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.8, 1.0],
        },
        "search_type": "random",
        "model_family": "ensemble",
    },
}


def get_model_spec(name: str) -> ModelSpec:
    """Retorna ModelSpec para o nome dado.

    Raises
    ------
    KeyError : se name nao existir no MODEL_REGISTRY.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Modelo '{name}' nao encontrado. Opcoes: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Retorna lista ordenada dos nomes dos modelos registrados."""
    return sorted(MODEL_REGISTRY.keys())
