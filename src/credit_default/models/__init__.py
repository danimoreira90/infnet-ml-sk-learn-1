"""Modulo de modelos."""

from credit_default.models.pipeline import build_pipeline
from credit_default.models.train import load_split_data, train_and_evaluate
from credit_default.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    list_models,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_model_spec",
    "list_models",
    "build_pipeline",
    "load_split_data",
    "train_and_evaluate",
]
