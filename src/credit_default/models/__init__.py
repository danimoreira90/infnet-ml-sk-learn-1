"""Modulo de modelos."""

from credit_default.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    list_models,
)

__all__ = ["MODEL_REGISTRY", "ModelSpec", "get_model_spec", "list_models"]
