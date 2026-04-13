"""Testes para o registro de modelos."""

import pytest
from sklearn.calibration import CalibratedClassifierCV

from credit_default.models.registry import (
    MODEL_REGISTRY,
    get_model_spec,
    list_models,
)


def test_model_registry_has_5_models():
    assert len(MODEL_REGISTRY) == 5


def test_all_model_specs_have_required_keys():
    for name, spec in MODEL_REGISTRY.items():
        assert "estimator" in spec, f"{name} sem estimator"
        assert "param_grid" in spec, f"{name} sem param_grid"
        assert "search_type" in spec, f"{name} sem search_type"
        assert "model_family" in spec, f"{name} sem model_family"


def test_get_model_spec_returns_correct_type():
    spec = get_model_spec("logreg")
    assert "estimator" in spec


def test_get_model_spec_raises_key_error():
    with pytest.raises(KeyError):
        get_model_spec("inexistente")


def test_list_models_returns_sorted_list():
    models = list_models()
    assert models == sorted(models)
    assert len(models) == 5


def test_perceptron_has_calibrated_classifier():
    spec = get_model_spec("perceptron")
    assert isinstance(spec["estimator"], CalibratedClassifierCV)


def test_perceptron_param_grid_uses_estimator_prefix():
    spec = get_model_spec("perceptron")
    for key in spec["param_grid"]:
        assert key.startswith("clf__estimator__"), f"Key {key} sem prefixo correto"


def test_all_estimators_have_predict_proba():
    for name, spec in MODEL_REGISTRY.items():
        assert hasattr(spec["estimator"], "predict_proba"), f"{name} sem predict_proba"
