"""Testes para construcao de pipelines."""

import pytest
from sklearn.pipeline import Pipeline

from credit_default.models.pipeline import build_pipeline
from credit_default.models.registry import list_models


def test_build_pipeline_returns_pipeline():
    pipe = build_pipeline("logreg")
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_pre_and_clf_steps():
    pipe = build_pipeline("logreg")
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["pre", "clf"]


def test_pipeline_fit_predict_on_minimal_data(minimal_df):
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    y = minimal_df[target_col]
    pipe = build_pipeline("logreg")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_pipeline_predict_proba_available(minimal_df):
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    y = minimal_df[target_col]
    pipe = build_pipeline("logreg")
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_build_pipeline_all_models():
    for model_name in list_models():
        pipe = build_pipeline(model_name)
        assert isinstance(pipe, Pipeline), f"Falha para {model_name}"
