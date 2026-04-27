"""Tests for Predictor class (mocked — no mlruns/ required).

Validates:
- Lazy loading semantics (model not loaded until needed)
- fail-fast on load failure
- predict / predict_proba output shapes
- is_ready flag
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from credit_default.serving.predictor import FEATURE_COLUMNS, Predictor

SAMPLE_RECORD = {col: 0 for col in FEATURE_COLUMNS}
SAMPLE_RECORD.update({"LIMIT_BAL": 30000, "AGE": 35, "SEX": 2})


class TestPredictorLazyLoad:
    def test_not_ready_before_load(self):
        p = Predictor(model_path="/fake/path")
        assert p.is_ready is False

    def test_ready_after_load(self):
        p = Predictor(model_path="/fake/path")
        mock_model = MagicMock()
        with patch("mlflow.sklearn.load_model", return_value=mock_model):
            p.load()
        assert p.is_ready is True

    def test_load_triggers_mlflow_call(self):
        p = Predictor(model_path="/fake/path")
        with patch("mlflow.sklearn.load_model") as mock_load:
            mock_load.return_value = MagicMock()
            p.load()
            mock_load.assert_called_once_with("/fake/path")


class TestPredictorFailFast:
    def test_load_failure_raises_runtime_error(self):
        p = Predictor(model_path="/nonexistent")
        with patch("mlflow.sklearn.load_model", side_effect=Exception("not found")):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                p.load()

    def test_is_ready_false_after_failed_load(self):
        p = Predictor(model_path="/nonexistent")
        with patch("mlflow.sklearn.load_model", side_effect=Exception("bad")):
            with pytest.raises(RuntimeError):
                p.load()
        assert p.is_ready is False


class TestPredictorInference:
    def _make_predictor(self) -> tuple[Predictor, MagicMock]:
        p = Predictor(model_path="/fake")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        with patch("mlflow.sklearn.load_model", return_value=mock_model):
            p.load()
        return p, mock_model

    def test_predict_returns_array(self):
        p, _ = self._make_predictor()
        result = p.predict([SAMPLE_RECORD, SAMPLE_RECORD])
        assert len(result) == 2

    def test_predict_proba_shape(self):
        p, _ = self._make_predictor()
        result = p.predict_proba([SAMPLE_RECORD, SAMPLE_RECORD])
        assert result.shape == (2, 2)

    def test_predict_auto_loads_if_not_ready(self):
        p = Predictor(model_path="/fake")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        with patch("mlflow.sklearn.load_model", return_value=mock_model):
            result = p.predict([SAMPLE_RECORD])
        assert p.is_ready is True
        assert len(result) == 1

    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS) == 23
