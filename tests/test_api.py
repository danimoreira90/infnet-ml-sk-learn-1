"""Integration tests for the FastAPI app (contract tests only).

Uses monkeypatch to replace the Predictor with a dummy that always returns
prediction=0, probability_default=0.2, probability_no_default=0.8.

Does NOT test numerical correctness (validated in Parte 5).
Tests ONLY: HTTP status codes, response schemas, error handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_default.serving.predictor import FEATURE_COLUMNS

FULL_RECORD = {col: 0 for col in FEATURE_COLUMNS}
FULL_RECORD.update({"LIMIT_BAL": 30000, "AGE": 35, "SEX": 2})


class DummyPredictor:
    """Always returns prediction=0 with fixed probabilities."""

    is_ready = True

    def load(self) -> None:  # noqa: D401
        pass

    def predict(self, records):
        return np.array([0] * len(records))

    def predict_proba(self, records):
        return np.array([[0.8, 0.2]] * len(records))


@pytest.fixture()
def client(monkeypatch):
    """TestClient with DummyPredictor injected — no model on disk required."""
    import credit_default.serving.app as app_module

    dummy = DummyPredictor()
    monkeypatch.setattr(app_module, "predictor", dummy)

    # Override lifespan so it doesn't call predictor.load() against real disk
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(app):  # type: ignore[misc]
        yield

    monkeypatch.setattr(app_module.app, "router", app_module.app.router)
    # Re-create TestClient with the patched app (lifespan bypassed via test_client)
    from credit_default.serving.app import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# -- /health ------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_has_model_uri(self, client):
        resp = client.get("/health")
        assert "model_uri" in resp.json()


# -- / (info) -----------------------------------------------------------------


class TestInfo:
    def test_info_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_info_has_feature_count(self, client):
        resp = client.get("/")
        assert resp.json()["feature_count"] == 23


# -- /predict -----------------------------------------------------------------


class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json={"record": FULL_RECORD})
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json={"record": FULL_RECORD})
        body = resp.json()
        assert "prediction" in body
        assert "probability_default" in body
        assert "probability_no_default" in body

    def test_predict_prediction_is_int(self, client):
        resp = client.post("/predict", json={"record": FULL_RECORD})
        assert isinstance(resp.json()["prediction"], int)

    def test_predict_probabilities_sum_to_one(self, client):
        resp = client.post("/predict", json={"record": FULL_RECORD})
        body = resp.json()
        total = body["probability_default"] + body["probability_no_default"]
        assert abs(total - 1.0) < 1e-6

    def test_predict_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in FULL_RECORD.items() if k != "AGE"}
        resp = client.post("/predict", json={"record": incomplete})
        assert resp.status_code == 422

    def test_predict_wrong_type_returns_422(self, client):
        bad = {**FULL_RECORD, "LIMIT_BAL": "not_a_number"}
        resp = client.post("/predict", json={"record": bad})
        assert resp.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# -- /predict/batch -----------------------------------------------------------


class TestPredictBatch:
    def test_batch_returns_200(self, client):
        resp = client.post("/predict/batch", json={"records": [FULL_RECORD, FULL_RECORD]})
        assert resp.status_code == 200

    def test_batch_response_length_matches_input(self, client):
        resp = client.post("/predict/batch", json={"records": [FULL_RECORD, FULL_RECORD]})
        assert len(resp.json()["predictions"]) == 2

    def test_batch_empty_records_returns_422(self, client):
        resp = client.post("/predict/batch", json={"records": []})
        assert resp.status_code == 422
