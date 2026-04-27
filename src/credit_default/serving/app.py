"""FastAPI application for credit default prediction.

Endpoints:
  GET  /health          -> {"status": "ok", "model_uri": str}
  GET  /                -> model info (name, model_path, feature_count)
  POST /predict         -> single-record prediction
  POST /predict/batch   -> batch prediction

Integrity controls:
- Predictor loaded at startup; if load fails, server exits (no silent stub).
- /predict and /predict/batch return 503 if predictor not ready.
- 422 returned automatically by FastAPI/Pydantic for malformed payloads.
- 500 returned with error_id for unexpected internal errors.
"""

from __future__ import annotations

import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from credit_default.serving.predictor import MODEL_URI, Predictor
from credit_default.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model at startup; fail-fast if unavailable."""
    logger.info("Loading model from: %s", MODEL_URI)
    predictor.load()  # raises RuntimeError → server won't start if model missing
    logger.info("Model loaded successfully.")
    yield


app = FastAPI(
    title="Credit Default Prediction API",
    description="GradientBoosting model treinado na Parte 5 do projeto INFNET.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _check_ready() -> None:
    """Raise 503 if predictor not ready (defensive — lifespan should guarantee this)."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.get("/")
def info() -> dict:
    return {
        "model_name": "GradientBoosting (baseline)",
        "model_uri": MODEL_URI,
        "feature_count": 23,
        "run_id": "6be94912218a4c51bd8297ac77719b7f",
        "test_metrics": {
            "roc_auc": 0.7682,
            "f1_macro": 0.6876,
            "accuracy": 0.8218,
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    _check_ready()
    try:
        record_dict = request.record.model_dump()
        predictions = predictor.predict([record_dict])
        probas = predictor.predict_proba([record_dict])
        return PredictResponse(
            prediction=int(predictions[0]),
            probability_default=float(probas[0][1]),
            probability_no_default=float(probas[0][0]),
        )
    except Exception as exc:
        error_id = str(uuid.uuid4())
        logger.error("Predict error [%s]: %s", error_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error. error_id={error_id}") from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    _check_ready()
    if not request.records:
        raise HTTPException(status_code=422, detail="records list must not be empty.")
    try:
        records = [r.model_dump() for r in request.records]
        predictions = predictor.predict(records)
        probas = predictor.predict_proba(records)
        results = [
            PredictResponse(
                prediction=int(predictions[i]),
                probability_default=float(probas[i][1]),
                probability_no_default=float(probas[i][0]),
            )
            for i in range(len(records))
        ]
        return BatchPredictResponse(predictions=results)
    except HTTPException:
        raise
    except Exception as exc:
        error_id = str(uuid.uuid4())
        logger.error("Batch predict error [%s]: %s", error_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error. error_id={error_id}") from exc
