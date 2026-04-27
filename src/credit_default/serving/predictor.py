"""Predictor: encapsulates MLflow model loading and inference.

MODEL_URI is the single source of truth for the model artifact location.
TRACKING_URI points to the local mlruns/ directory (or override via env var).

Both can be overridden via environment variables for Docker:
  MODEL_URI            -> defaults to "models:/m-4de1a2c47e7d40d9a679a40ba79c9c65"
  MLFLOW_TRACKING_URI  -> defaults to file:///abs/path/to/mlruns

Integrity controls:
- Lazy loading: model is loaded on first predict call (or explicit load()).
- Fail-fast: if load fails, raises RuntimeError immediately — no silent fallback.
- is_ready property: callers can check before predicting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Canonical model URI (registered in the local model registry)
_DEFAULT_MODEL_URI = "models:/m-4de1a2c47e7d40d9a679a40ba79c9c65"

# Tracking URI: file:///abs/path/to/mlruns (POSIX-safe, works on Windows and Docker)
# parents[3]: src/credit_default/serving/predictor.py -> repo root
_DEFAULT_TRACKING_URI: str = (Path(__file__).resolve().parents[3] / "mlruns").as_uri()

MODEL_URI: str = os.environ.get("MODEL_URI", _DEFAULT_MODEL_URI)
TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)

FEATURE_COLUMNS: List[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


class Predictor:
    """Wraps the GradientBoosting model with lazy loading and fail-fast semantics."""

    def __init__(
        self,
        model_uri: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self._model_uri = model_uri or MODEL_URI
        self._tracking_uri = tracking_uri or TRACKING_URI
        self._model = None

    def load(self) -> None:
        """Load the model. Raises RuntimeError on failure.

        When MODEL_URI starts with "models:" (registry URI), sets the tracking
        URI first so MLflow can resolve the registry. When MODEL_URI is a
        filesystem path (e.g., in Docker), skips set_tracking_uri — the path
        is already absolute POSIX and needs no registry resolution.
        """
        try:
            if self._model_uri.startswith("models:"):
                mlflow.set_tracking_uri(self._tracking_uri)
            self._model = mlflow.sklearn.load_model(self._model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self._model_uri}' "
                f"(tracking_uri='{self._tracking_uri}'): {exc}"
            ) from exc

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def _to_dataframe(self, records: List[dict]) -> pd.DataFrame:
        return pd.DataFrame(records, columns=FEATURE_COLUMNS)[FEATURE_COLUMNS]

    def predict(self, records: List[dict]) -> np.ndarray:
        """Return binary predictions (0/1) for a list of record dicts."""
        self._ensure_loaded()
        df = self._to_dataframe(records)
        return self._model.predict(df)  # type: ignore[union-attr]

    def predict_proba(self, records: List[dict]) -> np.ndarray:
        """Return probability arrays [[p_no_default, p_default], ...] for records."""
        self._ensure_loaded()
        df = self._to_dataframe(records)
        return self._model.predict_proba(df)  # type: ignore[union-attr]
