"""Predictor: encapsulates MLflow model loading and inference.

MODEL_PATH is the single source of truth for the model artifact location.
It can be overridden via the MODEL_PATH environment variable for Docker.

Integrity controls:
- Lazy loading: model is loaded on first predict call (or explicit load()).
- Fail-fast: if load fails, raises RuntimeError immediately — no silent fallback.
- is_ready property: callers can check before predicting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Default path relative to project root (works locally and in Docker after COPY)
_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[4]
    / "mlruns"
    / "236665223173386020"
    / "models"
    / "m-4de1a2c47e7d40d9a679a40ba79c9c65"
    / "artifacts"
)

MODEL_PATH: str = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)

FEATURE_COLUMNS: List[str] = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


class Predictor:
    """Wraps the GradientBoosting model with lazy loading and fail-fast semantics."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or MODEL_PATH
        self._model = None

    def load(self) -> None:
        """Load the model from disk. Raises RuntimeError on failure."""
        import mlflow.sklearn

        try:
            self._model = mlflow.sklearn.load_model(self._model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from '{self._model_path}': {exc}"
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
