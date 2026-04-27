"""Pydantic schemas for the credit default prediction API.

All 23 features derived from MLmodel signature (type: long -> int).
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict


class CreditRecord(BaseModel):
    """Single credit card holder record with 23 features."""

    model_config = ConfigDict(strict=True)

    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int


class PredictRequest(BaseModel):
    """Single-record prediction request."""

    model_config = ConfigDict(strict=True)

    record: CreditRecord


class PredictResponse(BaseModel):
    """Prediction result for a single record."""

    prediction: int
    probability_default: float
    probability_no_default: float


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(strict=True)

    records: List[CreditRecord]


class BatchPredictResponse(BaseModel):
    """Prediction results for a batch of records."""

    predictions: List[PredictResponse]
