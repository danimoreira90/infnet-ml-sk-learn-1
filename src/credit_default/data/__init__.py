"""Módulo de dados do projeto credit_default."""

from credit_default.data.diagnostics import run_all_diagnostics
from credit_default.data.fingerprint import (
    compute_fingerprint,
    save_fingerprint,
    short_hash,
)
from credit_default.data.ingest import load_cleaned, load_config, load_raw
from credit_default.data.schema import DataValidationError, save_schema, validate
from credit_default.data.splits import make_splits, save_split_indices, verify_splits

__all__ = [
    "load_raw",
    "load_cleaned",
    "load_config",
    "compute_fingerprint",
    "save_fingerprint",
    "short_hash",
    "validate",
    "DataValidationError",
    "save_schema",
    "make_splits",
    "verify_splits",
    "save_split_indices",
    "run_all_diagnostics",
]
