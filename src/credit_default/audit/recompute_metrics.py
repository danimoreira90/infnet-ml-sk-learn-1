"""Auditor de integridade de métricas MLflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RecomputeResult:
    run_id: str
    ok: bool
    mismatches: dict = field(default_factory=dict)


def recompute_run_metrics(run_id: str, **kwargs):
    raise NotImplementedError


def _load_splits(parquet_path: Path, split_path: Path, *, include_test: bool = False):
    raise NotImplementedError


def _cast_param(v: str):
    raise NotImplementedError
