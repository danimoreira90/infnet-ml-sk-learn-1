"""Fixtures compartilhadas para os testes da fundação de dados."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAW_PATH_ENV = "RAW_DATA_PATH"
_DEFAULT_RAW = (
    Path(__file__).resolve().parent.parent.parent / "data" / "default of credit card clients.xls"
)


@pytest.fixture(scope="session")
def raw_data_path() -> Path:
    p = Path(os.environ.get(RAW_PATH_ENV, str(_DEFAULT_RAW)))
    if not p.exists():
        pytest.skip(
            f"Dataset bruto nao encontrado em {p}. "
            "Defina RAW_DATA_PATH ou coloque o arquivo no path padrao."
        )
    return p


@pytest.fixture(scope="session")
def loaded_df(raw_data_path: Path) -> pd.DataFrame:
    from credit_default.data.ingest import load_raw

    return load_raw(raw_data_path)


@pytest.fixture(scope="session")
def cleaned_df(raw_data_path: Path) -> pd.DataFrame:
    """Dataset carregado, deduplicado e pronto para validate()."""
    from credit_default.data.ingest import load_raw

    df = load_raw(raw_data_path)
    return df.drop_duplicates()


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """DataFrame sintetico 500 linhas, distribuicao 80/20 (nao-default/default).

    Estrutura compativel com os modulos de Parte 2:
    - 23 features numericas + 1 target
    - Nenhum NaN, nenhuma duplicata
    """
    rng = np.random.default_rng(42)
    n = 500
    target = np.array([0] * 400 + [1] * 100)
    rng.shuffle(target)
    return pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(1, 5, n),
            "MARRIAGE": rng.integers(1, 4, n),
            "AGE": rng.integers(21, 79, n),
            **{f"PAY_{i}": rng.integers(-2, 9, n) for i in range(6)},
            **{f"BILL_AMT{i}": rng.integers(0, 100_000, n) for i in range(1, 7)},
            **{f"PAY_AMT{i}": rng.integers(0, 50_000, n) for i in range(1, 7)},
            "default payment next month": target,
        }
    )
