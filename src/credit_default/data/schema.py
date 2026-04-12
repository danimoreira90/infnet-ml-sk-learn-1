"""Validação de schema e qualidade do dataset."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Levantada para erros fatais de validação (estrutura, missing, duplicatas)."""


def validate(
    df: pd.DataFrame,
    *,
    expected_rows: int | None = 30000,
    expected_cols: int = 24,
    target_col: str = "default payment next month",
    education_valid_codes: list[int] | None = None,
    marriage_valid_codes: list[int] | None = None,
) -> list[str]:
    """Valida estrutura e qualidade do DataFrame pós-ingestão.

    Erros FATAIS → levanta DataValidationError imediatamente:
      - n_rows != expected_rows (pulado quando expected_rows=None)
      - n_cols != expected_cols
      - target_col ausente
      - qualquer NaN no DataFrame
      - linhas exatas duplicadas

    Warnings NÃO-FATAIS → retornados como list[str], DataFrame NÃO mutado:
      - EDUCATION contém códigos fora de education_valid_codes
      - MARRIAGE contém códigos fora de marriage_valid_codes

    Returns
    -------
    list[str] : lista de mensagens de warning (vazia se nenhuma anomalia).
    """
    if education_valid_codes is None:
        education_valid_codes = [1, 2, 3, 4]
    if marriage_valid_codes is None:
        marriage_valid_codes = [1, 2, 3]

    warnings: list[str] = []

    # --- Erros fatais ---
    if expected_rows is not None and df.shape[0] != expected_rows:
        raise DataValidationError(
            f"Número de linhas incorreto: esperado {expected_rows}, encontrado {df.shape[0]}"
        )
    if df.shape[1] != expected_cols:
        raise DataValidationError(
            f"Número de colunas incorreto: esperado {expected_cols}, encontrado {df.shape[1]}"
        )
    if target_col not in df.columns:
        raise DataValidationError(
            f"Coluna alvo '{target_col}' não encontrada. Colunas: {list(df.columns)}"
        )
    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        raise DataValidationError(f"DataFrame contém {n_missing} valor(es) ausente(s) (NaN).")
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        raise DataValidationError(
            f"DataFrame contém {n_duplicates} linha(s) exata(s) duplicada(s)."
        )

    # --- Warnings não-fatais ---
    if "EDUCATION" in df.columns:
        anomalous = sorted(set(df["EDUCATION"].unique()) - set(education_valid_codes))
        if anomalous:
            msg = (
                f"EDUCATION contém códigos fora da especificação UCI "
                f"{education_valid_codes}: {anomalous}. "
                "Dados preservados sem recodificação."
            )
            warnings.append(msg)
            logger.warning(msg)

    if "MARRIAGE" in df.columns:
        anomalous = sorted(set(df["MARRIAGE"].unique()) - set(marriage_valid_codes))
        if anomalous:
            msg = (
                f"MARRIAGE contém códigos fora da especificação UCI "
                f"{marriage_valid_codes}: {anomalous}. "
                "Dados preservados sem recodificação."
            )
            warnings.append(msg)
            logger.warning(msg)

    return warnings


def save_schema(
    df: pd.DataFrame,
    path: str | Path,
    *,
    warnings: list[str] | None = None,
) -> None:
    """Persiste schema como JSON.

    Campos:
      - columns, dtypes, shape, warnings, generated_at
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": list(df.shape),
        "warnings": warnings or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
