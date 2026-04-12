"""Fingerprint de dataset: SHA-256 de arquivo e metadados do DataFrame."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def compute_file_sha256(path: str | Path) -> str:
    """Calcula SHA-256 do arquivo em blocos de 64 KB.

    Returns
    -------
    str : hex digest completo (64 chars).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(hexdigest: str, n: int = 8) -> str:
    """Retorna os primeiros n caracteres do hexdigest.

    Usado como DATAHASH8 no naming de runs MLflow (Parte 3).
    """
    return hexdigest[:n]


def compute_fingerprint(
    df: pd.DataFrame,
    *,
    file_path: str | Path | None = None,
) -> dict[str, Any]:
    """Retorna fingerprint do dataset como dict serializável.

    Parâmetros
    ----------
    df        : DataFrame já carregado (pós drop-ID).
    file_path : se fornecido, calcula file_sha256 e file_short do arquivo.

    Returns
    -------
    dict com:
      - file_sha256 (str, 64 chars) — se file_path fornecido
      - file_short  (str, 8 chars)  — DATAHASH8
      - n_rows, n_cols
      - columns (list[str])
      - dtypes (dict[str, str])
      - generated_at (str ISO 8601 UTC)
    """
    result: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if file_path is not None:
        sha = compute_file_sha256(file_path)
        result["file_sha256"] = sha
        result["file_short"] = short_hash(sha)
    else:
        result["file_sha256"] = None
        result["file_short"] = None
    return result


def save_fingerprint(fingerprint: dict[str, Any], path: str | Path) -> None:
    """Persiste o fingerprint como JSON com indent=2.

    Cria diretórios pai se necessário.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, ensure_ascii=False)


def load_fingerprint(path: str | Path) -> dict[str, Any]:
    """Carrega fingerprint JSON previamente salvo."""
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)
