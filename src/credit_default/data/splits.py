"""Split determinístico e artefato de índices reutilizável."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    df: pd.DataFrame,
    target_col: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Retorna (train_idx, val_idx, test_idx) estratificados pelo target.

    Estratégia de dois passos:
      1. Separa test (test_ratio) estratificado do total.
      2. Do restante, separa val (val_ratio / (1 - test_ratio)) estratificado.
      3. O que sobra é train.

    Returns
    -------
    Três pd.Index disjuntos cuja união cobre o índice completo de df.

    Raises
    ------
    ValueError : se os ratios não somarem 1.0 (tolerância 1e-9) ou forem negativos.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios devem somar 1.0, mas somam {total}")
    for name, val in [
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ]:
        if val < 0:
            raise ValueError(f"{name} não pode ser negativo: {val}")

    y = df[target_col]

    # Passo 1: separa test
    idx_rest, idx_test = train_test_split(
        df.index,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )

    # Passo 2: separa val do restante
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    y_rest = y.loc[idx_rest]
    idx_train, idx_val = train_test_split(
        idx_rest,
        test_size=val_ratio_adj,
        stratify=y_rest,
        random_state=seed,
    )

    return idx_train, idx_val, idx_test


def verify_splits(
    df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
    *,
    tolerance: float = 0.02,
) -> None:
    """Valida disjunção, cobertura total e ratios aproximados.

    Raises
    ------
    AssertionError : se qualquer condição falhar.
    """
    n = len(df)
    # Disjunção
    assert len(set(train_idx) & set(val_idx)) == 0, "train e val se sobrepõem"
    assert len(set(train_idx) & set(test_idx)) == 0, "train e test se sobrepõem"
    assert len(set(val_idx) & set(test_idx)) == 0, "val e test se sobrepõem"
    # Cobertura total
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(
        df.index
    ), "union não cobre o índice completo"
    # Ratios aproximados
    assert (
        abs(len(train_idx) / n - 0.70) <= tolerance
    ), f"train ratio fora da tolerância: {len(train_idx)/n:.3f}"
    assert (
        abs(len(val_idx) / n - 0.15) <= tolerance
    ), f"val ratio fora da tolerância: {len(val_idx)/n:.3f}"
    assert (
        abs(len(test_idx) / n - 0.15) <= tolerance
    ), f"test ratio fora da tolerância: {len(test_idx)/n:.3f}"


def save_split_indices(
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
    *,
    seed: int,
    file_sha256: str,
    fingerprint_short: str,
    path: str | Path,
) -> None:
    """Persiste índices como JSON com metadados de auditoria.

    Campos:
      - seed, file_sha256 (64 chars), fingerprint_short (8 chars)
      - n_train, n_val, n_test
      - train_idx, val_idx, test_idx (lists of int)
      - generated_at (ISO 8601 UTC)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "file_sha256": file_sha256,
        "fingerprint_short": fingerprint_short,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
