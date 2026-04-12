"""Smoke tests dos contratos publicos dos modulos de Parte 2."""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Teste 1: load_raw — shape e colunas esperados
# ---------------------------------------------------------------------------
def test_load_raw_shape_and_target(loaded_df: pd.DataFrame) -> None:
    """load_raw() deve retornar 30000 linhas, 24 colunas, sem coluna ID."""
    assert loaded_df.shape == (30000, 24), f"Shape inesperado: {loaded_df.shape}"
    assert "ID" not in loaded_df.columns, "Coluna ID nao deve estar presente apos load_raw"
    assert "default payment next month" in loaded_df.columns, "Target ausente"


# ---------------------------------------------------------------------------
# Teste 2: fingerprint — determinismo e tamanho do file_short
# ---------------------------------------------------------------------------
def test_fingerprint_stable_across_runs(minimal_df: pd.DataFrame, tmp_path) -> None:
    """compute_fingerprint() deve retornar resultado identico em duas chamadas."""
    from credit_default.data.fingerprint import compute_fingerprint

    # Salva minimal_df como parquet temporario para obter file_sha256
    tmp_file = tmp_path / "minimal.parquet"
    minimal_df.to_parquet(tmp_file, engine="pyarrow", index=False)

    fp1 = compute_fingerprint(minimal_df, file_path=tmp_file)
    fp2 = compute_fingerprint(minimal_df, file_path=tmp_file)

    assert fp1["n_rows"] == fp2["n_rows"]
    assert fp1["n_cols"] == fp2["n_cols"]
    assert fp1["columns"] == fp2["columns"]
    assert fp1["file_sha256"] == fp2["file_sha256"], "SHA256 deve ser determinístico"
    assert len(fp1["file_short"]) == 8, f"file_short deve ter 8 chars: '{fp1['file_short']}'"


# ---------------------------------------------------------------------------
# Teste 3: validate — passa sem excecao no dataset real (deduplicado)
# ---------------------------------------------------------------------------
def test_validate_passes_on_real_dataset(cleaned_df: pd.DataFrame) -> None:
    """validate() nao deve levantar excecao para o dataset limpo (warnings sao ok)."""
    from credit_default.data.schema import validate

    result = validate(cleaned_df, expected_rows=None)
    assert isinstance(result, list), "validate() deve retornar list[str]"


# ---------------------------------------------------------------------------
# Teste 4: validate — detecta casos fatais (NaN, duplicata, target ausente)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutation,description",
    [
        ("nan", "DataFrame com NaN deve levantar DataValidationError"),
        ("duplicate", "DataFrame com duplicata deve levantar DataValidationError"),
        ("no_target", "DataFrame sem target deve levantar DataValidationError"),
    ],
)
def test_validate_flags_constructed_bad_cases(
    minimal_df: pd.DataFrame,
    mutation: str,
    description: str,
) -> None:
    from credit_default.data.schema import DataValidationError, validate

    df = minimal_df.copy()
    target_col = "default payment next month"

    if mutation == "nan":
        df.iloc[0, 0] = float("nan")
    elif mutation == "duplicate":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    elif mutation == "no_target":
        df = df.drop(columns=[target_col])
        target_col = "coluna_que_nao_existe"

    with pytest.raises(DataValidationError):
        validate(df, expected_rows=None, target_col=target_col)


# ---------------------------------------------------------------------------
# Teste 5: make_splits — tamanhos e estratificacao
# ---------------------------------------------------------------------------
def test_splits_sizes_and_stratification(minimal_df: pd.DataFrame) -> None:
    """make_splits() deve respeitar ratios 70/15/15 e manter distribuicao do target."""
    from credit_default.data.splits import make_splits

    target_col = "default payment next month"
    n = len(minimal_df)
    train_idx, val_idx, test_idx = make_splits(minimal_df, target_col, seed=42)

    # Tamanhos aproximados (tolerancia 3%)
    assert abs(len(train_idx) / n - 0.70) < 0.03, f"train ratio: {len(train_idx)/n:.3f}"
    assert abs(len(val_idx) / n - 0.15) < 0.03, f"val ratio: {len(val_idx)/n:.3f}"
    assert abs(len(test_idx) / n - 0.15) < 0.03, f"test ratio: {len(test_idx)/n:.3f}"

    # Estratificacao: minority_ratio ~0.20 em cada split (tolerancia 4%)
    expected_minority = 0.20
    for split_idx, split_name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
        ratio = float(minimal_df.loc[split_idx, target_col].mean())
        assert (
            abs(ratio - expected_minority) < 0.04
        ), f"Ratio minoritario em {split_name}: {ratio:.3f} (esperado ~{expected_minority})"


# ---------------------------------------------------------------------------
# Teste 6: make_splits — disjuncao e cobertura total
# ---------------------------------------------------------------------------
def test_splits_indices_disjoint_and_cover_full_dataset(minimal_df: pd.DataFrame) -> None:
    """Splits devem ser disjuntos e cobrir todos os indices do DataFrame."""
    from credit_default.data.splits import make_splits

    target_col = "default payment next month"
    train_idx, val_idx, test_idx = make_splits(minimal_df, target_col, seed=42)

    assert set(train_idx) & set(val_idx) == set(), "train e val se sobrepoem"
    assert set(train_idx) & set(test_idx) == set(), "train e test se sobrepoem"
    assert set(val_idx) & set(test_idx) == set(), "val e test se sobrepoem"
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(
        minimal_df.index
    ), "union nao cobre o indice completo"
