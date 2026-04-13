"""Testes para o modulo de preprocessing."""

import numpy as np
from sklearn.compose import ColumnTransformer

from credit_default.features.preprocessing import (
    CATEGORICAL,
    NUMERIC_CONTINUOUS,
    NUMERIC_ORDINAL,
    build_preprocessor,
)


def test_build_preprocessor_returns_column_transformer():
    """build_preprocessor() retorna ColumnTransformer."""
    ct = build_preprocessor()
    assert isinstance(ct, ColumnTransformer)


def test_preprocessor_has_three_transformers():
    """ColumnTransformer tem 3 transformers: num_cont, num_ord, cat."""
    ct = build_preprocessor()
    names = [name for name, _, _ in ct.transformers]
    assert names == ["num_cont", "num_ord", "cat"]


def test_preprocessor_fit_transform_shape(minimal_df):
    """fit_transform no minimal_df produz shape correto."""
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    ct = build_preprocessor()
    X_out = ct.fit_transform(X)
    # 14 scaled + 6 passthrough + num_onehot categories
    assert X_out.shape[0] == len(X)
    assert X_out.shape[1] > 20  # pelo menos 14+6+3 = 23


def test_preprocessor_no_nan_in_output(minimal_df):
    """Output do preprocessor nao contem NaN."""
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    ct = build_preprocessor()
    X_out = ct.fit_transform(X)
    assert not np.isnan(X_out).any()


def test_column_lists_cover_23_features():
    """NUMERIC_CONTINUOUS + NUMERIC_ORDINAL + CATEGORICAL == 23 features."""
    all_cols = NUMERIC_CONTINUOUS + NUMERIC_ORDINAL + CATEGORICAL
    assert len(all_cols) == 23
    assert len(set(all_cols)) == 23  # sem duplicatas


def test_preprocessor_handles_unknown_categories(minimal_df):
    """OneHotEncoder com handle_unknown='ignore' nao falha com categorias novas."""
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    ct = build_preprocessor()
    ct.fit(X)

    # Cria dados com categorias novas em SEX
    X_new = X.copy()
    X_new["SEX"] = 99  # categoria desconhecida
    result = ct.transform(X_new)
    assert not np.isnan(result).any()
