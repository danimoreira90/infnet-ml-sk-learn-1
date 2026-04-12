"""Pre-processamento de features para o pipeline de modelagem."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_CONTINUOUS: list[str] = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

NUMERIC_ORDINAL: list[str] = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

CATEGORICAL: list[str] = ["SEX", "EDUCATION", "MARRIAGE"]


def build_preprocessor() -> ColumnTransformer:
    """Retorna ColumnTransformer com 3 branches.

    - num_cont: StandardScaler para NUMERIC_CONTINUOUS
    - num_ord:  passthrough para NUMERIC_ORDINAL
    - cat:      OneHotEncoder(handle_unknown='ignore', sparse_output=False) para CATEGORICAL

    Returns
    -------
    ColumnTransformer configurado (nao fitado).
    """
    return ColumnTransformer(
        transformers=[
            ("num_cont", StandardScaler(), NUMERIC_CONTINUOUS),
            ("num_ord", "passthrough", NUMERIC_ORDINAL),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
