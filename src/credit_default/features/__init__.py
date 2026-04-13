"""Modulo de engenharia de features."""

from credit_default.features.dimred import (
    build_dimred_pipeline,
    compute_pca_n_components,
)
from credit_default.features.preprocessing import (
    CATEGORICAL,
    NUMERIC_CONTINUOUS,
    NUMERIC_ORDINAL,
    build_preprocessor,
)

__all__ = [
    "NUMERIC_CONTINUOUS",
    "NUMERIC_ORDINAL",
    "CATEGORICAL",
    "build_preprocessor",
    "build_dimred_pipeline",
    "compute_pca_n_components",
]
