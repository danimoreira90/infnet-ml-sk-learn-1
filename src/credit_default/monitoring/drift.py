"""Drift detection functions for data and model monitoring.

data_drift_report: per-feature statistical tests (KS for continuous,
    chi2 for categorical). Returns p-values and drift flags.

model_drift_report: flags roc_auc degradation > 5 percentage points
    vs a reference baseline.

No hardcoded always-pass thresholds. alpha is parametrizable (default 0.05).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

# Features by type (based on dataset schema)
_CONTINUOUS_FEATURES: List[str] = [
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

_CATEGORICAL_FEATURES: List[str] = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-feature drift statistics between reference and current datasets.

    Uses Kolmogorov-Smirnov for continuous features and chi2_contingency
    for categorical features.

    Args:
        reference_df: Historical data (e.g., train + val splits).
        current_df: Incoming data to test for drift (e.g., simulated test set).
        alpha: Significance level for drift_detected flag (default 0.05).

    Returns:
        Dict mapping feature name to:
            {
                "test": "ks" | "chi2",
                "statistic": float,
                "p_value": float,
                "drift_detected": bool,
            }
    """
    results: Dict[str, Dict[str, Any]] = {}

    for feature in _CONTINUOUS_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref = reference_df[feature].dropna().values
        cur = current_df[feature].dropna().values
        stat, p_value = ks_2samp(ref, cur)
        results[feature] = {
            "test": "ks",
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < alpha),
        }

    for feature in _CATEGORICAL_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref_counts = reference_df[feature].value_counts()
        cur_counts = current_df[feature].value_counts()
        # Align categories — fill missing with 0
        all_cats = ref_counts.index.union(cur_counts.index)
        ref_aligned = ref_counts.reindex(all_cats, fill_value=0).values
        cur_aligned = cur_counts.reindex(all_cats, fill_value=0).values
        contingency = [ref_aligned, cur_aligned]
        try:
            stat, p_value, _, _ = chi2_contingency(contingency)
        except ValueError:
            # Degenerate case (e.g., single category) — no drift detectable
            stat, p_value = 0.0, 1.0
        results[feature] = {
            "test": "chi2",
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < alpha),
        }

    return results


def model_drift_report(
    reference_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    roc_auc_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Flag model performance drift based on roc_auc degradation.

    Args:
        reference_metrics: Baseline metrics dict with at least "roc_auc".
        current_metrics: Current metrics dict with at least "roc_auc".
        roc_auc_threshold: Drop in roc_auc (absolute) that triggers drift flag.
            Default 0.05 (5 percentage points).

    Returns:
        {
            "reference_roc_auc": float,
            "current_roc_auc": float,
            "delta_roc_auc": float,       # current - reference (negative = degradation)
            "threshold": float,
            "drift_detected": bool,
        }
    """
    ref_auc = float(reference_metrics["roc_auc"])
    cur_auc = float(current_metrics["roc_auc"])
    delta = cur_auc - ref_auc
    return {
        "reference_roc_auc": ref_auc,
        "current_roc_auc": cur_auc,
        "delta_roc_auc": round(delta, 6),
        "threshold": roc_auc_threshold,
        "drift_detected": bool(delta < -roc_auc_threshold),
    }
