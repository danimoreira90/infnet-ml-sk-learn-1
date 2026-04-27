"""Unit tests for drift detection functions.

Uses synthetic DataFrames — no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_default.monitoring.drift import (
    _CATEGORICAL_FEATURES,
    _CONTINUOUS_FEATURES,
    data_drift_report,
    model_drift_report,
)

RNG = np.random.default_rng(42)


def _make_df(n: int = 500, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Synthetic DataFrame with all 23 features."""
    rng = np.random.default_rng(seed)
    data = {}
    for feat in _CONTINUOUS_FEATURES:
        data[feat] = rng.normal(loc=50_000 + shift, scale=10_000, size=n).astype(int)
    for feat in _CATEGORICAL_FEATURES:
        data[feat] = rng.choice([0, 1, 2, 3], size=n)
    return pd.DataFrame(data)


class TestDataDriftReport:
    def test_returns_dict_with_all_features(self):
        ref = _make_df()
        cur = _make_df()
        result = data_drift_report(ref, cur)
        expected = set(_CONTINUOUS_FEATURES) | set(_CATEGORICAL_FEATURES)
        assert set(result.keys()) == expected

    def test_each_entry_has_required_keys(self):
        ref = _make_df()
        cur = _make_df()
        result = data_drift_report(ref, cur)
        for feat, entry in result.items():
            assert "test" in entry, feat
            assert "statistic" in entry, feat
            assert "p_value" in entry, feat
            assert "drift_detected" in entry, feat

    def test_continuous_features_use_ks(self):
        ref = _make_df()
        cur = _make_df()
        result = data_drift_report(ref, cur)
        for feat in _CONTINUOUS_FEATURES:
            assert result[feat]["test"] == "ks"

    def test_categorical_features_use_chi2(self):
        ref = _make_df()
        cur = _make_df()
        result = data_drift_report(ref, cur)
        for feat in _CATEGORICAL_FEATURES:
            assert result[feat]["test"] == "chi2"

    def test_no_drift_on_identical_data(self):
        ref = _make_df(seed=1)
        result = data_drift_report(ref, ref)
        # Identical data: all p_values should be 1.0 (no drift)
        for feat, entry in result.items():
            assert (
                entry["drift_detected"] is False
            ), f"{feat}: p={entry['p_value']:.4f} flagged on identical data"

    def test_drift_detected_on_large_shift(self):
        ref = _make_df(n=1000, seed=0, shift=0.0)
        cur = _make_df(n=1000, seed=1, shift=500_000.0)  # massive shift
        result = data_drift_report(ref, cur)
        drifted = [f for f, e in result.items() if e["drift_detected"] and e["test"] == "ks"]
        assert len(drifted) > 0, "Expected KS drift on large-shifted data"

    def test_p_value_is_float(self):
        ref = _make_df()
        cur = _make_df()
        result = data_drift_report(ref, cur)
        for entry in result.values():
            assert isinstance(entry["p_value"], float)

    def test_drift_detected_is_bool(self):
        ref = _make_df()
        cur = _make_df()
        result = data_drift_report(ref, cur)
        for entry in result.values():
            assert isinstance(entry["drift_detected"], bool)

    def test_alpha_parametrizable(self):
        ref = _make_df(n=1000, seed=0)
        cur = _make_df(n=1000, seed=99)  # slightly different
        result_strict = data_drift_report(ref, cur, alpha=1e-10)
        result_loose = data_drift_report(ref, cur, alpha=1.0)
        strict_count = sum(e["drift_detected"] for e in result_strict.values())
        loose_count = sum(e["drift_detected"] for e in result_loose.values())
        assert loose_count >= strict_count

    def test_missing_column_skipped(self):
        ref = _make_df()[_CONTINUOUS_FEATURES[:3]]
        cur = _make_df()[_CONTINUOUS_FEATURES[:3]]
        # Should not raise, just skip absent features
        result = data_drift_report(ref, cur)
        assert set(result.keys()) == set(_CONTINUOUS_FEATURES[:3])


class TestModelDriftReport:
    def test_no_drift_when_auc_stable(self):
        ref = {"roc_auc": 0.77}
        cur = {"roc_auc": 0.77}
        result = model_drift_report(ref, cur)
        assert result["drift_detected"] is False

    def test_drift_detected_when_auc_drops_more_than_threshold(self):
        ref = {"roc_auc": 0.77}
        cur = {"roc_auc": 0.70}  # drop of 0.07 > threshold 0.05
        result = model_drift_report(ref, cur)
        assert result["drift_detected"] is True

    def test_no_drift_when_auc_drop_below_threshold(self):
        ref = {"roc_auc": 0.77}
        cur = {"roc_auc": 0.73}  # drop of 0.04, below threshold of 0.05
        result = model_drift_report(ref, cur)
        assert result["drift_detected"] is False

    def test_no_drift_when_auc_improves(self):
        ref = {"roc_auc": 0.70}
        cur = {"roc_auc": 0.80}
        result = model_drift_report(ref, cur)
        assert result["drift_detected"] is False

    def test_delta_calculated_correctly(self):
        ref = {"roc_auc": 0.77}
        cur = {"roc_auc": 0.70}
        result = model_drift_report(ref, cur)
        assert abs(result["delta_roc_auc"] - (-0.07)) < 1e-6

    def test_custom_threshold(self):
        ref = {"roc_auc": 0.77}
        cur = {"roc_auc": 0.74}  # drop 0.03
        result_strict = model_drift_report(ref, cur, roc_auc_threshold=0.02)
        result_loose = model_drift_report(ref, cur, roc_auc_threshold=0.05)
        assert result_strict["drift_detected"] is True
        assert result_loose["drift_detected"] is False

    def test_result_contains_all_keys(self):
        ref = {"roc_auc": 0.77}
        cur = {"roc_auc": 0.75}
        result = model_drift_report(ref, cur)
        for key in (
            "reference_roc_auc",
            "current_roc_auc",
            "delta_roc_auc",
            "threshold",
            "drift_detected",
        ):
            assert key in result
