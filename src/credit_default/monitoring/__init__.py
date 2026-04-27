"""Monitoring package: data drift and model drift detection."""

from credit_default.monitoring.drift import data_drift_report, model_drift_report

__all__ = ["data_drift_report", "model_drift_report"]
