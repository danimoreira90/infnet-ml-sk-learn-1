"""Treina os 5 modelos em modo baseline (sem tuning) e loga no MLflow.

Uso: uv run python scripts/train_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import mlflow  # noqa: E402

mlflow.set_tracking_uri((repo_root / "mlruns").as_uri())

from credit_default.models.registry import list_models  # noqa: E402
from credit_default.models.train import load_split_data, train_and_evaluate  # noqa: E402


def main() -> None:
    parquet_path = repo_root / "data" / "credit_card_cleaned.parquet"
    split_indices_path = repo_root / "artifacts" / "splits" / "split_indices.json"
    fingerprint_path = repo_root / "artifacts" / "data_fingerprint.json"

    X_train, X_val, _, y_train, y_val, _ = load_split_data(parquet_path, split_indices_path)

    with open(fingerprint_path) as f:
        fp = json.load(f)
    datahash8: str = fp["file_short"]

    githash7 = (
        subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], cwd=repo_root)
        .decode()
        .strip()
    )

    print(f"\n{'='*70}")
    print(f"Baseline Training | datahash8={datahash8} | git={githash7}")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'roc_auc [PRIMARY]':>18} {'f1_macro':>10} {'train_s':>8}")
    print(f"{'-'*55}")

    results = []
    for model_name in list_models():
        with tempfile.TemporaryDirectory() as tmp:
            result = train_and_evaluate(
                model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tune=False,
                datahash8=datahash8,
                githash7=githash7,
                tmp_dir=Path(tmp),
            )
        roc = result["metrics"]["roc_auc"]
        f1 = result["metrics"]["f1_macro"]
        print(f"{model_name:<15} {roc:>18.4f} {f1:>10.4f}")
        results.append((model_name, roc, f1))

    print(f"\n{'='*70}")
    print("Resumo final (ordenado por roc_auc desc — PRIMARY):")
    results.sort(key=lambda x: x[1], reverse=True)
    for model_name, roc, f1 in results:
        print(f"  {model_name:<15} roc_auc={roc:.4f}  f1_macro={f1:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
