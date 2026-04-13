"""Treina 5 modelos com 3 configs de dimred: pca_k10, pca_k15, lda_k1 = 15 runs.

Protecoes de integridade:
- Guard 1: data fingerprint verificado antes de qualquer run (sys.exit(1) em mismatch)
- Guard 2: params MLflow verificados apos cada run (sys.exit(1) se params ausentes)

NAO deleta runs da Parte 3. Adiciona 15 runs ao experimento existente.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from credit_default.models.train import load_split_data, train_dimred_and_evaluate  # noqa: E402
from credit_default.tracking.mlflow_utils import get_or_create_experiment  # noqa: E402

# ─── Constantes ────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "infnet-ml-sistema"
MODELS = ["perceptron", "logreg", "dtree", "rf", "gb"]
DIMRED_CONFIGS: list[tuple[str, int, str]] = [
    ("pca", 10, "pca_k10"),
    ("pca", 15, "pca_k15"),
    ("lda", 1, "lda_k1"),
]
REQUIRED_PARAMS = {
    "model_name",
    "seed",
    "cv_folds",
    "dimred_method",
    "dimred_n_components",
    "scoring_primary",
    "search_type",
}


# ─── Guards de Integridade ─────────────────────────────────────────────────────


def _verify_data_fingerprint() -> str:
    """Verifica SHA-256 do parquet contra artifacts/data_fingerprint.json.

    sys.exit(1) em mismatch.
    """
    fp_path = repo_root / "artifacts" / "data_fingerprint.json"
    with open(fp_path) as f:
        fp = json.load(f)
    expected = fp["file_short"]  # "30c6be3a"

    parquet_path = repo_root / "artifacts" / "data" / "credit_card_cleaned.parquet"
    actual = hashlib.sha256(parquet_path.read_bytes()).hexdigest()[:8]

    if actual != expected:
        print(
            f"ERRO INTEGRIDADE: fingerprint mismatch! " f"expected={expected}, actual={actual}",
            flush=True,
        )
        sys.exit(1)

    print(f"[OK] fingerprint={expected}", flush=True)
    return expected


def _assert_params_not_empty(client: MlflowClient, run_id: str) -> None:
    """Verifica params MLflow obrigatorios. sys.exit(1) se ausentes."""
    params = client.get_run(run_id).data.params
    missing = REQUIRED_PARAMS - set(params.keys())
    if missing:
        print(
            f"ERRO INTEGRIDADE: params ausentes no run {run_id}: {missing}",
            flush=True,
        )
        sys.exit(1)


def _get_baseline_run_id(client: MlflowClient, experiment_id: str, model_name: str) -> str:
    """Busca run_id do baseline P3 equivalente (stage=baseline, mesmo model_name).

    Retorna "" se nao encontrado (nao falha).
    """
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(f"tags.stage = 'baseline' AND params.model_name = '{model_name}'"),
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0].info.run_id if runs else ""
    except Exception:
        return ""


def _get_githash7() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ─── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Guard 1: data fingerprint
    datahash8 = _verify_data_fingerprint()
    githash7 = _get_githash7()

    # Tracking URI Windows-safe
    mlflow.set_tracking_uri((repo_root / "mlruns").absolute().as_uri())
    client = MlflowClient()
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    # Dados (carregados uma vez)
    parquet_path = repo_root / "artifacts" / "data" / "credit_card_cleaned.parquet"
    split_path = repo_root / "artifacts" / "splits" / "split_indices.json"
    X_train, X_val, _X_test, y_train, y_val, _y_test = load_split_data(parquet_path, split_path)

    print(f"\n{'=' * 70}")
    print(f"Dimred Training | datahash8={datahash8} | git={githash7}")
    print(f"Configs: {[c[2] for c in DIMRED_CONFIGS]} | Modelos: {MODELS}")
    print(f"{'=' * 70}")

    results: list[dict[str, Any]] = []

    for dimred_method, n_components, dimred_label in DIMRED_CONFIGS:
        print(f"\n--- {dimred_label.upper()} ---")
        for model_name in MODELS:
            baseline_rid = _get_baseline_run_id(client, experiment_id, model_name)
            with tempfile.TemporaryDirectory() as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                result = train_dimred_and_evaluate(
                    model_name,
                    dimred_method,
                    n_components,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    seed=42,
                    cv_folds=5,
                    experiment_id=experiment_id,
                    datahash8=datahash8,
                    githash7=githash7,
                    tmp_dir=tmp_dir,
                    baseline_run_id=baseline_rid,
                )

            # Guard 2: params nao-vazios
            _assert_params_not_empty(client, result["run_id"])

            ev = result["dimred_explained_variance"]
            ev_str = f"{ev:.4f}" if isinstance(ev, float) else ev
            print(
                f"  {model_name:<12} roc_auc={result['metrics']['roc_auc']:.4f}"
                f"  f1_macro={result['metrics']['f1_macro']:.4f}"
                f"  ev={ev_str}",
                flush=True,
            )
            results.append(
                {
                    "model": model_name,
                    "dimred": dimred_label,
                    "roc_auc": result["metrics"]["roc_auc"],
                    "f1_macro": result["metrics"]["f1_macro"],
                }
            )

    # Resumo final
    print(f"\n{'=' * 70}")
    print("Resumo final (ordenado por roc_auc desc — PRIMARY):")
    for r in sorted(results, key=lambda x: x["roc_auc"], reverse=True):
        print(
            f"  {r['dimred']:<10} {r['model']:<12}"
            f" roc_auc={r['roc_auc']:.4f}  f1_macro={r['f1_macro']:.4f}"
        )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
