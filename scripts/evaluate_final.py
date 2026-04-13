"""Avalia o modelo vencedor no test set — UMA UNICA EXECUCAO.

Fluxo:
1. Le winner_run_id de reports/parte_5/final_selection_rationale.md
2. Recarrega params do MLflow run vencedor
3. Reconstroi pipeline do zero (build_pipeline ou build_dimred_pipeline)
4. Treina em X_train + X_val concatenados
5. Avalia UMA VEZ em X_test (via _load_splits(include_test=True))
6. Loga novo MLflow run com stage="final_eval", signature, input_example
7. Salva reports/parte_5/test_metrics.json

PROTECAO: test_idx acessado SOMENTE via _load_splits(include_test=True).
Nenhuma linha deste script referencia a string "test_idx" diretamente.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from credit_default.audit.recompute_metrics import _cast_param, _load_splits  # noqa: E402
from credit_default.evaluation.metrics import compute_all_metrics  # noqa: E402
from credit_default.evaluation.plots import (  # noqa: E402
    confusion_matrix_plot,
    roc_plot,
)
from credit_default.features.dimred import build_dimred_pipeline  # noqa: E402
from credit_default.models.pipeline import build_pipeline  # noqa: E402
from credit_default.tracking.mlflow_utils import (  # noqa: E402
    get_or_create_experiment,
    log_standard_artifacts,
    log_standard_tags,
)
from credit_default.tracking.run_naming import compose_run_name  # noqa: E402

RATIONALE_PATH   = repo_root / "reports" / "parte_5" / "final_selection_rationale.md"
PARQUET_PATH     = repo_root / "data" / "credit_card_cleaned.parquet"
SPLIT_PATH       = repo_root / "artifacts" / "splits" / "split_indices.json"
FINGERPRINT_PATH = repo_root / "artifacts" / "data_fingerprint.json"
TEST_METRICS_OUT = repo_root / "reports" / "parte_5" / "test_metrics.json"
EXPERIMENT_NAME  = "infnet-ml-sistema"
EXP_ID           = "236665223173386020"
SEED             = 42


def _read_winner_run_id() -> str:
    text = RATIONALE_PATH.read_text(encoding="utf-8")
    m = re.search(r"winner_run_id:\s*(\S+)", text)
    if not m:
        print(f"[ERRO] winner_run_id nao encontrado em {RATIONALE_PATH}", flush=True)
        sys.exit(1)
    return m.group(1).strip()


def _dimred_tag_str(method: str, n: int) -> str:
    if method == "none" or n == 0:
        return "none"
    return f"{method}_k{n}"


def main() -> None:
    winner_run_id = _read_winner_run_id()
    print(f"[INFO] winner_run_id: {winner_run_id}", flush=True)

    client = MlflowClient()
    run = client.get_run(winner_run_id)
    params  = run.data.params
    val_metrics = run.data.metrics
    model_family = run.data.tags.get("model_family", "ensemble")

    model_name    = params["model_name"]
    dimred_method = params.get("dimred_method", "none")
    dimred_n      = int(params.get("dimred_n_components", "0"))
    clf_params    = {k: _cast_param(v) for k, v in params.items() if k.startswith("clf__")}

    # Construir pipeline
    if dimred_method and dimred_method != "none" and dimred_n > 0:
        pipeline = build_dimred_pipeline(model_name, dimred_method, dimred_n, seed=SEED)
    else:
        pipeline = build_pipeline(model_name, seed=SEED)
    if clf_params:
        pipeline.set_params(**clf_params)

    # Carregar splits (include_test=True — unico uso do test set neste projeto)
    X_train, X_val, X_test, y_train, y_val, y_test = _load_splits(
        PARQUET_PATH, SPLIT_PATH, include_test=True
    )
    X_trainval = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)

    print(f"[INFO] Treinando em X_trainval: {len(X_trainval)} amostras ...", flush=True)
    t0 = time.perf_counter()
    pipeline.fit(X_trainval, y_trainval)
    training_time_s = time.perf_counter() - t0
    print(f"[INFO] Treino concluido em {training_time_s:.2f}s", flush=True)

    # Avaliar UMA VEZ no test set
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_all_metrics(y_test.to_numpy(), y_pred, y_proba)

    # Latencia de inferencia
    n_test = len(X_test)
    t_inf = time.perf_counter()
    pipeline.predict_proba(X_test)
    inference_latency_ms = (time.perf_counter() - t_inf) / n_test * 1000

    # Git hash e data fingerprint
    githash7 = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], cwd=repo_root
    ).decode().strip()
    with open(FINGERPRINT_PATH) as f:
        datahash8 = json.load(f)["file_short"]

    dimred_tag = _dimred_tag_str(dimred_method, dimred_n)
    run_name = compose_run_name(
        "final_eval", model_name,
        dimred=dimred_tag,
        seed=SEED,
        datahash8=datahash8,
        githash7=githash7,
    )

    # Signature para Parte 6
    signature = infer_signature(X_test.head(5), y_pred[:5])

    # Gerar artefatos em tmpdir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        confusion_matrix_plot(y_test.to_numpy(), y_pred, output_path=tmp / "confusion_matrix_test.png")
        roc_plot(y_test.to_numpy(), y_proba, output_path=tmp / "roc_curve_test.png")

        metrics_payload = {
            **test_metrics,
            "training_time_s": training_time_s,
            "inference_latency_ms": inference_latency_ms,
        }
        (tmp / "test_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )

        exp_id = get_or_create_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as mlrun:
            log_standard_tags(
                mlrun,
                model_family=model_family,
                git_commit=githash7,
                dataset_fingerprint=datahash8,
                compute_profile_s=training_time_s,
                project_part="parte_5",
            )
            mlflow.set_tags({
                "evaluation_set":    "test",
                "candidate_run_id":  winner_run_id,
                "criterion_source":  "docs/final_selection_criteria.md",
            })
            mlflow.log_metrics({
                **test_metrics,
                "training_time_s":      training_time_s,
                "inference_latency_ms": inference_latency_ms,
            })
            log_standard_artifacts(mlrun, tmp)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(3),
            )
            final_run_id = mlrun.info.run_id

    # Salva copia local
    TEST_METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_METRICS_OUT.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Tabela val vs test
    print("", flush=True)
    print(f"MLflow run final_eval: {final_run_id}", flush=True)
    print(f"Run name: {run_name}", flush=True)
    print("", flush=True)
    print(f"{'Metrica':<20} | {'Val (candidato)':>16} | {'Test (final)':>12} | {'Delta':>8}", flush=True)
    print("-" * 66, flush=True)
    for m in ["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"]:
        val_v  = val_metrics.get(m, float("nan"))
        test_v = test_metrics.get(m, float("nan"))
        delta  = test_v - val_v
        print(f"{m:<20} | {val_v:>16.6f} | {test_v:>12.6f} | {delta:>+8.6f}", flush=True)
    print("", flush=True)
    print(f"[OK] test_metrics salvo em {TEST_METRICS_OUT}", flush=True)


if __name__ == "__main__":
    main()
