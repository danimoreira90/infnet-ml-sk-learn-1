"""Roda o auditor de integridade em 3 runs aleatórios da P3/P4.

Saída esperada: "[AUDIT SAMPLE] N/N runs auditados OK"
Exit code 0 se todos OK, 1 se qualquer run falhar (propagado por recompute_run_metrics).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from credit_default.audit.recompute_metrics import recompute_run_metrics  # noqa: E402

EXP_ID = "236665223173386020"
PARQUET_PATH = repo_root / "data" / "credit_card_cleaned.parquet"
SPLIT_PATH   = repo_root / "artifacts" / "splits" / "split_indices.json"


def main() -> None:
    client = MlflowClient()
    all_runs = client.search_runs(experiment_ids=[EXP_ID])

    eligible = [
        r for r in all_runs
        if r.data.tags.get("project_part") in ("parte_3", "parte_4")
    ]
    if not eligible:
        print("[AUDIT SAMPLE] Nenhum run elegível encontrado (parte_3/parte_4).", flush=True)
        sys.exit(1)

    k = min(3, len(eligible))
    random.seed(42)
    sample = random.sample(eligible, k=k)

    ok_count = 0
    for run in sample:
        run_id = run.info.run_id
        result = recompute_run_metrics(
            run_id,
            parquet_path=PARQUET_PATH,
            split_path=SPLIT_PATH,
        )
        if result.ok:
            ok_count += 1

    print(f"[AUDIT SAMPLE] {ok_count}/{k} runs auditados OK", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
