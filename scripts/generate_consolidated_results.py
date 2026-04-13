"""Gera reports/parte_5/consolidated_results.md com os 25 runs ordenados pelo critério."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from mlflow.tracking import MlflowClient

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

EXP_ID = "236665223173386020"
OUTPUT_PATH = repo_root / "reports" / "parte_5" / "consolidated_results.md"
CRITERIA_PATH = repo_root / "docs" / "final_selection_criteria.md"

COMPLEXITY_RANK = {"linear": 0, "tree": 1, "ensemble": 2}


def _complexity(model_family: str) -> int:
    return COMPLEXITY_RANK.get(model_family, 99)


def main() -> None:
    client = MlflowClient()
    all_runs = client.search_runs(experiment_ids=[EXP_ID])

    rows = []
    for r in all_runs:
        m = r.data.metrics
        t = r.data.tags
        rows.append({
            "run_id_short": r.info.run_id[:8],
            "run_id":       r.info.run_id,
            "run_name":     r.info.run_name or "",
            "stage":        t.get("stage", ""),
            "model_family": t.get("model_family", ""),
            "dimred_method":       t.get("dimred_method", "none"),
            "dimred_n_components": t.get("dimred_n_components", "0"),
            "roc_auc":             m.get("roc_auc", float("nan")),
            "cv_roc_auc_mean":     m.get("cv_roc_auc_mean", float("nan")),
            "cv_roc_auc_std":      m.get("cv_roc_auc_std", float("inf")),
            "inference_latency_ms": m.get("inference_latency_ms", float("inf")),
            "training_time_s":     m.get("training_time_s", float("inf")),
        })

    rows.sort(key=lambda x: (
        -x["roc_auc"],
         x["cv_roc_auc_std"],
         x["inference_latency_ms"],
         x["training_time_s"],
         _complexity(x["model_family"]),
        x["run_id"],
    ))

    lines = [
        f"# Resultados Consolidados — Parte 5",
        f"",
        f"**Data**: {date.today().isoformat()}",
        f"**Critério**: `{CRITERIA_PATH.relative_to(repo_root)}`",
        f"**Experimento MLflow**: `{EXP_ID}`",
        f"**Total de runs**: {len(rows)}",
        f"",
        f"Ordenação: `roc_auc↓` → `cv_roc_auc_std↑` → `latency_ms↑` → `train_s↑` → complexidade↑",
        f"",
        "| rank | run_id | stage | model_family | dimred | roc_auc_val | cv_roc_auc_std | latency_ms | train_s |",
        "|------|--------|-------|--------------|--------|-------------|----------------|------------|---------|",
    ]
    for i, r in enumerate(rows, 1):
        dimred = r["dimred_method"] if r["dimred_method"] != "none" else "—"
        if dimred != "—":
            dimred = f"{r['dimred_method']}_k{r['dimred_n_components']}"
        lines.append(
            f"| {i} | `{r['run_id_short']}` | {r['stage']} | {r['model_family']} | {dimred} "
            f"| {r['roc_auc']:.6f} | {r['cv_roc_auc_std']:.6f} | {r['inference_latency_ms']:.3f} | {r['training_time_s']:.3f} |"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {OUTPUT_PATH} gerado com {len(rows)} runs.", flush=True)


if __name__ == "__main__":
    main()
