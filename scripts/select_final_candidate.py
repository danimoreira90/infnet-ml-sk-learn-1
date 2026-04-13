"""Aplica o critério de seleção em cascata aos 25 runs e gera final_selection_rationale.md.

NÃO toca test_idx. NÃO cria MLflow run.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

EXP_ID = "236665223173386020"
CRITERIA_PATH  = repo_root / "docs" / "final_selection_criteria.md"
OUTPUT_PATH    = repo_root / "reports" / "parte_5" / "final_selection_rationale.md"
COMPLEXITY_RANK = {"linear": 0, "tree": 1, "ensemble": 2}
TIE_DELTA = 1e-4


def _complexity(run: Run) -> int:
    fam = run.data.tags.get("model_family", "ensemble")
    return COMPLEXITY_RANK.get(fam, 99)


def _metric(run: Run, key: str, default: float = float("inf")) -> float:
    v = run.data.metrics.get(key)
    return v if v is not None else default


def apply_criterion(runs: list[Run]) -> tuple[Run, int, str]:
    """Retorna (vencedor, decision_step, decision_reason)."""
    candidates = list(runs)

    # Step 1: maior roc_auc
    max_roc = max(_metric(r, "roc_auc", 0.0) for r in candidates)
    tied = [r for r in candidates if abs(_metric(r, "roc_auc", 0.0) - max_roc) <= TIE_DELTA]
    if len(tied) == 1:
        return tied[0], 1, f"roc_auc={max_roc:.6f} (único após step 1)"
    candidates = tied

    # Step 2: menor cv_roc_auc_std
    min_std = min(_metric(r, "cv_roc_auc_std") for r in candidates)
    tied = [r for r in candidates if abs(_metric(r, "cv_roc_auc_std") - min_std) <= TIE_DELTA]
    if len(tied) == 1:
        return tied[0], 2, f"cv_roc_auc_std={min_std:.6f} (menor, step 2)"
    candidates = tied

    # Step 3: menor inference_latency_ms
    min_lat = min(_metric(r, "inference_latency_ms") for r in candidates)
    tied = [r for r in candidates if abs(_metric(r, "inference_latency_ms") - min_lat) <= TIE_DELTA]
    if len(tied) == 1:
        return tied[0], 3, f"inference_latency_ms={min_lat:.4f} (menor, step 3)"
    candidates = tied

    # Step 4: menor training_time_s
    min_train = min(_metric(r, "training_time_s") for r in candidates)
    tied = [r for r in candidates if abs(_metric(r, "training_time_s") - min_train) <= TIE_DELTA]
    if len(tied) == 1:
        return tied[0], 4, f"training_time_s={min_train:.4f} (menor, step 4)"
    candidates = tied

    # Step 5: menor complexidade nominal + run_id lexicográfico como desempate final
    winner = min(candidates, key=lambda r: (_complexity(r), r.info.run_id))
    fam = winner.data.tags.get("model_family", "?")
    return winner, 5, f"complexidade_nominal={fam} (step 5; desempate final por run_id)"


def main() -> None:
    print(f"[AUDITORIA] Critério fonte: {CRITERIA_PATH}", flush=True)

    client = MlflowClient()
    all_runs = client.search_runs(experiment_ids=[EXP_ID])

    # Filtra runs sem training_time_s
    eligible = [
        r for r in all_runs
        if r.data.metrics.get("training_time_s") is not None
        and r.data.metrics.get("roc_auc") is not None
    ]
    print(f"[INFO] {len(eligible)}/{len(all_runs)} runs elegíveis (com training_time_s e roc_auc)", flush=True)

    if not eligible:
        print("[ERRO] Nenhum run elegível encontrado.", flush=True)
        sys.exit(1)

    winner, step, reason = apply_criterion(eligible)

    # Top-5 para o relatório
    def sort_key(r: Run):
        return (
            -_metric(r, "roc_auc", 0.0),
             _metric(r, "cv_roc_auc_std"),
             _metric(r, "inference_latency_ms"),
             _metric(r, "training_time_s"),
             _complexity(r),
             r.info.run_id,
        )
    top5 = sorted(eligible, key=sort_key)[:5]

    # Gerar relatório
    top5_lines = [
        "| rank | run_id | roc_auc_val | cv_roc_auc_std | latency_ms | train_s | model_family |",
        "|------|--------|-------------|----------------|------------|---------|--------------|",
    ]
    for i, r in enumerate(top5, 1):
        top5_lines.append(
            f"| {i} | `{r.info.run_id[:8]}` "
            f"| {_metric(r, 'roc_auc', 0):.6f} "
            f"| {_metric(r, 'cv_roc_auc_std', float('nan')):.6f} "
            f"| {_metric(r, 'inference_latency_ms', float('nan')):.3f} "
            f"| {_metric(r, 'training_time_s', float('nan')):.3f} "
            f"| {r.data.tags.get('model_family', '?')} |"
        )

    content = f"""# Seleção Final do Modelo — Rationale

**Data**: {date.today().isoformat()}
**Critério fonte**: `docs/final_selection_criteria.md`

## Vencedor

- winner_run_id: {winner.info.run_id}
- winner_run_name: {winner.info.run_name or ""}
- decision_step: {step}
- decision_reason: {reason}

## Interpretações Aplicadas

1. predict_proba "nativo": inclui CalibratedClassifierCV (parte do pipeline treinado, não pós-processamento do usuário).
2. Desempate intra-tier no step 5: ordem lexicográfica do run_id (determinista).

## Top-5 Candidatos

{chr(10).join(top5_lines)}
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(content, encoding="utf-8")

    print("", flush=True)
    print("=" * 52, flush=True)
    print(f"  VENCEDOR: {winner.info.run_id[:8]}", flush=True)
    print(f"  Run name: {(winner.info.run_name or '')[:50]}", flush=True)
    print(f"  Step decisivo: {step} -- {reason[:40]}", flush=True)
    print("=" * 52, flush=True)
    print("", flush=True)
    print(f"[OK] {OUTPUT_PATH} gerado.", flush=True)


if __name__ == "__main__":
    main()
