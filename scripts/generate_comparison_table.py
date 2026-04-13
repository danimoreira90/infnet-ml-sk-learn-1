"""Gera tabelas comparativas de resultados a partir do MLflow.

Uso: uv run python scripts/generate_comparison_table.py

Saidas:
- reports/parte_3/comparison_baseline.md
- reports/parte_3/comparison_tuned.md
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import mlflow  # noqa: E402

mlflow.set_tracking_uri((repo_root / "mlruns").as_uri())

from credit_default.tracking.compare import consolidated_results_table  # noqa: E402

COLUMNS = [
    "run_name",
    "model_family",
    "roc_auc",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "accuracy",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "cv_f1_mean",
    "cv_f1_std",
    "training_time_s",
]

FLOAT_COLS = {
    "roc_auc",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "accuracy",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "cv_f1_mean",
    "cv_f1_std",
    "training_time_s",
}


def df_to_markdown(df) -> str:
    """Converte DataFrame para tabela markdown."""
    cols = [c for c in COLUMNS if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if c in FLOAT_COLS and not isinstance(v, str):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def write_comparison(stage: str, output_path: Path) -> None:
    df = consolidated_results_table(stage=stage)
    header = f"# Comparacao de Modelos — {stage.capitalize()}\n\n"
    header += "> Ordenado por roc_auc descendente (metrica primaria).\n"
    header += "> Gerado automaticamente por `scripts/generate_comparison_table.py`.\n\n"
    if df.empty:
        content = header + "_Nenhum run encontrado._\n"
    else:
        content = header + df_to_markdown(df) + "\n"
    output_path.write_text(content, encoding="utf-8")
    print(f"Gerado: {output_path.relative_to(repo_root)}")


def main() -> None:
    out_dir = repo_root / "reports" / "parte_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_comparison("baseline", out_dir / "comparison_baseline.md")
    write_comparison("tune", out_dir / "comparison_tuned.md")


if __name__ == "__main__":
    main()
