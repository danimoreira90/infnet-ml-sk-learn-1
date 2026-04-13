"""Gera tabelas comparativas Parte 4 (dimred) vs Parte 3 (baseline).

Outputs:
- reports/parte_4/comparison_dimred.md   : todos os runs dimred + baseline P3
- reports/parte_4/comparison_pca_vs_lda.md : pivot por modelo
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from credit_default.tracking.compare import consolidated_results_table  # noqa: E402
from credit_default.tracking.mlflow_utils import get_or_create_experiment  # noqa: E402

EXPERIMENT_NAME = "infnet-ml-sistema"
OUTPUT_DIR = repo_root / "reports" / "parte_4"

MODELS = ["perceptron", "logreg", "dtree", "rf", "gb"]
DIMRED_CONFIGS = ["pca_k10", "pca_k15", "lda_k1"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri((repo_root / "mlruns").absolute().as_uri())
    get_or_create_experiment(EXPERIMENT_NAME)

    # Carregar todos os runs (baseline P3 + dimred P4)
    df_all = consolidated_results_table(EXPERIMENT_NAME)

    # Adicionar colunas de dimred a partir de params (via MLflow direto)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    all_runs = client.search_runs(exp.experiment_id, max_results=200)

    # Mapa run_name -> params extras
    extra: dict[str, dict] = {}
    for r in all_runs:
        p = r.data.params
        extra[r.info.run_name or ""] = {
            "model_name": p.get("model_name", ""),
            "dimred_method": p.get("dimred_method", "none"),
            "dimred_n_components": p.get("dimred_n_components", "0"),
            "dimred_tag": r.data.tags.get("dimred_method", "none"),
            "ev_tag": r.data.tags.get("dimred_explained_variance", "na"),
        }

    df_all["model_name"] = df_all["run_name"].map(lambda n: extra.get(n, {}).get("model_name", ""))
    df_all["dimred_method"] = df_all["run_name"].map(
        lambda n: extra.get(n, {}).get("dimred_method", "none")
    )
    df_all["dimred_n_components"] = df_all["run_name"].map(
        lambda n: extra.get(n, {}).get("dimred_n_components", "0")
    )
    df_all["dimred_explained_variance"] = df_all["run_name"].map(
        lambda n: extra.get(n, {}).get("ev_tag", "na")
    )

    # Separar baseline P3 e dimred P4
    p3 = df_all[df_all["stage"] == "baseline"].copy()
    p4 = df_all[df_all["stage"] == "dimred"].copy()

    # ── Tabela 1: comparison_dimred.md ─────────────────────────────────────────
    cols = [
        "run_name",
        "model_name",
        "model_family",
        "stage",
        "roc_auc",
        "f1_macro",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "training_time_s",
        "dimred_method",
        "dimred_n_components",
        "dimred_explained_variance",
    ]

    combined = pd.concat([p3, p4], ignore_index=True).sort_values("roc_auc", ascending=False)
    available = [c for c in cols if c in combined.columns]
    md1 = combined[available].to_markdown(index=False, floatfmt=".4f")

    out1 = OUTPUT_DIR / "comparison_dimred.md"
    out1.write_text(
        "# Comparativo Parte 4 — Dimred vs Baseline (Parte 3)\n\n"
        "Ordenado por `roc_auc` desc (metrica primaria).\n\n"
        f"{md1}\n",
        encoding="utf-8",
    )
    print(f"Gerado: {out1}")

    # ── Tabela 2: comparison_pca_vs_lda.md ─────────────────────────────────────
    pivot_rows = []
    for model in MODELS:
        row: dict = {"model": model}

        # P3 baseline
        p3_m = p3[p3["model_name"] == model]
        row["none (P3)"] = f"{p3_m['roc_auc'].values[0]:.4f}" if not p3_m.empty else "n/a"

        # P4 dimred configs
        for cfg in DIMRED_CONFIGS:
            # Match run_name containing the dimred config label
            p4_cfg = p4[p4["run_name"].str.contains(f"__{cfg}__", na=False)]
            p4_m = p4_cfg[p4_cfg["model_name"] == model]
            row[cfg] = f"{p4_m['roc_auc'].values[0]:.4f}" if not p4_m.empty else "n/a"

        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows, columns=["model", "none (P3)"] + DIMRED_CONFIGS)
    md2 = pivot_df.to_markdown(index=False)

    out2 = OUTPUT_DIR / "comparison_pca_vs_lda.md"
    out2.write_text(
        "# Pivot: ROC-AUC por Modelo e Tecnica de Dimred\n\n"
        "Colunas: baseline sem dimred (P3) + 3 configs Parte 4.\n"
        "Metrica primaria: `roc_auc`.\n\n"
        f"{md2}\n",
        encoding="utf-8",
    )
    print(f"Gerado: {out2}")


if __name__ == "__main__":
    main()
