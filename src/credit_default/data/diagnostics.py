"""Diagnósticos de qualidade do dataset — 7 checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend não-interativo: sem janelas, seguro em CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def check_missing(df: pd.DataFrame) -> dict[str, Any]:
    """Conta valores ausentes por coluna.

    Returns
    -------
    dict com: n_missing_total (int), n_cols_with_missing (int),
              details ({col: count} apenas para colunas com missing > 0).
    """
    missing = df.isnull().sum()
    details = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
    return {
        "n_missing_total": int(missing.sum()),
        "n_cols_with_missing": len(details),
        "details": details,
    }


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Conta linhas exatamente duplicadas.

    Returns
    -------
    dict com: n_duplicate_rows (int), duplicate_ratio (float).
    """
    n_dup = int(df.duplicated().sum())
    return {
        "n_duplicate_rows": n_dup,
        "duplicate_ratio": round(n_dup / len(df), 6) if len(df) > 0 else 0.0,
    }


def check_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    *,
    figures_dir: Path,
    minority_ratio_threshold: float = 0.20,
) -> dict[str, Any]:
    """Distribuição das classes do target; salva gráfico de barras.

    Parameters
    ----------
    minority_ratio_threshold : float
        Proporção mínima da classe minoritária abaixo da qual emite imbalance_warning.

    Returns
    -------
    dict com: class_counts, class_ratios, minority_class, minority_ratio,
              imbalance_warning (bool).
    Salva: figures_dir/target_distribution.png
    """
    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    ratios = (counts / total).round(4)
    minority_class = int(counts.idxmin())
    minority_ratio = float(ratios[minority_class])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k in counts.index], counts.values, color=["steelblue", "tomato"])
    ax.set_xlabel(target_col)
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuicao do Target")
    for i, v in enumerate(counts.values):
        ax.text(i, v + total * 0.005, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "target_distribution.png", dpi=100)
    plt.close(fig)

    return {
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "class_ratios": {int(k): float(v) for k, v in ratios.items()},
        "minority_class": minority_class,
        "minority_ratio": minority_ratio,
        "imbalance_warning": minority_ratio < minority_ratio_threshold,
    }


def check_outliers(
    df: pd.DataFrame,
    target_col: str,
    *,
    figures_dir: Path,
) -> dict[str, Any]:
    """Detecta outliers por IQR em colunas numéricas (exceto target).

    Um valor é outlier se estiver abaixo de Q1 - 1.5*IQR ou
    acima de Q3 + 1.5*IQR.

    Returns
    -------
    dict com: cols_with_outliers ({col: n_outliers}).
    Salva: figures_dir/outlier_counts.png
    """
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    cols_with_outliers: dict[str, int] = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            cols_with_outliers[col] = n_out

    # Gráfico: apenas top-20 colunas com mais outliers para legibilidade
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if cols_with_outliers:
        sorted_items = sorted(cols_with_outliers.items(), key=lambda x: x[1], reverse=True)[:20]
        cols_plot, counts_plot = zip(*sorted_items)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(cols_plot, counts_plot, color="coral")
        ax.set_xlabel("Coluna")
        ax.set_ylabel("N outliers (IQR)")
        ax.set_title("Outliers por Coluna (metodo IQR)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(figures_dir / "outlier_counts.png", dpi=100)
        plt.close(fig)
    else:
        # Salva figura vazia com nota
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Nenhum outlier IQR detectado", ha="center", va="center")
        ax.axis("off")
        fig.savefig(figures_dir / "outlier_counts.png", dpi=100)
        plt.close(fig)

    return {"cols_with_outliers": cols_with_outliers}


def check_correlations(
    df: pd.DataFrame,
    target_col: str,
    *,
    figures_dir: Path,
    top_n: int = 10,
) -> dict[str, Any]:
    """Pearson e Spearman entre features numéricas e o target.

    Returns
    -------
    dict com: top_pearson (list[{col, corr}]), top_spearman (list[{col, corr}]).
    Salva: figures_dir/correlation_heatmap.png
    """
    numeric_df = df.select_dtypes(include="number")
    feature_cols = [c for c in numeric_df.columns if c != target_col]

    pearson = (
        numeric_df[feature_cols]
        .corrwith(df[target_col], method="pearson")
        .abs()
        .sort_values(ascending=False)
    )
    spearman = (
        numeric_df[feature_cols]
        .corrwith(df[target_col], method="spearman")
        .abs()
        .sort_values(ascending=False)
    )

    top_pearson = [
        {"col": col, "corr": round(float(val), 4)} for col, val in pearson.head(top_n).items()
    ]
    top_spearman = [
        {"col": col, "corr": round(float(val), 4)} for col, val in spearman.head(top_n).items()
    ]

    # Heatmap com as top-N features (Pearson)
    top_cols = pearson.head(top_n).index.tolist()
    corr_matrix = numeric_df[top_cols + [target_col]].corr(method="pearson")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(f"Heatmap Pearson (top-{top_n} features + target)")
    plt.tight_layout()
    fig.savefig(figures_dir / "correlation_heatmap.png", dpi=100)
    plt.close(fig)

    return {"top_pearson": top_pearson, "top_spearman": top_spearman}


def check_leakage_risk(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Documenta risco de data leakage identificado nos notebooks legados.

    APENAS DOCUMENTA — nao corrige. A correcao e responsabilidade da Parte 3.

    O risco detectado e a ausencia de Pipeline/ColumnTransformer nos notebooks
    01-06: o StandardScaler possivelmente foi fitado em train+test, introduzindo
    leakage de normalizacao.

    Returns
    -------
    dict com: leakage_risk_detected (bool), severity (str), details (str),
              action (str).
    """
    _ = df, target_col  # parametros aceitos para assinatura uniforme; sem inspecao aqui
    return {
        "leakage_risk_detected": True,
        "severity": "HIGH",
        "details": (
            "Notebooks 02-06 utilizam StandardScaler sem Pipeline/ColumnTransformer. "
            "O scaler provavelmente foi fitado antes do split ou sobre train+test, "
            "introduzindo leakage de normalizacao. "
            "As metricas reportadas nos notebooks podem estar otimistas."
        ),
        "action": "Corrigir na Parte 3 encapsulando transformacoes em sklearn.pipeline.Pipeline.",
    }


def check_bias_risk(
    df: pd.DataFrame,
    target_col: str,
    *,
    figures_dir: Path,
    sensitive_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Distribuicao do target por features sensiveis (SEX, EDUCATION, MARRIAGE).

    Returns
    -------
    dict com: sensitive_features ({col: {str(value): default_rate}}).
    Salva: figures_dir/bias_by_sensitive_feature.png
    """
    if sensitive_cols is None:
        sensitive_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in df.columns]

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict[str, float]] = {}
    for col in sensitive_cols:
        rates = df.groupby(col)[target_col].mean().round(4)
        result[col] = {str(int(k)): round(float(v), 4) for k, v in rates.items()}

    if sensitive_cols:
        n_cols = len(sensitive_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
        if n_cols == 1:
            axes = [axes]
        for ax, col in zip(axes, sensitive_cols):
            vals = result[col]
            ax.bar(list(vals.keys()), list(vals.values()), color="mediumseagreen")
            ax.set_xlabel(col)
            ax.set_ylabel("Taxa de inadimplência")
            ax.set_title(f"Inadimplência por {col}")
            ax.set_ylim(0, 1)
        plt.suptitle("Risco de Vies por Features Sensiveis", fontsize=12)
        plt.tight_layout()
        fig.savefig(figures_dir / "bias_by_sensitive_feature.png", dpi=100)
        plt.close(fig)

    return {"sensitive_features": result}


def run_all_diagnostics(
    df: pd.DataFrame,
    target_col: str,
    *,
    figures_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Orquestra todos os 7 checks de qualidade.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpo (sem duplicatas).
    target_col : str
        Nome da coluna alvo.
    figures_dir : str | Path
        Diretório onde as figuras PNG serao salvas. Criado se nao existir.
    seed : int
        Semente para reproducibilidade (reservado para uso futuro).

    Returns
    -------
    dict completo com resultados de todos os checks, pronto para JSON.
    """
    _ = seed  # seed reservado para futuras operações estocásticas
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "missing": check_missing(df),
        "duplicates": check_duplicates(df),
        "target_distribution": check_target_distribution(df, target_col, figures_dir=figures_dir),
        "outliers": check_outliers(df, target_col, figures_dir=figures_dir),
        "correlations": check_correlations(df, target_col, figures_dir=figures_dir),
        "leakage_risk": check_leakage_risk(df, target_col),
        "bias_risk": check_bias_risk(df, target_col, figures_dir=figures_dir),
    }
