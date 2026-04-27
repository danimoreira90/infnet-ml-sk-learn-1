"""Gera relatório de drift simulando produção.

Referência histórica: train + val splits (como o modelo conhece).
Dados novos simulados: test set (dados que o modelo nunca viu em produção).

Saída:
  - reports/parte_6/drift_report.md — tabela com p-values e flags por feature.
  - MLflow run com tag stage=drift_report no experimento principal.

CONTRATO DE INTEGRIDADE:
  - _load_splits(include_test=True) é chamado APENAS aqui (drift simulation).
  - NÃO treina nenhum modelo. NÃO modifica nenhum run existente.
  - Cria exatamente 1 run MLflow novo com tag stage=drift_report.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

from credit_default.audit.recompute_metrics import _load_splits
from credit_default.monitoring.drift import data_drift_report, model_drift_report

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARQUET_PATH = _REPO_ROOT / "data" / "credit_card_cleaned.parquet"
_SPLIT_PATH = _REPO_ROOT / "artifacts" / "splits" / "split_indices.json"
_REPORTS_DIR = _REPO_ROOT / "reports" / "parte_6"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_EXPERIMENT_NAME = "credit-default-prediction"
_TRACKING_URI = (_REPO_ROOT / "mlruns").as_uri()

# Métricas do modelo final (test set, Parte 5) — baseline imutável
_BASELINE_METRICS = {
    "roc_auc": 0.7682,
    "f1_macro": 0.6876,
    "accuracy": 0.8218,
}


def _build_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega splits e monta DataFrames de referência e corrente."""
    X_train, X_val, X_test, y_train, y_val, y_test = _load_splits(
        parquet_path=_PARQUET_PATH,
        split_path=_SPLIT_PATH,
        include_test=True,
    )
    # Referência histórica: train + val (o que o modelo conhece)
    reference_df = pd.concat([X_train, X_val], ignore_index=True)
    # Dados "novos" simulados de produção: test set
    current_df = X_test.copy()
    return reference_df, current_df


def _generate_markdown(
    drift_result: dict,
    model_result: dict,
    reference_n: int,
    current_n: int,
) -> str:
    rows = []
    for feature, entry in sorted(drift_result.items()):
        flag = "🚨 SIM" if entry["drift_detected"] else "✅ NÃO"
        rows.append(
            f"| {feature} | {entry['test'].upper()} "
            f"| {entry['statistic']:.4f} | {entry['p_value']:.4f} | {flag} |"
        )
    table = "\n".join(rows)

    drifted_features = [f for f, e in drift_result.items() if e["drift_detected"]]
    model_flag = "SIM" if model_result["drift_detected"] else "NAO"
    ref_auc = f"{model_result['reference_roc_auc']:.4f}"
    cur_auc = f"{model_result['current_roc_auc']:.4f}"
    delta_auc = f"{model_result['delta_roc_auc']:+.4f}"
    model_row = f"| roc_auc | {ref_auc} | {cur_auc} | {delta_auc} | {model_flag} |"
    n_drifted = len(drifted_features)
    n_total = len(drift_result)
    affected = ", ".join(drifted_features) if drifted_features else "Nenhuma"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# Relatorio de Drift - Parte 6

Gerado em: {generated_at}

## Configuracao da Simulacao

| Parametro | Valor |
|-----------|-------|
| Referencia historica | train + val ({reference_n:,} registros) |
| Dados correntes simulados | test set ({current_n:,} registros) |
| Nivel de significancia (alpha) | 0.05 |
| Teste features continuas | Kolmogorov-Smirnov (KS) |
| Teste features categoricas | chi-quadrado (chi2) |

## Drift de Dados por Feature

| Feature | Teste | Estatistica | p-value | Drift? |
|---------|-------|-------------|---------|--------|
{table}

## Resumo de Drift de Dados

- **Features com drift detectado:** {n_drifted} / {n_total}
- **Features afetadas:** {affected}

## Drift de Modelo (Performance)

| Metrica | Baseline (Parte 5) | Corrente (simulado) | Delta | Drift? |
|---------|-------------------|---------------------|-------|--------|
{model_row}

> **Nota:** As metricas "correntes" nesta simulacao sao as mesmas do test set
> da Parte 5 (o modelo foi avaliado exatamente uma vez sobre esses dados).
> Em producao real, as metricas correntes seriam recomputadas periodicamente
> com dados rotulados acumulados.

## Plano de Retreinamento

### Criterios de Gatilho

| Criterio | Limiar | Acao |
|----------|--------|------|
| Drift em features criticas (PAY_0, LIMIT_BAL) | p < 0.05 | Avaliar imediatamente |
| Drift em 5+ features simultaneas | qualquer | Iniciar retreinamento |
| Queda de roc_auc | > 5 pp vs baseline | Retreinamento obrigatorio |
| Queda de roc_auc | > 3 pp vs baseline | Alerta + monitoramento intensivo |
| Latencia de inferencia | > 500 ms (P95) | Investigar degradacao |

### Frequencia Minima Sugerida

- **Drift de dados:** verificar semanalmente (ou a cada 10k novos registros)
- **Avaliacao de modelo:** mensal (requer dados rotulados acumulados)
- **Retreinamento completo:** trimestral ou quando gatilho acionado

### Estimativa de Custo por Retreinamento

| Etapa | Tempo estimado | Custo computacional |
|-------|---------------|---------------------|
| Preparacao de dados | ~5 min | Baixo |
| GridSearch (25 combinacoes) | ~15-30 min | Medio (CPU local) |
| Avaliacao no test set | ~1 min | Baixo |
| Deploy da nova versao | ~5 min | Baixo |
| **Total** | **~25-40 min** | **Baixo (sem GPU)** |

> O custo e aceitavel para um sistema de cartao de credito onde erros
> de classificacao tem impacto financeiro direto.
"""


def main() -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    experiment = mlflow.set_experiment(_EXPERIMENT_NAME)

    print("Carregando splits...")
    reference_df, current_df = _build_dataframes()
    print(f"  Referência: {len(reference_df):,} registros")
    print(f"  Corrente:   {len(current_df):,} registros")

    print("Calculando drift de dados...")
    drift_result = data_drift_report(reference_df, current_df, alpha=0.05)

    print("Calculando drift de modelo...")
    # Para simulação, usamos as métricas do test set como "correntes"
    test_metrics_path = _REPO_ROOT / "reports" / "parte_5" / "test_metrics.json"
    if test_metrics_path.exists():
        current_metrics = json.loads(test_metrics_path.read_text())
    else:
        current_metrics = _BASELINE_METRICS  # fallback: sem degradação
    model_result = model_drift_report(_BASELINE_METRICS, current_metrics)

    # Sumário no terminal
    drifted = [f for f, e in drift_result.items() if e["drift_detected"]]
    print(f"\nFeatures com drift (alpha=0.05): {len(drifted)} / {len(drift_result)}")
    for f in drifted:
        e = drift_result[f]
        print(f"  {f}: {e['test'].upper()} p={e['p_value']:.4f}")
    print(f"Drift de modelo: {model_result['drift_detected']} "
          f"(delta_roc_auc={model_result['delta_roc_auc']:+.4f})")

    # Gerar relatório Markdown
    report_md = _generate_markdown(
        drift_result, model_result, len(reference_df), len(current_df)
    )
    report_path = _REPORTS_DIR / "drift_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nRelatório gerado: {report_path}")

    # Logar no MLflow
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="drift_report",
        tags={"stage": "drift_report", "parte": "6"},
    ) as run:
        mlflow.log_param("reference_n", len(reference_df))
        mlflow.log_param("current_n", len(current_df))
        mlflow.log_param("alpha", 0.05)
        mlflow.log_param("n_features_tested", len(drift_result))
        mlflow.log_metric("n_features_drifted", len(drifted))
        mlflow.log_metric("model_drift_detected", int(model_result["drift_detected"]))
        mlflow.log_metric("delta_roc_auc", model_result["delta_roc_auc"])
        # Log p-values por feature
        for feat, entry in drift_result.items():
            mlflow.log_metric(f"pval_{feat}", entry["p_value"])
        mlflow.log_artifact(str(report_path))
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
