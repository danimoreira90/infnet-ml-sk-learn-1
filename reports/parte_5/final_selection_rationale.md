# Seleção Final do Modelo — Rationale

**Data**: 2026-04-13
**Critério fonte**: `docs/final_selection_criteria.md`

## Vencedor

- winner_run_id: 6743e23b1b4a4e2aa15201cebb394a07
- winner_run_name: baseline__gb__numstd_catoh__none__none__seed42__data30c6be3a__code6ea3d3f
- decision_step: 3
- decision_reason: inference_latency_ms=0.0019 (menor, step 3)

## Interpretações Aplicadas

1. predict_proba "nativo": inclui CalibratedClassifierCV (parte do pipeline treinado, não pós-processamento do usuário).
2. Desempate intra-tier no step 5: ordem lexicográfica do run_id (determinista).

## Top-5 Candidatos

| rank | run_id | roc_auc_val | cv_roc_auc_std | latency_ms | train_s | model_family |
|------|--------|-------------|----------------|------------|---------|--------------|
| 1 | `6743e23b` | 0.779480 | 0.004798 | 0.002 | 6.683 | ensemble |
| 2 | `09502f9a` | 0.779427 | 0.004798 | 0.003 | 365.425 | ensemble |
| 3 | `bbc1d933` | 0.776724 | 0.006165 | 0.023 | 77.050 | ensemble |
| 4 | `db540919` | 0.766071 | 0.004992 | 0.002 | 9.004 | ensemble |
| 5 | `97fa3043` | 0.762029 | 0.005592 | 0.002 | 5.957 | ensemble |
