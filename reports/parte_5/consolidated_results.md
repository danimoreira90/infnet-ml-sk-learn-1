# Resultados Consolidados — Parte 5

**Data**: 2026-04-13
**Critério**: `docs\final_selection_criteria.md`
**Experimento MLflow**: `236665223173386020`
**Total de runs**: 25

Ordenação: `roc_auc↓` → `cv_roc_auc_std↑` → `latency_ms↑` → `train_s↑` → complexidade↑

| rank | run_id | stage | model_family | dimred | roc_auc_val | cv_roc_auc_std | latency_ms | train_s |
|------|--------|-------|--------------|--------|-------------|----------------|------------|---------|
| 1 | `6743e23b` | baseline | ensemble | — | 0.779480 | 0.004798 | 0.002 | 6.683 |
| 2 | `09502f9a` | tune | ensemble | — | 0.779427 | 0.004798 | 0.003 | 365.425 |
| 3 | `bbc1d933` | tune | ensemble | — | 0.776724 | 0.006165 | 0.023 | 77.050 |
| 4 | `db540919` | dimred | ensemble | pca_k15 | 0.766071 | 0.004992 | 0.002 | 9.004 |
| 5 | `97fa3043` | dimred | ensemble | pca_k10 | 0.762029 | 0.005592 | 0.002 | 5.957 |
| 6 | `01734f47` | baseline | ensemble | — | 0.757295 | 0.006165 | 0.016 | 4.097 |
| 7 | `2fffbb48` | tune | tree | — | 0.751438 | 0.004675 | 0.001 | 6.902 |
| 8 | `e9286b68` | dimred | ensemble | pca_k10 | 0.742007 | 0.005478 | 0.016 | 5.888 |
| 9 | `12ac360f` | dimred | ensemble | pca_k15 | 0.741710 | 0.005684 | 0.016 | 6.099 |
| 10 | `bc7e94ae` | tune | linear | — | 0.723362 | 0.002823 | 0.001 | 12.135 |
| 11 | `78b180c7` | baseline | linear | — | 0.723231 | 0.002823 | 0.001 | 0.154 |
| 12 | `59cf2495` | dimred | ensemble | lda_k1 | 0.718911 | 0.004816 | 0.002 | 1.074 |
| 13 | `059b3398` | dimred | linear | lda_k1 | 0.717060 | 0.003321 | 0.001 | 0.082 |
| 14 | `a198b736` | dimred | linear | lda_k1 | 0.717060 | 0.003321 | 0.001 | 0.110 |
| 15 | `c217c3fd` | dimred | linear | pca_k15 | 0.716700 | 0.000879 | 0.001 | 0.076 |
| 16 | `3c0441cf` | dimred | linear | pca_k10 | 0.704539 | 0.001208 | 0.001 | 0.050 |
| 17 | `66c09f7c` | baseline | linear | — | 0.693095 | 0.009622 | 0.001 | 0.086 |
| 18 | `71066fc9` | tune | linear | — | 0.691170 | 0.009622 | 0.001 | 3.170 |
| 19 | `8caca812` | dimred | linear | pca_k15 | 0.669995 | 0.018272 | 0.002 | 0.073 |
| 20 | `446257ea` | dimred | linear | pca_k10 | 0.659140 | 0.013516 | 0.002 | 0.090 |
| 21 | `fa32287d` | dimred | ensemble | lda_k1 | 0.652970 | 0.005331 | 0.018 | 3.332 |
| 22 | `85b50ee1` | dimred | tree | pca_k15 | 0.609648 | 0.009501 | 0.001 | 0.576 |
| 23 | `3cbf8e73` | dimred | tree | pca_k10 | 0.600569 | 0.006568 | 0.002 | 0.331 |
| 24 | `59f2da77` | baseline | tree | — | 0.598675 | 0.004675 | 0.001 | 0.431 |
| 25 | `be384749` | dimred | tree | lda_k1 | 0.596775 | 0.006325 | 0.001 | 0.126 |
