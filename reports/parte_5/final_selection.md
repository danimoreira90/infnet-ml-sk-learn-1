# Seleção Final do Modelo — Parte 5

**Data**: 2026-04-13
**Experimento MLflow**: `infnet-ml-sistema` (id `236665223173386020`)
**Final eval run_id**: `6be94912218a4c51bd8297ac77719b7f`

---

## Critério de Seleção

> Transcrito de `docs/final_selection_criteria.md` (arquivo imutável, commitado antes de qualquer avaliação no test set).

### Requisitos obrigatórios

1. `predict_proba` nativo (sem calibração post-hoc adicional).
2. Modelo deve ser serializável via `joblib` ou `mlflow.sklearn.log_model`.
3. `training_time_s` finito e registrado.

### Métricas de decisão (ordem estrita)

1. **Maior `roc_auc`** (validação) — métrica primária.
2. **Empate** (até a 4ª casa decimal): menor `cv_roc_auc_std` — estabilidade em CV.
3. **Empate persistente**: menor `inference_latency_ms` — viabilidade operacional.
4. **Empate persistente**: menor `training_time_s` — custo de retreino.
5. **Empate persistente**: menor complexidade nominal (linear < árvore única < ensemble).

### Interpretações aplicadas

1. `predict_proba` "nativo": inclui `CalibratedClassifierCV` (parte do pipeline treinado, não pós-processamento do usuário).
2. Desempate intra-tier no step 5: `run_id` lexicográfico (determinista). Lacuna não prevista no contrato — registrada aqui sem alterar o arquivo original.

---

## Universo de Candidatos (25 runs, val set)

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

---

## Vencedor

**Run**: `6743e23b1b4a4e2aa15201cebb394a07`
**Run name**: `baseline__gb__numstd_catoh__none__none__seed42__data30c6be3a__code6ea3d3f`
**Modelo**: GradientBoostingClassifier (baseline, Parte 3 — sem tuning, sem dimred)

### Percurso da cascata de desempate

| Step | Critério | Valor vencedor | Situação |
|------|----------|----------------|----------|
| 1 | `roc_auc` maior | 0.779480 | Empate com `09502f9a` (0.779427, delta=5.3e-5 ≤ 1e-4) |
| 2 | `cv_roc_auc_std` menor | 0.004798 | Empate (ambos 0.004798, delta=0) |
| **3** | **`inference_latency_ms` menor** | **0.0019 ms** | **Vencedor** (`09502f9a` = 0.003 ms) |

O GradientBoosting baseline vence o GradientBoosting tunado (que levou 365s de treinamento vs 6.7s) exclusivamente pela latência de inferência. O tuning não trouxe ganho mensurável em `roc_auc` nem em estabilidade — resultado consistente com o esperado para um ensemble robusto no dataset deste tamanho.

---

## Métricas no Test Set

| Métrica | Valor |
|---------|-------|
| roc_auc | **0.768213** |
| f1_macro | 0.687623 |
| precision_macro | 0.758758 |
| recall_macro | 0.662225 |
| accuracy | 0.821802 |
| training_time_s | 8.50 s |
| inference_latency_ms | 0.0021 ms/amostra |

---

## Comparação Val vs Test

| Métrica | Val (candidato) | Test (final) | Delta |
|---------|-----------------|--------------|-------|
| roc_auc | 0.779480 | 0.768213 | **-0.011267** |
| f1_macro | 0.673333 | 0.687623 | +0.014290 |
| precision_macro | 0.750524 | 0.758758 | +0.008234 |
| recall_macro | 0.649051 | 0.662225 | +0.013174 |
| accuracy | 0.816908 | 0.821802 | +0.004894 |

---

## Análise

O `roc_auc` caiu de 0.779 no val para 0.768 no test (delta -1.1 pp) — queda esperada e dentro do intervalo normal para este tipo de modelo e dataset. Não há sinal de overfitting severo.

As demais métricas (f1_macro, precision, recall, accuracy) melhoraram ligeiramente no test set. Isso é explicado pelo fato de o modelo final ser retreinado em X_trainval (25.470 amostras) em vez de X_train (20.975): mais dados produzem melhor generalização para métricas de classificação, enquanto o `roc_auc` — métrica de ranking — é naturalmente mais sensível à distribuição específica do conjunto de avaliação.

**Conclusão**: o modelo opera conforme o esperado, sem ajuste post-hoc. As métricas são reportadas as-is, conforme o contrato de `docs/final_selection_criteria.md`.
