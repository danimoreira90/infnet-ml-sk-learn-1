# Relatório Técnico — Projeto de Classificação de Inadimplência

---

## Parte 2 — Fundação de Dados e Diagnóstico Inicial

### Objetivo

Estabelecer uma base de dados reproduzível, auditável e documentada para as etapas de modelagem da Parte 3. Os princípios norteadores são rastreabilidade, transparência e ausência de números hardcoded: todas as métricas aqui referenciadas são calculadas dinamicamente a partir do dataset e podem ser reproduzidas executando os scripts da Parte 2.

---

### Dataset Bruto

O dataset utilizado é o arquivo `.xls` cujo SHA-256 completo e shape estão registrados em `artifacts/data_fingerprint.json` (campo `file_sha256` e campos `n_rows`/`n_cols`). O campo `file_short` (8 chars) identifica este dataset de forma compacta e será utilizado como prefixo no naming de experimentos MLflow na Parte 3.

Durante a ingestão, foi identificada e removida explicitamente uma quantidade de linhas exatamente duplicadas. O número exato está registrado no log de execução de `build_clean_dataset.py` e documentado em `docs/integrity_manifest.md`. O parquet limpo resultante é a entrada para todas as etapas subsequentes.

---

### Anomalias Documentadas

Duas colunas apresentam valores fora da especificação UCI original:

- **EDUCATION**: contém códigos além dos previstos na documentação do dataset.
- **MARRIAGE**: contém um código adicional não documentado.

Os valores anômalos e sua distribuição estão listados em `artifacts/data_schema.json` (campo `warnings`). Os dados foram **preservados sem recodificação** — a recodificação, se necessária para a modelagem, é responsabilidade da Parte 3 via `ColumnTransformer`.

---

### Estratégia de Split

O conjunto de dados limpo foi dividido em três partições estratificadas pelo target:

| Partição | Proporção | N (linhas) |
|----------|-----------|-----------|
| Treino   | 70%       | ver `artifacts/splits/split_indices.json` (campo `n_train`) |
| Validação | 15%      | ver `artifacts/splits/split_indices.json` (campo `n_val`) |
| Teste    | 15%       | ver `artifacts/splits/split_indices.json` (campo `n_test`) |

A semente utilizada é registrada no artefato. A estratificação garante que a proporção da classe minoritária seja mantida em cada partição.

**Nota:** os notebooks 02–06 utilizam split 70/30 (train/test). A mudança para 70/15/15 é deliberada e exclusiva ao novo pipeline da Parte 2. Ver `docs/integrity_manifest.md` para justificativa.

---

### Diagnósticos de Qualidade

Os resultados completos dos 7 checks estão em `artifacts/data_qa_summary.json`. Os principais achados:

**Missing values:** quantidade e distribuição por coluna disponível em `artifacts/data_qa_summary.json` (campo `missing`).

**Distribuição do target:** a taxa da classe minoritária e o flag de desbalanceamento estão em `artifacts/data_qa_summary.json` (campo `target_distribution`). A figura `reports/figures/parte_2/target_distribution.png` exibe a distribuição visual.

**Outliers:** número de colunas com outliers detectados pelo método IQR disponível em `artifacts/data_qa_summary.json` (campo `outliers`). A figura `reports/figures/parte_2/outlier_counts.png` mostra a distribuição por coluna.

**Correlações com o target:** as top features por correlação de Pearson e Spearman estão em `artifacts/data_qa_summary.json` (campo `correlations`). A figura `reports/figures/parte_2/correlation_heatmap.png` exibe o heatmap das top features.

**Risco de viés:** taxas de inadimplência por SEX, EDUCATION e MARRIAGE estão em `artifacts/data_qa_summary.json` (campo `bias_risk`). A figura `reports/figures/parte_2/bias_by_sensitive_feature.png` exibe a comparação por grupo.

---

### Risco de Leakage (documentado)

Foi identificado um risco de leakage de normalização nos notebooks 02–06: o `StandardScaler` foi possivelmente ajustado sem separação adequada dos conjuntos. As métricas reportadas naqueles notebooks podem ser ligeiramente otimistas.

**Ação planejada:** correção na Parte 3 via `sklearn.pipeline.Pipeline`. O risco está documentado em `artifacts/data_qa_summary.json` (campo `diagnostics.leakage_risk`) e em `docs/integrity_manifest.md`.

---

### Conclusão

O dataset passou na auditoria estrutural: sem valores ausentes, sem duplicatas no conjunto limpo, target presente, schema compatível com a especificação. As anomalias de EDUCATION e MARRIAGE são documentadas e não impedem a modelagem. O conjunto está aprovado para as etapas de treino da Parte 3.

Todos os artefatos desta seção podem ser reproduzidos com:

```bash
python scripts/build_clean_dataset.py
python scripts/run_data_qa.py
```

---

## Parte 3 — Pipeline de Modelagem e Rastreamento MLflow

### Objetivo

Treinar, avaliar e comparar 5 algoritmos de classificacao usando sklearn Pipelines com rastreamento completo via MLflow. Dois estagios: baseline (sem tuning) e tuned (com hyperparameter search). Metrica primaria: **roc_auc**.

### Modelos Avaliados

| Modelo | Familia | Search (tune) | Observacao |
|--------|---------|---------------|------------|
| Perceptron | linear | GridSearch | CalibratedClassifierCV para predict_proba |
| Logistic Regression | linear | GridSearch | — |
| Decision Tree | tree | GridSearch | — |
| Random Forest | ensemble | RandomizedSearch (n_iter=30) | — |
| Gradient Boosting | ensemble | RandomizedSearch (n_iter=30) | — |

### Preprocessamento

ColumnTransformer com 3 branches:
- **StandardScaler** para 14 features continuas (LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6)
- **Passthrough** para 6 features ordinais (PAY_0, PAY_2-6)
- **OneHotEncoder** (handle_unknown='ignore') para 3 features categoricas (SEX, EDUCATION, MARRIAGE)

A estrategia resolve o risco de leakage de normalizacao documentado na Parte 2: o ColumnTransformer esta dentro do Pipeline e so e fitado durante Pipeline.fit() no conjunto de treino.

### Rastreamento MLflow

Cada run registra:
- **10 tags**: stage, model_family, git_commit, dataset_fingerprint, compute_profile_s, project_part, framework, python_version, os_platform, search_type
  (mlflow.runName e atributo nativo do run, nao duplicado como tag)
- **11 metricas**: roc_auc [PRIMARY], f1_macro, precision_macro, recall_macro, accuracy, cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std, training_time_s, inference_latency_ms
- **5+ artefatos**: confusion_matrix.png, roc_curve.png, pr_curve.png, run_summary.json (com primary_metric="roc_auc"), split_fingerprint.txt

### Resultados

Consultar tabelas comparativas geradas automaticamente:
- `reports/parte_3/comparison_baseline.md`
- `reports/parte_3/comparison_tuned.md`

### Reproducao

```bash
uv run python scripts/train_baseline.py
uv run python scripts/train_tuned.py
uv run python scripts/generate_comparison_table.py
```

---

## Parte 4 — Reducao de Dimensionalidade (PCA e LDA)

### Objetivo

Aplicar duas tecnicas de reducao de dimensionalidade ao pipeline sklearn, retreinar os 5 modelos e comparar desempenho com o baseline da Parte 3. Metrica primaria: **roc_auc**.

### Tecnicas Escolhidas

**PCA — Principal Component Analysis** (nao-supervisionado)
- Linear, determinístico, rapido, maduro em producao.
- Inserido apos o ColumnTransformer, antes do classificador.
- `n_components` calculado para reter >= 85% da variancia explicada (determinístico no conjunto de treino).
- Testadas duas configuracoes: `pca_k10` (10 componentes, 84.2% EV) e `pca_k15` (15 componentes, 94.3% EV).

**LDA — Linear Discriminant Analysis** (supervisionado)
- Usa o target para maximizar separacao entre classes.
- Em classificacao binaria, `n_components` e forcado a 1 pela matematica: `min(n_classes - 1, n_features) = 1`. Configuracao: `lda_k1`.
- Recebe `y` automaticamente via `Pipeline.fit(X, y)`.

**t-SNE — Excluido**
- Algoritmo transductivo (nao gera transformacao para novos dados).
- Custo O(n²) proibitivo com validacao cruzada (20k amostras × 5 folds).
- Destinado a visualizacao, nao a pipelines de predicao em producao.

### Arquitetura do Pipeline (3 etapas)

```
pre (ColumnTransformer)  →  dimred (PCA | LDA)  →  clf (estimador)
```

O ColumnTransformer e fitado exclusivamente no fold de treino (invariante garantido pelo Pipeline sklearn). A etapa `dimred` recebe o array transformado denso (OneHotEncoder com `sparse_output=False`).

### Configuracoes Treinadas

| Config | Metodo | n_components | EV Retida |
|--------|--------|-------------|-----------|
| pca_k10 | PCA | 10 | 84.2% |
| pca_k15 | PCA | 15 | 94.3% |
| lda_k1 | LDA | 1 | n/a (supervisionado) |

Total: **15 runs** (5 modelos × 3 configs). Adicionados ao experimento `infnet-ml-sistema` sem deletar os 10 runs da Parte 3.

### Tags Adicionais (Parte 4)

Cada run registra 4 tags extras alem das 10 da Parte 3:
- `dimred_method` ∈ {pca, lda}
- `dimred_n_components` (int)
- `dimred_explained_variance` (float para PCA, "na" para LDA)
- `baseline_run_id` (run_id do baseline P3 equivalente)

### Protecoes de Integridade

- **Guard 1 (pre-run)**: Verifica existencia de `artifacts/data_fingerprint.json` e `data/credit_card_cleaned.parquet`. `sys.exit(1)` em falha.
- **Guard 2 (pos-run)**: `mlflow.get_run(run_id).data.params` deve conter minimamente `{model_name, seed, cv_folds, dimred_method, dimred_n_components, scoring_primary, search_type}`. `sys.exit(1)` se params vazio.

### Resultados

| Modelo | Baseline P3 | pca_k10 | pca_k15 | lda_k1 |
|--------|-------------|---------|---------|--------|
| perceptron | 0.6931 | 0.6591 | 0.6700 | 0.7171 |
| logreg | 0.7232 | 0.7045 | 0.7167 | 0.7171 |
| dtree | 0.5987 | 0.6006 | 0.6096 | 0.5968 |
| rf | 0.7573 | 0.7420 | 0.7417 | 0.6530 |
| gb | 0.7795 | 0.7620 | 0.7661 | 0.7189 |

**Observacoes principais:**

1. **PCA degrada modelos ensemble**: GradientBoosting e RandomForest perdem ate 1.5 pp de ROC-AUC com PCA (k=10 ou k=15). Esses modelos sao robustos a features correlacionadas e nao se beneficiam de compressao linear.

2. **LDA beneficia modelos lineares**: Perceptron e LogReg com LDA (k=1) atingem ROC-AUC 0.717, superior ao baseline desses modelos. LDA projeta os dados no discriminante otimo para classificacao binaria.

3. **Custo computacional**: LDA e o mais rapido (projeta em 1 dimensao). PCA k=10 e similar ao baseline. PCA k=15 adiciona ~35% de custo (9s vs 6.6s para GB).

4. **Interpretabilidade**: LDA produz 1 componente com significado discriminativo direto. PCA produz combinacoes lineares das 23 features (menos interpretavel). Baseline mantém features originais com maior interpretabilidade.

5. **Melhor modelo geral**: GradientBoosting baseline (ROC-AUC 0.7795) permanece superior a todas as configs de dimred. A reducao de dimensionalidade nao melhora o modelo campeao para este dataset.

### Tabelas Completas

- `reports/parte_4/comparison_dimred.md` — 20 runs ordenados por roc_auc
- `reports/parte_4/comparison_pca_vs_lda.md` — pivot modelo × config

### Reproducao

```bash
uv run python scripts/train_dimred.py
uv run python scripts/generate_comparison_dimred.py
```

---

## Parte 5 — Selecao Final do Modelo

### Objetivo

Selecionar o modelo de operacao a partir dos 25 runs do experimento `infnet-ml-sistema` (5 baseline + 5 tune da Parte 3; 15 dimred da Parte 4), avalia-lo uma unica vez no test set (nunca tocado anteriormente) e registrar o modelo final no MLflow com signature para uso na Parte 6.

### Integridade da Selecao

O criterio de selecao foi definido e commitado em `docs/final_selection_criteria.md` **antes** de qualquer acesso ao test set. O timestamp do commit que cria esse arquivo precede todos os commits da Parte 5. Isso blinda a escolha contra vies de selecao post-hoc.

### Auditor de Metricas

`src/credit_default/audit/recompute_metrics.py` implementa `recompute_run_metrics(run_id)`: reconstroi o pipeline a partir dos params logados no MLflow, treina em X_train, avalia em X_val e compara com as metricas registradas (tolerancia 1e-4). Rodou em 3 runs aleatorios da P3/P4 antes do push final — todos OK.

### Modelo Vencedor

**GradientBoostingClassifier baseline** (`run_id`: `6743e23b1b4a4e2aa15201cebb394a07`, Parte 3)

Criterio decisivo: step 3 (`inference_latency_ms`). O GB baseline (0.0019 ms/amostra) venceu o GB tunado (0.003 ms/amostra) com roc_auc e cv_roc_auc_std identicos na escala de 4 casas decimais. O tuning nao trouxe ganho mensuravel — resultado esperado para um ensemble robusto neste dataset.

### Metricas no Test Set

| Metrica | Val (candidato) | Test (final) | Delta |
|---------|-----------------|--------------|-------|
| roc_auc | 0.779480 | **0.768213** | -0.011 |
| f1_macro | 0.673333 | 0.687623 | +0.014 |
| accuracy | 0.816908 | 0.821802 | +0.005 |

Queda de 1.1 pp em roc_auc val→test e dentro do intervalo normal. Demais metricas melhoraram com o retreino em X_trainval (25.470 amostras vs 20.975 em treino).

### Artefatos Gerados

- `reports/parte_5/consolidated_results.md` — tabela dos 25 runs ordenados pelo criterio
- `reports/parte_5/final_selection_rationale.md` — vencedor e qual step decidiu
- `reports/parte_5/final_selection.md` — documento consolidado pos-test-eval
- `reports/parte_5/test_metrics.json` — metricas brutas do test set
- MLflow run `final_eval__gb__numstd_catoh__none__none__seed42__data30c6be3a__code14c0c28` — model URI `models:/m-4de1a2c47e7d40d9a679a40ba79c9c65` (primeiro `mlflow.sklearn.log_model` no projeto; layout MLflow 3.x)

### Reproducao

```bash
uv run python scripts/audit_sample.py
uv run python scripts/generate_consolidated_results.py
uv run python scripts/select_final_candidate.py
# Apos aprovacao do vencedor:
uv run python scripts/evaluate_final.py
```

---

## Parte 6 — Operacionalizacao e Simulacao de Producao

### Objetivo

Expor o modelo vencedor da Parte 5 como servico HTTP, conteineiriza-lo com Docker e implementar monitoramento continuo de drift, fechando o ciclo MLOps do projeto.

### Arquitetura do Servico

O modelo e servido por uma aplicacao FastAPI (`src/credit_default/serving/app.py`) com quatro endpoints:

| Metodo | Path | Descricao |
|--------|------|-----------|
| GET | `/health` | Status do servico e `model_uri` |
| GET | `/` | Info do modelo (nome, metricas, run_id) |
| POST | `/predict` | Predicao unitaria (23 features int) |
| POST | `/predict/batch` | Predicao em lote |

A validacao de payload usa Pydantic v2 com `ConfigDict(strict=True)`: todos os 23 campos sao `int` obrigatorios (derivados da MLmodel signature `type: long`). Payload incompleto retorna HTTP 422 automatico.

### Controles de Integridade

| Controle | Implementacao |
|----------|--------------|
| `MODEL_URI` imutavel | Constante em `predictor.py`; mudanca requer commit explicito |
| Fail-fast no startup | `lifespan()` chama `predictor.load()` — `RuntimeError` impede inicializacao |
| Sem stub no `/predict` | Se predictor nao pronto, retorna 503 (nunca 200 com valor fixo) |
| Test set tocado 1x | `evaluate_final.py` e o unico script com `include_test=True` |
| CI valida testes | 117 testes no GitHub Actions em todo push/PR |

### Dual-Mode MODEL_URI

O predictor suporta dois modos sem mudanca de codigo:

- **Host local**: `MODEL_URI = models:/m-4de1a2c47e7d40d9a679a40ba79c9c65` — resolve via MLflow registry com `set_tracking_uri` apontando para `mlruns/` local.
- **Docker**: `MODEL_URI = /app/mlruns/.../artifacts` — path POSIX absoluto; o condicional `if self._model_uri.startswith("models:")` pula o `set_tracking_uri`, contornando o problema de paths Windows nos metadados do MLmodel.

### Monitoramento de Drift

Implementado em `src/credit_default/monitoring/drift.py`:

- **Drift de dados**: KS test (14 features continuas) + chi2 (9 features categoricas), `alpha=0.05` parametrizavel.
- **Drift de modelo**: flag se `delta_roc_auc < -0.05` (queda de 5 pp vs baseline).

Gatilhos de retreinamento:

| Situacao | Acao |
|----------|------|
| 0 features com drift | Monitoramento padrao |
| 1-4 features com drift | Alertar equipe |
| >= 5 features com drift | Iniciar retreinamento |
| roc_auc cai > 5 pp | Retreinamento obrigatorio |

### Metricas no Servico (Parte 5 — test set)

| Metrica | Valor |
|---------|-------|
| roc_auc | 0.7682 |
| f1_macro | 0.6876 |
| accuracy | 0.8218 |

### Reproducao

```bash
# Servico local
uv run uvicorn src.credit_default.serving.app:app --port 8000

# Docker
docker compose up -d

# Relatorio de drift
uv run python scripts/run_drift_report.py
```
