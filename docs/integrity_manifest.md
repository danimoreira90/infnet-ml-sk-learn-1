# Manifesto de Integridade — Parte 2

Registro dos controles de integridade, anomalias documentadas e decisões de design da fundação de dados.

---

## Política de Histórico Git

**O histórico git não é reescrito após commits serem feitos. Fixes são commits atômicos.**

Razão: cada commit — incluindo commits `fix(scope)` — representa um evento real de engenharia com causa identificável, sintoma e correção documentados. Reescrever o histórico (rebase interativo, amend em commits já existentes) apagaria rastreabilidade que este manifesto exige preservar.

Convenção adotada: `fix(scope): descrição` é um tipo válido na spec de Conventional Commits e carrega o mesmo peso de auditoria que `feat` ou `docs`. A contagem de commits da Parte 2 é **15 (13 features/docs + 2 fixes atômicos)**:

| Commit | Tipo | Causa do fix |
|--------|------|-------------|
| `61d929f` | `fix(repo)` | `build-backend = "setuptools.backends.legacy:build"` não existe em setuptools moderno |
| `f9004cb` | `fix(repo)` | `.gitignore` sem âncora `/` bloqueava `src/credit_default/data/` |

Ambos os fixes estão detalhados em `IMPLEMENTATION_REPORT.md`.

---

## Controles de Integridade

| Artefato | Script gerador | Verificação |
|----------|---------------|-------------|
| `data/credit_card_cleaned.parquet` | `build_clean_dataset.py` | SHA-256 do `.xls` verificado antes da leitura |
| `artifacts/data_fingerprint.json` | `run_data_qa.py` | Gerado exclusivamente por código; nunca editado à mão |
| `artifacts/data_schema.json` | `run_data_qa.py` | `validate()` garante ausência de NaN e duplicatas |
| `artifacts/splits/split_indices.json` | `run_data_qa.py` | `verify_splits()` garante disjunção e cobertura total antes de salvar |
| `artifacts/data_qa_summary.json` | `run_data_qa.py` | Agregação dos 7 checks; todos calculados a partir do DataFrame |
| `reports/figures/parte_2/*.png` | `run_data_qa.py` | Gerados em runtime; `.gitkeep` versiona o diretório |
| `uv.lock` | `uv lock` | Versionado no repo; consistência verificável com `uv lock --check` |

---

## DATAHASH8 — Uso Futuro na Parte 3

O campo `file_short` (primeiros 8 chars do SHA-256 do arquivo bruto) está presente em:
- `artifacts/data_fingerprint.json` → campo `file_short`
- `artifacts/splits/split_indices.json` → campo `fingerprint_short`

Finalidade: será usado como prefixo no naming de MLflow runs na Parte 3
(`DATAHASH8_<modelo>_<timestamp>`), garantindo rastreabilidade entre experimentos e o dataset exato que os produziu.

SHA-256 completo do arquivo bruto (64 chars) também está em `split_indices.json` para verificação por auditores.

---

## Anomalias Documentadas (não recodificadas)

### EDUCATION — códigos fora da especificação UCI

A especificação UCI define: `{1=graduate school, 2=university, 3=high school, 4=others}`.

O dataset contém adicionalmente os valores `{0, 5, 6}`, não documentados na especificação original.

**Decisão:** os dados são preservados sem recodificação. `validate()` emite warning explícito. A recodificação, se necessária, é responsabilidade da Parte 3 (pipeline com `ColumnTransformer`).

### MARRIAGE — códigos fora da especificação UCI

A especificação UCI define: `{1=married, 2=single, 3=others}`.

O dataset contém adicionalmente o valor `{0}`, não documentado.

**Decisão:** idem EDUCATION — preservado, warning emitido, nunca silenciado.

---

## Divergência de Split — não é inconsistência

Os notebooks `02_baseline_perceptron.ipynb` a `06_final_report.ipynb` utilizam split **70/30** (train/test).

O código `src/credit_default/data/splits.py` implementa split **70/15/15** (train/val/test).

**Isso é uma mudança deliberada da Parte 2, não um bug.**

Motivação: a introdução de um conjunto de validação separado é necessária para o ciclo de tuning de hiperparâmetros da Parte 3 sem contaminar o conjunto de teste.

Os notebooks existentes **não foram modificados** e continuam utilizando o split original para garantir reprodutibilidade dos resultados já reportados.

---

## `check_leakage_risk` — apenas documentação

A função `diagnostics.check_leakage_risk()` **não corrige** o risco que documenta.

**Risco identificado:** os notebooks 02–06 utilizam `StandardScaler` sem `sklearn.pipeline.Pipeline`. Há risco de que o scaler tenha sido ajustado (`.fit()`) antes do split ou sobre train+test, introduzindo leakage de normalização. As métricas reportadas nos notebooks podem estar ligeiramente otimistas.

**Ação:** a correção é responsabilidade da Parte 3, que encapsulará todas as transformações em `Pipeline` com `ColumnTransformer`, garantindo que nenhuma estatística de transformação vaze do conjunto de treino para validação ou teste.

---

## Duplicatas no Dataset Bruto

**Causa confirmada por inspeção:** as 35 duplicatas são **aparentes, não estruturais**.

Verificação executada:

```python
import pandas as pd
df = pd.read_excel('../data/default of credit card clients.xls', header=1)
print('com ID:', df.duplicated().sum())   # → 0
print('sem ID:', df.drop(columns=['ID']).duplicated().sum())  # → 35
```

Com o campo `ID` presente nenhuma linha é duplicata (`0` duplicatas). Ao dropar `ID`, 35 pares de clientes **distintos** (IDs diferentes) compartilham exatamente os mesmos valores nas outras 23 features — colisão de features, não corrupção de dados.

**Impacto em modelagem:** 35 de 30000 = 0,12% das linhas. Desprezível em magnitude, mas documentado para rastreabilidade completa.

**Tratamento:** `build_clean_dataset.py` remove as 35 linhas redundantes pós-drop-ID com log visível (`AVISO: 35 linha(s) exatamente duplicada(s) removida(s) (30000 -> 29965 linhas)`). O parquet limpo tem **29965 linhas**.

Este fato está registrado aqui e no log de execução do script. Nunca foi silenciado.

---

## Parte 3 — Controle: Parametros MLflow Verificados em Runtime

**Controle introduzido:** `fix(tracking)` — commit 6ea3d3f

**Problema detectado:** os 10 runs MLflow gerados na implementacao inicial (C11+C12) tinham `params/` vazia (`ParamsCount=0`). A funcao `log_standard_params()` nao existia em `mlflow_utils.py` e nao era chamada em `train.py`.

**Correcao aplicada:**
- `log_standard_params()` adicionada a `src/credit_default/tracking/mlflow_utils.py`: loga 8 params meta (model_name, seed, cv_folds, scoring_primary="roc_auc", split_strategy, search_type, n_train, n_val) + todos clf__* hiperparametros do estimador (escalares).
- `train.py` extrai `clf_params` pos-treino (baseline: `pipeline.get_params()` filtrado; tuned: `search_obj.best_params_`) e chama `log_standard_params()` dentro do `start_run`.
- 5 testes em `tests/test_mlflow_utils.py` verificam presenca dos params obrigatorios.

**Verificacao pos-correcao (mlruns regenerado):**

| Run | ParamsCount | clf__keys | scoring_primary |
|-----|-------------|-----------|-----------------|
| tune__rf | 12 | 4 | roc_auc |
| tune__perceptron | 11 | 3 | roc_auc |
| tune__logreg | 11 | 3 | roc_auc |
| tune__gb | 12 | 4 | roc_auc |
| tune__dtree | 11 | 3 | roc_auc |
| baseline__rf | 27 | 19 | roc_auc |
| baseline__perceptron | 28 | 20 | roc_auc |
| baseline__logreg | 22 | 14 | roc_auc |
| baseline__gb | 28 | 20 | roc_auc |
| baseline__dtree | 21 | 13 | roc_auc |

**Regra:** params loggeados sao verificados por script (`uv run python -c "..."`) e por suite de testes — nunca apenas por documentacao. O DoD exige `ParamsCount >= 8` e `clf__keys >= 3` por run.
