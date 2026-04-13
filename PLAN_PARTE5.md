# PLAN: Parte 5 — Consolidação e Seleção Final do Modelo de Operação

**Date**: 2026-04-13
**Status**: ready-for-approval
**Branch**: `main` (HEAD: commits da Parte 4 publicados)
**Parte anterior**: Parte 4 completa e publicada em origin/main.

> **For agentic workers:** REQUIRED: Use subagent-driven-development for every task.
> Each task = one fresh subagent. Spec compliance review → code quality review → done.

---

## 1. Confirmação de Contexto (read-only, 5 linhas)

1. `docs/final_selection_criteria.md` — **CONTRATO VINCULANTE** commitado antes desta thread.
   Critério em cascata de 5 desempates: `roc_auc` (val) > `cv_roc_auc_std` > `inference_latency_ms` > `training_time_s` > complexidade nominal (linear < árvore única < ensemble). Requisitos obrigatórios: `predict_proba` nativo, serializável via joblib/mlflow, `training_time_s` registrado.
2. `artifacts/splits/split_indices.json` — `train_idx` (20975), `val_idx` (4495), `test_idx` (4495).
   **`test_idx` NUNCA foi tocado até aqui.** Parte 5 é o único e único uso do test set.
3. MLflow experimento `"infnet-ml-sistema"` (id `236665223173386020`) — 25 runs existentes: 5 baseline (P3) + 5 tune (P3) + 15 dimred (P4). Todos têm params/metrics logados, **mas nenhum tem `mlflow.sklearn.log_model`**. Retreino do zero obrigatório em `evaluate_final.py`.
4. `src/credit_default/models/registry.py` — MODEL_REGISTRY com 5 modelos. Complexidade nominal para desempate: perceptron/logreg = linear; dtree = árvore única; rf/gb = ensemble. Perceptron usa `CalibratedClassifierCV` — tem `predict_proba`.
5. `src/credit_default/tracking/run_naming.py` — `compose_run_name()` gera nome padrão 8-partes. Final eval usa stage=`"final_eval"` e 3 tags extras obrigatórias: `evaluation_set`, `candidate_run_id`, `criterion_source`.

---

## 2. Arquitetura de Módulos e Scripts Novos

```
src/credit_default/audit/
├── __init__.py                           <- exporta RecomputeResult, recompute_run_metrics
└── recompute_metrics.py                  <- auditor de integridade de métricas

scripts/
├── generate_consolidated_results.py      <- lê 25 runs MLflow, gera tabela ordenada
├── select_final_candidate.py             <- aplica critério cascata, SEM tocar test_idx
├── evaluate_final.py                     <- avalia no test set (UMA VEZ, retreina do zero)
└── audit_sample.py                       <- roda auditor em 3+ runs aleatórios P3/P4

tests/
├── test_recompute_metrics.py             <- TDD para o auditor (RED->GREEN->REFACTOR)
└── test_parte5_smoke.py                  <- smoke tests para scripts e artefatos novos

reports/parte_5/
├── consolidated_results.md               <- tabela dos 25 runs ordenados por critério
├── final_selection_rationale.md          <- vencedor + qual desempate decidiu
└── final_selection.md                    <- documento consolidado pós-test-eval
```

---

## 3. Diff-Plan: 12 Commits em Conventional Commits

### Commit 1 — Estrutura de diretórios e módulo audit stub
```
chore(parte5): cria diretórios reports/parte_5 e src/credit_default/audit
```
Arquivos criados:
- `reports/parte_5/.gitkeep`
- `src/credit_default/audit/__init__.py` (stub vazio com docstring)
- `src/credit_default/audit/recompute_metrics.py` (stub: funções com `raise NotImplementedError`)

---

### Commit 2 — TDD RED: testes falhos para recompute_metrics
```
test(parte5): adiciona testes TDD red para recompute_metrics
```
Arquivo: `tests/test_recompute_metrics.py`

6 testes que DEVEM FALHAR neste commit (antes da implementação):

```python
# test_recompute_result_is_dataclass
# Importa RecomputeResult, verifica campos: run_id (str), ok (bool), mismatches (dict)

# test_recompute_matching_run_returns_ok
# Mock MlflowClient.get_run com params de um run válido e métricas corretas.
# recompute_run_metrics(run_id, parquet_path=..., split_path=...) -> ok=True

# test_recompute_metric_mismatch_exits_nonzero
# Métricas logadas diferem das recomputadas por 0.01 (> tolerância 1e-4).
# Espera SystemExit com código != 0.

# test_recompute_tolerates_float_rounding
# Diferença de 1e-5 (< tolerância) -> ok=True, mismatches={}

# test_recompute_unknown_model_raises_keyerror
# params["model_name"] = "modelo_inexistente" -> KeyError

# test_recompute_missing_split_artifact_raises
# split_path aponta para arquivo inexistente -> FileNotFoundError
```

Verificação obrigatória: `pytest tests/test_recompute_metrics.py -v` → todos FAIL

---

### Commit 3 — TDD GREEN: implementação do auditor
```
feat(parte5): implementa recompute_metrics (auditor de integridade)
```

**`src/credit_default/audit/recompute_metrics.py`**:

```python
"""Auditor de integridade de métricas MLflow."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.tracking import MlflowClient

from credit_default.evaluation.metrics import compute_all_metrics
from credit_default.features.dimred import build_dimred_pipeline
from credit_default.models.pipeline import build_pipeline

TOLERANCE = 1e-4
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PARQUET = _REPO_ROOT / "data" / "credit_card_cleaned.parquet"
DEFAULT_SPLIT   = _REPO_ROOT / "artifacts" / "splits" / "split_indices.json"
TARGET_COL = "default payment next month"
VAL_METRICS = ["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"]


@dataclass
class RecomputeResult:
    run_id: str
    ok: bool
    mismatches: dict[str, tuple[float, float]] = field(default_factory=dict)


def recompute_run_metrics(
    run_id: str,
    *,
    parquet_path: Path = DEFAULT_PARQUET,
    split_path: Path = DEFAULT_SPLIT,
    tolerance: float = TOLERANCE,
) -> RecomputeResult:
    """Reconstrói pipeline do run_id, treina em X_train, avalia em X_val.

    Compara métricas recomputadas com as logadas (tolerância = 1e-4).
    Chama sys.exit(1) se qualquer métrica exceder tolerância.
    Nunca lê test_idx.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {parquet_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices não encontrado: {split_path}")

    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    logged_metrics = run.data.metrics

    # Carregar dados (somente train_idx e val_idx — test_idx nunca lido aqui)
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    with open(split_path) as f:
        split_info = json.load(f)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train = X.iloc[split_info["train_idx"]].reset_index(drop=True)
    y_train = y.iloc[split_info["train_idx"]].reset_index(drop=True)
    X_val   = X.iloc[split_info["val_idx"]].reset_index(drop=True)
    y_val   = y.iloc[split_info["val_idx"]].reset_index(drop=True)

    # Reconstruir pipeline
    model_name    = params["model_name"]
    seed          = int(params.get("seed", "42"))
    dimred_method = params.get("dimred_method", "none")
    dimred_n      = int(params.get("dimred_n_components", "0"))

    if dimred_method and dimred_method != "none" and dimred_n > 0:
        pipeline = build_dimred_pipeline(model_name, dimred_method, dimred_n, seed=seed)
    else:
        pipeline = build_pipeline(model_name, seed=seed)

    # Aplicar best_params_ (parâmetros clf__* logados no MLflow)
    clf_params = {k: _cast_param(v) for k, v in params.items() if k.startswith("clf__")}
    if clf_params:
        pipeline.set_params(**clf_params)

    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    recomputed = compute_all_metrics(y_val.to_numpy(), y_pred, y_proba)

    mismatches: dict[str, tuple[float, float]] = {}
    for metric in VAL_METRICS:
        logged_val = logged_metrics.get(metric)
        if logged_val is None:
            continue
        recomp_val = recomputed.get(metric, float("nan"))
        if abs(logged_val - recomp_val) > tolerance:
            mismatches[metric] = (logged_val, recomp_val)

    ok = len(mismatches) == 0
    result = RecomputeResult(run_id=run_id, ok=ok, mismatches=mismatches)

    if not ok:
        print(f"[MISMATCH] run_id={run_id}", flush=True)
        for m, (log_v, rec_v) in mismatches.items():
            diff = abs(log_v - rec_v)
            print(f"  {m}: logged={log_v:.6f} recomputed={rec_v:.6f} diff={diff:.2e}", flush=True)
        sys.exit(1)

    print(f"[OK] run_id={run_id} — todas métricas dentro da tolerância {tolerance}", flush=True)
    return result


def _cast_param(v: str) -> Any:
    """Converte string de param MLflow para int, float, bool ou None."""
    if v == "None":
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    for fn in (int, float):
        try:
            return fn(v)
        except (ValueError, TypeError):
            pass
    return v
```

**`src/credit_default/audit/__init__.py`**:
```python
"""Módulo de auditoria de integridade de métricas MLflow."""
from credit_default.audit.recompute_metrics import RecomputeResult, recompute_run_metrics

__all__ = ["RecomputeResult", "recompute_run_metrics"]
```

Verificação obrigatória: `pytest tests/test_recompute_metrics.py -v` → todos PASS

---

### Commit 4 — TDD REFACTOR do auditor (função reutilizável em evaluate_final)
```
refactor(parte5): extrai _load_splits com include_test=False no auditor
```
Sem mudança de comportamento externo. Testes ainda PASS.

Extrai função privada reutilizável em `recompute_metrics.py`:

```python
def _load_splits(
    parquet_path: Path,
    split_path: Path,
    *,
    include_test: bool = False,
) -> tuple:
    """Carrega splits do dataset a partir de split_indices.json.

    Quando include_test=False (padrão): retorna (X_train, X_val, y_train, y_val).
    Quando include_test=True: retorna (X_train, X_val, X_test, y_train, y_val, y_test).

    CONTRATO ANTI-LEAKAGE: test_idx só é lido quando include_test=True.
    O auditor (recompute_run_metrics) NUNCA passa include_test=True.
    Somente evaluate_final.py passa include_test=True.
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    with open(split_path) as f:
        split_info = json.load(f)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train = X.iloc[split_info["train_idx"]].reset_index(drop=True)
    y_train = y.iloc[split_info["train_idx"]].reset_index(drop=True)
    X_val   = X.iloc[split_info["val_idx"]].reset_index(drop=True)
    y_val   = y.iloc[split_info["val_idx"]].reset_index(drop=True)
    if not include_test:
        return X_train, X_val, y_train, y_val
    X_test = X.iloc[split_info["test_idx"]].reset_index(drop=True)
    y_test = y.iloc[split_info["test_idx"]].reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

`recompute_run_metrics` passa a chamar: `X_train, X_val, y_train, y_val = _load_splits(parquet_path, split_path)` (include_test=False implícito).
`evaluate_final.py` (commit 9) importa `_load_splits` de `credit_default.audit.recompute_metrics` e chama com `include_test=True`.

Benefício concreto: a função é testada via `test_recompute_matching_run_returns_ok` (auditor), e reutilizada em `evaluate_final.py` — o refactor tem usuário real, não é abstração prematura.

---

### Commit 5 — script audit_sample.py
```
feat(parte5): audit_sample.py roda auditor em 3 runs aleatórios P3/P4
```
Arquivo: `scripts/audit_sample.py`

Lógica:
1. `EXP_ID = "236665223173386020"`
2. `MlflowClient().search_runs(experiment_ids=[EXP_ID])` → todos os runs
3. Filtra runs com tag `project_part` in `["parte_3", "parte_4"]`
4. `random.seed(42); sample = random.sample(eligible_runs, k=min(3, len(eligible_runs)))`
5. Para cada run: `recompute_run_metrics(run.info.run_id)` (exit 1 se falhar)
6. Imprime: `"[AUDIT SAMPLE] N/N runs auditados OK"`
7. `sys.exit(0)` ao final

---

### Commit 6 — generate_consolidated_results.py
```
feat(parte5): gera tabela consolidada dos 25 runs ordenada pelo critério
```
Arquivo: `scripts/generate_consolidated_results.py`

Lógica:
1. `MlflowClient().search_runs(experiment_ids=[EXP_ID])` → 25 runs
2. Para cada run extrai: `run_id` (8 chars), `run_name`, `stage` tag, `model_family` tag,
   `dimred_method` tag, `dimred_n_components` tag, `roc_auc`, `cv_roc_auc_mean`,
   `cv_roc_auc_std`, `inference_latency_ms`, `training_time_s`
3. `complexity_rank(model_family) -> int`: `"linear"→0, "tree"→1, "ensemble"→2`
4. Ordena: `roc_auc desc`, `cv_roc_auc_std asc`, `inference_latency_ms asc`,
   `training_time_s asc`, `complexity_rank asc`
5. Gera `reports/parte_5/consolidated_results.md`:
   - Header com data e fonte do critério
   - Tabela markdown de 25 linhas com colunas:
     `rank | run_id | stage | model_family | dimred | roc_auc_val | cv_roc_auc_std | latency_ms | train_s`

---

### Commit 7 — select_final_candidate.py
```
feat(parte5): select_final_candidate aplica critério cascata e gera rationale
```
Arquivo: `scripts/select_final_candidate.py`

**PROTEÇÃO HARD**: nenhuma linha do arquivo referencia `"test_idx"`.

Lógica:
1. Imprime caminho absoluto de `docs/final_selection_criteria.md` (trilha de auditoria)
2. `MlflowClient().search_runs(experiment_ids=[EXP_ID])` → 25 runs
3. Verifica requisitos obrigatórios: elimina runs com `training_time_s` ausente ou NaN
4. Aplica cascata de 5 critérios em estrito (delta de empate = `1e-4`):
   - **Step 1**: maior `roc_auc` → se único, encerra
   - **Step 2**: menor `cv_roc_auc_std` entre empatados → se único, encerra
   - **Step 3**: menor `inference_latency_ms` → se único, encerra
   - **Step 4**: menor `training_time_s` → se único, encerra
   - **Step 5**: menor complexidade nominal (linear < tree < ensemble); desempate final por `run_id` lexicográfico se mesmo tier
5. Registra `decision_step` (1–5) e `decision_reason` (texto descritivo)
6. Gera `reports/parte_5/final_selection_rationale.md`:

```markdown
# Seleção Final do Modelo — Rationale

**Critério fonte**: docs/final_selection_criteria.md

## Vencedor

- winner_run_id: <run_id completo>
- winner_run_name: <run_name>
- decision_step: <1-5>
- decision_reason: <texto>

## Interpretações Aplicadas

1. predict_proba "nativo": inclui CalibratedClassifierCV (parte do pipeline treinado, não pós-processamento do usuário).
2. Desempate intra-tier no step 5: ordem lexicográfica do run_id (determinista).

## Top-5 Candidatos

| rank | run_id | roc_auc_val | cv_roc_auc_std | latency_ms | train_s | model_family |
|------|--------|-------------|----------------|------------|---------|--------------|
| ...  | ...    | ...         | ...            | ...        | ...     | ...          |
```

7. Imprime sumário no stdout.

---

### Commit 8 — artefatos gerados (PAUSA obrigatória)
```
docs(parte5): consolidated_results e rationale gerados
```
Arquivos commitados:
- `reports/parte_5/consolidated_results.md`
- `reports/parte_5/final_selection_rationale.md`

**⚠️ PAUSA OBRIGATÓRIA APÓS ESTE COMMIT.**

Mostrar ao usuário:
1. Conteúdo completo de `reports/parte_5/consolidated_results.md`
2. Conteúdo completo de `reports/parte_5/final_selection_rationale.md`
3. Aguardar confirmação explícita: *"pode prosseguir para evaluate_final"*

---

### Commit 9 — evaluate_final.py (SOMENTE após confirmação)
```
feat(parte5): evaluate_final retreina vencedor e avalia no test set (única execução)
```
Arquivo: `scripts/evaluate_final.py`

**PROTEÇÃO HARD**: `test_idx` acessado exclusivamente via `_load_splits(..., include_test=True)` — importado de `credit_default.audit.recompute_metrics`. Nenhuma outra linha do repo referencia `"test_idx"` diretamente.

Lógica completa:
1. Lê `reports/parte_5/final_selection_rationale.md`, extrai `winner_run_id:` via regex
2. `MlflowClient().get_run(winner_run_id)` → params e tags do vencedor
3. Determina arquitetura do pipeline:
   - `model_name = params["model_name"]`
   - `dimred_method = params["dimred_method"]`
   - `dimred_n = int(params["dimred_n_components"])`
   - Se `dimred_method != "none" and dimred_n > 0`: `build_dimred_pipeline(model_name, dimred_method, dimred_n, seed=42)`
   - Caso contrário: `build_pipeline(model_name, seed=42)`
4. Aplica `clf__*` params do run vencedor: `pipeline.set_params(**clf_params)`
5. Carrega splits via `_load_splits(parquet_path, split_path, include_test=True)`:
   - Retorna `(X_train, X_val, X_test, y_train, y_val, y_test)`
   - X_trainval = `pd.concat([X_train, X_val]).reset_index(drop=True)`
   - y_trainval = `pd.concat([y_train, y_val]).reset_index(drop=True)`
   - X_test e y_test usados exclusivamente para avaliação final
6. `t0 = time.perf_counter()` → `pipeline.fit(X_trainval, y_trainval)` → `training_time_s`
7. Avalia UMA VEZ em X_test:
   - `y_pred = pipeline.predict(X_test)`
   - `y_proba = pipeline.predict_proba(X_test)[:, 1]`
   - `test_metrics = compute_all_metrics(y_test.to_numpy(), y_pred, y_proba)`
8. Mede `inference_latency_ms` (tempo médio por amostra em X_test × 1000)
9. Gera artefatos em `tempfile.mkdtemp()`:
   - `confusion_matrix_test.png` (via `confusion_matrix_plot`)
   - `roc_curve_test.png` (via `roc_plot`)
   - `test_metrics.json` (dict com `test_metrics` + `training_time_s` + `inference_latency_ms`)
10. Constrói `dimred_tag_str` a partir dos params do vencedor:
    - `"pca"` + n=10 → `"pca_k10"` | n=15 → `"pca_k15"`
    - `"lda"` + n=1 → `"lda_k1"`
    - `"none"` → `"none"`
11. `githash7` via `subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"])`
12. `datahash8` de `artifacts/data_fingerprint.json["file_short"]`
13. `run_name = compose_run_name("final_eval", model_name, dimred=dimred_tag_str, seed=42, datahash8=datahash8, githash7=githash7)`
14. MLflow run:
    ```python
    from mlflow.models import infer_signature

    signature = infer_signature(X_test.head(5), y_pred[:5])

    with mlflow.start_run(run_name=run_name, experiment_id=EXP_ID) as mlrun:
        log_standard_tags(mlrun, model_family=..., git_commit=githash7,
                          dataset_fingerprint=datahash8,
                          compute_profile_s=training_time_s, project_part="parte_5")
        mlflow.set_tags({
            "evaluation_set": "test",
            "candidate_run_id": winner_run_id,
            "criterion_source": "docs/final_selection_criteria.md",
        })
        mlflow.log_metrics({**test_metrics, "training_time_s": training_time_s,
                            "inference_latency_ms": inference_latency_ms})
        log_standard_artifacts(mlrun, Path(tmp_dir))
        mlflow.sklearn.log_model(          # PRIMEIRA VEZ no projeto
            pipeline,
            artifact_path="model",
            signature=signature,           # schema de input/output para Parte 6
            input_example=X_test.head(3), # exemplo real para validação de payload
        )
    ```
    Nota: `infer_signature` recebe `X_test.head(5)` (DataFrame) e `y_pred[:5]` (array 1-D).
    O `input_example` com 3 linhas reais do test set é suficiente para o FastAPI da Parte 6
    validar o schema do payload sem expor o test set completo.
15. Salva cópia local: `reports/parte_5/test_metrics.json`
16. Imprime tabela val vs test:
    ```
    Métrica          | Val (candidato) | Test (final)  | Delta
    -----------------|-----------------|---------------|-------
    roc_auc          | 0.XXXXXX        | 0.XXXXXX      | -0.XXX
    f1_macro         | 0.XXXXXX        | 0.XXXXXX      | -0.XXX
    ...
    ```

---

### Commit 10 — final_selection.md
```
docs(parte5): relatório final_selection com tabela 25 runs e métricas test set
```
Arquivo: `reports/parte_5/final_selection.md`

Seções obrigatórias:
1. `## Critério de Seleção` — transcrito de `docs/final_selection_criteria.md` (sem edição)
2. `## Universo de Candidatos` — tabela dos 25 runs (val metrics), da consolidated_results.md
3. `## Vencedor` — extraído de `final_selection_rationale.md`
4. `## Métricas no Test Set` — de `test_metrics.json`
5. `## Comparação Val vs Test` — tabela: roc_auc, f1_macro, precision_macro, recall_macro, accuracy + delta
6. `## Análise` — comentário sobre delta val→test (esperado: test ≤ val em roc_auc; sinalizar se contrário mas NÃO ajustar)

---

### Commit 11 — relatorio_tecnico.md + README
```
docs(parte5): adiciona seção Parte 5 ao relatório técnico e README
```
Arquivos modificados:
- `reports/relatorio_tecnico.md` — nova seção `## Parte 5 — Seleção Final do Modelo`
  Subseções: Metodologia de seleção, Integridade (timestamping do critério), Auditor de métricas, Vencedor, Métricas no test set.
- `README.md` — nova seção `## Como rodar a Parte 5`:
  ```bash
  # 1. Tabela consolidada dos 25 runs
  python scripts/generate_consolidated_results.py

  # 2. Selecionar candidato final (não toca test_idx)
  python scripts/select_final_candidate.py

  # 3. [Após aprovação do usuário] Avaliar no test set — UMA VEZ
  python scripts/evaluate_final.py

  # 4. Auditar integridade de 3 runs aleatórios P3/P4
  python scripts/audit_sample.py
  ```

---

### Commit 12 — smoke tests
```
test(parte5): smoke tests para módulo audit e artefatos gerados
```
Arquivo: `tests/test_parte5_smoke.py`

Testes:
- `test_audit_module_importable` — `from credit_default.audit import RecomputeResult, recompute_run_metrics`
- `test_consolidated_results_file_exists` — `Path("reports/parte_5/consolidated_results.md").exists()`
- `test_rationale_has_winner_run_id` — arquivo existe e contém linha `winner_run_id:`
- `test_test_metrics_json_exists_and_valid` — `reports/parte_5/test_metrics.json` existe e é JSON com chave `roc_auc`
- `test_cast_param_int` — `_cast_param("42") == 42`
- `test_cast_param_float` — `_cast_param("0.1") == pytest.approx(0.1)`
- `test_cast_param_none` — `_cast_param("None") is None`
- `test_cast_param_bool_true` — `_cast_param("True") is True`

Verificação final: `pytest -v --tb=short` → TODOS PASS (37+ existentes + novos)

---

## 4. Riscos e Lacunas do Critério

### Riscos Técnicos

| # | Risco | Prob. | Mitigação |
|---|-------|-------|-----------|
| R1 | Empate total nos 5 steps | Muito baixa | Desempate por `run_id` lexicográfico (determinista). Registrar no rationale. |
| R2 | `cv_roc_auc_std` ausente em algum run | Muito baixa | Guard antes do select: verificar presença. Todos os runs P3/P4 logam. |
| R3 | Retreino com search_type="random" (rf/gb) produz resultado diferente | Média | `random_state=42` sempre setado via `build_pipeline`/`build_dimred_pipeline`. Determinista. |
| R4 | `infer_signature` falha se `y_pred[:5]` for array 2-D (ex: predict retorna shape (n,1)) | Baixa | `y_pred = pipeline.predict(X_test)` → shape `(n,)` para todos os classificadores sklearn. Se necessário: `y_pred[:5].ravel()`. |
| R5 | Test metrics muito melhores que val (leakage suspeito) | Muito baixa | Reportar as-is. NUNCA ajustar. Investigar separadamente se ocorrer. |
| R6 | `dimred_tag_str` no run_name do final_eval não combina com padrão P4 | Média | Construir explicitamente: `"pca"+"k10"→"pca_k10"`, `"lda"+"k1"→"lda_k1"`, senão `"none"`. |

### Lacunas no Critério (SINALIZADAS — sem alterar `docs/final_selection_criteria.md`)

1. **Step 5 não define desempate intra-tier**: Se dois runs do mesmo tier de complexidade empatam nos 4 primeiros critérios, o contrato não especifica desempate. **Tratamento**: `run_id` lexicográfico como desempate final determinista. Registrado no `final_selection_rationale.md`. O arquivo de critério **NÃO será modificado**.

2. **`predict_proba` "nativo" vs `CalibratedClassifierCV`**: Perceptron usa `CalibratedClassifierCV(Perceptron(...))`. O contrato diz "nativo (sem calibração post-hoc adicional)". **Tratamento**: a calibração é parte do pipeline treinado (step `"clf"`), não pós-processamento do usuário. Interpretação: "nativo" = "disponível via `pipeline.predict_proba()` sem etapas externas". Registrado no rationale.

---

## 5. Checklist DoD (Definition of Done)

- [ ] `docs/final_selection_criteria.md` — NÃO modificado (`git log docs/final_selection_criteria.md` mostra exatamente 1 commit)
- [ ] `src/credit_default/audit/__init__.py` criado, exporta `RecomputeResult` e `recompute_run_metrics`
- [ ] `src/credit_default/audit/recompute_metrics.py` implementado
- [ ] `tests/test_recompute_metrics.py` — 6 testes, todos PASS
- [ ] Commit 2 verificado como RED (testes falhavam antes do commit 3)
- [ ] `scripts/audit_sample.py` — roda em ≥3 runs P3/P4, exit code 0
- [ ] `scripts/generate_consolidated_results.py` — gera `reports/parte_5/consolidated_results.md` com 25 runs
- [ ] `scripts/select_final_candidate.py` — NÃO contém referência a `"test_idx"` (grep verifica)
- [ ] `reports/parte_5/final_selection_rationale.md` — contém `winner_run_id:` e `decision_step:`
- [ ] **PAUSA EXPLÍCITA** — usuário aprovou vencedor antes do commit 9
- [ ] Inventário `test_idx`: exatamente 4 arquivos referenciam o padrão — `run_data_qa.py`, `test_data_foundation.py` (ambos Parte 2, legítimos), `recompute_metrics.py`, `evaluate_final.py` (Guard 1A)
- [ ] `include_test=True` aparece em exatamente 1 arquivo: `scripts/evaluate_final.py` (Guard 1B)
- [ ] MLflow final_eval run criado com tags: `evaluation_set="test"`, `candidate_run_id=<id>`, `criterion_source="docs/final_selection_criteria.md"`
- [ ] `mlflow.sklearn.log_model` chamado exatamente 1 vez no projeto inteiro (grep verifica)
- [ ] `reports/parte_5/test_metrics.json` — existe e é JSON válido com chave `roc_auc`
- [ ] `reports/parte_5/final_selection.md` — existe com seções val vs test
- [ ] `reports/relatorio_tecnico.md` — contém seção `## Parte 5`
- [ ] `README.md` — contém seção "Como rodar a Parte 5"
- [ ] `pytest -v --tb=short` — TODOS PASS (37+ existentes + novos)
- [ ] `python scripts/audit_sample.py` — exit code 0 (antes do push final)

---

## 6. Guards de Integridade — Comandos de Verificação

```bash
# Guard 1A: inventário descritivo de arquivos que referenciam "test_idx" (não é alarme)
# PowerShell (prioritário):
Select-String -Path scripts\*.py, src\**\*.py, tests\*.py -Pattern "test_idx" |
    Select-Object -ExpandProperty Path -Unique
# Git Bash:
grep -rln "test_idx" scripts/ src/ tests/ | grep -v "__pycache__" | sort -u
# Esperado EXATAMENTE 4 paths:
#   scripts/run_data_qa.py                         (Parte 2 — gera o split artifact)
#   tests/test_data_foundation.py                  (Parte 2 — valida geometria do split)
#   src/credit_default/audit/recompute_metrics.py  (Parte 5 — _load_splits, única definição)
#   scripts/evaluate_final.py                      (Parte 5 — único treino/eval no test set)

# Guard 1B: controle de integridade real — include_test=True em exatamente 1 arquivo
# PowerShell (prioritário):
Select-String -Path scripts\*.py, src\**\*.py, tests\*.py -Pattern "include_test=True"
# Git Bash:
grep -rn "include_test=True" scripts/ src/ tests/
# Esperado: somente scripts/evaluate_final.py (1 única ocorrência)

# Guard 2: final_selection_criteria.md com exatamente 1 commit
git log --oneline docs/final_selection_criteria.md
# Esperado: exatamente 1 linha (commit do usuário, antes desta thread)

# Guard 1C: mlflow.sklearn.log_model em exatamente 1 arquivo
# PowerShell (prioritário):
Select-String -Path scripts\*.py, src\**\*.py -Pattern "mlflow.sklearn.log_model"
# Git Bash:
grep -rn "mlflow.sklearn.log_model" scripts/ src/
# Esperado: somente scripts/evaluate_final.py

# Guard 4: auditor passa em sample
python scripts/audit_sample.py
# Esperado: "[AUDIT SAMPLE] 3/3 runs auditados OK", exit code 0

# Guard 5: todos os testes passam
pytest -v --tb=short
# Esperado: todos PASS, 0 SKIP, 0 FAIL
```

---

## 7. Sequência de Execução (PHASE B — resumo operacional)

```
Commit 1  → chore: estrutura dirs + stubs
Commit 2  → test: RED (pytest mostra FAIL)       ← verificar RED obrigatório
Commit 3  → feat: GREEN (pytest mostra PASS)     ← verificar PASS obrigatório
Commit 4  → refactor: auditor limpo
Commit 5  → feat: audit_sample.py
Commit 6  → feat: generate_consolidated_results.py
Commit 7  → feat: select_final_candidate.py

  [EXECUTAR]: python scripts/audit_sample.py                    (exit 0)
  [EXECUTAR]: python scripts/generate_consolidated_results.py
  [EXECUTAR]: python scripts/select_final_candidate.py

Commit 8  → docs: commit dos artefatos gerados
  *** PAUSA: mostrar rationale ao usuário, aguardar confirmação ***

Commit 9  → feat: evaluate_final.py                            ← SOMENTE após confirmação
  [EXECUTAR UMA VEZ]: python scripts/evaluate_final.py

Commit 10 → docs: final_selection.md
Commit 11 → docs: relatorio_tecnico + README
Commit 12 → test: smoke tests

  [VERIFICAR]: pytest -v --tb=short                            (todos PASS)
  [VERIFICAR]: python scripts/audit_sample.py                  (exit 0)
  [GUARDS 1A/1B/1C]: rodar os 3 comandos PowerShell da seção 6 e validar contagens esperadas (4 / 1 / 1)
  → push para origin/main
```

---

## 8. Critérios de Sucesso da Parte 5

- [ ] 1 novo MLflow run com `stage="final_eval"` no experimento existente
- [ ] 25 runs P3/P4 preservados intactos (não deletados, não modificados)
- [ ] `mlflow.sklearn.log_model` executado pela primeira vez no projeto
- [ ] `include_test=True` invocado exatamente 1 vez, em `scripts/evaluate_final.py` (Guard 1B); `test_idx` referenciado em exatamente 4 arquivos esperados (Guard 1A)
- [ ] `docs/final_selection_criteria.md` inalterado durante toda a Parte 5
- [ ] Todos os testes existentes + novos passam: `pytest -v`
- [ ] Auditor em ≥3 runs: exit code 0
- [ ] Relatório final, relatório técnico e README atualizados
