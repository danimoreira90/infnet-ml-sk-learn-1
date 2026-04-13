# PLAN_PARTE3.md — Parte 3: Pipeline de Modelagem e Rastreamento MLflow

> **Execucao somente apos aprovacao explicita do usuario.**
> **Para workers agenticos:** OBRIGATORIO: usar subagent-driven-development para cada task.
> Cada task = um subagent novo. Revisao de conformidade com spec + revisao de qualidade de codigo antes de marcar como concluido.

**Date**: 2026-04-12
**Status**: ready
**Spec**: N/A — derivado do PLAN.md (Parte 2) e requisitos do projeto Infnet
**Predecessora**: Parte 2 (HEAD 1c9dc04, 23 commits)

---

## 1. Confirmacao de Contexto

| Item | Esperado | Verificado |
|------|----------|-----------|
| HEAD | `1c9dc04` | Confirmar com `git log -1 --oneline` |
| Working tree | limpo | Confirmar com `git status -s` (sem output) |
| src/credit_default/data/ | ingest, fingerprint, schema, splits, diagnostics | 5 modulos + __init__.py |
| artifacts/splits/split_indices.json | presente, n_train=20975, n_val=4495, n_test=4495 | Presente |
| artifacts/data_fingerprint.json | file_short="30c6be3a" | Presente |
| data/credit_card_cleaned.parquet | 29965 x 24 | gitignored, gerado localmente |
| mlruns/ no .gitignore | presente | Sim (linha 31 do .gitignore) |
| Python | 3.11.x via uv | Confirmar `uv run python --version` |
| conftest.py minimal_df | PAY_0..PAY_5 (BUG: deveria ser PAY_0,PAY_2..PAY_6) | Corrigido no C0 |

---

## 2. Diff-Plan por Commit (20 commits: C0–C19)

### Nota sobre conftest.py

**Bug identificado**: `tests/conftest.py` linha 64 gera `PAY_0, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5` mas o dataset real tem `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` (sem PAY_1). Isso nao afetou testes da Parte 2 (que nao dependem de nomes exatos de colunas PAY), mas e bloqueante para testes da Parte 3 que usam `build_preprocessor()` com nomes de colunas reais. Corrigido no C0.

---

### C0: `fix(tests): corrige colunas PAY no minimal_df do conftest`

**Arquivos modificados:**
- `tests/conftest.py`

**Alteracao exata:**
Linha 64 — substituir:
```python
**{f"PAY_{i}": rng.integers(-2, 9, n) for i in range(6)},
```
por:
```python
**{f"PAY_{i}": rng.integers(-2, 9, n) for i in [0, 2, 3, 4, 5, 6]},
```

**Justificativa:** O dataset UCI nao possui coluna PAY_1 (pula de PAY_0 para PAY_2). A fixture deve espelhar a estrutura real para que testes da Parte 3 funcionem corretamente com `NUMERIC_ORDINAL`.

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
cols = {f'PAY_{i}': rng.integers(-2,9,5) for i in [0,2,3,4,5,6]}
assert 'PAY_1' not in cols and 'PAY_6' in cols; print('OK')
"
uv run pytest -q tests/test_data_foundation.py
```

---

### C1: `chore(deps): adiciona mlflow>=2.9 ao pyproject.toml`

**Arquivos modificados:**
- `pyproject.toml`
- `uv.lock` (regenerado pelo `uv lock` abaixo — DEVE ser commitado junto)

**Alteracao exata:**
Na secao `[project] dependencies`, adicionar apos `"seaborn>=0.13",`:
```
"mlflow>=2.9",
```

**Apos edicao, executar (ANTES do commit — na ordem):**
```bash
uv lock
uv lock --check
uv sync
```

**Confirmar que mlflow esta no lockfile antes de commitar:**
```bash
grep -c "mlflow" uv.lock
```
Esperado: > 0 (mlflow e dependencias transitivas presentes)

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv lock --check
uv run python -c "import mlflow; print(mlflow.__version__)"
```

---

### C2: `feat(features): cria preprocessing.py com build_preprocessor()`

**Arquivos criados:**
- `src/credit_default/features/__init__.py`
- `src/credit_default/features/preprocessing.py`

**API publica — `preprocessing.py`:**
```python
"""Pre-processamento de features para o pipeline de modelagem."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_CONTINUOUS: list[str] = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

NUMERIC_ORDINAL: list[str] = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

CATEGORICAL: list[str] = ["SEX", "EDUCATION", "MARRIAGE"]


def build_preprocessor() -> ColumnTransformer:
    """Retorna ColumnTransformer com 3 branches:
    - num_cont: StandardScaler para NUMERIC_CONTINUOUS
    - num_ord:  passthrough para NUMERIC_ORDINAL
    - cat:      OneHotEncoder(handle_unknown='ignore', sparse_output=False) para CATEGORICAL

    Returns
    -------
    ColumnTransformer configurado (nao fitado).
    """
    ...
```

**API publica — `features/__init__.py`:**
```python
"""Modulo de engenharia de features."""

from credit_default.features.preprocessing import (
    CATEGORICAL,
    NUMERIC_CONTINUOUS,
    NUMERIC_ORDINAL,
    build_preprocessor,
)

__all__ = [
    "NUMERIC_CONTINUOUS",
    "NUMERIC_ORDINAL",
    "CATEGORICAL",
    "build_preprocessor",
]
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "from credit_default.features.preprocessing import build_preprocessor; print(type(build_preprocessor()))"
```
Esperado: `<class 'sklearn.compose._column_transformer.ColumnTransformer'>`

---

### C3: `feat(models): cria registry.py com os 5 modelos e param grids`

**Arquivos criados:**
- `src/credit_default/models/__init__.py`
- `src/credit_default/models/registry.py`

**API publica — `registry.py`:**
```python
"""Registro central de modelos e seus hyperparameter grids."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier


class ModelSpec(TypedDict):
    estimator: Any
    param_grid: dict[str, list]
    search_type: Literal["none", "grid", "random"]
    model_family: Literal["linear", "tree", "ensemble"]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "perceptron": {
        "estimator": CalibratedClassifierCV(
            Perceptron(random_state=42), cv=3, method="sigmoid"
        ),
        "param_grid": {
            "clf__estimator__max_iter": [300, 1000],
            "clf__estimator__eta0": [0.01, 0.1, 1.0],
            "clf__estimator__penalty": ["l2", "elasticnet"],
        },
        "search_type": "grid",
        "model_family": "linear",
    },
    "logreg": {
        "estimator": LogisticRegression(random_state=42),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "saga"],
            "clf__max_iter": [500],
        },
        "search_type": "grid",
        "model_family": "linear",
    },
    "dtree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "clf__max_depth": [3, 5, 10, None],
            "clf__min_samples_split": [2, 10, 50],
            "clf__criterion": ["gini", "entropy"],
        },
        "search_type": "grid",
        "model_family": "tree",
    },
    "rf": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 10],
            "clf__max_features": ["sqrt", "log2"],
        },
        "search_type": "random",
        "model_family": "ensemble",
    },
    "gb": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.8, 1.0],
        },
        "search_type": "random",
        "model_family": "ensemble",
    },
}


def get_model_spec(name: str) -> ModelSpec:
    """Retorna ModelSpec para o nome dado.

    Raises
    ------
    KeyError : se name nao existir no MODEL_REGISTRY.
    """
    ...


def list_models() -> list[str]:
    """Retorna lista ordenada dos nomes dos modelos registrados."""
    ...
```

**Nota sobre Perceptron:** O Perceptron nao possui `predict_proba` nativo. O estimator registrado e `CalibratedClassifierCV(Perceptron(...), cv=3, method='sigmoid')` para garantir `predict_proba` disponivel. O param_grid usa prefixo `clf__estimator__` para acessar os hyperparameters do Perceptron interno do CalibratedClassifierCV.

**API publica — `models/__init__.py`:**
```python
"""Modulo de modelos."""

from credit_default.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    list_models,
)

__all__ = ["MODEL_REGISTRY", "ModelSpec", "get_model_spec", "list_models"]
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "from credit_default.models.registry import MODEL_REGISTRY; assert len(MODEL_REGISTRY)==5; print(sorted(MODEL_REGISTRY.keys()))"
```
Esperado: `['dtree', 'gb', 'logreg', 'perceptron', 'rf']`

---

### C4: `feat(models): cria pipeline.py com build_pipeline()`

**Arquivos criados:**
- `src/credit_default/models/pipeline.py`

**Arquivo modificado:**
- `src/credit_default/models/__init__.py` — adicionar import de `build_pipeline`

**API publica — `pipeline.py`:**
```python
"""Construcao de pipelines sklearn com preprocessor + modelo."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from credit_default.features.preprocessing import build_preprocessor
from credit_default.models.registry import get_model_spec


def build_pipeline(model_name: str, *, seed: int = 42) -> Pipeline:
    """Retorna Pipeline([('pre', build_preprocessor()), ('clf', estimator)]).

    O estimator e obtido de MODEL_REGISTRY[model_name].
    Se o estimator aceitar random_state, ele e setado com seed via set_params.

    Parameters
    ----------
    model_name : chave do MODEL_REGISTRY (perceptron, logreg, dtree, rf, gb)
    seed       : semente para reprodutibilidade

    Returns
    -------
    Pipeline com steps 'pre' e 'clf'

    Raises
    ------
    KeyError : se model_name nao existir no MODEL_REGISTRY.
    """
    ...
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "
from credit_default.models.pipeline import build_pipeline
pipe = build_pipeline('logreg')
print([s[0] for s in pipe.steps])
"
```
Esperado: `['pre', 'clf']`

---

### C5: `feat(evaluation): cria metrics.py com compute_all_metrics()`

**Arquivos criados:**
- `src/credit_default/evaluation/__init__.py`
- `src/credit_default/evaluation/metrics.py`

**API publica — `metrics.py`:**
```python
"""Calculo centralizado de metricas de classificacao."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas padrao de classificacao binaria.

    Parameters
    ----------
    y_true  : labels reais (0/1)
    y_pred  : predicoes binarias (0/1)
    y_proba : probabilidades da classe positiva (shape (n,))

    Returns
    -------
    dict com chaves: roc_auc, f1_macro, precision_macro, recall_macro, accuracy
    Todos os valores sao float.
    """
    ...
```

**API publica — `evaluation/__init__.py`:**
```python
"""Modulo de avaliacao de modelos."""

from credit_default.evaluation.metrics import compute_all_metrics

__all__ = ["compute_all_metrics"]
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "
import numpy as np
from credit_default.evaluation.metrics import compute_all_metrics
y_true = np.array([0,0,1,1])
y_pred = np.array([0,1,1,1])
y_proba = np.array([0.1,0.6,0.8,0.9])
m = compute_all_metrics(y_true, y_pred, y_proba)
assert set(m.keys()) == {'roc_auc','f1_macro','precision_macro','recall_macro','accuracy'}
print('OK:', m)
"
```

---

### C6: `feat(evaluation): cria plots.py (confusion_matrix, roc, pr)`

**Arquivos criados:**
- `src/credit_default/evaluation/plots.py`

**Arquivo modificado:**
- `src/credit_default/evaluation/__init__.py` — adicionar imports de plots

**API publica — `plots.py`:**
```python
"""Geracao de graficos de avaliacao de modelos."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    output_path: Path,
) -> Path:
    """Salva confusion matrix como PNG em output_path. Retorna output_path.

    Fecha a figura apos salvar (plt.close()).
    """
    ...


def roc_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    output_path: Path,
) -> Path:
    """Salva ROC curve como PNG em output_path. Retorna output_path.

    Fecha a figura apos salvar (plt.close()).
    """
    ...


def pr_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    output_path: Path,
) -> Path:
    """Salva Precision-Recall curve como PNG em output_path. Retorna output_path.

    Fecha a figura apos salvar (plt.close()).
    """
    ...
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "from credit_default.evaluation.plots import confusion_matrix_plot, roc_plot, pr_plot; print('OK')"
```

---

### C7: `feat(tracking): cria run_naming.py com compose_run_name()`

**Arquivos criados:**
- `src/credit_default/tracking/__init__.py`
- `src/credit_default/tracking/run_naming.py`

**API publica — `run_naming.py`:**
```python
"""Convencao de nomes para runs MLflow."""

from __future__ import annotations


def compose_run_name(
    stage: str,
    model: str,
    preproc: str = "numstd_catoh",
    dimred: str = "none",
    search: str = "none",
    seed: int = 42,
    datahash8: str = "",
    githash7: str = "",
) -> str:
    """Gera nome padronizado para run MLflow.

    Formato:
    <stage>__<model>__<preproc>__<dimred>__<search>__seed<seed>__data<DATAHASH8>__code<GITHASH7>

    Parameters
    ----------
    stage     : 'baseline' ou 'tune'
    model     : 'perceptron', 'logreg', 'dtree', 'rf', 'gb'
    preproc   : descritor do preprocessamento (default: 'numstd_catoh')
    dimred    : descritor de reducao dimensional (default: 'none')
    search    : 'none', 'grid', 'random'
    seed      : semente usada
    datahash8 : primeiros 8 chars do SHA-256 do dataset
    githash7  : primeiros 7 chars do git commit hash

    Returns
    -------
    str no formato padronizado
    """
    ...
```

**API publica — `tracking/__init__.py`:**
```python
"""Modulo de rastreamento MLflow."""

from credit_default.tracking.run_naming import compose_run_name

__all__ = ["compose_run_name"]
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "
from credit_default.tracking.run_naming import compose_run_name
name = compose_run_name('baseline','rf',datahash8='30c6be3a',githash7='1c9dc04')
print(name)
assert name == 'baseline__rf__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04'
print('OK')
"
```

---

### C8: `feat(tracking): cria mlflow_utils.py (log_standard_tags/metrics/artifacts)`

**Arquivos criados:**
- `src/credit_default/tracking/mlflow_utils.py`

**Arquivo modificado:**
- `src/credit_default/tracking/__init__.py` — adicionar imports

**API publica — `mlflow_utils.py`:**
```python
"""Utilitarios padronizados para logging no MLflow."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(name: str = "infnet-ml-sistema") -> str:
    """Retorna experiment_id. Cria o experiment se nao existir.

    Returns
    -------
    str : experiment_id
    """
    ...


def log_standard_tags(
    run: mlflow.ActiveRun,
    *,
    model_family: str,
    git_commit: str,
    dataset_fingerprint: str,
    compute_profile_s: float,
) -> None:
    """Loga tags padronizadas no run ativo.

    Tags logadas (10 total):
    1. model_family
    2. git_commit
    3. dataset_fingerprint
    4. compute_profile_s (str do float)
    5. project_part = "parte_3"
    6. framework = "scikit-learn"
    7. python_version (sys.version)
    8. os_platform (platform.platform())
    9. stage (extraido do run_name: primeira parte antes de '__')
    10. search_type (extraido do run_name: quinta parte)

    Nota: mlflow.runName e atributo nativo do run (setado em start_run como run_name=run_name).
    NAO duplicar como tag explicita.
    """
    ...


def log_standard_metrics(
    run: mlflow.ActiveRun,
    metrics: dict[str, float],
    *,
    cv_roc_auc_mean: float,
    cv_roc_auc_std: float,
    cv_f1_mean: float,
    cv_f1_std: float,
    training_time_s: float,
    inference_latency_ms: float,
) -> None:
    """Loga metricas padronizadas no run ativo.

    Metricas logadas (11 total):
    1-5. roc_auc, f1_macro, precision_macro, recall_macro, accuracy (do dict metrics)
    6.   cv_roc_auc_mean  (metrica PRIMARIA de CV — scoring='roc_auc')
    7.   cv_roc_auc_std
    8.   cv_f1_mean       (metrica secundaria de CV — scoring='f1_macro')
    9.   cv_f1_std
    10.  training_time_s
    11.  inference_latency_ms
    """
    ...


def log_standard_artifacts(
    run: mlflow.ActiveRun,
    tmp_dir: Path,
    *,
    has_feature_importances: bool = False,
) -> None:
    """Loga artefatos padronizados do tmp_dir no run ativo.

    Artefatos esperados no tmp_dir:
    - confusion_matrix.png
    - roc_curve.png
    - pr_curve.png
    - run_summary.json
    - split_fingerprint.txt
    - feature_importances.csv (se has_feature_importances=True)

    Usa mlflow.log_artifacts(str(tmp_dir)).
    """
    ...
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "
from credit_default.tracking.mlflow_utils import (
    get_or_create_experiment,
    log_standard_tags,
    log_standard_metrics,
    log_standard_artifacts,
)
print('OK')
"
```

---

### C9: `feat(tracking): cria compare.py com consolidated_results_table()`

**Arquivos criados:**
- `src/credit_default/tracking/compare.py`

**Arquivo modificado:**
- `src/credit_default/tracking/__init__.py` — adicionar import

**API publica — `compare.py`:**
```python
"""Consolidacao de resultados de experimentos MLflow."""

from __future__ import annotations

import pandas as pd
from mlflow.tracking import MlflowClient


def consolidated_results_table(
    experiment_name: str = "infnet-ml-sistema",
    *,
    stage: str | None = None,
) -> pd.DataFrame:
    """Query MLflow via MlflowClient, retorna DataFrame com todos os runs.

    Parameters
    ----------
    experiment_name : nome do experiment MLflow
    stage           : se fornecido, filtra runs pela tag 'stage'

    Returns
    -------
    pd.DataFrame com colunas:
    - run_name, model_family, stage, search_type
    - roc_auc, f1_macro, precision_macro, recall_macro, accuracy
    - cv_roc_auc_mean, cv_roc_auc_std
    - cv_f1_mean, cv_f1_std
    - training_time_s, inference_latency_ms
    Ordenado por roc_auc descendente (metrica primaria).

    Raises
    ------
    ValueError : se experiment nao existir.
    """
    ...
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "from credit_default.tracking.compare import consolidated_results_table; print('OK')"
```

---

### C10: `feat(models): cria train.py com train_and_evaluate()`

**Arquivos criados:**
- `src/credit_default/models/train.py`

**Arquivo modificado:**
- `src/credit_default/models/__init__.py` — adicionar imports de `load_split_data` e `train_and_evaluate`

**API publica — `train.py`:**
```python
"""Orquestracao de treino, avaliacao e logging MLflow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from credit_default.evaluation.metrics import compute_all_metrics
from credit_default.evaluation.plots import confusion_matrix_plot, pr_plot, roc_plot
from credit_default.models.pipeline import build_pipeline
from credit_default.models.registry import get_model_spec
from credit_default.tracking.mlflow_utils import (
    get_or_create_experiment,
    log_standard_artifacts,
    log_standard_metrics,
    log_standard_tags,
)
from credit_default.tracking.run_naming import compose_run_name


def load_split_data(
    parquet_path: Path,
    split_indices_path: Path,
    target_col: str = "default payment next month",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Carrega dataset e retorna splits usando indices de split_indices.json.

    Retorna X_train, X_val, X_test, y_train, y_val, y_test.
    Usa SOMENTE train_idx e val_idx para treino/validacao.
    test_idx e carregado apenas para completude da tupla mas NUNCA usado em treino.

    IMPORTANTE: Esta funcao le os indices do JSON — nunca regenera splits.

    Parameters
    ----------
    parquet_path       : caminho para credit_card_cleaned.parquet
    split_indices_path : caminho para split_indices.json
    target_col         : nome da coluna target

    Returns
    -------
    tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    ...


def train_and_evaluate(
    model_name: str,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
    cv_folds: int = 5,
    tune: bool = False,
    experiment_name: str = "infnet-ml-sistema",
    datahash8: str,
    githash7: str,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Pipeline completo: train -> CV -> [tune] -> evaluate -> MLflow log.

    Fluxo:
    1. build_pipeline(model_name, seed=seed)
    2. Cross-validation com cross_validate() usando
       scoring={'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'},
       cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
       no X_train. Extrai cv_roc_auc_mean, cv_roc_auc_std (PRIMARIA) e
       cv_f1_mean, cv_f1_std (secundaria).
    3. Se tune=True E search_type != 'none':
       - GridSearchCV ou RandomizedSearchCV conforme registry.search_type
       - Para RandomizedSearchCV: n_iter=30, random_state=seed
       - scoring='roc_auc' (metrica PRIMARIA), cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
       - Refit=True (default) — ja refita no X_train completo
       - pipeline = search.best_estimator_
    4. Se tune=False: fit simples no X_train
    5. Predict no X_val: y_pred e y_proba
    6. compute_all_metrics(y_val, y_pred, y_proba)
    7. Medir inference_latency_ms: tempo medio por amostra no X_val
    8. Gerar plots (confusion_matrix, roc, pr) no tmp_dir
    9. Gerar run_summary.json no tmp_dir com todas as metricas, params e "primary_metric": "roc_auc"
    10. Gerar split_fingerprint.txt no tmp_dir com datahash8
    11. Se modelo tree/ensemble com feature_importances_: gerar feature_importances.csv
    12. compose_run_name() para gerar run_name
    13. MLflow: start_run(run_name=run_name, experiment_id=...)
    14. log_standard_tags, log_standard_metrics, log_standard_artifacts
    15. end_run
    16. Retorna dict com best_pipeline, metrics, run_id, run_name,
        cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std

    Parameters
    ----------
    model_name       : chave do MODEL_REGISTRY
    X_train, y_train : dados de treino (indices de split_indices.json train_idx)
    X_val, y_val     : dados de validacao (indices de split_indices.json val_idx)
    seed             : semente para reprodutibilidade
    cv_folds         : numero de folds para cross-validation
    tune             : se True, executa hyperparameter search
    experiment_name  : nome do experiment MLflow
    datahash8        : primeiros 8 chars do SHA-256 do dataset
    githash7         : primeiros 7 chars do git commit hash
    tmp_dir          : diretorio temporario para artefatos (caller gerencia lifecycle)

    Returns
    -------
    dict com chaves: best_pipeline, metrics, run_id, run_name,
                     cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std
    """
    ...
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python -c "from credit_default.models.train import load_split_data, train_and_evaluate; print('OK')"
```

---

### C11: `feat(scripts): cria train_baseline.py (5 modelos, stage=baseline)`

**Arquivos criados:**
- `scripts/train_baseline.py`

**Comportamento do script:**
```python
"""Treina os 5 modelos em modo baseline (sem tuning) e loga no MLflow.

Uso: uv run python scripts/train_baseline.py

Fluxo:
1. Resolve repo_root = Path(__file__).resolve().parent.parent
2. sys.path.insert(0, str(repo_root / 'src'))
3. mlflow.set_tracking_uri(str(repo_root / 'mlruns'))  # PATH ABSOLUTO
4. Carrega split data via load_split_data(
       parquet_path=repo_root / 'data' / 'credit_card_cleaned.parquet',
       split_indices_path=repo_root / 'artifacts' / 'splits' / 'split_indices.json',
   )
5. Le datahash8 de artifacts/data_fingerprint.json campo 'file_short'
6. Obtem githash7 via subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD'])
7. Para cada modelo em list_models():
   a. Cria tempfile.TemporaryDirectory() como ctx manager
   b. Chama train_and_evaluate(model_name, tune=False, ...)
   c. Imprime resumo: run_name, roc_auc [PRIMARY], f1_macro [secondary]
8. Imprime tabela resumo final ordenada por roc_auc desc (PRIMARY)
"""
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python scripts/train_baseline.py
uv run python -c "
import mlflow
from pathlib import Path
mlflow.set_tracking_uri(str(Path('mlruns').resolve()))
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('infnet-ml-sistema')
runs = client.search_runs(exp.experiment_id, filter_string=\"tags.stage = 'baseline'\")
assert len(runs) == 5, f'Esperado 5 runs baseline, encontrado {len(runs)}'
print(f'OK: {len(runs)} runs baseline')
"
```

---

### C12: `feat(scripts): cria train_tuned.py (5 modelos, stage=tune)`

**Arquivos criados:**
- `scripts/train_tuned.py`

**Comportamento:** Identico ao `train_baseline.py` exceto:
- `tune=True` na chamada de `train_and_evaluate()`
- O search_type de cada modelo e determinado pelo registry:
  - perceptron, logreg, dtree: GridSearchCV (scoring='roc_auc')
  - rf, gb: RandomizedSearchCV(n_iter=30, scoring='roc_auc')
- Tabela resumo final ordenada por roc_auc desc (PRIMARY)

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python scripts/train_tuned.py
uv run python -c "
import mlflow
from pathlib import Path
mlflow.set_tracking_uri(str(Path('mlruns').resolve()))
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('infnet-ml-sistema')
runs = client.search_runs(exp.experiment_id, filter_string=\"tags.stage = 'tune'\")
assert len(runs) == 5, f'Esperado 5 runs tune, encontrado {len(runs)}'
print(f'OK: {len(runs)} runs tune')
"
```

---

### C13: `feat(scripts): cria generate_comparison_table.py`

**Arquivos criados:**
- `scripts/generate_comparison_table.py`

**Comportamento do script:**
```python
"""Gera tabelas comparativas de resultados a partir do MLflow.

Uso: uv run python scripts/generate_comparison_table.py

Saidas:
- reports/parte_3/comparison_baseline.md
- reports/parte_3/comparison_tuned.md

Fluxo:
1. Resolve repo_root
2. sys.path.insert(0, str(repo_root / 'src'))
3. mlflow.set_tracking_uri(str(repo_root / 'mlruns'))
4. Chama consolidated_results_table(stage='baseline') -> df_baseline
5. Chama consolidated_results_table(stage='tune') -> df_tuned
6. Converte cada df para markdown table
7. Salva em reports/parte_3/comparison_baseline.md e comparison_tuned.md
8. Formato markdown: header + tabela com colunas:
   run_name | model_family | roc_auc | f1_macro | precision_macro | recall_macro | accuracy | cv_roc_auc_mean | cv_roc_auc_std | cv_f1_mean | cv_f1_std | training_time_s
"""
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run python scripts/generate_comparison_table.py
test -f reports/parte_3/comparison_baseline.md && echo "baseline OK"
test -f reports/parte_3/comparison_tuned.md && echo "tuned OK"
head -5 reports/parte_3/comparison_baseline.md
```

---

### C14: `test(features): test_preprocessing.py`

**Arquivos criados:**
- `tests/test_preprocessing.py`

**Testes (todos usam fixture `minimal_df` do conftest — ja corrigida no C0):**

```python
"""Testes para o modulo de preprocessing."""

from credit_default.features.preprocessing import (
    CATEGORICAL,
    NUMERIC_CONTINUOUS,
    NUMERIC_ORDINAL,
    build_preprocessor,
)


def test_build_preprocessor_returns_column_transformer():
    """build_preprocessor() retorna ColumnTransformer."""
    ...


def test_preprocessor_has_three_transformers():
    """ColumnTransformer tem 3 transformers: num_cont, num_ord, cat."""
    ct = build_preprocessor()
    names = [name for name, _, _ in ct.transformers]
    assert names == ["num_cont", "num_ord", "cat"]


def test_preprocessor_fit_transform_shape(minimal_df):
    """fit_transform no minimal_df produz shape correto.
    500 linhas. Colunas = 14 (scaled) + 6 (passthrough) + N_onehot.
    """
    ...


def test_preprocessor_no_nan_in_output(minimal_df):
    """Output do preprocessor nao contem NaN."""
    ...


def test_column_lists_cover_23_features():
    """NUMERIC_CONTINUOUS + NUMERIC_ORDINAL + CATEGORICAL == 23 features."""
    all_cols = NUMERIC_CONTINUOUS + NUMERIC_ORDINAL + CATEGORICAL
    assert len(all_cols) == 23
    assert len(set(all_cols)) == 23  # sem duplicatas


def test_preprocessor_handles_unknown_categories(minimal_df):
    """OneHotEncoder com handle_unknown='ignore' nao falha com categorias novas."""
    ...
```

**TDD Steps:**
1. RED: Criar arquivo de teste, rodar `uv run pytest tests/test_preprocessing.py -v` — confirmar falhas
2. GREEN: Implementacao ja feita no C2 — testes devem passar
3. Commit: `test(features): test_preprocessing.py`

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run pytest tests/test_preprocessing.py -v
```

---

### C15: `test(models): test_pipeline.py + test_registry.py`

**Arquivos criados:**
- `tests/test_pipeline.py`
- `tests/test_registry.py`

**Testes em `test_registry.py`:**
```python
"""Testes para o registro de modelos."""

from credit_default.models.registry import (
    MODEL_REGISTRY,
    get_model_spec,
    list_models,
)
import pytest


def test_model_registry_has_5_models():
    assert len(MODEL_REGISTRY) == 5


def test_all_model_specs_have_required_keys():
    for name, spec in MODEL_REGISTRY.items():
        assert "estimator" in spec, f"{name} sem estimator"
        assert "param_grid" in spec, f"{name} sem param_grid"
        assert "search_type" in spec, f"{name} sem search_type"
        assert "model_family" in spec, f"{name} sem model_family"


def test_get_model_spec_returns_correct_type():
    spec = get_model_spec("logreg")
    assert "estimator" in spec


def test_get_model_spec_raises_key_error():
    with pytest.raises(KeyError):
        get_model_spec("inexistente")


def test_list_models_returns_sorted_list():
    models = list_models()
    assert models == sorted(models)
    assert len(models) == 5


def test_perceptron_has_calibrated_classifier():
    from sklearn.calibration import CalibratedClassifierCV
    spec = get_model_spec("perceptron")
    assert isinstance(spec["estimator"], CalibratedClassifierCV)


def test_perceptron_param_grid_uses_estimator_prefix():
    spec = get_model_spec("perceptron")
    for key in spec["param_grid"]:
        assert key.startswith("clf__estimator__"), f"Key {key} sem prefixo correto"


def test_all_estimators_have_predict_proba():
    for name, spec in MODEL_REGISTRY.items():
        assert hasattr(spec["estimator"], "predict_proba"), f"{name} sem predict_proba"
```

**Testes em `test_pipeline.py`:**
```python
"""Testes para construcao de pipelines."""

import pytest
from sklearn.pipeline import Pipeline

from credit_default.models.pipeline import build_pipeline
from credit_default.models.registry import list_models


def test_build_pipeline_returns_pipeline():
    pipe = build_pipeline("logreg")
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_pre_and_clf_steps():
    pipe = build_pipeline("logreg")
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["pre", "clf"]


def test_pipeline_fit_predict_on_minimal_data(minimal_df):
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    y = minimal_df[target_col]
    pipe = build_pipeline("logreg")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_pipeline_predict_proba_available(minimal_df):
    target_col = "default payment next month"
    X = minimal_df.drop(columns=[target_col])
    y = minimal_df[target_col]
    pipe = build_pipeline("logreg")
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_build_pipeline_all_models():
    for model_name in list_models():
        pipe = build_pipeline(model_name)
        assert isinstance(pipe, Pipeline), f"Falha para {model_name}"
```

**TDD Steps:**
1. RED: Criar arquivos de teste, rodar — esperar que passem (implementacao ja feita em C3-C4)
2. GREEN: Confirmar todos verdes
3. Commit: `test(models): test_pipeline.py + test_registry.py`

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run pytest tests/test_pipeline.py tests/test_registry.py -v
```

---

### C16: `test(tracking): test_run_naming.py`

**Arquivos criados:**
- `tests/test_run_naming.py`

**Testes:**
```python
"""Testes para convencao de nomes de runs MLflow."""

from credit_default.tracking.run_naming import compose_run_name


def test_compose_run_name_baseline_format():
    name = compose_run_name("baseline", "rf", datahash8="30c6be3a", githash7="1c9dc04")
    assert name == "baseline__rf__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04"


def test_compose_run_name_tune_format():
    name = compose_run_name(
        "tune", "rf", search="random", datahash8="30c6be3a", githash7="1c9dc04"
    )
    assert name == "tune__rf__numstd_catoh__none__random__seed42__data30c6be3a__code1c9dc04"


def test_compose_run_name_custom_params():
    name = compose_run_name(
        "baseline", "logreg",
        preproc="custom", dimred="pca", seed=123,
        datahash8="abcd1234", githash7="abc1234",
    )
    assert name == "baseline__logreg__custom__pca__none__seed123__dataabcd1234__codeabc1234"


def test_compose_run_name_defaults():
    name = compose_run_name("baseline", "dtree")
    assert "__numstd_catoh__" in name
    assert "__none__none__" in name
    assert "__seed42__" in name


def test_compose_run_name_all_models():
    for model in ["perceptron", "logreg", "dtree", "rf", "gb"]:
        name = compose_run_name("baseline", model)
        assert f"__{model}__" in name
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
uv run pytest tests/test_run_naming.py -v
```

---

### C17: `docs(parte3): comparison_baseline.md e comparison_tuned.md (templates)`

**Arquivos criados:**
- `reports/parte_3/.gitkeep`
- `reports/parte_3/comparison_baseline.md`
- `reports/parte_3/comparison_tuned.md`
- `reports/figures/parte_3/.gitkeep`

**Arquivo modificado:**
- `.gitignore` — adicionar apos a linha `!reports/figures/parte_2/.gitkeep`:
```
reports/figures/parte_3/*.png
!reports/figures/parte_3/.gitkeep
```

**Conteudo de `comparison_baseline.md` (template):**
```markdown
# Comparacao de Modelos — Baseline

> Este arquivo e gerado automaticamente por `scripts/generate_comparison_table.py`.
> NAO editar manualmente. Para atualizar, executar o script.

_Tabela sera preenchida apos execucao do script._
```

**Conteudo de `comparison_tuned.md` (template):**
```markdown
# Comparacao de Modelos — Tuned

> Este arquivo e gerado automaticamente por `scripts/generate_comparison_table.py`.
> NAO editar manualmente. Para atualizar, executar o script.

_Tabela sera preenchida apos execucao do script._
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
test -f reports/parte_3/comparison_baseline.md && echo "baseline template OK"
test -f reports/parte_3/comparison_tuned.md && echo "tuned template OK"
test -f reports/parte_3/.gitkeep && echo "gitkeep OK"
```

---

### C18: `docs(relatorio): atualiza relatorio_tecnico.md com secao Parte 3`

**Arquivos modificados:**
- `reports/relatorio_tecnico.md`

**Conteudo a adicionar apos a ultima linha do arquivo (apos a secao Parte 2):**

```markdown
---

## Parte 3 — Pipeline de Modelagem e Rastreamento MLflow

### Objetivo

Treinar, avaliar e comparar 5 algoritmos de classificacao usando sklearn Pipelines com rastreamento completo via MLflow. Dois estagios: baseline (sem tuning) e tuned (com hyperparameter search).

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

\```bash
uv run python scripts/train_baseline.py
uv run python scripts/train_tuned.py
uv run python scripts/generate_comparison_table.py
\```
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
grep -c "Parte 3" reports/relatorio_tecnico.md
```
Esperado: >= 2 (titulo + mencoes)

---

### C19: `docs(readme): atualiza README com "Como rodar a Parte 3"`

**Arquivos modificados:**
- `README.md`

**Conteudo a adicionar (nova secao, antes do final do arquivo):**

```markdown
## Parte 3 — Modelagem e MLflow

### Pre-requisitos

\```bash
uv sync  # instala dependencias incluindo mlflow
\```

### Execucao

\```bash
# 1. Treino baseline (5 modelos, ~3-5 min)
uv run python scripts/train_baseline.py

# 2. Treino com tuning (5 modelos, ~15-25 min)
uv run python scripts/train_tuned.py

# 3. Gerar tabelas comparativas
uv run python scripts/generate_comparison_table.py
\```

### Visualizar resultados no MLflow UI

\```bash
uv run mlflow ui --backend-store-uri mlruns/
# Abrir http://localhost:5000
\```

### Testes da Parte 3

\```bash
uv run pytest tests/test_preprocessing.py tests/test_pipeline.py tests/test_registry.py tests/test_run_naming.py -v
\```
```

**Rollback:** `git revert HEAD`

**Validacao:**
```bash
grep -c "Parte 3" README.md
```
Esperado: >= 1

---

## 3. Public APIs — Resumo de Assinaturas

| Modulo | Funcoes/Constantes | Commit |
|--------|--------------------|--------|
| `src/credit_default/features/preprocessing.py` | `NUMERIC_CONTINUOUS`, `NUMERIC_ORDINAL`, `CATEGORICAL`, `build_preprocessor() -> ColumnTransformer` | C2 |
| `src/credit_default/models/registry.py` | `ModelSpec`, `MODEL_REGISTRY`, `get_model_spec(name: str) -> ModelSpec`, `list_models() -> list[str]` | C3 |
| `src/credit_default/models/pipeline.py` | `build_pipeline(model_name: str, *, seed: int = 42) -> Pipeline` | C4 |
| `src/credit_default/evaluation/metrics.py` | `compute_all_metrics(y_true, y_pred, y_proba) -> dict[str, float]` | C5 |
| `src/credit_default/evaluation/plots.py` | `confusion_matrix_plot(...)`, `roc_plot(...)`, `pr_plot(...)` — todos retornam `Path` | C6 |
| `src/credit_default/tracking/run_naming.py` | `compose_run_name(stage, model, ...) -> str` | C7 |
| `src/credit_default/tracking/mlflow_utils.py` | `get_or_create_experiment(name) -> str`, `log_standard_tags(...)`, `log_standard_metrics(...)`, `log_standard_artifacts(...)` | C8 |
| `src/credit_default/tracking/compare.py` | `consolidated_results_table(experiment_name, *, stage) -> pd.DataFrame` | C9 |
| `src/credit_default/models/train.py` | `load_split_data(...) -> tuple[6]`, `train_and_evaluate(...) -> dict[str, Any]` | C10 |

---

## 4. MLflow Run Name Convention — Exemplos

```
baseline__perceptron__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04
baseline__logreg__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04
baseline__dtree__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04
baseline__rf__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04
baseline__gb__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04
tune__perceptron__numstd_catoh__none__grid__seed42__data30c6be3a__code1c9dc04
tune__logreg__numstd_catoh__none__grid__seed42__data30c6be3a__code1c9dc04
tune__dtree__numstd_catoh__none__grid__seed42__data30c6be3a__code1c9dc04
tune__rf__numstd_catoh__none__random__seed42__data30c6be3a__code1c9dc04
tune__gb__numstd_catoh__none__random__seed42__data30c6be3a__code1c9dc04
```

---

## 5. Riscos e Mitigacoes

| Risco | Severidade | Mitigacao |
|-------|------------|-----------|
| MLflow tracking URI no Windows depende de CWD | Alta | Usar `mlflow.set_tracking_uri(str(repo_root / 'mlruns'))` com Path absoluto em todos os scripts e em `get_or_create_experiment()` |
| GB RandomizedSearch memoria | Media | n_estimators max 300, subsample 0.8 caps data por arvore |
| RF com n_iter=30 em 29965 amostras | Baixa | Tempo aceitavel ~5-10 min por modelo |
| OneHotEncoder com EDUCATION codigos {0,5,6} | Media | `handle_unknown='ignore'` configurado no build_preprocessor (documentado na Parte 2) |
| Perceptron sem predict_proba nativo | Alta | Usar `CalibratedClassifierCV(Perceptron(...), cv=3, method='sigmoid')`; param_grid usa prefixo `clf__estimator__` |
| MLflow artifact logging precisa de diretorio temporario | Baixa | Usar `tempfile.TemporaryDirectory()` por run, sempre limpo via context manager |
| conftest.py minimal_df com colunas PAY erradas | Alta | Corrigido no C0 antes de qualquer teste da Parte 3 |
| Notebooks e utils.py nao devem ser tocados | Alta | Validacao: `git diff HEAD -- "*.ipynb" utils.py` deve retornar vazio |
| Modulos da Parte 2 (src/credit_default/data/) congelados | Alta | Validacao: `git diff HEAD -- src/credit_default/data/` deve retornar vazio |

---

## 6. DoD Checklist — Parte 3

- [ ] mlflow>=2.9 em pyproject.toml; `uv lock --check` passa
- [ ] `uv run python -c "from credit_default.features.preprocessing import build_preprocessor; print('OK')"` passa
- [ ] `uv run python -c "from credit_default.models.registry import MODEL_REGISTRY; assert len(MODEL_REGISTRY)==5"` passa
- [ ] `uv run python -c "from credit_default.tracking.run_naming import compose_run_name; print(compose_run_name('baseline','rf',datahash8='30c6be3a',githash7='1c9dc04'))"` produz `baseline__rf__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04`
- [ ] `uv run python scripts/train_baseline.py` completa sem erro; 5 runs no MLflow com tag stage=baseline
- [ ] `uv run python scripts/train_tuned.py` completa sem erro; 5 runs adicionais no MLflow com tag stage=tune
- [ ] `uv run python scripts/generate_comparison_table.py` gera `reports/parte_3/comparison_baseline.md` e `comparison_tuned.md` com tabelas preenchidas
- [ ] `uv run pytest -q tests/test_preprocessing.py tests/test_pipeline.py tests/test_registry.py tests/test_run_naming.py` — todos verdes
- [ ] `uv run ruff check src/ scripts/ tests/` sem erros
- [ ] `uv run black --check src/ scripts/ tests/` sem reformatacoes
- [ ] Cada run MLflow tem: 10 tags, 11 metricas, 5+ artefatos (confusion_matrix.png, roc_curve.png, pr_curve.png, run_summary.json, split_fingerprint.txt)
- [ ] Cada run tem pasta params/ nao-vazia com minimo 8 params (8 meta: model_name, seed, cv_folds, scoring_primary, split_strategy, search_type, n_train, n_val + >= 3 clf__ hiperparametros do classificador)
- [ ] run_summary.json de cada run contem "primary_metric": "roc_auc"
- [ ] split_fingerprint.txt em cada run confirma fingerprint_short = "30c6be3a"
- [ ] `git diff 516306a -- "*.ipynb" utils.py` retorna vazio (notebooks intocados)
- [ ] `git diff HEAD -- src/credit_default/data/` retorna vazio (Parte 2 congelada)
- [ ] test_idx NUNCA tocado em treino (validar: train_and_evaluate() nao usa X_test/y_test)

---

## 7. Integrity Checklist — Parte 3

- [ ] `train_and_evaluate()` nunca acessa test_idx — somente X_train, y_train, X_val, y_val nos parametros
- [ ] `build_preprocessor()` nunca fitado fora do `Pipeline.fit()` — retorna ColumnTransformer nao-fitado
- [ ] Split indices lidos de `artifacts/splits/split_indices.json` (nunca regenerados)
- [ ] datahash8 lido de `artifacts/data_fingerprint.json` campo `file_short` (nunca hardcoded)
- [ ] githash7 obtido via `git rev-parse --short=7 HEAD` em runtime (nunca hardcoded)
- [ ] Todas as metricas em comparison tables vem de MLflow via `consolidated_results_table()` (nunca hardcoded)
- [ ] `mlruns/` no `.gitignore` (verificar: `git check-ignore mlruns/` retorna `mlruns/`)
- [ ] Perceptron usa `CalibratedClassifierCV` para predict_proba — verificavel em `MODEL_REGISTRY["perceptron"]["estimator"]`

---

## 8. Estimativa de Tempo de Execucao

| Etapa | Tempo estimado |
|-------|---------------|
| Baseline (5 modelos x CV 5-fold, sem tuning) | ~3-5 min |
| Tuning (Perceptron+LogReg+DTree GridSearch + RF+GB RandomSearch n_iter=30) | ~15-25 min |
| Testes (pytest) | ~30 seg |
| Geracao de tabelas comparativas | ~10 seg |
| **Validacao DoD completa** | **~25-35 min** |

---

## 9. Estrutura Final de Diretorios (Parte 3)

```
src/credit_default/
├── __init__.py                          # existente (inalterado)
├── data/                                # CONGELADO (Parte 2) — nenhum arquivo tocado
│   ├── __init__.py
│   ├── ingest.py
│   ├── fingerprint.py
│   ├── schema.py
│   ├── splits.py
│   └── diagnostics.py
├── features/                            # NOVO (C2)
│   ├── __init__.py
│   └── preprocessing.py
├── models/                              # NOVO (C3, C4, C10)
│   ├── __init__.py
│   ├── registry.py
│   ├── pipeline.py
│   └── train.py
├── evaluation/                          # NOVO (C5, C6)
│   ├── __init__.py
│   ├── metrics.py
│   └── plots.py
└── tracking/                            # NOVO (C7, C8, C9)
    ├── __init__.py
    ├── run_naming.py
    ├── mlflow_utils.py
    └── compare.py

scripts/
├── build_clean_dataset.py               # existente (inalterado)
├── run_data_qa.py                       # existente (inalterado)
├── train_baseline.py                    # NOVO (C11)
├── train_tuned.py                       # NOVO (C12)
└── generate_comparison_table.py         # NOVO (C13)

tests/
├── conftest.py                          # MODIFICADO (C0 — fix PAY columns)
├── __init__.py                          # existente (inalterado)
├── test_data_foundation.py              # existente (inalterado)
├── test_preprocessing.py                # NOVO (C14)
├── test_pipeline.py                     # NOVO (C15)
├── test_registry.py                     # NOVO (C15)
└── test_run_naming.py                   # NOVO (C16)

reports/
├── relatorio_tecnico.md                 # MODIFICADO (C18 — adiciona secao Parte 3)
├── figures/
│   ├── parte_2/                         # existente (inalterado)
│   └── parte_3/
│       └── .gitkeep                     # NOVO (C17)
└── parte_3/                             # NOVO (C17)
    ├── .gitkeep
    ├── comparison_baseline.md           # template (C17), preenchido pelo script (C13)
    └── comparison_tuned.md              # template (C17), preenchido pelo script (C13)
```

---

## 10. Ordem de Execucao e Dependencias

```
C0  fix(tests)     ────────── sem dependencias ───────────────────────────────
C1  chore(deps)    ────────── sem dependencias ───────────────────────────────

C2  feat(features) ────────── depende de C1 ──────────────────────────────────
C3  feat(models/registry) ── depende de C1 ──────────────────────────────────
C5  feat(eval/metrics) ───── depende de C1 ──────────────────────────────────
C6  feat(eval/plots) ─────── depende de C1 ──────────────────────────────────
C7  feat(track/naming) ───── sem dependencias (puro string formatting) ──────

C4  feat(models/pipeline) ── depende de C2, C3 ──────────────────────────────
C8  feat(track/mlflow) ───── depende de C1, C7 ──────────────────────────────
C9  feat(track/compare) ──── depende de C1, C8 ──────────────────────────────

C10 feat(models/train) ───── depende de C4, C5, C6, C7, C8 ──────────────────

C11 feat(scripts/baseline) ─ depende de C10 ──────────────────────────────────
C12 feat(scripts/tuned) ──── depende de C10 ──────────────────────────────────

C13 feat(scripts/compare) ── depende de C9 (runtime: tambem C11, C12) ───────

C14 test(features) ────────── depende de C0, C2 ──────────────────────────────
C15 test(models) ──────────── depende de C0, C3, C4 ──────────────────────────
C16 test(tracking) ────────── depende de C7 ──────────────────────────────────

C17 docs(templates) ───────── sem dependencias ───────────────────────────────
C18 docs(relatorio) ───────── sem dependencias ───────────────────────────────
C19 docs(readme) ──────────── sem dependencias ───────────────────────────────
```

**Fases de paralelismo:**

| Fase | Commits | Prerequisito |
|------|---------|-------------|
| 1 — Fundacao | C0, C1 | nenhum (executar C0 antes de C1) |
| 2 — Modulos independentes | C2, C3, C5, C6, C7 | C1 completo |
| 3 — Modulos dependentes | C4, C8, C9 | predecessores da Fase 2 |
| 4 — Orquestrador | C10 | C4, C5, C6, C7, C8 completos |
| 5 — Scripts | C11, C12 | C10 completo (paralelo entre si) |
| 6 — Script comparacao | C13 | C9 completo (runtime: C11, C12 ja executados) |
| 7 — Testes | C14, C15, C16 | modulos implementados (paralelo entre si) |
| 8 — Documentacao | C17, C18, C19 | sem dependencias (paralelo entre si) |

---

## 11. Rollback Plan

Cada commit e atomico e revertivel individualmente:
```bash
git revert HEAD  # reverte ultimo commit
```

Para rollback completo da Parte 3:
```bash
# Identificar primeiro commit da Parte 3
FIRST_P3=$(git log --oneline --reverse --ancestry-path 1c9dc04..HEAD | head -1 | cut -d' ' -f1)
# Reverter todos os commits da Parte 3
git revert --no-commit ${FIRST_P3}..HEAD
git commit -m "revert: remove Parte 3 completa"
```

Alternativa mais segura (reset para estado da Parte 2):
```bash
git checkout 1c9dc04 -- .
git commit -m "revert: restaura estado da Parte 2"
```

**Nota:** `mlruns/` e gitignored e deve ser removido manualmente se necessario:
```bash
rm -rf mlruns/
```