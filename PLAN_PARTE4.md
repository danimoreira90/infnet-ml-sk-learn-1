# PLAN: Parte 4 — Redução de Dimensionalidade (PCA + LDA)

**Date**: 2026-04-13
**Status**: ready-for-approval
**Branch**: `main` (HEAD: `ceb6129`, 46 commits sobre 516306a)
**Parte anterior**: Parte 3 completa e publicada em origin/main.

**Critérios de Sucesso**:
- [ ] 15 MLflow runs novos (5 modelos × 3 configs dimred) em "infnet-ml-sistema"
- [ ] 10 runs da Parte 3 preservados intactos (não deletar)
- [ ] 37 testes existentes + novos testes todos verdes
- [ ] Relatório técnico e README atualizados com Parte 4
- [ ] DoD checklist 100% verde

> **For agentic workers:** REQUIRED: Use subagent-driven-development for every task.
> Each task = one fresh subagent. Spec compliance review → code quality review → done.

---

## 1. Confirmação de Contexto (read-only, 5 linhas)

1. `src/credit_default/features/__init__.py` — exporta `NUMERIC_CONTINUOUS`, `NUMERIC_ORDINAL`, `CATEGORICAL`, `build_preprocessor`. A ser estendido com `build_dimred_pipeline`, `compute_pca_n_components`.
2. `src/credit_default/models/pipeline.py` — `build_pipeline(model_name, seed=42)` retorna `Pipeline([("pre", build_preprocessor()), ("clf", estimator)])`. Parte 4 NÃO modifica esta função.
3. `src/credit_default/tracking/mlflow_utils.py` — tem `log_standard_params` (8 meta params + clf__*) e `log_standard_tags` (10 tags). Parte 4 estende AMBAS com kwargs opcionais dimred.
4. `artifacts/data_fingerprint.json` — `file_short="30c6be3a"`, `file_sha256` completo. Guard de integridade usa `file_short`.
5. Experimento MLflow: `"infnet-ml-sistema"` com 10 runs existentes (code6ea3d3f). Parte 4 adiciona 15 runs ao MESMO experimento sem deletar os existentes.

---

## 2. Escolha de Técnicas e Justificativa

### Técnicas Escolhidas: **PCA + LDA**

**PCA (Principal Component Analysis)**
- Não supervisionado, linear, `O(n·p²)`, maduro em produção.
- Suporta `transform()` — pode ser encapsulado diretamente em `Pipeline.fit()`.
- `n_components` escolhido em dois valores fixos: **k=10** e **k=15**, para comparar dois pontos do trade-off variance-explained vs dimensionalidade.
- `sklearn.decomposition.PCA(n_components=k, random_state=42)`.

**LDA (Linear Discriminant Analysis)**
- Supervisionado — usa o target `y` para maximizar separabilidade entre classes.
- Restrição matemática em classificação binária: `n_components = min(n_classes - 1, n_features) = min(1, p) = 1`. **Não é uma escolha — é um fato.**
- Sklearn propaga `y` automaticamente no `Pipeline.fit(X, y)`.
- `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)`.

### t-SNE Excluído

**Motivo técnico**: t-SNE é **transdutivo** — não possui `transform()`, apenas `fit_transform()`. Não pode ser encapsulado em `Pipeline` para CV. Custo `O(n²)` inviável com 20975 amostras de treino em 5 folds. Uso legítimo: visualização exploratória, não pipeline de produção. Justificativa documentada em `reports/relatorio_tecnico.md` seção Parte 4.

---

## 3. Arquitetura do Pipeline

```
Pipeline(steps=[
    ("pre",    ColumnTransformer([              ← Parte 3, inalterado
                 ("num", StandardScaler(), NUMERIC_CONTINUOUS),
                 ("ord", "passthrough",    NUMERIC_ORDINAL),
                 ("cat", OneHotEncoder(),  CATEGORICAL),
               ])),
    ("dimred", PCA(n_components=k) | LDA(n_components=1)),   ← NOVO Parte 4
    ("clf",    estimator),                      ← Parte 3, inalterado
])
```

**Posição**: dimred entra DEPOIS do ColumnTransformer e ANTES do classificador. O scaler e OneHot são fitados apenas nos dados de treino (garantido pelo Pipeline).

**LDA e arrays densos**: `ColumnTransformer` com `remainder="drop"` (configuração atual de `build_preprocessor()`) produz array numpy denso — compatível com LDA.

---

## 4. Convenção MLflow — Parte 4

**Formato run name** (unchanged from master prompt):
```
{stage}__{model}__{preproc}__{dimred}__{search}__seed{seed}__data{DATAHASH8}__code{GITHASH7}
```

**Valores Parte 4**:
- `stage` = `dimred`
- `model` ∈ `{perceptron, logreg, dtree, rf, gb}`
- `preproc` = `numstd_catoh`
- `dimred` ∈ `{pca_k10, pca_k15, lda_k1}`
- `search` = `none`

**Total**: 5 modelos × 3 configs = **15 runs**.

**Tags adicionais obrigatórias** (além das 10 da Parte 3):
| Tag | Tipo | Exemplo |
|-----|------|---------|
| `dimred_method` | str | `"pca"` \| `"lda"` |
| `dimred_n_components` | str(int) | `"10"` |
| `dimred_explained_variance` | str(float) | `"0.8734"` (PCA) \| `"na"` (LDA) |
| `baseline_run_id` | str | run_id do run P3 baseline equivalente |

---

## 5. Proteções de Integridade

**Guard 1 — Data Fingerprint** (antes de qualquer run):
```python
def _verify_data_fingerprint(repo_root: Path) -> str:
    """Verifica SHA-256 do parquet. Falha com sys.exit(1) se mismatch."""
    import hashlib, json
    fp = json.loads((repo_root / "artifacts" / "data_fingerprint.json").read_text())
    expected = fp["file_short"]   # "30c6be3a"
    parquet_path = repo_root / "artifacts" / "data" / "credit_card_cleaned.parquet"
    actual = hashlib.sha256(parquet_path.read_bytes()).hexdigest()[:8]
    if actual != expected:
        print(f"ERRO: fingerprint mismatch: expected={expected}, actual={actual}", flush=True)
        sys.exit(1)
    return expected
```

**Guard 2 — Params não-vazios** (após cada run):
```python
REQUIRED_PARAMS = {
    "model_name", "seed", "cv_folds", "dimred_method",
    "dimred_n_components", "scoring_primary", "search_type",
}

def _assert_params_not_empty(client: MlflowClient, run_id: str) -> None:
    """Verifica que params obrigatórios foram logados. sys.exit(1) se faltarem."""
    params = client.get_run(run_id).data.params
    missing = REQUIRED_PARAMS - set(params.keys())
    if missing:
        print(f"ERRO: params ausentes no run {run_id}: {missing}", flush=True)
        sys.exit(1)
```

---

## 6. Diff-Plan por Commit

### C0 — `test: add failing tests for dimred module (RED)`

**Arquivo criado**: `tests/test_dimred.py`

```python
"""Testes RED para build_dimred_pipeline e compute_pca_n_components."""
from __future__ import annotations
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

# Esses imports vão FALHAR até C1 criar o módulo:
from credit_default.features.dimred import build_dimred_pipeline, compute_pca_n_components


@pytest.fixture
def small_X():
    rng = np.random.default_rng(42)
    return rng.standard_normal((30, 10))


@pytest.fixture
def small_y():
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, 30)


def test_build_dimred_pipeline_pca_returns_3step_pipeline():
    """Pipeline tem 3 steps: pre, dimred, clf; dimred é PCA."""
    pipe = build_dimred_pipeline("logreg", "pca", n_components=5)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps.keys()) == ["pre", "dimred", "clf"]
    assert isinstance(pipe.named_steps["dimred"], PCA)
    assert pipe.named_steps["dimred"].n_components == 5


def test_build_dimred_pipeline_lda_returns_3step_pipeline():
    """Pipeline tem 3 steps: pre, dimred, clf; dimred é LDA com n_components=1."""
    pipe = build_dimred_pipeline("logreg", "lda", n_components=1)
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps["dimred"], LinearDiscriminantAnalysis)
    assert pipe.named_steps["dimred"].n_components == 1


def test_build_dimred_pipeline_invalid_method_raises():
    """dimred_method inválido levanta ValueError."""
    with pytest.raises(ValueError, match="pca.*lda"):
        build_dimred_pipeline("logreg", "tsne", n_components=2)


def test_compute_pca_n_components_returns_valid_int(small_X):
    """compute_pca_n_components retorna int >= 1 e <= n_features."""
    k = compute_pca_n_components(small_X, threshold=0.85)
    assert isinstance(k, int)
    assert 1 <= k <= small_X.shape[1]


def test_compute_pca_n_components_higher_threshold_gives_more_components(small_X):
    """threshold maior → mais componentes."""
    k85 = compute_pca_n_components(small_X, threshold=0.85)
    k95 = compute_pca_n_components(small_X, threshold=0.95)
    assert k95 >= k85


def test_dimred_pipeline_fit_predict_smoke():
    """Pipeline PCA fit+predict_proba não levanta exceção em dados pequenos."""
    import pandas as pd
    from credit_default.features.preprocessing import NUMERIC_CONTINUOUS, NUMERIC_ORDINAL, CATEGORICAL
    # Cria DataFrame mínimo com todas as colunas esperadas
    rng = np.random.default_rng(0)
    n = 40
    cols = NUMERIC_CONTINUOUS + NUMERIC_ORDINAL + CATEGORICAL
    data = {c: rng.integers(0, 5, n) for c in cols}
    X = pd.DataFrame(data).astype(float)
    # Usar apenas colunas numéricas contínuas para simplificar
    X_simple = pd.DataFrame(rng.standard_normal((n, 5)),
                             columns=[f"f{i}" for i in range(5)])
    y = rng.integers(0, 2, n)
    # Smoke test com pipeline de 3 steps mínimo (pula pre pois X já é numérico)
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pipe = SKPipeline([
        ("pre", StandardScaler()),
        ("dimred", PCA(n_components=2, random_state=42)),
        ("clf", LogisticRegression(random_state=42)),
    ])
    pipe.fit(X_simple, y)
    proba = pipe.predict_proba(X_simple)
    assert proba.shape == (n, 2)
```

**Verificação**: `uv run pytest tests/test_dimred.py -v` → FALHA com `ImportError: cannot import name 'build_dimred_pipeline'` (RED confirmado).

---

### C1 — `feat: implement dimred module with build_dimred_pipeline and compute_pca_n_components (GREEN)`

**Arquivo criado**: `src/credit_default/features/dimred.py`

```python
"""Reducao de dimensionalidade: PCA e LDA integrados ao pipeline sklearn."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from credit_default.features.preprocessing import build_preprocessor
from credit_default.models.registry import get_model_spec


def compute_pca_n_components(
    X_transformed: np.ndarray,
    threshold: float = 0.85,
) -> int:
    """Retorna menor k tal que sum(explained_variance_ratio_[:k]) >= threshold.

    Parameters
    ----------
    X_transformed : array numpy já escalado (saída de pre.transform()).
    threshold     : variância acumulada mínima (default 0.85).

    Returns
    -------
    int : número de componentes PCA, entre 1 e n_features.
    """
    pca_full = PCA(random_state=42)
    pca_full.fit(X_transformed)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, X_transformed.shape[1])


def build_dimred_pipeline(
    model_name: str,
    dimred_method: str,
    n_components: int,
    *,
    seed: int = 42,
) -> Pipeline:
    """Constrói Pipeline de 3 etapas: pre → dimred → clf.

    Parameters
    ----------
    model_name    : chave do MODEL_REGISTRY (perceptron, logreg, dtree, rf, gb)
    dimred_method : "pca" ou "lda"
    n_components  : número de componentes (PCA: k>=1; LDA binário: sempre 1)
    seed          : random_state para PCA e estimador

    Returns
    -------
    sklearn.pipeline.Pipeline com steps ["pre", "dimred", "clf"]

    Raises
    ------
    ValueError : se dimred_method não for "pca" ou "lda"
    """
    spec = get_model_spec(model_name)
    # Clonar estimador via set_params para não mutar o registry
    import copy
    estimator = copy.deepcopy(spec["estimator"])

    pre = build_preprocessor()

    if dimred_method == "pca":
        dimred_step = PCA(n_components=n_components, random_state=seed)
    elif dimred_method == "lda":
        dimred_step = LinearDiscriminantAnalysis(n_components=n_components)
    else:
        raise ValueError(
            f"dimred_method deve ser 'pca' ou 'lda', recebido: {dimred_method!r}"
        )

    return Pipeline(steps=[
        ("pre", pre),
        ("dimred", dimred_step),
        ("clf", estimator),
    ])
```

**Arquivo modificado**: `src/credit_default/features/__init__.py`

Adicionar ao bloco de imports existente:
```python
from credit_default.features.dimred import (
    build_dimred_pipeline,
    compute_pca_n_components,
)
```
Adicionar ao `__all__`:
```python
"build_dimred_pipeline",
"compute_pca_n_components",
```

**Verificação**: `uv run pytest tests/test_dimred.py -v` → todos os testes passam (GREEN).

---

### C2 — `test: add failing tests for dimred params in log_standard_params (RED)`

**Arquivo modificado**: `tests/test_mlflow_utils.py`

Adicionar 2 testes ao final do arquivo (vão FALHAR até C3):

```python
def test_log_params_dimred_pca_keys_present():
    """Params dimred PCA: dimred_method e dimred_n_components presentes."""
    params = _run_with_params(
        {"clf__C": 1.0},
        search_type="none",
        dimred_method="pca",
        dimred_n_components=10,
    )
    assert params["dimred_method"] == "pca"
    assert params["dimred_n_components"] == "10"
    assert params["scoring_primary"] == "roc_auc"


def test_log_params_dimred_lda_keys_present():
    """Params dimred LDA: dimred_method=lda e dimred_n_components=1."""
    params = _run_with_params(
        {"clf__C": 1.0},
        search_type="none",
        dimred_method="lda",
        dimred_n_components=1,
    )
    assert params["dimred_method"] == "lda"
    assert params["dimred_n_components"] == "1"
```

NOTA: `_run_with_params` também precisa ser atualizada para aceitar os novos kwargs — mas isso só vai compilar após C3. O test file vai ter erro de sintaxe/argumento até lá.

Atualizar `_run_with_params` para aceitar kwargs opcionais:
```python
def _run_with_params(
    clf_params: dict,
    search_type: str = "none",
    dimred_method: str = "none",
    dimred_n_components: int = 0,
) -> dict:
    exp_id = mlflow.create_experiment(f"test-exp-{mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME}")
    # usar uuid para evitar colisão entre testes
    import uuid
    exp_id = mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")
    with mlflow.start_run(experiment_id=exp_id) as run:
        log_standard_params(
            run,
            model_name="logreg",
            seed=42,
            cv_folds=5,
            n_train=20975,
            n_val=4495,
            search_type=search_type,
            clf_params=clf_params,
            dimred_method=dimred_method,
            dimred_n_components=dimred_n_components,
        )
        run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    return client.get_run(run_id).data.params
```

**NOTA IMPORTANTE**: Os 5 testes existentes usam `_run_with_params` sem os novos kwargs — eles continuam funcionando porque os novos params têm defaults.

**Verificação**: `uv run pytest tests/test_mlflow_utils.py -v` → novos testes FALHAM (argumento inesperado), existentes passam.

---

### C3 — `feat: extend log_standard_params with dimred kwargs (GREEN)`

**Arquivo modificado**: `src/credit_default/tracking/mlflow_utils.py`

Alterar assinatura de `log_standard_params`:
```python
def log_standard_params(
    run: mlflow.ActiveRun,
    *,
    model_name: str,
    seed: int,
    cv_folds: int,
    n_train: int,
    n_val: int,
    search_type: str,
    clf_params: dict[str, Any],
    dimred_method: str = "none",
    dimred_n_components: int = 0,
) -> None:
    """Loga parametros padronizados no run ativo.

    Params logados:
    Meta (10): model_name, seed, cv_folds, scoring_primary, split_strategy,
               search_type, n_train, n_val, dimred_method, dimred_n_components
    Classifier (clf__*): clf_params — defaults para baseline, best_params_ para tuned
    """
    meta: dict[str, str] = {
        "model_name": model_name,
        "seed": str(seed),
        "cv_folds": str(cv_folds),
        "scoring_primary": "roc_auc",
        "split_strategy": "stratified_70_15_15_from_part2",
        "search_type": search_type,
        "n_train": str(n_train),
        "n_val": str(n_val),
        "dimred_method": dimred_method,
        "dimred_n_components": str(dimred_n_components),
    }
    clf_params_str = {k: str(v) for k, v in clf_params.items()}
    mlflow.log_params({**meta, **clf_params_str})
```

**ATENÇÃO**: Os callers existentes em `train.py` (Parte 3) não passam `dimred_method`/`dimred_n_components`, então receberão defaults `"none"` e `"0"` respectivamente — comportamento retrocompatível correto.

**Verificação**: `uv run pytest tests/test_mlflow_utils.py -v` → 7/7 testes passam.

---

### C4 — `feat: extend log_standard_tags with dimred tags`

**Arquivo modificado**: `src/credit_default/tracking/mlflow_utils.py`

Alterar assinatura de `log_standard_tags`:
```python
def log_standard_tags(
    run: mlflow.ActiveRun,
    *,
    model_family: str,
    git_commit: str,
    dataset_fingerprint: str,
    compute_profile_s: float,
    dimred_method: str = "none",
    dimred_n_components: int = 0,
    dimred_explained_variance: float | str = "na",
    baseline_run_id: str = "",
) -> None:
    """Loga tags padronizadas no run ativo.

    Tags logadas (14 total para runs Parte 4; 10 para Parte 3 com defaults):
    1-10. tags existentes da Parte 3 (model_family, git_commit, ...)
    11. dimred_method
    12. dimred_n_components
    13. dimred_explained_variance
    14. baseline_run_id
    """
    run_name = run.info.run_name or ""
    parts = run_name.split("__")
    stage = parts[0] if len(parts) > 0 else ""
    search_type_tag = parts[4] if len(parts) > 4 else "none"

    mlflow.set_tags(
        {
            "model_family": model_family,
            "git_commit": git_commit,
            "dataset_fingerprint": dataset_fingerprint,
            "compute_profile_s": str(compute_profile_s),
            "project_part": "parte_4",
            "framework": "scikit-learn",
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "stage": stage,
            "search_type": search_type_tag,
            "dimred_method": dimred_method,
            "dimred_n_components": str(dimred_n_components),
            "dimred_explained_variance": str(dimred_explained_variance),
            "baseline_run_id": baseline_run_id,
        }
    )
```

**ATENÇÃO**: Callers existentes em `train.py` (Parte 3) não passam os novos kwargs — recebem defaults. O campo `project_part` muda para `"parte_4"` apenas em runs que usam os novos kwargs explicitamente. Para os runs Parte 3 existentes, isso não importa (já estão logados). Para o `train.py` de Parte 3, o caller continua passando apenas os 4 kwargs originais.

**CORREÇÃO**: Para não quebrar retrocompatibilidade, manter `project_part` como valor fixo `"parte_3"` na função existente (ou torná-lo um parâmetro). Decisão: tornar `project_part` um parâmetro opcional:

```python
def log_standard_tags(
    run: mlflow.ActiveRun,
    *,
    model_family: str,
    git_commit: str,
    dataset_fingerprint: str,
    compute_profile_s: float,
    dimred_method: str = "none",
    dimred_n_components: int = 0,
    dimred_explained_variance: float | str = "na",
    baseline_run_id: str = "",
    project_part: str = "parte_3",   # ← novo, default preserva comportamento P3
) -> None:
```

E usar `"project_part": project_part` no dict de tags.

**Verificação**: `uv run pytest --tb=short -q` → 37 testes existentes passam (retrocompatibilidade OK).

---

### C5 — `test: add failing integration tests for train_dimred_and_evaluate (RED)`

**Arquivo criado**: `tests/test_train_dimred.py`

```python
"""Testes de integração para train_dimred_and_evaluate."""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# Import vai FALHAR até C6:
from credit_default.models.train import train_dimred_and_evaluate


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    mlflow.set_tracking_uri(tmp_path.as_uri())
    yield
    mlflow.set_tracking_uri("")


@pytest.fixture
def minimal_df_dimred():
    """DataFrame mínimo com todas as colunas do dataset real."""
    from tests.conftest import minimal_df  # reutilizar fixture
    # Nota: usar o fixture diretamente via parametrize não é possível aqui,
    # então recriamos o mesmo padrão de conftest.py
    from credit_default.features.preprocessing import (
        NUMERIC_CONTINUOUS, NUMERIC_ORDINAL, CATEGORICAL
    )
    rng = np.random.default_rng(7)
    n = 50  # suficiente para LDA (precisa de >1 amostra por classe)
    data = {}
    for col in NUMERIC_CONTINUOUS:
        data[col] = rng.integers(1000, 100000, n).astype(float)
    for col in NUMERIC_ORDINAL:
        data[col] = rng.integers(-2, 9, n)
    for col in CATEGORICAL:
        data[col] = rng.integers(1, 4, n)
    return pd.DataFrame(data)


@pytest.fixture
def minimal_y():
    rng = np.random.default_rng(7)
    y = pd.Series(rng.integers(0, 2, 50))
    # Garantir ao menos 2 amostras de cada classe para LDA
    y.iloc[:25] = 0
    y.iloc[25:] = 1
    return y


def test_train_dimred_pca_returns_dict_with_required_keys(
    minimal_df_dimred, minimal_y, tmp_path
):
    """train_dimred_and_evaluate com PCA retorna dict com chaves obrigatórias."""
    import uuid
    exp_id = mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")
    result = train_dimred_and_evaluate(
        "logreg",
        "pca",
        n_components=2,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_name=None,  # usa experiment_id diretamente
        experiment_id=exp_id,
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    required_keys = {
        "best_pipeline", "metrics", "run_id", "run_name",
        "cv_roc_auc_mean", "cv_roc_auc_std", "cv_f1_mean", "cv_f1_std",
        "dimred_explained_variance",
    }
    assert required_keys <= set(result.keys())


def test_train_dimred_lda_n_components_is_1(
    minimal_df_dimred, minimal_y, tmp_path
):
    """LDA pipeline: step dimred tem n_components == 1."""
    import uuid
    exp_id = mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")
    result = train_dimred_and_evaluate(
        "logreg",
        "lda",
        n_components=1,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=exp_id,
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    pipeline = result["best_pipeline"]
    assert isinstance(pipeline.named_steps["dimred"], LinearDiscriminantAnalysis)
    assert pipeline.named_steps["dimred"].n_components == 1


def test_train_dimred_params_not_empty(minimal_df_dimred, minimal_y, tmp_path):
    """Após run PCA, params MLflow contêm dimred_method e dimred_n_components."""
    import uuid
    exp_id = mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")
    result = train_dimred_and_evaluate(
        "logreg",
        "pca",
        n_components=2,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=exp_id,
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(result["run_id"]).data.params
    assert params.get("dimred_method") == "pca"
    assert params.get("dimred_n_components") == "2"
    assert params.get("scoring_primary") == "roc_auc"


def test_train_dimred_pca_explained_variance_is_float(
    minimal_df_dimred, minimal_y, tmp_path
):
    """result['dimred_explained_variance'] é float entre 0 e 1 para PCA."""
    import uuid
    exp_id = mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")
    result = train_dimred_and_evaluate(
        "logreg",
        "pca",
        n_components=2,
        X_train=minimal_df_dimred,
        y_train=minimal_y,
        X_val=minimal_df_dimred,
        y_val=minimal_y,
        seed=42,
        cv_folds=2,
        experiment_id=exp_id,
        datahash8="30c6be3a",
        githash7="test123",
        tmp_dir=tmp_path,
        baseline_run_id="",
    )
    ev = result["dimred_explained_variance"]
    assert isinstance(ev, float)
    assert 0.0 < ev <= 1.0
```

**Verificação**: `uv run pytest tests/test_train_dimred.py -v` → FALHA com `ImportError` (RED confirmado).

---

### C6 — `feat: add train_dimred_and_evaluate to train.py (GREEN)`

**Arquivo modificado**: `src/credit_default/models/train.py`

Adicionar nova função ao final do arquivo:

```python
def train_dimred_and_evaluate(
    model_name: str,
    dimred_method: str,
    n_components: int,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
    cv_folds: int = 5,
    experiment_name: str = "infnet-ml-sistema",
    experiment_id: str | None = None,
    datahash8: str,
    githash7: str,
    tmp_dir: Path,
    baseline_run_id: str = "",
) -> dict[str, Any]:
    """Pipeline com dimred: pre → dimred → clf, sem tuning (search=none).

    Fluxo:
    1. build_dimred_pipeline(model_name, dimred_method, n_components, seed=seed)
    2. cross_validate() com scoring={roc_auc, f1_macro},
       cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    3. pipeline.fit(X_train, y_train)
    4. Predict X_val: y_pred, y_proba
    5. compute_all_metrics(y_val, y_pred, y_proba)
    6. inference_latency_ms: tempo médio por amostra no X_val
    7. Plots: confusion_matrix, roc, pr no tmp_dir
    8. Para PCA: dimred_explained_variance = pipeline["dimred"].explained_variance_ratio_.sum()
       Para LDA: dimred_explained_variance = "na" (string)
    9. run_summary.json com primary_metric="roc_auc", dimred_method, dimred_n_components
    10. split_fingerprint.txt
    11. feature_importances.csv se clf tem feature_importances_
    12. dimred_label = f"{dimred_method}_k{n_components}"
    13. compose_run_name(stage="dimred", model=model_name, preproc="numstd_catoh",
                         dimred=dimred_label, search="none", seed=seed,
                         datahash8=datahash8, githash7=githash7)
    14. MLflow start_run → log_standard_tags (com dimred + baseline_run_id, project_part="parte_4")
                         → log_standard_params (com dimred_method, dimred_n_components)
                         → log_standard_metrics → log_standard_artifacts
    15. Retorna dict: best_pipeline, metrics, run_id, run_name,
                      cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std,
                      dimred_explained_variance (float ou "na")

    Parameters
    ----------
    model_name     : chave do MODEL_REGISTRY
    dimred_method  : "pca" ou "lda"
    n_components   : número de componentes (LDA: sempre 1)
    X_train, y_train : dados de treino
    X_val, y_val     : dados de validação
    seed           : semente
    cv_folds       : número de folds CV
    experiment_name: nome do experiment MLflow (ignorado se experiment_id fornecido)
    experiment_id  : ID direto do experiment (para testes)
    datahash8      : primeiros 8 chars SHA-256 do dataset
    githash7       : primeiros 7 chars do git commit hash
    tmp_dir        : diretório temporário para artefatos
    baseline_run_id: run_id do run P3 baseline equivalente (tag baseline_run_id)

    Returns
    -------
    dict com chaves: best_pipeline, metrics, run_id, run_name,
                     cv_roc_auc_mean, cv_roc_auc_std, cv_f1_mean, cv_f1_std,
                     dimred_explained_variance
    """
    from credit_default.features.dimred import build_dimred_pipeline

    spec = get_model_spec(model_name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # 1. Build pipeline com dimred
    pipeline = build_dimred_pipeline(model_name, dimred_method, n_components, seed=seed)

    # 2. Cross-validation
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    cv_roc_auc_mean = float(np.mean(cv_results["test_roc_auc"]))
    cv_roc_auc_std = float(np.std(cv_results["test_roc_auc"]))
    cv_f1_mean = float(np.mean(cv_results["test_f1_macro"]))
    cv_f1_std = float(np.std(cv_results["test_f1_macro"]))

    # 3. Fit simples no X_train (sem tuning)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time_s = time.perf_counter() - t0

    # 4-5. Predict e métricas
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = compute_all_metrics(np.array(y_val), y_pred, y_proba)

    # 6. Inference latency
    n_samples = len(X_val)
    t_inf = time.perf_counter()
    pipeline.predict_proba(X_val)
    inference_latency_ms = (time.perf_counter() - t_inf) / n_samples * 1000

    # 7. Plots
    confusion_matrix_plot(np.array(y_val), y_pred, output_path=tmp_dir / "confusion_matrix.png")
    roc_plot(np.array(y_val), y_proba, output_path=tmp_dir / "roc_curve.png")
    pr_plot(np.array(y_val), y_proba, output_path=tmp_dir / "pr_curve.png")

    # 8. Explained variance
    if dimred_method == "pca":
        dimred_ev: float | str = float(
            pipeline.named_steps["dimred"].explained_variance_ratio_.sum()
        )
    else:
        dimred_ev = "na"

    # 9. run_summary.json
    dimred_label = f"{dimred_method}_k{n_components}"
    summary = {
        "model_name": model_name,
        "primary_metric": "roc_auc",
        "metrics": metrics,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "dimred_method": dimred_method,
        "dimred_n_components": n_components,
        "dimred_explained_variance": dimred_ev,
        "seed": seed,
        "training_time_s": training_time_s,
        "inference_latency_ms": inference_latency_ms,
    }
    with open(tmp_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 10. split_fingerprint.txt
    (tmp_dir / "split_fingerprint.txt").write_text(
        f"datahash8={datahash8}\ngithash7={githash7}\n"
    )

    # 11. feature_importances.csv
    has_fi = False
    clf_step = pipeline.named_steps.get("clf")
    estimator_inner = getattr(clf_step, "estimator", clf_step)
    if hasattr(estimator_inner, "feature_importances_"):
        has_fi = True
        fi = pd.DataFrame({"importance": estimator_inner.feature_importances_})
        fi.to_csv(tmp_dir / "feature_importances.csv", index=False)

    # 12. clf_params (baseline: get_params filtrado)
    _SCALAR_TYPES = (str, int, float, bool, type(None))
    clf_params: dict[str, Any] = {
        k: v
        for k, v in pipeline.get_params(deep=True).items()
        if k.startswith("clf__") and isinstance(v, _SCALAR_TYPES)
    }

    # 13. Run name
    run_name = compose_run_name(
        stage="dimred",
        model=model_name,
        preproc="numstd_catoh",
        dimred=dimred_label,
        search="none",
        seed=seed,
        datahash8=datahash8,
        githash7=githash7,
    )

    # 14. MLflow logging
    if experiment_id is None:
        experiment_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        log_standard_tags(
            run,
            model_family=spec["model_family"],
            git_commit=githash7,
            dataset_fingerprint=datahash8,
            compute_profile_s=training_time_s,
            dimred_method=dimred_method,
            dimred_n_components=n_components,
            dimred_explained_variance=dimred_ev,
            baseline_run_id=baseline_run_id,
            project_part="parte_4",
        )
        log_standard_params(
            run,
            model_name=model_name,
            seed=seed,
            cv_folds=cv_folds,
            n_train=len(X_train),
            n_val=len(X_val),
            search_type="none",
            clf_params=clf_params,
            dimred_method=dimred_method,
            dimred_n_components=n_components,
        )
        log_standard_metrics(
            run,
            metrics,
            cv_roc_auc_mean=cv_roc_auc_mean,
            cv_roc_auc_std=cv_roc_auc_std,
            cv_f1_mean=cv_f1_mean,
            cv_f1_std=cv_f1_std,
            training_time_s=training_time_s,
            inference_latency_ms=inference_latency_ms,
        )
        log_standard_artifacts(run, tmp_dir, has_feature_importances=has_fi)
        run_id = run.info.run_id

    return {
        "best_pipeline": pipeline,
        "metrics": metrics,
        "run_id": run_id,
        "run_name": run_name,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "dimred_explained_variance": dimred_ev,
    }
```

Adicionar import no topo de `train.py` (na seção de imports locais):
```python
# Nota: build_dimred_pipeline importado localmente dentro da função
# para evitar importação circular potencial
```

**Verificação**: `uv run pytest tests/test_train_dimred.py -v` → 4 testes passam (GREEN).
`uv run pytest --tb=short -q` → todos os testes passam (37 + 4 novos = 41).

---

### C7 — `feat: scripts/train_dimred.py — 15 MLflow runs (5 modelos × 3 configs)`

**Arquivo criado**: `scripts/train_dimred.py`

```python
#!/usr/bin/env python
"""Treina 5 modelos com 3 configs de dimred: pca_k10, pca_k15, lda_k1 = 15 runs.

Proteções de integridade:
- Guard 1: data fingerprint verificado antes de qualquer run (sys.exit(1) em mismatch)
- Guard 2: params MLflow verificados após cada run (sys.exit(1) se params ausentes)

NÃO deleta runs da Parte 3. Adiciona 15 runs ao experimento existente.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from credit_default.models.train import load_split_data, train_dimred_and_evaluate
from credit_default.tracking.mlflow_utils import get_or_create_experiment

# ─── Constantes ────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "infnet-ml-sistema"
MODELS = ["perceptron", "logreg", "dtree", "rf", "gb"]
DIMRED_CONFIGS = [
    ("pca", 10, "pca_k10"),
    ("pca", 15, "pca_k15"),
    ("lda", 1,  "lda_k1"),
]
REQUIRED_PARAMS = {
    "model_name", "seed", "cv_folds", "dimred_method",
    "dimred_n_components", "scoring_primary", "search_type",
}

# ─── Guards de Integridade ─────────────────────────────────────────────────────

def _verify_data_fingerprint() -> str:
    """Verifica SHA-256 do parquet contra artifacts/data_fingerprint.json.
    sys.exit(1) em mismatch.
    """
    fp_path = repo_root / "artifacts" / "data_fingerprint.json"
    with open(fp_path) as f:
        fp = json.load(f)
    expected = fp["file_short"]  # "30c6be3a"

    parquet_path = repo_root / "artifacts" / "data" / "credit_card_cleaned.parquet"
    actual = hashlib.sha256(parquet_path.read_bytes()).hexdigest()[:8]

    if actual != expected:
        print(
            f"ERRO INTEGRIDADE: fingerprint mismatch! "
            f"expected={expected}, actual={actual}",
            flush=True,
        )
        sys.exit(1)
    print(f"[OK] fingerprint={expected}", flush=True)
    return expected


def _assert_params_not_empty(client: MlflowClient, run_id: str) -> None:
    """Verifica params MLflow obrigatórios. sys.exit(1) se ausentes."""
    params = client.get_run(run_id).data.params
    missing = REQUIRED_PARAMS - set(params.keys())
    if missing:
        print(
            f"ERRO INTEGRIDADE: params ausentes no run {run_id}: {missing}",
            flush=True,
        )
        sys.exit(1)


def _get_baseline_run_id(client: MlflowClient, experiment_id: str, model_name: str) -> str:
    """Busca run_id do baseline P3 equivalente (stage=baseline, mesmo model_name).
    Retorna "" se não encontrado (não falha).
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.stage = 'baseline' AND params.model_name = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else ""


def _get_githash7() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Guard 1: data fingerprint
    datahash8 = _verify_data_fingerprint()
    githash7 = _get_githash7()

    # Tracking URI Windows-safe
    mlflow.set_tracking_uri((repo_root / "mlruns").absolute().as_uri())
    client = MlflowClient()
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    # Dados (carregados uma vez)
    parquet_path = repo_root / "artifacts" / "data" / "credit_card_cleaned.parquet"
    split_path = repo_root / "artifacts" / "splits" / "split_indices.json"
    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data(parquet_path, split_path)

    print(f"\n{'='*70}")
    print(f"Dimred Training | datahash8={datahash8} | git={githash7}")
    print(f"Configs: {[c[2] for c in DIMRED_CONFIGS]} | Modelos: {MODELS}")
    print(f"{'='*70}")

    results: list[dict[str, Any]] = []

    for dimred_method, n_components, dimred_label in DIMRED_CONFIGS:
        print(f"\n--- {dimred_label.upper()} ---")
        for model_name in MODELS:
            baseline_rid = _get_baseline_run_id(client, experiment_id, model_name)
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = train_dimred_and_evaluate(
                    model_name,
                    dimred_method,
                    n_components,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    seed=42,
                    cv_folds=5,
                    experiment_id=experiment_id,
                    datahash8=datahash8,
                    githash7=githash7,
                    tmp_dir=Path(tmp_dir),
                    baseline_run_id=baseline_rid,
                )

            # Guard 2: params não-vazios
            _assert_params_not_empty(client, result["run_id"])

            ev = result["dimred_explained_variance"]
            ev_str = f"{ev:.4f}" if isinstance(ev, float) else ev
            print(
                f"  {model_name:<12} roc_auc={result['metrics']['roc_auc']:.4f}"
                f"  f1_macro={result['metrics']['f1_macro']:.4f}"
                f"  ev={ev_str}"
            )
            results.append({
                "model": model_name,
                "dimred": dimred_label,
                "roc_auc": result["metrics"]["roc_auc"],
                "f1_macro": result["metrics"]["f1_macro"],
            })

    # Resumo final
    print(f"\n{'='*70}")
    print("Resumo final (ordenado por roc_auc desc — PRIMARY):")
    for r in sorted(results, key=lambda x: x["roc_auc"], reverse=True):
        print(f"  {r['dimred']:<10} {r['model']:<12} roc_auc={r['roc_auc']:.4f}  f1_macro={r['f1_macro']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
```

**Verificação**:
```bash
uv run python scripts/train_dimred.py
# Deve completar sem erro, imprimir resumo de 15 runs
# Verificar: ls mlruns/<exp_id>/ | wc -l >= 25 (10 P3 + 15 P4)
```

**PONTO DE PARADA**: Após este commit e execução do script, **PARAR e aguardar confirmação do usuário** sobre ParamsCount disk check antes de continuar para docs.

---

### C8 — `feat: scripts/generate_comparison_dimred.py`

**Arquivo criado**: `scripts/generate_comparison_dimred.py`

```python
"""Gera tabelas comparativas Parte 4 (dimred) vs Parte 3 (baseline).

Outputs:
- reports/parte_4/comparison_dimred.md   : todos os 15 runs dimred + 5 P3 baseline
- reports/parte_4/comparison_pca_vs_lda.md : pivot por modelo (colunas = none/pca_k10/pca_k15/lda_k1)
"""
from __future__ import annotations
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

import mlflow
import pandas as pd

from credit_default.tracking.compare import consolidated_results_table
from credit_default.tracking.mlflow_utils import get_or_create_experiment

EXPERIMENT_NAME = "infnet-ml-sistema"
OUTPUT_DIR = repo_root / "reports" / "parte_4"

def main() -> None:
    mlflow.set_tracking_uri((repo_root / "mlruns").absolute().as_uri())
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    # Carregar todos os runs
    df = consolidated_results_table(experiment_id)

    # Separar P3 baseline e P4 dimred
    p3 = df[df["run_name"].str.startswith("baseline__")].copy()
    p4 = df[df["run_name"].str.startswith("dimred__")].copy()

    # ── Tabela 1: comparison_dimred.md ──────────────────────────────────────────
    cols = ["run_name", "model_name", "model_family", "roc_auc", "f1_macro",
            "cv_roc_auc_mean", "cv_roc_auc_std", "training_time_s",
            "dimred_method", "dimred_n_components", "dimred_explained_variance"]

    # Adicionar cols dimred ao p3 (preenchidas como "none"/"0"/"na")
    for col in ["dimred_method", "dimred_n_components", "dimred_explained_variance"]:
        if col not in p3.columns:
            p3[col] = "none" if "method" in col or "variance" in col else "0"

    combined = pd.concat([p3, p4], ignore_index=True).sort_values("roc_auc", ascending=False)

    available_cols = [c for c in cols if c in combined.columns]
    md1 = combined[available_cols].to_markdown(index=False, floatfmt=".4f")

    out1 = OUTPUT_DIR / "comparison_dimred.md"
    out1.write_text(
        f"# Comparativo Parte 4 — Dimred vs Baseline (Parte 3)\n\n"
        f"Ordenado por `roc_auc` desc (métrica primária).\n\n"
        f"{md1}\n",
        encoding="utf-8",
    )
    print(f"Gerado: {out1}")

    # ── Tabela 2: comparison_pca_vs_lda.md ──────────────────────────────────────
    pivot_rows = []
    models = ["perceptron", "logreg", "dtree", "rf", "gb"]
    configs = ["none (P3)", "pca_k10", "pca_k15", "lda_k1"]

    for model in models:
        row = {"model": model}
        # P3 baseline
        p3_model = p3[p3["model_name"] == model]
        row["none (P3)"] = f"{p3_model['roc_auc'].values[0]:.4f}" if not p3_model.empty else "n/a"
        # P4 dimred configs
        for cfg in ["pca_k10", "pca_k15", "lda_k1"]:
            p4_cfg = p4[p4["run_name"].str.contains(f"__{cfg}__")]
            p4_model = p4_cfg[p4_cfg["run_name"].str.contains(f"__{model}__")]
            row[cfg] = f"{p4_model['roc_auc'].values[0]:.4f}" if not p4_model.empty else "n/a"
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows, columns=["model"] + configs)
    md2 = pivot_df.to_markdown(index=False)

    out2 = OUTPUT_DIR / "comparison_pca_vs_lda.md"
    out2.write_text(
        f"# Pivot: ROC-AUC por Modelo e Técnica de Dimred\n\n"
        f"Colunas: baseline sem dimred (P3) + 3 configs Parte 4.\n\n"
        f"{md2}\n",
        encoding="utf-8",
    )
    print(f"Gerado: {out2}")


if __name__ == "__main__":
    main()
```

**Verificação**:
```bash
uv run python scripts/generate_comparison_dimred.py
ls reports/parte_4/
# comparison_dimred.md  comparison_pca_vs_lda.md
```

---

### C9 — `test: smoke tests for generate_comparison_dimred`

**Arquivo criado**: `tests/test_generate_comparison_dimred.py`

```python
"""Smoke tests para generate_comparison_dimred.py."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
SCRIPT = repo_root / "scripts" / "generate_comparison_dimred.py"
OUT_DIR = repo_root / "reports" / "parte_4"


@pytest.mark.skipif(
    not (OUT_DIR / "comparison_dimred.md").exists(),
    reason="Requires train_dimred.py to have been run first",
)
def test_comparison_dimred_md_exists():
    """comparison_dimred.md existe após execução do script."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert (OUT_DIR / "comparison_dimred.md").exists()
    assert (OUT_DIR / "comparison_pca_vs_lda.md").exists()


@pytest.mark.skipif(
    not (OUT_DIR / "comparison_dimred.md").exists(),
    reason="Requires train_dimred.py to have been run first",
)
def test_comparison_dimred_has_data_rows():
    """comparison_dimred.md tem pelo menos 15 linhas de dados."""
    md = (OUT_DIR / "comparison_dimred.md").read_text(encoding="utf-8")
    data_lines = [
        ln for ln in md.split("\n")
        if ln.strip().startswith("|") and "---" not in ln and "run_name" not in ln
    ]
    assert len(data_lines) >= 15, f"Esperado >= 15 linhas, encontrado {len(data_lines)}"
```

**Verificação**: `uv run pytest tests/test_generate_comparison_dimred.py -v`

---

### C10 — `feat: create reports/parte_4/ structure and update .gitignore`

**Arquivos**:
- Criar `reports/parte_4/.gitkeep` (arquivo vazio para rastrear diretório no git)
- Atualizar `.gitignore`: adicionar linha `reports/parte_4/figures/` se não existir

**Verificação**:
```bash
git check-ignore reports/parte_4/figures/nonexistent.png  # deve retornar o path
ls reports/parte_4/
```

---

### C11 — `docs: add Parte 4 section to relatorio_tecnico.md`

**Arquivo modificado**: `reports/relatorio_tecnico.md`

Adicionar seção ao final:

```markdown
## Parte 4 — Redução de Dimensionalidade

### Escolha das Técnicas

Foram aplicadas exatamente duas técnicas de redução de dimensionalidade: **PCA** e **LDA**.

**t-SNE foi excluído** pelos seguintes motivos técnicos:
1. **Algoritmo transductivo**: não possui `transform()`, apenas `fit_transform()`. Impossível encapsular em `sklearn.Pipeline` para validação cruzada.
2. **Custo computacional O(n²)**: com 20.975 amostras de treino em 5 folds, o tempo seria proibitivo (~horas).
3. **Finalidade**: t-SNE é uma técnica de visualização exploratória, não um extrator de features para pipelines de classificação.

### PCA — Análise de Componentes Principais

- **Tipo**: não supervisionado, linear
- **Implementação**: `sklearn.decomposition.PCA`
- **Configurações testadas**: k=10 e k=15 componentes
- **Posição no pipeline**: após `ColumnTransformer`, antes do classificador
- **Variance explicada**: logada por run via `explained_variance_ratio_.sum()`

### LDA — Linear Discriminant Analysis

- **Tipo**: supervisionado (usa o target y), linear
- **Implementação**: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
- **Restrição binária**: `n_components = min(n_classes - 1, n_features) = 1`
  - Para classificação binária, LDA produz exatamente **1 componente discriminante**.
  - Não é uma escolha de hiperparâmetro — é uma consequência matemática.
- **Vantagem**: maximiza a separabilidade entre classes (ratio variância-entre/variância-dentro)

### Arquitetura do Pipeline

```
Pipeline(steps=[
    ("pre",    ColumnTransformer([...]))   # Parte 3 inalterado
    ("dimred", PCA(k) | LDA(1))            # Parte 4 — novo
    ("clf",    estimador)                  # Parte 3 inalterado
])
```

### Experimento MLflow

- **Experimento**: `infnet-ml-sistema` (mesmo da Parte 3)
- **Runs adicionados**: 15 (5 modelos × 3 configs: pca_k10, pca_k15, lda_k1)
- **Runs Parte 3**: preservados intactos (não deletados)
- **Métrica primária**: `roc_auc` (consistente com Parte 3)

### Resultados

[tabela gerada por `scripts/generate_comparison_dimred.py`]

Ver: `reports/parte_4/comparison_dimred.md` e `reports/parte_4/comparison_pca_vs_lda.md`

### Trade-offs Observados

| Aspecto | PCA k=10 | PCA k=15 | LDA k=1 |
|---------|----------|----------|---------|
| Dimensionalidade final | 10 | 15 | 1 |
| Informação usada | não supervisionado | não supervisionado | supervisionado |
| Interpretabilidade | componentes ortogonais (difícil) | idem | 1 eixo discriminante |
| Custo fit | baixo | baixo | baixo |
| Risco overfitting por dimred | baixo | baixo | médio (leakage se mal aplicado) |
| Suporte Pipeline.fit() | ✓ | ✓ | ✓ (y propagado) |
```

---

### C12 — `docs: add Parte 4 section to README.md`

**Arquivo modificado**: `README.md`

Adicionar seção `## Parte 4 — Redução de Dimensionalidade`:

```markdown
## Parte 4 — Redução de Dimensionalidade

Aplica PCA (k=10, k=15) e LDA (k=1) ao pipeline sklearn. 15 runs MLflow.

### Como executar

```bash
# 1. Treinar 15 modelos com dimred (5 modelos × 3 configs)
uv run python scripts/train_dimred.py

# 2. Gerar tabelas comparativas
uv run python scripts/generate_comparison_dimred.py

# 3. Ver resultados
cat reports/parte_4/comparison_dimred.md
cat reports/parte_4/comparison_pca_vs_lda.md
```

### Resultados

- `reports/parte_4/comparison_dimred.md` — todos os 20 runs (15 P4 + 5 P3 baseline) ordenados por roc_auc
- `reports/parte_4/comparison_pca_vs_lda.md` — pivot: modelos × técnicas

### MLflow UI

```bash
uv run mlflow ui --backend-store-uri mlruns/
# Acessar http://localhost:5000
# Filtrar por tag stage=dimred para ver apenas Parte 4
```
```

---

## 7. Arquitetura — Diagrama de Módulos

```
src/credit_default/
├── data/               CONGELADO (Parte 2)
├── features/
│   ├── __init__.py     MODIFICADO: +build_dimred_pipeline, +compute_pca_n_components
│   ├── preprocessing.py INALTERADO
│   └── dimred.py        NOVO (C1)
├── models/
│   ├── pipeline.py     INALTERADO
│   ├── registry.py     INALTERADO
│   └── train.py        MODIFICADO: +train_dimred_and_evaluate (C6)
├── tracking/
│   ├── mlflow_utils.py MODIFICADO: log_standard_params +dimred kwargs (C3)
│   │                               log_standard_tags +dimred tags (C4)
│   ├── compare.py      INALTERADO
│   └── run_naming.py   INALTERADO
└── evaluation/         INALTERADO

scripts/
├── train_baseline.py   INALTERADO
├── train_tuned.py      INALTERADO
├── train_dimred.py     NOVO (C7)
└── generate_comparison_dimred.py  NOVO (C8)

tests/
├── conftest.py         INALTERADO
├── test_dimred.py      NOVO (C0+C1)
├── test_train_dimred.py NOVO (C5+C6)
├── test_generate_comparison_dimred.py NOVO (C9)
├── test_mlflow_utils.py MODIFICADO: +2 testes dimred (C2+C3)
└── (outros inalterados)

reports/
└── parte_4/
    ├── .gitkeep         NOVO (C10)
    ├── comparison_dimred.md      (gerado por script, não commitado)
    └── comparison_pca_vs_lda.md  (gerado por script, não commitado)
```

---

## 8. Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| LDA espera array denso (não sparse) | Média | `build_preprocessor()` retorna array denso com `remainder="drop"` — verificar em teste de smoke |
| `explained_variance_ratio_` acessado antes de fit | Alta (bug comum) | Acessado APÓS `pipeline.fit()` |
| LDA n_components > 1 em classificação binária | Alta (erro conceitual) | Hardcoded n_components=1; teste `test_lda_n_components_is_1` garante |
| baseline_run_id não encontrado (P3 runs ausentes) | Baixa | Default `""`, warning logado, não falha |
| CalibratedClassifierCV + LDA 1D emite warning de calibração | Alta (esperado) | Não é erro — apenas warning; não suprimir |
| `compose_run_name` não aceita kwarg `dimred` | Verificar | Ler assinatura atual de `run_naming.py` antes de C6 |
| Parte 3 runs deletados acidentalmente | Baixa | Script não chama `mlflow.delete_run()`, só lê o experimento existente |
| PCA k=15 > features disponíveis após OneHot | Possível | Dataset tem ~23 features originais; após OneHot (SEX, EDUCATION, MARRIAGE) tem ~26 features. k=15 é seguro. |

**ATENÇÃO — `compose_run_name` assinatura atual**:
```python
def compose_run_name(stage, model, preproc="numstd_catoh", dimred="none",
                     search="none", seed=42, datahash8="", githash7="") -> str:
```
O kwarg `dimred` já existe. Confirmar antes de C6.

---

## 9. DoD Checklist — Parte 4

```markdown
- [ ] `uv run pytest tests/test_dimred.py tests/test_train_dimred.py tests/test_mlflow_utils.py -q` — todos verdes
- [ ] `uv run pytest --tb=short -q` — 41+ testes passam
- [ ] `uv run ruff check src/ scripts/ tests/` sem erros
- [ ] `uv run black --check src/ scripts/ tests/` sem reformatações
- [ ] `uv run python scripts/train_dimred.py` completa sem erro; 15 runs novos em "infnet-ml-sistema"
- [ ] 10 runs Parte 3 preservados (não deletados)
- [ ] Guard 1 executado: "fingerprint=30c6be3a" impresso no início do script
- [ ] Guard 2 executado: nenhum "ERRO INTEGRIDADE" nos logs de 15 runs
- [ ] Params verificados: 15/15 runs têm dimred_method e dimred_n_components em params
- [ ] PCA explained_variance logado (float) nos 10 runs PCA; "na" nos 5 runs LDA
- [ ] LDA runs: pipeline["dimred"].n_components == 1 (verificado em test_train_dimred.py)
- [ ] `uv run python scripts/generate_comparison_dimred.py` gera comparison_dimred.md e comparison_pca_vs_lda.md
- [ ] comparison_dimred.md tem ≥ 15 linhas de dados (15 P4 + 5 P3 = 20)
- [ ] run_summary.json de cada run contém primary_metric="roc_auc" e dimred_method
- [ ] split_fingerprint.txt em 15/15 runs confirma datahash8=30c6be3a
- [ ] test_idx nunca tocado (train_dimred_and_evaluate não usa X_test/y_test)
- [ ] relatorio_tecnico.md tem seção Parte 4 com justificativa t-SNE exclusão
- [ ] README.md tem seção Parte 4 com instruções de execução
```

---

## 10. Checklist de Integridade — Parte 4

```markdown
- [ ] `_verify_data_fingerprint()` executado antes de qualquer run (sys.exit(1) em mismatch)
- [ ] `_assert_params_not_empty()` executado após CADA um dos 15 runs
- [ ] Experimento "infnet-ml-sistema" não recriado (get_or_create_experiment reutiliza existente)
- [ ] mlruns/ da Parte 3 não deletado (confirmar: `ls mlruns/<exp_id>/ | wc -l >= 25`)
- [ ] `build_dimred_pipeline()` retorna Pipeline com 3 steps ["pre", "dimred", "clf"]
- [ ] Split indices lidos de split_indices.json (nunca regenerados) — reutiliza load_split_data()
- [ ] datahash8 lido de artifacts/data_fingerprint.json (nunca hardcoded no código de produção)
- [ ] githash7 obtido via git rev-parse em runtime
- [ ] LDA em Pipeline recebe y via Pipeline.fit(X, y) automaticamente
- [ ] PCA explained_variance_ratio_.sum() acessado APÓS pipeline.fit()
```

---

## 11. Estimativa de Tempo

| Etapa | Tempo estimado |
|-------|----------------|
| C0+C1: dimred.py + testes | ~15 min |
| C2+C3+C4: extensão mlflow_utils | ~10 min |
| C5+C6: train_dimred_and_evaluate | ~20 min |
| C7: train_dimred.py + execução (15 runs, CV 5-fold, sem tuning) | ~10-20 min execução |
| **PAUSA**: aguardar confirmação ParamsCount do usuário | — |
| C8+C9: comparison script + testes | ~15 min |
| C10: estrutura reports/parte_4/ | ~2 min |
| C11+C12: docs | ~15 min |
| **DoD completo** | **~30-45 min** |

---

## 12. Sequência de Commits

```
C0  test: add failing tests for dimred module (RED)
C1  feat: implement build_dimred_pipeline and compute_pca_n_components
C2  test: add failing tests for dimred params in log_standard_params (RED)
C3  feat: extend log_standard_params with dimred_method and dimred_n_components
C4  feat: extend log_standard_tags with dimred and baseline_run_id tags
C5  test: add failing integration tests for train_dimred_and_evaluate (RED)
C6  feat: add train_dimred_and_evaluate to train.py
C7  feat: scripts/train_dimred.py with integrity guards
    ← PARAR e aguardar confirmação do usuário ←
C8  feat: scripts/generate_comparison_dimred.py
C9  test: smoke tests for generate_comparison_dimred
C10 feat: create reports/parte_4 structure and update .gitignore
C11 docs: add Parte 4 section to relatorio_tecnico.md
C12 docs: add Parte 4 section to README.md
```
