# PLAN.md — Parte 2: Fundação de Dados e Diagnóstico Inicial

> **Execução somente após aprovação explícita do usuário.**

---

## 1. Confirmação de Contexto

| Item | Esperado | Verificado |
|------|----------|-----------|
| HEAD | `516306a` | `516306a89ed650b36a6bc639f5955c31d0ee1f71` ✅ |
| Working tree | limpo | limpo (sem output em `git status -s`) ✅ |
| Layout | flat, 6 notebooks, utils.py, 3 docs, .gitignore | confirmado ✅ |
| SHA256 do .xls | `30c6be3a…` | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` ✅ |
| .gitignore atual | bloqueia data/, .venv/ | bloqueia data/, models/, results/, .venv/ — **FALTA artifacts/ e mlruns/** → corrigido no Commit 1 |
| uv disponível | uv instalado | ⚠️ não verificado — confirmar com `uv --version` antes do Commit 1 |
| Pastas src/, configs/, tests/ | ausentes | ausentes ✅ |

---

## PRÉ-REQUISITO — Verificar uv e Python 3.11 (antes do Commit 1)

```bash
# 1. Verificar uv instalado
uv --version
# Se ausente: instalar em https://docs.astral.sh/uv/getting-started/installation/

# 2. Verificar Python 3.11 disponível via uv
uv python list
# Deve mostrar 3.11.x como instalado (não apenas 'download available')

# 3. Se 3.11 não estiver listado como instalado:
uv python install 3.11
uv python list   # confirmar que 3.11 aparece sem 'download available'
```

> uv gerencia a instalação do Python internamente — não é necessário instalar via python.org.

---

## 2. Diff-Plan por Commit

### Commit 1 — `chore(repo): adiciona pyproject, uv, ruff, black, pytest e .gitignore`

**Arquivos criados/modificados:**
```
pyproject.toml          (NOVO)
uv.lock                 (NOVO — gerado por `uv lock`, versionado no repo)
.gitignore              (MODIFICADO — append)
```

**Responsabilidades do `pyproject.toml`:**
- `name = "infnet-sk-learn-models"`, `version = "0.2.0"`
- `requires-python = ">=3.10,<3.13"` (exclui 3.13+ onde wheels podem faltar)
- Dependências de runtime: `pandas`, `numpy`, `scikit-learn`, `xlrd>=2.0.1`, `pyarrow`, `pyyaml`, `matplotlib`, `seaborn`
- Dependências de dev (extras `[dev]`): `pytest>=7.4`, `pytest-cov`, `ruff>=0.4`, `black>=24.0`
- `[tool.setuptools.package-dir]`: `"" = "src"` (src-layout)
- `[tool.setuptools.packages.find]`: `where = ["src"]`
- `[tool.ruff]`: `line-length = 99`, rules `["E","F","W","I"]`, `exclude = ["*.ipynb", "utils.py"]`
- `[tool.black]`: `line-length = 99`, `exclude = '(\.ipynb|utils\.py)'`
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`
- `[tool.uv]`: `python-version = "3.11"`

**Processo do Commit 1** (ordem obrigatória):
```bash
# 1. Escrever pyproject.toml e .gitignore
# 2. Gerar lockfile
uv lock
# 3. Staged todos os três arquivos
git add pyproject.toml uv.lock .gitignore
git commit ...
```

`uv.lock` é **versionado** (NÃO entra no `.gitignore`).

**Modificações em `.gitignore`** (append ao final):
```
# Artifacts de execução (gerados localmente, nunca versionados)
artifacts/
mlruns/
```

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
chore(repo): adiciona pyproject, uv, ruff, black, pytest e .gitignore

Estabelece a base de tooling reproduzível para a Parte 2.
pyproject.toml limita Python a >=3.10,<3.13; [tool.uv] pina
python-version=3.11 para garantir wheels (pyarrow, scikit-learn,
pandas). uv lock gera uv.lock versionado para reprodutibilidade
determinística de dependências. ruff e black excluem notebooks
e utils.py legado. .gitignore estendido para artifacts/ e mlruns/.
```

---

### Commit 2 — `feat(configs): adiciona configs/data.yaml com paths, seed e thresholds de QA`

**Arquivos criados:**
```
configs/data.yaml       (NOVO)
```

**Conteúdo de `configs/data.yaml`:**
```yaml
data:
  raw_path: "../data/default of credit card clients.xls"  # relativo à raiz do repo
  cleaned_path: "data/credit_card_cleaned.parquet"        # local, gitignored
  target_column: "default payment next month"
  id_column: "ID"
  expected_sha256: "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
  read_excel_header: 1   # row 0 é título; row 1 é o header real

split:
  seed: 42
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

qa:
  expected_rows: 30000
  expected_raw_cols: 25        # antes de dropar ID
  expected_clean_cols: 24      # após dropar ID
  minority_ratio_min: 0.20
  minority_ratio_max: 0.25
  education_valid_codes: [1, 2, 3, 4]   # UCI spec; {0,5,6} são anomalias documentadas
  marriage_valid_codes: [1, 2, 3]       # UCI spec; {0} é anomalia documentada

artifacts:
  base_dir: "artifacts"
  fingerprint_path: "artifacts/data_fingerprint.json"
  schema_path: "artifacts/data_schema.json"
  splits_dir: "artifacts/splits"
  split_indices_path: "artifacts/splits/split_indices.json"
  qa_summary_path: "artifacts/data_qa_summary.json"

reports:
  figures_dir: "reports/figures/parte_2"
```

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(configs): adiciona configs/data.yaml com paths, seed e thresholds de QA

Centraliza toda configuração de dados em um único arquivo YAML
para eliminar valores mágicos espalhados em scripts e notebooks.
O raw_path usa caminho relativo à raiz do repo (../data/…),
consistente com a convenção já usada nos notebooks. Os thresholds
de QA (education_valid_codes, marriage_valid_codes) documentam
explicitamente as anomalias do dataset UCI sem silenciosamente
recodificá-las.
```

---

### Commit 3 — `feat(data): cria src/credit_default/data/ingest.py como entrypoint canônico`

**Arquivos criados:**
```
src/__init__.py                         (NOVO — vazio)
src/credit_default/__init__.py          (NOVO — vazio)
src/credit_default/data/__init__.py     (NOVO — reexporta API pública)
src/credit_default/data/ingest.py       (NOVO)
```

> ⚠️ **Cascade para commits 4–8:** todos os módulos residem em
> `src/credit_default/data/`. Imports em scripts e testes:
> `from credit_default.data.<módulo> import ...`

**API pública de `src/credit_default/data/ingest.py`:**
```python
def load_raw(
    path: str | Path,
    *,
    header: int = 1,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Carrega o dataset bruto .xls, dropa a coluna ID e retorna
    DataFrame com 30000 linhas e 24 colunas.
    Raises: FileNotFoundError se path não existir.
    Raises: ValueError se id_col não estiver presente.
    """

def load_cleaned(path: str | Path) -> pd.DataFrame:
    """
    Carrega o parquet limpo produzido por build_clean_dataset.py.
    Raises: FileNotFoundError se parquet não existir.
    """

def load_config(config_path: str | Path | None = None) -> dict:
    """
    Carrega configs/data.yaml. Se config_path=None, resolve relativo ao repo root.
    """
```

**`src/credit_default/data/__init__.py`** reexporta: `load_raw`, `load_cleaned`, `load_config`.

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(data): cria src/credit_default/data/ingest.py como entrypoint canônico

Pacote nomeado credit_default para evitar colisão com o diretório
data/ local e com pacotes genéricos de terceiros. load_raw()
encapsula leitura do .xls (header=1), drop da coluna ID e
FileNotFoundError explícita. Notebooks existentes permanecem
inalterados nesta parte.
```

---

### Commit 4 — `feat(data): implementa fingerprint de dataset (sha256 + schema) para MLflow`

**Arquivos criados:**
```
src/credit_default/data/fingerprint.py  (NOVO)
```

**API pública de `src/credit_default/data/fingerprint.py`:**
```python
def compute_file_sha256(path: str | Path) -> str:
    """SHA-256 do arquivo em blocos de 64 KB. Retorna hex digest (64 chars)."""

def short_hash(hexdigest: str, n: int = 8) -> str:
    """Retorna primeiros n chars. Usado como DATAHASH8 no naming MLflow (Parte 3)."""

def compute_fingerprint(
    df: pd.DataFrame,
    *,
    file_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Retorna dict com:
    {
      "file_sha256": str,         # SHA-256 completo do arquivo (64 chars)
      "file_short": str,          # primeiros 8 chars (DATAHASH8)
      "n_rows": int,
      "n_cols": int,
      "columns": list[str],
      "dtypes": dict[str, str],
      "generated_at": str,        # ISO 8601 UTC
    }
    """

def save_fingerprint(fingerprint: dict[str, Any], path: str | Path) -> None:
    """Persiste como JSON com indent=2. Cria diretórios pai se necessário."""

def load_fingerprint(path: str | Path) -> dict[str, Any]:
    """Carrega fingerprint JSON previamente salvo."""
```

**`src/credit_default/data/__init__.py`** atualizado: reexporta `compute_fingerprint`, `short_hash`, `save_fingerprint`.

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(data): implementa fingerprint de dataset (sha256 + schema) para MLflow

O fingerprint garante rastreabilidade: qualquer alteração no
arquivo bruto invalida o hash e é detectada na próxima execução.
O campo file_short (8 chars) será usado como DATAHASH8 no naming
de MLflow runs na Parte 3. O cálculo do SHA-256 é feito em blocos
para ser seguro com arquivos de qualquer tamanho.
```

---

### Commit 5 — `feat(data): adiciona schema.py com expectativas e validate()`

**Arquivos criados:**
```
src/credit_default/data/schema.py       (NOVO)
```

**API pública de `src/credit_default/data/schema.py`:**
```python
class DataValidationError(Exception):
    """Levantada para erros fatais de validação."""

def validate(
    df: pd.DataFrame,
    *,
    expected_rows: int = 30000,
    expected_cols: int = 24,
    target_col: str = "default payment next month",
    education_valid_codes: list[int] | None = None,
    marriage_valid_codes: list[int] | None = None,
) -> list[str]:
    """
    Erros FATAIS → levanta DataValidationError imediatamente:
    - n_rows != expected_rows
    - n_cols != expected_cols
    - target_col ausente
    - qualquer NaN no DataFrame
    - linhas exatas duplicadas

    Warnings NÃO-FATAIS → retornados como list[str], DataFrame NÃO mutado:
    - EDUCATION contém códigos fora de education_valid_codes
    - MARRIAGE contém códigos fora de marriage_valid_codes
    """

def save_schema(
    df: pd.DataFrame,
    path: str | Path,
    *,
    warnings: list[str] | None = None,
) -> None:
    """
    Persiste JSON:
    {"columns": list, "dtypes": dict, "shape": [int, int],
     "warnings": list, "generated_at": str}
    """
```

**`src/credit_default/data/__init__.py`** atualizado: reexporta `validate`, `DataValidationError`, `save_schema`.

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(data): adiciona schema.py com expectativas e validate()

validate() implementa filosofia fail-fast: erros estruturais
(shape errado, target ausente, missing, duplicatas) levantam
DataValidationError imediatamente. Anomalias documentadas do UCI
(EDUCATION={0,5,6}, MARRIAGE={0}) geram warnings explícitos mas
não mutam o DataFrame, preservando os dados originais para
auditoria. Isso é documentado, não silenciado.
```

---

### Commit 6 — `feat(data): implementa split determinístico com artefato de índices reutilizável`

**Arquivos criados:**
```
src/credit_default/data/splits.py       (NOVO)
```

**API pública de `src/credit_default/data/splits.py`:**
```python
def make_splits(
    df: pd.DataFrame,
    target_col: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Retorna (train_idx, val_idx, test_idx) estratificados.
    Estratégia:
      1. Separa test (test_ratio) estratificado do total.
      2. Do restante, separa val (val_ratio/(1-test_ratio)) estratificado.
      3. O que sobra é train.
    Raises: ValueError se ratios não somarem 1.0 (tol 1e-9).
    """

def verify_splits(
    df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
    *,
    tolerance: float = 0.02,
) -> None:
    """Valida disjunção, cobertura total e ratios. Levanta AssertionError se falhar."""

def save_split_indices(
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
    *,
    seed: int,
    file_sha256: str,        # hash completo (64 chars) — para verificação do auditor
    fingerprint_short: str,  # 8 chars — para uso no run_name MLflow (Parte 3)
    path: str | Path,
) -> None:
    """
    Persiste JSON:
    {
      "seed": int,
      "file_sha256": str,          # 64 chars — hash completo para auditoria
      "fingerprint_short": str,    # 8 chars — DATAHASH8 para MLflow run names
      "n_train": int,
      "n_val": int,
      "n_test": int,
      "train_idx": list[int],
      "val_idx": list[int],
      "test_idx": list[int],
      "generated_at": str,
    }
    """
```

**`src/credit_default/data/__init__.py`** atualizado: reexporta `make_splits`, `verify_splits`, `save_split_indices`.

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(data): implementa split determinístico com artefato de índices reutilizável

Muda o split de 70/30 (notebooks) para 70/15/15 (train/val/test)
conforme aprovado na Parte 2. split_indices.json guarda tanto o
file_sha256 completo (64 chars, para verificação do auditor) quanto
o fingerprint_short (8 chars, para run_name MLflow na Parte 3).
verify_splits() garante disjunção e cobertura total antes de salvar.
```

---

### Commit 7 — `feat(scripts): adiciona build_clean_dataset.py para reconstruir dataset limpo`

**Arquivos criados:**
```
scripts/build_clean_dataset.py   (NOVO)
```

**Responsabilidades do script:**
1. Resolve `configs/data.yaml` relativo ao repo root via `__file__`
2. Verifica SHA-256 do .xls contra `expected_sha256` do config — falha se divergir
3. Chama `load_raw(raw_path, header=cfg["data"]["read_excel_header"])`
4. Chama `validate(df, ...)` — fail-fast; exibe warnings para EDUCATION/MARRIAGE
5. Cria diretório `data/` se não existir
6. Salva `data/credit_card_cleaned.parquet` com `engine="pyarrow"`, `index=False`
7. Imprime: path de saída, shape, SHA256 do arquivo bruto

**Imports do script:**
```python
from credit_default.data.ingest import load_raw, load_config
from credit_default.data.schema import validate
from credit_default.data.fingerprint import compute_file_sha256
```

**Interface CLI:**
```
python scripts/build_clean_dataset.py [--config PATH] [--raw-path PATH]
```

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(scripts): adiciona build_clean_dataset.py para reconstruir dataset limpo

Substitui o credit_card_cleaned.csv legacy (não rastreável) por
um parquet reproduzível gerado a partir do .xls bruto verificado.
Qualquer analista com acesso ao arquivo bruto original pode
recriar o dataset limpo executando um único comando, com
verificação automática de SHA-256 para garantir integridade.
```

---

### Commit 8 — `feat(data): implementa diagnostics.py com 7 checks de qualidade`

**Arquivos criados:**
```
src/credit_default/data/diagnostics.py  (NOVO)
```

**API pública de `src/credit_default/data/diagnostics.py`:**
```python
def check_missing(df: pd.DataFrame) -> dict[str, Any]:
    """Retorna {n_missing_total, n_cols_with_missing, details: {col: count}}."""

def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Retorna {n_duplicate_rows, duplicate_ratio}."""

def check_target_distribution(
    df: pd.DataFrame, target_col: str, *, figures_dir: Path
) -> dict[str, Any]:
    """
    Retorna {class_counts, class_ratios, minority_class, minority_ratio,
             imbalance_warning: bool}.
    Salva: figures_dir/target_distribution.png
    """

def check_outliers(
    df: pd.DataFrame, target_col: str, *, figures_dir: Path
) -> dict[str, Any]:
    """
    IQR por coluna numérica.
    Retorna {cols_with_outliers: {col: n_outliers}}.
    Salva: figures_dir/outlier_counts.png
    """

def check_correlations(
    df: pd.DataFrame, target_col: str, *, figures_dir: Path
) -> dict[str, Any]:
    """
    Pearson e Spearman com o target.
    Retorna {top_pearson: list, top_spearman: list}.
    Salva: figures_dir/correlation_heatmap.png
    """

def check_leakage_risk(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """
    APENAS DOCUMENTA — não corrige.
    Risco identificado: ausência de Pipeline/ColumnTransformer nos notebooks
    → scaler possivelmente fitado em train+test (leakage de normalização).
    Retorna {leakage_risk_detected: True, severity: "HIGH", details: str,
             action: "ver Parte 3"}.
    """

def check_bias_risk(
    df: pd.DataFrame, target_col: str, *, figures_dir: Path
) -> dict[str, Any]:
    """
    Distribuição do target por SEX, EDUCATION, MARRIAGE.
    Retorna {sensitive_features: {col: {value: default_rate}}}.
    Salva: figures_dir/bias_by_sensitive_feature.png
    """

def run_all_diagnostics(
    df: pd.DataFrame,
    target_col: str,
    *,
    figures_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Orquestra todos os checks. Cria figures_dir se não existir.
    Retorna dict completo para serialização JSON.
    """
```

**`src/credit_default/data/__init__.py`** atualizado: reexporta `run_all_diagnostics`.

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(data): implementa diagnostics.py com 7 checks de qualidade

Cobre: missing, duplicatas, distribuição do alvo, outliers (IQR),
correlações com o target, risco de leakage (ausência de Pipeline
nos notebooks — documentado para Parte 3), e risco de viés por
features sensíveis (SEX, EDUCATION, MARRIAGE). Nenhum valor
numérico é hardcoded; tudo é calculado a partir do DataFrame.
```

---

### Commit 9 — `feat(scripts): adiciona run_data_qa.py que orquestra fingerprint, schema, split e diagnósticos`

**Arquivos criados:**
```
scripts/run_data_qa.py       (NOVO)
```

**Imports do script:**
```python
from credit_default.data.ingest import load_raw, load_cleaned, load_config
from credit_default.data.fingerprint import compute_fingerprint, save_fingerprint
from credit_default.data.schema import validate, save_schema
from credit_default.data.splits import make_splits, verify_splits, save_split_indices
from credit_default.data.diagnostics import run_all_diagnostics
```

**Responsabilidades:**
1. Carrega config via `load_config()`
2. Carrega dataset via `load_cleaned()` (ou `load_raw()` se parquet não existir — avisa)
3. Computa e salva `artifacts/data_fingerprint.json`
4. Valida schema e salva `artifacts/data_schema.json`
5. Gera splits; chama `save_split_indices(..., file_sha256=fp["file_sha256"], fingerprint_short=fp["file_short"])`; salva `artifacts/splits/split_indices.json`
6. Cria `reports/figures/parte_2/` se não existir
7. Executa `run_all_diagnostics()` com figures_dir correto
8. Salva `artifacts/data_qa_summary.json` agregando todos os resultados
9. Imprime resumo formatado no stdout (zero hardcoded)

**Artifacts gerados (em runtime, gitignored):**
- `artifacts/data_fingerprint.json`
- `artifacts/data_schema.json`
- `artifacts/splits/split_indices.json`
- `artifacts/data_qa_summary.json`
- `reports/figures/parte_2/*.png` (≥4 figuras)

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
feat(scripts): adiciona run_data_qa.py que orquestra todo o QA de dados

Ponto de entrada único para reproduzir a auditoria completa de
dados: fingerprint, schema, splits determinísticos e diagnósticos
visuais. Todos os artefatos são gerados por código a partir do
dataset bruto — nenhum valor é hardcoded ou pré-computado.
```

---

### Commit 10 — `test(data): adiciona smoke tests da fundação de dados`

**Arquivos criados:**
```
tests/__init__.py             (NOVO — vazio)
tests/conftest.py             (NOVO)
tests/test_data_foundation.py (NOVO)
```

**`tests/conftest.py`:**
```python
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH_ENV = "RAW_DATA_PATH"
_DEFAULT_RAW = Path(__file__).resolve().parent.parent / ".." / "data" / \
               "default of credit card clients.xls"

@pytest.fixture(scope="session")
def raw_data_path() -> Path:
    p = Path(os.environ.get(RAW_PATH_ENV, str(_DEFAULT_RAW)))
    if not p.exists():
        pytest.skip(
            f"Dataset bruto não encontrado em {p}. "
            "Defina RAW_DATA_PATH ou coloque o arquivo no path padrão."
        )
    return p

@pytest.fixture(scope="session")
def loaded_df(raw_data_path: Path) -> pd.DataFrame:
    from credit_default.data.ingest import load_raw
    return load_raw(raw_data_path)

@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """
    DataFrame sintético com 500 linhas e distribuição 80/20 (não-default/default)
    para tornar o teste de estratificação significativo.
    Estrutura compatível com os módulos de Parte 2.
    """
    rng = np.random.default_rng(42)
    n = 500
    # 80/20: 400 classe 0, 100 classe 1
    target = np.array([0] * 400 + [1] * 100)
    rng.shuffle(target)
    return pd.DataFrame({
        "LIMIT_BAL":  rng.integers(10_000, 500_000, n),
        "SEX":        rng.integers(1, 3, n),
        "EDUCATION":  rng.integers(1, 5, n),
        "MARRIAGE":   rng.integers(1, 4, n),
        "AGE":        rng.integers(21, 79, n),
        **{f"PAY_{i}":    rng.integers(-2, 9, n) for i in range(6)},
        **{f"BILL_AMT{i}": rng.integers(0, 100_000, n) for i in range(1, 7)},
        **{f"PAY_AMT{i}":  rng.integers(0, 50_000, n) for i in range(1, 7)},
        "default payment next month": target,
    })
```

**`tests/test_data_foundation.py`** — 6 testes:

| Teste | Fixture usada | Usa raw file? |
|-------|--------------|---------------|
| `test_load_raw_shape_and_target` | `loaded_df` | Sim (skip declarado se ausente) |
| `test_fingerprint_stable_across_runs` | `minimal_df` + `tmp_path` | Não |
| `test_validate_passes_on_real_dataset` | `loaded_df` | Sim (skip declarado se ausente) |
| `test_validate_flags_constructed_bad_cases` | `minimal_df` | Não |
| `test_splits_sizes_and_stratification` | `minimal_df` | Não |
| `test_splits_indices_disjoint_and_cover_full_dataset` | `minimal_df` | Não |

**Asserts críticos por teste:**
```python
# test_load_raw_shape_and_target
assert df.shape == (30000, 24)
assert "ID" not in df.columns
assert "default payment next month" in df.columns

# test_fingerprint_stable_across_runs
assert fp1 == fp2                      # determinismo
assert len(fp1["file_short"]) == 8    # DATAHASH8

# test_validate_passes_on_real_dataset
warnings = validate(df, ...)
assert isinstance(warnings, list)     # nenhuma exceção; warnings permitidos

# test_validate_flags_constructed_bad_cases  (3 casos via @pytest.mark.parametrize)
# caso NaN, caso duplicata, caso sem target → cada um levanta DataValidationError

# test_splits_sizes_and_stratification
# minority_ratio no minimal_df = 0.20; verificar que cada split mantém ~0.20 ±0.04
assert abs(len(train_idx) / n - 0.70) < 0.03
assert abs(len(val_idx)   / n - 0.15) < 0.03
assert abs(len(test_idx)  / n - 0.15) < 0.03
# class ratio estratificado: |ratio_split - 0.20| < 0.04 para cada split

# test_splits_indices_disjoint_and_cover_full_dataset
assert set(train_idx) & set(val_idx)  == set()
assert set(train_idx) & set(test_idx) == set()
assert set(val_idx)   & set(test_idx) == set()
assert set(train_idx) | set(val_idx)  | set(test_idx) == set(df.index)
```

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
test(data): adiciona smoke tests da fundação de dados

Seis testes cobrem os contratos públicos dos módulos de Parte 2.
minimal_df usa distribuição 80/20 para tornar o teste de
estratificação significativo. Testes de lógica pura (fingerprint,
validate bad cases, splits) usam DataFrame sintético e sempre
rodam. Testes que requerem o .xls bruto usam pytest.skip explícito
com mensagem — skip declarado, não silencioso.
```

---

### Commit 11 — `docs(data): adiciona docs/data_foundation.md e docs/integrity_manifest.md`

**Arquivos criados:**
```
docs/data_foundation.md      (NOVO — PT-BR)
docs/integrity_manifest.md   (NOVO — PT-BR)
```

**`docs/data_foundation.md`** — seções:
- Visão geral do pacote `src/credit_default/data/`
- Descrição de cada módulo: ingest, fingerprint, schema, splits, diagnostics
- Como os artefatos são gerados e onde ficam
- Referências cruzadas para `artifacts/data_fingerprint.json` e `docs/integrity_manifest.md`

**`docs/integrity_manifest.md`** — seções:
- Controles de integridade da Parte 2 (tabela: artefato → script gerador → verificação)
- Nota sobre DATAHASH8 (uso futuro na Parte 3)
- Anomalias documentadas: EDUCATION={0,5,6}, MARRIAGE={0} (UCI out-of-spec, não recodificados)
- **"Divergência de split — não é inconsistência":** notebooks 02–06 mantêm 70/30 até Parte 3;
  o split 70/15/15 é adotado apenas pelo novo código `src/credit_default/data/splits.py`
- **"check_leakage_risk — apenas documentação":** a função registra o risco; correção na Parte 3

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
docs(data): adiciona data_foundation.md e integrity_manifest.md

Documentação em PT-BR dos módulos src/credit_default/data/ e dos
controles de integridade da Parte 2. O integrity_manifest esclarece
explicitamente: (1) split 70/30 nos notebooks não é bug — mudança
para 70/15/15 é deliberada e só vale para o novo código; (2)
check_leakage_risk documenta mas não corrige — correção é da Parte 3.
```

---

### Commit 12 — `docs(report): adiciona seção "Parte 2 — Fundação de Dados" ao relatório técnico`

**Arquivos criados/modificados:**
```
reports/relatorio_tecnico.md              (NOVO)
reports/figures/parte_2/.gitkeep          (NOVO — mantém diretório no git)
.gitignore                                (MODIFICADO — padrões específicos de PNG)
```

**Modificações em `.gitignore`** (append — padrões específicos, NÃO todo o diretório):
```
# Figuras geradas em runtime (parte_2); .gitkeep permanece versionado
reports/figures/parte_2/*.png
!reports/figures/parte_2/.gitkeep
# (Partes 3+ poderão versionar figuras finais se necessário)
```

**Conteúdo de `reports/relatorio_tecnico.md`** — seção "Parte 2" (PT-BR, sem números hardcoded):
- Objetivo da fundação de dados
- Dataset bruto: SHA-256 e shape → "ver `artifacts/data_fingerprint.json`"
- Anomalias documentadas → "ver `artifacts/data_schema.json` (campo warnings)"
- Split strategy: 70/15/15, seed=42, estratificado pelo target
- Diagnósticos: subseção por check → "ver `artifacts/data_qa_summary.json`"
- Risco de leakage: documentado; verificação real na Parte 3
- Conclusão: dataset aprovado para modelagem

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
docs(report): adiciona seção Parte 2 ao relatorio_tecnico.md

A seção documenta a fundação de dados em PT-BR com referências
explícitas aos artefatos gerados por código. .gitignore atualizado
com padrões específicos (reports/figures/parte_2/*.png) em vez
de ignorar todo o diretório reports/figures/, deixando espaço
para versionar figuras finais de Partes 3+ quando fizer sentido.
```

---

### Commit 13 — `docs(repo): atualiza README com comandos de setup e execução da Parte 2`

**Arquivos modificados:**
```
README.md       (APPEND — não reescrever)
```

**Seção appended (PT-BR):**
````markdown
## Como rodar a Parte 2 — Fundação de Dados

### Pré-requisitos
- **uv** (gestor de ambientes e pacotes Python)
  Verificar: `uv --version`
  Instalar: https://docs.astral.sh/uv/getting-started/installation/
  Python 3.11 é gerenciado pelo uv: `uv python install 3.11`
- Dataset bruto em `../data/default of credit card clients.xls`

### Instalação
```bash
uv venv --python 3.11
source .venv/Scripts/activate      # Git Bash (Windows)
# ou: .venv\Scripts\Activate.ps1  # PowerShell (Windows)
# ou: source .venv/bin/activate   # Linux/macOS
uv pip install -e ".[dev]"
```

### Construir dataset limpo
```bash
python scripts/build_clean_dataset.py
# Saída: data/credit_card_cleaned.parquet (shape esperado: 30000 x 24)
```

### Executar QA completo de dados
```bash
python scripts/run_data_qa.py
# Saídas: artifacts/, reports/figures/parte_2/
```

### Rodar testes
```bash
pytest -q
```

### Verificar estilo
```bash
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
```
````

**Rollback:** `git revert HEAD`

**Mensagem exata de commit:**
```
docs(repo): atualiza README com setup e execução da Parte 2

Adiciona seção "Como rodar a Parte 2" com uv como gestor de
ambientes. Não reescreve o README — apenas appenda a nova seção.
```

---

## 3. Plano de Execução (validação por grupo de commits)

### Pré-Commit 1 — Verificar uv e Python 3.11
```bash
uv --version
uv python list
# Se 3.11 não listado como instalado: uv python install 3.11
```

### Após Commit 1 (tooling)
```bash
uv venv --python 3.11
source .venv/Scripts/activate   # Git Bash / ou .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
python -c "import credit_default; print('package OK')"
ruff --version && black --version && pytest --version
uv lock --check
```

### Após Commits 2–6 (módulos src/credit_default/data/)
```bash
python -c "from credit_default.data.ingest import load_raw, load_config; print('ingest OK')"
python -c "from credit_default.data.fingerprint import compute_fingerprint, short_hash; print('fingerprint OK')"
python -c "from credit_default.data.schema import validate, DataValidationError; print('schema OK')"
python -c "from credit_default.data.splits import make_splits, verify_splits; print('splits OK')"
ruff check src/ && black --check src/
```

### Após Commit 7 (build_clean_dataset.py)
```bash
python scripts/build_clean_dataset.py
python -c "import pandas as pd; df = pd.read_parquet('data/credit_card_cleaned.parquet'); print(df.shape)"
# Esperado: (30000, 24)
```

### Após Commit 8 (diagnostics.py)
```bash
python -c "from credit_default.data.diagnostics import run_all_diagnostics; print('OK')"
ruff check src/credit_default/data/diagnostics.py
black --check src/credit_default/data/diagnostics.py
```

### Após Commit 9 (run_data_qa.py)
```bash
python scripts/run_data_qa.py
python -c "import json; fp=json.load(open('artifacts/data_fingerprint.json')); print(fp['file_sha256'][:8])"
# Esperado: 30c6be3a
ls artifacts/ && ls artifacts/splits/ && ls reports/figures/parte_2/
```

### Após Commit 10 (testes)
```bash
pytest -q --tb=short
# Esperado: 6 passed (ou ≤2 skipped com motivo declarado se .xls não encontrado via env)
```

### Validação Final (DoD completo)
```bash
# Venv limpo com uv
uv venv --python 3.11
source .venv/Scripts/activate   # Git Bash / ou .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
uv lock --check

python scripts/build_clean_dataset.py
python scripts/run_data_qa.py
pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/

# Verificar artefatos obrigatórios
test -f artifacts/data_fingerprint.json  && echo "fingerprint OK"
test -f artifacts/data_schema.json       && echo "schema OK"
test -f artifacts/splits/split_indices.json && echo "splits OK"
test -f artifacts/data_qa_summary.json   && echo "qa_summary OK"
test -f data/credit_card_cleaned.parquet && echo "parquet OK"
test -f uv.lock                          && echo "uv.lock OK"

# Verificar integridade do split_indices
python -c "
import json
si = json.load(open('artifacts/splits/split_indices.json'))
assert len(si['file_sha256']) == 64, 'SHA256 deve ter 64 chars'
assert len(si['fingerprint_short']) == 8, 'short deve ter 8 chars'
print('split_indices integridade OK')
"

# Verificar notebooks intocados (saída deve ser vazia)
git diff 516306a -- "*.ipynb" utils.py
# deve retornar saída vazia

# Contar commits novos (verificação visual)
git log 516306a..HEAD --oneline
# deve listar exatamente 13 linhas
```

---

## 4. Riscos e Mitigações

| # | Risco | Impacto | Mitigação |
|---|-------|---------|-----------|
| 1 | Espaço no nome `"default of credit card clients.xls"` | Alto | Usar `pathlib.Path()` em todo código Python; nunca concatenar strings de path |
| 2 | `xlrd >= 2.0.1` requerido para `.xls` | Alto | Declarar em `[project.dependencies]`; build script valida import antes de ler |
| 3 | uv ou Python 3.11 não instalado | **BLOQUEADOR** | Pré-requisito documentado; `uv python install 3.11` resolve sem python.org |
| 4 | Parquet engine — fastparquet vs pyarrow | Baixo | Fixar `engine="pyarrow"` explicitamente em todo `to_parquet`/`read_parquet` |
| 5 | Colisão de nome de pacote (resolvido) | — | Renomeado para `credit_default` — colisão eliminada |
| 6 | PNGs de Parte 2 indo ao git | Médio | Padrão específico `reports/figures/parte_2/*.png` no .gitignore; `!.gitkeep` |
| 7 | `header=1` errado produz colunas incorretas | Alto | Verificado via `test_load_raw_shape_and_target` (30000 linhas + colunas esperadas) |
| 8 | Split 70/15/15 diverge de notebooks 70/30 | Médio | Documentado no `integrity_manifest.md` como mudança deliberada |
| 9 | EDUCATION/MARRIAGE recodificados silenciosamente | Alto | `validate()` retorna warnings, nunca muta; `build_clean_dataset.py` preserva originais |
| 10 | split_indices.json sem hash completo (resolvido) | — | `save_split_indices()` inclui `file_sha256` (64) E `fingerprint_short` (8) |

---

## 5. Checklist — Definition of Done (Parte 2)

- [ ] `uv --version` retorna versão válida
- [ ] `uv python list` inclui 3.11.x instalado (não apenas 'download available')
- [ ] `uv pip install -e ".[dev]"` conclui sem erros em venv criado por `uv venv --python 3.11`
- [ ] `uv.lock` existe, está versionado, e `uv lock --check` passa
- [ ] `python scripts/build_clean_dataset.py` produz `data/credit_card_cleaned.parquet` com shape `(29965, 24)` (35 duplicatas removidas do raw)
- [ ] `python scripts/run_data_qa.py` produz os 4 artefatos JSON + figuras PNG
- [ ] `artifacts/data_fingerprint.json` → `file_sha256` começa com `30c6be3a`
- [ ] `artifacts/splits/split_indices.json` → contém `file_sha256` (64 chars) E `fingerprint_short` (8 chars)
- [ ] `artifacts/splits/split_indices.json` → `train_idx`, `val_idx`, `test_idx` disjuntos
- [ ] `pytest -q` verde (≥6 testes; skips apenas com motivo declarado)
- [ ] `ruff check src/ scripts/ tests/` sem erros
- [ ] `black --check src/ scripts/ tests/` sem erros
- [ ] `reports/relatorio_tecnico.md` tem seção "Parte 2" sem números hardcoded
- [ ] `README.md` tem seção "Como rodar a Parte 2" com comandos `uv`
- [ ] `git diff 516306a -- "*.ipynb" utils.py` retorna saída vazia (notebooks intocados)
- [ ] Nenhuma atribuição de ferramenta/vendor em qualquer documento
- [ ] `git log 516306a..HEAD --oneline` lista 15 commits (13 features/docs + 2 fix atômicos — ver integrity_manifest.md)

---

## 6. Checklist — Integridade

- [ ] SHA-256 do .xls verificado em `build_clean_dataset.py` contra `configs/data.yaml`
- [ ] `artifacts/data_fingerprint.json` gerado exclusivamente por código (nunca editado à mão)
- [ ] `artifacts/splits/split_indices.json` inclui `file_sha256` (64 chars) para auditoria completa
- [ ] `artifacts/splits/split_indices.json` inclui `fingerprint_short` (8 chars) para MLflow run name
- [ ] `validate()` levanta `DataValidationError` em 3 casos fatais distintos (testado)
- [ ] `validate()` retorna warnings sem mutar DataFrame (EDUCATION/MARRIAGE)
- [ ] `artifacts/` no `.gitignore` → `git check-ignore artifacts/` retorna `artifacts/`
- [ ] `mlruns/` no `.gitignore` → `git check-ignore mlruns/` retorna `mlruns/`
- [ ] `data/credit_card_cleaned.parquet` não vai ao git (`data/` gitignored)
- [ ] PNGs em `reports/figures/parte_2/` não vão ao git (padrão `*.png` gitignored)
- [ ] `.gitkeep` em `reports/figures/parte_2/` **vai** ao git (exceção `!.gitkeep`)
- [ ] Nenhum número em `relatorio_tecnico.md` sem referência a artefato computado
- [ ] `integrity_manifest.md` documenta: split 70/30 nos notebooks é deliberado (não bug)
- [ ] `integrity_manifest.md` documenta: `check_leakage_risk` é somente documentação
- [ ] `uv.lock` versionado e consistente com pyproject.toml (verificado por `uv lock --check`)

---

## Contrato de Execução

Quando o usuário disser **"aprovado, pode implementar"**:

1. **Verificar uv e Python 3.11:** `uv --version` + `uv python list` deve conter 3.11.
   Se uv ausente: instalar em https://docs.astral.sh/uv/getting-started/installation/ e parar.
   Se 3.11 ausente: `uv python install 3.11` — confirmar instalação antes de prosseguir.
2. Implementar commits 1 a 13 em ordem estrita.
3. Após cada commit: mostrar diff resumido, mensagem usada e output do comando de validação.
4. Parar e reportar se qualquer validação falhar — não avançar para o próximo commit.
5. Não modificar notebooks (`01`–`06`) ou `utils.py` em nenhum dos 13 commits.
6. Não gerar números ou métricas hardcoded em nenhum arquivo de documentação.
7. Usar `uv venv --python 3.11` + `uv pip install` em todos os ambientes.
