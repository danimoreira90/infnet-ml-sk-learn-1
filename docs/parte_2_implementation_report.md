# Relatório de Implementação — Parte 2: Fundação de Dados

---

## Sumário Executivo

Implementação completa da Parte 2 do projeto de inadimplência de cartão de crédito, com 15 commits sobre o HEAD `516306a`. O plano original previa 13 commits; dois commits de correção foram necessários durante a execução (detalhados abaixo). Todos os critérios do DoD foram atendidos.

**Resultado final:**
- 8 testes passando (0 falhas, 0 skips)
- ruff: sem erros | black: sem reformatações necessárias
- uv lock --check: OK
- Notebooks 01–06 e utils.py: intocados
- Todos os 4 artefatos JSON + 4 figuras PNG gerados com sucesso

---

## Histórico de Commits

| # | Hash | Tipo | Descrição | Arquivos | Linhas |
|---|------|------|-----------|----------|--------|
| 1 | `ae9a915` | chore | Tooling: pyproject, uv, ruff, black, pytest, .gitignore | 4 | +1159 |
| 1b | `61d929f` | fix | Corrige build-backend para setuptools.build_meta | 1 | +1/-1 |
| 2 | `9b6f688` | feat | configs/data.yaml com paths, seed e thresholds de QA | 1 | +33 |
| 3 | `e1bbe17` | feat | src/credit_default/data/ — stubs iniciais + ingest.py | 3 | +4/-1 |
| 3b | `f9004cb` | fix | Ancora /data/ /models/ /results/ ao root no .gitignore | 3 | +84/-3 |
| 4 | `cf66b16` | feat | fingerprint.py (SHA-256 + DATAHASH8 + save/load) | 2 | +105/-1 |
| 5 | `251e7b6` | feat | schema.py (DataValidationError + validate + save_schema) | 1 | +121 |
| 6 | `a8a6807` | feat | splits.py (make/verify/save_split_indices 70/15/15) | 1 | +139 |
| 7 | `c1e6d87` | feat | build_clean_dataset.py + fix validate() expected_rows→None | 2 | +97/-3 |
| 8 | `3130682` | feat | diagnostics.py (7 checks) + __init__.py consolidado | 2 | +327 |
| 9 | `d207e3c` | feat | run_data_qa.py (orquestrador) | 1 | +155 |
| 10 | `1e4b0c2` | test | Smoke tests: conftest.py + test_data_foundation.py (8 testes) | 3 | +193 |
| 11 | `0e31f72` | docs | data_foundation.md + integrity_manifest.md | 2 | +233 |
| 12 | `0d19627` | docs | relatorio_tecnico.md + .gitkeep + .gitignore PNG | 3 | +85 |
| 13 | `b376382` | docs | README.md — seção "Como rodar a Parte 2" | 1 | +49 |

**Total:** 2.524 linhas adicionadas | 9 linhas removidas

---

## Explicação dos Dois Commits de Correção

### Commit `61d929f` — `build-backend` inválido

**Causa:** Commit 1 usou `setuptools.backends.legacy:build` como build-backend, que não existe em versões modernas do setuptools.

**Sintoma:** `uv pip install -e ".[dev]"` falhava com `ModuleNotFoundError: No module named 'setuptools.backends'`.

**Correção:** Uma linha em `pyproject.toml`:
```diff
-build-backend = "setuptools.backends.legacy:build"
+build-backend = "setuptools.build_meta"
```

### Commit `f9004cb` — `.gitignore` bloqueando o pacote Python

**Causa:** O `.gitignore` original usava `data/` (sem âncora), que o git interpretava como "qualquer diretório chamado `data/` em qualquer nível", bloqueando `src/credit_default/data/` ao tentar adicionar os arquivos.

**Sintoma:** `git add src/credit_default/data/ingest.py` resultava em "The following paths are ignored by one of your .gitignore files: src/credit_default/data/".

**Correção:** Ancoragem ao root do repositório:
```diff
-data/
-models/
-results/
+/data/
+/models/
+/results/
```

---

## Decisões Técnicas Relevantes

### 1. Deduplicação explícita em `build_clean_dataset.py`

O dataset UCI bruto contém **35 linhas exatamente duplicadas** — não documentado na especificação original. A política adotada foi transparência total: o script remove as duplicatas com log visível (`30000 → 29965 linhas`) antes de chamar `validate()`. Isso também exigiu tornar `expected_rows` opcional (`int | None`) em `validate()`, já que o número de linhas depois da dedup difere do total bruto.

### 2. `.gitignore` ancorado ao root

Padrões sem prefixo `/` no `.gitignore` correspondem a qualquer diretório na árvore inteira. Usar `/data/` em vez de `data/` é necessário para ignorar apenas o diretório de dados na raiz sem afetar o pacote Python `src/credit_default/data/`.

### 3. `expected_rows: int | None` em `validate()`

A função `validate()` recebe `expected_rows=None` quando o pipeline não conhece a contagem exata de antemão (pós-dedup). Isso mantém a API flexível para reutilização futura sem quebrar o contrato de tipo.

### 4. `cleaned_df` fixture separada de `loaded_df`

O `loaded_df` carrega os dados brutos via `load_raw()` — com as 35 duplicatas. `validate()` rejeita duplicatas como erro fatal. A fixture `cleaned_df` foi adicionada ao `conftest.py` para refletir o fluxo real do pipeline (load → deduplicate → validate).

### 5. `matplotlib.use("Agg")`

Backend não-interativo declarado no topo de `diagnostics.py` para garantir que a geração de figuras funcione em ambientes sem display (CI, servidores), evitando `_tkinter.TclError` no Windows e erros similares em Linux headless.

### 6. `[tool.uv] python-preference = "managed"` + `.python-version`

O campo `python-version` em `[tool.uv]` não é reconhecido pelo uv (reporta `Unknown field`). A solução correta é `python-preference = "managed"` no `[tool.uv]` combinado com um arquivo `.python-version` contendo `3.11`, que o uv respeita ao criar o venv.

---

## Diffs Detalhados por Arquivo

### `pyproject.toml` (commits 1 + 1b)

```diff
+[build-system]
+requires = ["setuptools>=68"]
+build-backend = "setuptools.build_meta"
+
+[project]
+name = "infnet-sk-learn-models"
+version = "0.2.0"
+requires-python = ">=3.10,<3.13"
+dependencies = [
+    "pandas>=2.0", "numpy>=1.24", "scikit-learn>=1.3",
+    "xlrd>=2.0.1", "pyarrow>=14.0", "pyyaml>=6.0",
+    "matplotlib>=3.7", "seaborn>=0.13",
+]
+
+[tool.setuptools.package-dir]
+"" = "src"
+
+[tool.ruff]
+line-length = 99
+exclude = ["*.ipynb", "utils.py"]
+
+[tool.ruff.lint]
+select = ["E", "F", "W", "I"]
+
+[tool.black]
+line-length = 99
+target-version = ["py311"]
+
+[tool.pytest.ini_options]
+testpaths = ["tests"]
+
+[tool.uv]
+python-preference = "managed"
```

### `.gitignore` (commit 1 + 3b + 12)

**Commit 1 — padrões de dados (com bug):**
```diff
-data/
-models/
-results/
+data/
+models/
+results/
+artifacts/
+mlruns/
```

**Commit 3b — correção de ancoragem:**
```diff
-data/
-models/
-results/
+/data/
+/models/
+/results/
```

**Commit 12 — PNG específicos:**
```diff
+reports/figures/parte_2/*.png
+!reports/figures/parte_2/.gitkeep
```

### `src/credit_default/data/schema.py` — `expected_rows: int | None` (commit 7)

```diff
 def validate(
     df: pd.DataFrame,
     *,
-    expected_rows: int = 30000,
+    expected_rows: int | None = 30000,
     expected_cols: int = 24,
 ...
-    if df.shape[0] != expected_rows:
+    if expected_rows is not None and df.shape[0] != expected_rows:
```

### `scripts/build_clean_dataset.py` — deduplicação explícita (commit 7)

```diff
+    # Remover duplicatas exatas antes da validação
+    n_before = len(df)
+    df = df.drop_duplicates()
+    n_dropped = n_before - len(df)
+    if n_dropped > 0:
+        print(
+            f"[build] AVISO: {n_dropped} linha(s) exatamente duplicada(s) removida(s) "
+            f"({n_before} -> {len(df)} linhas). Fato registrado; dataset ainda valido."
+        )
+
     # Validar (expected_rows=None pois contagem pode diferir após dedup)
-    warnings = validate(df, expected_rows=qa_cfg["expected_rows"], ...)
+    warnings = validate(df, expected_rows=None, ...)
```

---

## Verificações DoD — Estado Final

| Critério | Status |
|----------|--------|
| `uv --version` retorna versão válida | ✅ 0.10.9 |
| `uv python list` inclui 3.11.x instalado | ✅ 3.11.15 |
| `uv pip install -e ".[dev]"` sem erros | ✅ |
| `uv.lock` existe e `uv lock --check` passa | ✅ 45 pacotes |
| `build_clean_dataset.py` produz parquet | ✅ (29965, 24) |
| `run_data_qa.py` produz 4 artefatos JSON + 4 figuras | ✅ |
| `file_sha256` começa com `30c6be3a` | ✅ |
| `split_indices.json` tem `file_sha256` (64 chars) | ✅ |
| `split_indices.json` tem `fingerprint_short` (8 chars) | ✅ |
| `pytest -q` verde | ✅ 8 passed |
| `ruff check` sem erros | ✅ |
| `black --check` sem reformatações | ✅ |
| `relatorio_tecnico.md` sem números hardcoded | ✅ |
| README com seção "Como rodar a Parte 2" | ✅ |
| Notebooks intocados (`git diff 516306a -- "*.ipynb"` vazio) | ✅ |
| Nenhuma atribuição de ferramenta/vendor | ✅ |
| `git log 516306a..HEAD` lista commits Parte 2 | ✅ 15 (13 + 2 fixes) |

---

## Arquivos Criados/Modificados

### Novos arquivos versionados (13 arquivos)

```
pyproject.toml
.python-version
uv.lock
configs/data.yaml
src/__init__.py
src/credit_default/__init__.py
src/credit_default/data/__init__.py
src/credit_default/data/ingest.py
src/credit_default/data/fingerprint.py
src/credit_default/data/schema.py
src/credit_default/data/splits.py
src/credit_default/data/diagnostics.py
scripts/build_clean_dataset.py
scripts/run_data_qa.py
tests/__init__.py
tests/conftest.py
tests/test_data_foundation.py
docs/data_foundation.md
docs/integrity_manifest.md
reports/relatorio_tecnico.md
reports/figures/parte_2/.gitkeep
```

### Arquivos modificados

```
.gitignore    — /data/, /models/, /results/ ancorados; artifacts/; mlruns/; PNGs parte_2
README.md     — seção "Como rodar a Parte 2" appendada
```

### Arquivos gerados em runtime (gitignored)

```
data/credit_card_cleaned.parquet          — (29965, 24)
artifacts/data_fingerprint.json           — SHA-256 + DATAHASH8
artifacts/data_schema.json                — columns, dtypes, warnings
artifacts/splits/split_indices.json       — 20975/4495/4495 com hashes
artifacts/data_qa_summary.json            — resultados dos 7 checks
reports/figures/parte_2/target_distribution.png
reports/figures/parte_2/outlier_counts.png
reports/figures/parte_2/correlation_heatmap.png
reports/figures/parte_2/bias_by_sensitive_feature.png
```
