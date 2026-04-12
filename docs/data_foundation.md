# Fundação de Dados — Parte 2

Documentação do pacote `src/credit_default/data/` e do pipeline de auditoria de dados.

---

## Visão Geral

O pacote `credit_default.data` encapsula todas as operações sobre o dataset de inadimplência de cartão de crédito: ingestão, fingerprinting, validação de schema, splits determinísticos e diagnósticos de qualidade.

Princípios:
- **Fail-fast**: erros estruturais levantam exceção imediatamente; o pipeline não avança com dados corrompidos.
- **Rastreabilidade**: todo artefato gerado por código carrega metadados de auditoria (SHA-256, timestamp UTC, semente).
- **Sem números hardcoded**: nenhuma métrica é fixada no código fonte; tudo é calculado a partir dos dados.
- **Transparência**: anomalias documentadas (EDUCATION/MARRIAGE fora da especificação UCI) são registradas como warnings, nunca silenciadas ou recodificadas.

---

## Módulos

### `ingest.py` — Ingestão Canônica

| Função | Descrição |
|--------|-----------|
| `load_raw(path, *, header, id_col)` | Carrega `.xls` bruto, descarta coluna ID, retorna DataFrame 30000×24 |
| `load_cleaned(path)` | Carrega parquet limpo via pyarrow |
| `load_config(config_path)` | Carrega `configs/data.yaml`; resolve relativo ao repo root se `None` |

**Nota:** `load_raw()` encapsula `header=1` porque a linha 0 do arquivo Excel é um título descritivo e a linha 1 é o cabeçalho real das colunas.

---

### `fingerprint.py` — Rastreabilidade de Arquivo

| Função | Descrição |
|--------|-----------|
| `compute_file_sha256(path)` | SHA-256 do arquivo em blocos de 64 KB; retorna 64 chars |
| `short_hash(hexdigest, n=8)` | Primeiros `n` chars — usado como `DATAHASH8` no naming de runs MLflow (Parte 3) |
| `compute_fingerprint(df, *, file_path)` | Dict com `file_sha256`, `file_short`, `n_rows`, `n_cols`, `columns`, `dtypes`, `generated_at` |
| `save_fingerprint(fingerprint, path)` | Persiste como JSON com `indent=2` |
| `load_fingerprint(path)` | Carrega JSON previamente salvo |

**Artefato gerado:** `artifacts/data_fingerprint.json`

---

### `schema.py` — Validação de Schema

| Símbolo | Descrição |
|---------|-----------|
| `DataValidationError` | Exceção para erros fatais de validação |
| `validate(df, *, expected_rows, expected_cols, target_col, ...)` | Erros fatais (shape, target, NaN, duplicatas) → exceção; warnings (EDUCATION/MARRIAGE) → `list[str]` |
| `save_schema(df, path, *, warnings)` | Persiste `{columns, dtypes, shape, warnings, generated_at}` como JSON |

**Política de `expected_rows`:** aceita `int | None`. Quando `None`, a verificação de contagem de linhas é pulada — usado após deduplicação explícita no pipeline de build.

**Artefato gerado:** `artifacts/data_schema.json`

---

### `splits.py` — Split Determinístico

| Função | Descrição |
|--------|-----------|
| `make_splits(df, target_col, *, seed, train_ratio, val_ratio, test_ratio)` | Split estratificado 70/15/15 em dois passos |
| `verify_splits(df, train_idx, val_idx, test_idx, *, tolerance)` | Valida disjunção, cobertura total e ratios aproximados |
| `save_split_indices(train_idx, val_idx, test_idx, *, seed, file_sha256, fingerprint_short, path)` | Persiste índices com metadados de auditoria |

**Estratégia de dois passos:**
1. Separa `test` (15%) do total de forma estratificada.
2. Do restante (85%), separa `val` (15%/85% ≈ 17.6%) de forma estratificada.
3. O que sobra é `train` (~70%).

**Artefato gerado:** `artifacts/splits/split_indices.json`

Ver `docs/integrity_manifest.md` para nota sobre a divergência 70/30 (notebooks) × 70/15/15 (Parte 2).

---

### `diagnostics.py` — 7 Checks de Qualidade

| Função | Descrição | Figura gerada |
|--------|-----------|---------------|
| `check_missing(df)` | Valores ausentes por coluna | — |
| `check_duplicates(df)` | Contagem e ratio de linhas duplicadas | — |
| `check_target_distribution(df, target_col, *, figures_dir)` | Distribuição das classes | `target_distribution.png` |
| `check_outliers(df, target_col, *, figures_dir)` | Outliers por IQR em colunas numéricas | `outlier_counts.png` |
| `check_correlations(df, target_col, *, figures_dir)` | Pearson e Spearman com o target | `correlation_heatmap.png` |
| `check_leakage_risk(df, target_col)` | Documenta risco de leakage (apenas documentação) | — |
| `check_bias_risk(df, target_col, *, figures_dir)` | Default rate por SEX, EDUCATION, MARRIAGE | `bias_by_sensitive_feature.png` |
| `run_all_diagnostics(df, target_col, *, figures_dir, seed)` | Orquestra todos os checks | todos acima |

**Artefato gerado:** `artifacts/data_qa_summary.json` (via `scripts/run_data_qa.py`)

---

## Scripts

### `scripts/build_clean_dataset.py`

Reconstrói `data/credit_card_cleaned.parquet` a partir do `.xls` bruto verificado.

```
Fluxo:
  1. Carrega configs/data.yaml
  2. Verifica SHA-256 do .xls contra expected_sha256
  3. load_raw() → DataFrame 30000×24
  4. Remove 35 duplicatas exatas com logging explícito
  5. validate() — fail-fast; exibe warnings EDUCATION/MARRIAGE
  6. Salva parquet (engine=pyarrow, index=False)
```

```bash
python scripts/build_clean_dataset.py [--config PATH] [--raw-path PATH]
```

### `scripts/run_data_qa.py`

Auditoria completa de dados. Ponto de entrada único para reprodução.

```
Fluxo:
  1. Carrega parquet limpo (fallback: .xls bruto com aviso)
  2. Computa e salva fingerprint
  3. Valida schema e salva artifacts/data_schema.json
  4. Gera splits 70/15/15 e salva artifacts/splits/split_indices.json
  5. Executa run_all_diagnostics() → 4 figuras PNG
  6. Salva artifacts/data_qa_summary.json
  7. Imprime resumo formatado no stdout
```

```bash
python scripts/run_data_qa.py [--config PATH]
```

---

## Artefatos Gerados (em Runtime, Gitignored)

| Artefato | Script gerador | Conteúdo |
|----------|---------------|----------|
| `artifacts/data_fingerprint.json` | `run_data_qa.py` | SHA-256, DATAHASH8, shape, dtypes, timestamp |
| `artifacts/data_schema.json` | `run_data_qa.py` | columns, dtypes, shape, warnings |
| `artifacts/splits/split_indices.json` | `run_data_qa.py` | índices train/val/test, seed, SHA-256 completo |
| `artifacts/data_qa_summary.json` | `run_data_qa.py` | resultados agregados dos 7 checks |
| `data/credit_card_cleaned.parquet` | `build_clean_dataset.py` | dataset limpo, sem duplicatas |
| `reports/figures/parte_2/*.png` | `run_data_qa.py` | 4 figuras diagnósticas |

Para detalhes de integridade, ver `docs/integrity_manifest.md`.
