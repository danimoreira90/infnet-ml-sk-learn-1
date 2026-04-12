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
