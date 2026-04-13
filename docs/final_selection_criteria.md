# Criterio de Selecao do Modelo Final — Parte 5

> Este documento foi escrito **antes** de qualquer avaliacao no conjunto de
> teste (`test_idx`). Seu objetivo e blindar a escolha do modelo final contra
> vies de selecao post-hoc.

## Universo de candidatos

Os 25 runs do experimento `infnet-ml-sistema`:
- 5 runs baseline (Parte 3, sem tuning, sem reducao de dimensionalidade)
- 5 runs tune (Parte 3, com tuning, sem reducao de dimensionalidade)
- 15 runs dimred (Parte 4, sem tuning, com PCA ou LDA)

## Criterio — regra de desempate em cascata

O modelo final e o run que satisfaz **todos** os requisitos e vence nos
criterios abaixo, avaliados estritamente na ordem indicada.

### Requisitos obrigatorios

1. `predict_proba` nativo (sem calibracao post-hoc adicional).
2. Modelo deve ser serializavel via `joblib` ou `mlflow.sklearn.log_model`.
3. `training_time_s` finito e registrado.

### Metricas de decisao (ordem estrita)

1. **Maior `roc_auc`** (validacao), metrica primaria do projeto.
2. **Empate** (ate a 4a casa decimal): menor `cv_roc_auc_std` — estabilidade
   em validacao cruzada.
3. **Empate persistente**: menor `inference_latency_ms` — viabilidade
   operacional.
4. **Empate persistente**: menor `training_time_s` — custo de retreino.
5. **Empate persistente**: menor complexidade nominal, nesta ordem de
   preferencia: linear < arvore unica < ensemble.

## Avaliacao no conjunto de teste

Apos a selecao, o modelo vencedor e **retreinado do zero** sobre `X_train`
combinado com `X_val` (concatenados), usando os mesmos hiperparametros do
run vencedor. O modelo retreinado e avaliado **uma unica vez** em `X_test`
(`test_idx` de `artifacts/splits/split_indices.json`).

A metrica de teste e **reportada como esta**, sem ajuste posterior. Nao ha
"revisao do criterio" apos ver a metrica de teste.

## Auditoria

- Este arquivo e versionado no repositorio antes do script
  `scripts/select_final_candidate.py` ser implementado.
- O timestamp do commit que cria este arquivo deve **preceder** o timestamp
  de qualquer commit que toque `test_idx`.
- Qualquer alteracao neste criterio apos a fase de selecao requer commit
  explicito com justificativa no corpo do commit.
