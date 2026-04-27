# Operacionalização — Parte 6

Projeto: Previsão de Inadimplência em Cartão de Crédito
Modelo: GradientBoosting (baseline) | Run: `6be94912218a4c51bd8297ac77719b7f`

---

## Arquitetura do Serviço

```
┌─────────────────────────────────────────────────────────────┐
│                    Cliente (curl / Postman)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI — credit_default.serving.app           │
│                                                             │
│  GET  /health          → status + model_uri                 │
│  GET  /                → info do modelo                     │
│  POST /predict         → predicao unitaria                  │
│  POST /predict/batch   → predicao em lote                   │
│                                                             │
│  Pydantic (strict=True) — validacao de 23 features          │
│  CORS: localhost permitido                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Predictor (lazy load, fail-fast)              │
│                                                             │
│  LOCAL:   MODEL_URI = models:/m-4de1a2c47e7d40d9a679a40ba79c9c65
│           TRACKING_URI = file:///abs/path/to/mlruns         │
│                                                             │
│  DOCKER:  MODEL_URI = /app/mlruns/.../artifacts (POSIX)     │
│           (sem set_tracking_uri — path direto)              │
└──────────────────────────┬──────────────────────────────────┘
                           │ mlflow.sklearn.load_model
                           ▼
┌─────────────────────────────────────────────────────────────┐
│   mlruns/236665223173386020/models/m-4de1a2c47e7d40d9a679a40ba79c9c65/artifacts/
│   ├── MLmodel            (metadados e flavor sklearn)       │
│   ├── model.pkl          (GradientBoostingClassifier)       │
│   └── serving_input_example.json (schema de entrada)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Schemas

### Request — POST /predict

```json
{
  "record": {
    "LIMIT_BAL": 30000,
    "SEX": 2,
    "EDUCATION": 1,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": -1,
    "PAY_2": -1,
    "PAY_3": -1,
    "PAY_4": -2,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 390,
    "BILL_AMT2": 780,
    "BILL_AMT3": 0,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 780,
    "PAY_AMT2": 0,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
  }
}
```

Todos os campos são `int` (derivados da MLmodel signature — type: long).
Todos os campos são obrigatórios. Payload incompleto retorna HTTP 422.

### Response — POST /predict

```json
{
  "prediction": 0,
  "probability_default": 0.2765047603683503,
  "probability_no_default": 0.7234952396316496
}
```

### Request — POST /predict/batch

```json
{
  "records": [
    { "LIMIT_BAL": 30000, "SEX": 2, ... },
    { "LIMIT_BAL": 100000, "SEX": 1, ... }
  ]
}
```

### Response — POST /predict/batch

```json
{
  "predictions": [
    { "prediction": 0, "probability_default": 0.2765, "probability_no_default": 0.7235 },
    { "prediction": 0, "probability_default": 0.1832, "probability_no_default": 0.8168 }
  ]
}
```

---

## Endpoints

| Método | Path | Descrição | Status de Erro |
|--------|------|-----------|---------------|
| GET | `/health` | Status do serviço e model_uri | — |
| GET | `/` | Info do modelo (nome, métricas, run_id) | — |
| POST | `/predict` | Predição unitária | 422 payload inválido, 503 modelo não carregado, 500 erro interno |
| POST | `/predict/batch` | Predição em lote | 422 lista vazia ou payload inválido, 503/500 idem |

---

## Exemplos de curl

### Health check

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Resposta esperada:
```json
{"status": "ok", "model_uri": "models:/m-4de1a2c47e7d40d9a679a40ba79c9c65"}
```

### Informações do modelo

```bash
curl -s http://localhost:8000/ | python -m json.tool
```

### Predição unitária

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "record": {
      "LIMIT_BAL":30000,"SEX":2,"EDUCATION":1,"MARRIAGE":1,"AGE":35,
      "PAY_0":-1,"PAY_2":-1,"PAY_3":-1,"PAY_4":-2,"PAY_5":-2,"PAY_6":-2,
      "BILL_AMT1":390,"BILL_AMT2":780,"BILL_AMT3":0,"BILL_AMT4":0,
      "BILL_AMT5":0,"BILL_AMT6":0,
      "PAY_AMT1":780,"PAY_AMT2":0,"PAY_AMT3":0,"PAY_AMT4":0,
      "PAY_AMT5":0,"PAY_AMT6":0
    }
  }' | python -m json.tool
```

### Payload inválido (deve retornar 422)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"record": {"LIMIT_BAL": "invalido"}}' | python -m json.tool
```

---

## Execução Local (uvicorn)

```bash
# Instalar dependências
uv sync

# Subir o servidor
uv run uvicorn src.credit_default.serving.app:app --port 8000

# Variáveis de ambiente opcionais (defaults já configurados)
# MODEL_URI=models:/m-4de1a2c47e7d40d9a679a40ba79c9c65
# MLFLOW_TRACKING_URI=file:///abs/path/mlruns
```

O servidor inicializa em ~3s (carregamento do modelo GradientBoosting via MLflow).

---

## Execução via Docker

```bash
# Build da imagem
docker build -t infnet-ml-api .

# Subir via docker-compose
docker compose up -d

# Verificar status
docker compose ps

# Testar (PowerShell)
Invoke-RestMethod -Uri http://localhost:8000/health

# Logs
docker compose logs --tail=20

# Parar
docker compose down
```

A variável `MODEL_URI` no `docker-compose.yml` aponta para o path POSIX absoluto
dentro do container, contornando o problema de paths Windows nos metadados do MLmodel.

---

## Monitoramento de Drift

### Gerar relatório de drift

```bash
uv run python scripts/run_drift_report.py
```

Saída: `reports/parte_6/drift_report.md` + MLflow run com `stage=drift_report`.

### Interpretação dos resultados

| Situação | Ação recomendada |
|----------|-----------------|
| 0 features com drift | Sistema estável, monitoramento padrão |
| 1–4 features com drift | Alertar equipe, aumentar frequência de monitoramento |
| ≥ 5 features com drift | Iniciar processo de retreinamento |
| roc_auc cai > 5pp | Retreinamento obrigatório |

---

## Plano de Monitoramento e Retreinamento

Ver `reports/parte_6/drift_report.md` para o plano completo com:
- Critérios de gatilho por feature e por métrica
- Frequência mínima de verificação
- Estimativa de custo por retreinamento (~25–40 min, sem GPU)

---

## Controles de Integridade

| Controle | Implementação |
|----------|--------------|
| MODEL_URI imutável | Constante em `predictor.py`; mudança requer commit explícito |
| Fail-fast no startup | `lifespan()` chama `predictor.load()` — RuntimeError impede startup |
| Sem stub no /predict | Se predictor não pronto, retorna 503 (não 200 com valor fixo) |
| Test set tocado 1x | `evaluate_final.py` é o único script com `include_test=True` |
| CI valida testes | 117 testes no GitHub Actions em todo push/PR |
