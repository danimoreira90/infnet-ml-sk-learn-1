# Roteiro de Vídeo — Projeto INFNET: sklearn Models
**Duração estimada:** 5–7 minutos | **Idioma:** Português

---

## Estrutura Geral

| Bloco | Duração | Tema |
|-------|---------|------|
| 1 | 30s | Apresentação |
| 2 | 1min | Estrutura do repositório + arquitetura |
| 3 | 1min | MLflow UI: 25 runs + final_eval |
| 4 | 1min | Seleção final + critério escrito antes |
| 5 | 1min | Inferência: uvicorn + PowerShell /predict |
| 6 | 45s | Drift report executado |
| 7 | 45s | GitHub Actions |
| 8 | 30s | Encerramento |

---

## Bloco 1 — Apresentação (30s)

**Tela:** IDE ou terminal com o projeto aberto.

**Narração:**
> "Olá! Neste vídeo apresento o projeto final da disciplina de Machine Learning
> Aplicado do INFNET. O objetivo foi construir um sistema completo de previsão
> de inadimplência em cartão de crédito, cobrindo desde a exploração dos dados
> até a operacionalização do modelo em produção."

**Pontos a narrar:**
- Dataset: UCI Taiwan 2005, 30.000 clientes, 23 features
- Problema: classificação binária (inadimplência no próximo mês)
- 6 partes: EDA → Baseline → Tuning → Dim. Reduction → Seleção Final → Deploy

---

## Bloco 2 — Estrutura do Repositório + Arquitetura (1min)

**Tela:** Terminal no diretório raiz.

**Comandos a digitar:**
```powershell
# Mostrar estrutura de alto nível
ls

# Mostrar pacote principal
ls src/credit_default/

# Mostrar que há 117 testes passando
uv run pytest -q --tb=no 2>&1 | Select-Object -Last 3
```

**Narração:**
> "O repositório segue uma estrutura de pacote Python com submódulos por
> responsabilidade: data, features, models, tracking, audit, serving e monitoring.
> A Parte 6 adicionou o módulo serving, com o FastAPI, e o módulo monitoring,
> com detecção de drift estatístico."

**Pontos a narrar:**
- `src/credit_default/serving/` → API de produção
- `src/credit_default/monitoring/` → detecção de drift
- `scripts/` → scripts executáveis de cada fase
- 117 testes cobrindo todo o pipeline

---

## Bloco 3 — MLflow UI: 25 runs + final_eval (1min)

**Tela:** Browser com MLflow UI.

**Comandos a digitar (antes de abrir o browser):**
```powershell
uv run mlflow ui --port 5000
# Abrir: http://localhost:5000
```

**O que mostrar:**
1. Experimento `credit-default-prediction` → 25+ runs visíveis
2. Ordenar por `roc_auc` decrescente
3. Apontar o run `final_eval` (run_id: `6be94912...`)
4. Clicar no run → mostrar métricas: roc_auc=0.7682, f1_macro=0.6876
5. Mostrar a aba "Artifacts" → `model.pkl`, `MLmodel`, `serving_input_example.json`

**Narração:**
> "Todas as 25 combinações de modelos e hiperparâmetros foram rastreadas no
> MLflow. O run final_eval corresponde ao modelo retreinado em treino + validação
> e avaliado uma única vez no test set — essa é a nossa estimativa de
> generalização real."

---

## Bloco 4 — Seleção Final + Critério Escrito Antes (1min)

**Tela:** Terminal PowerShell.

**Comandos a digitar:**
```powershell
# Mostrar o critério escrito ANTES da execução
Get-Content docs/final_selection_criteria.md

# Mostrar o relatório final com o vencedor identificado
Get-Content reports/parte_5/final_selection.md | Select-Object -First 60
```

**Narração:**
> "Um ponto crítico do projeto foi garantir que o critério de seleção foi
> definido ANTES de ver os resultados — evitando data snooping. O arquivo
> final_selection_criteria.md tem o timestamp de commit anterior à execução.
> O vencedor foi o GradientBoosting baseline, selecionado por roc_auc superior
> com menor variância entre as folds."

**Pontos a narrar:**
- Critério em cascata: roc_auc → f1_macro → complexidade
- Test set tocado exatamente 1 vez (evaluate_final.py)
- run_id imutável: `6be94912218a4c51bd8297ac77719b7f`

---

## Bloco 5 — Inferência: uvicorn + PowerShell /predict (1min)

**Tela:** Dois terminais PowerShell lado a lado (split screen).

**Opção de servidor:** Para o vídeo, recomenda-se **uvicorn local** (startup visível
ao vivo, ~3s). Alternativa: `docker compose up -d` (container já pronto, sem log
de startup na tela — útil se o modelo local não estiver configurado).

**Terminal 1 — subir o servidor (uvicorn local):**
```powershell
uv run uvicorn credit_default.serving.app:app --port 8000
# Aguardar: "Application startup complete."
```

**Terminal 2 — testar endpoints:**
```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8000/health | ConvertTo-Json

# Informações do modelo
Invoke-RestMethod -Uri http://localhost:8000/ | ConvertTo-Json

# Predição: cliente com LIMIT_BAL=30k, sem atrasos (deve retornar prediction=0)
$body = @{
  record = @{
    LIMIT_BAL=30000; SEX=2; EDUCATION=1; MARRIAGE=1; AGE=35
    PAY_0=-1; PAY_2=-1; PAY_3=-1; PAY_4=-2; PAY_5=-2; PAY_6=-2
    BILL_AMT1=390; BILL_AMT2=780; BILL_AMT3=0; BILL_AMT4=0; BILL_AMT5=0; BILL_AMT6=0
    PAY_AMT1=780; PAY_AMT2=0; PAY_AMT3=0; PAY_AMT4=0; PAY_AMT5=0; PAY_AMT6=0
  }
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST `
  -ContentType "application/json" -Body $body | ConvertTo-Json

# Payload inválido → deve retornar erro 422
try {
  Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST `
    -ContentType "application/json" `
    -Body '{"record":{"LIMIT_BAL":"errado"}}'
} catch {
  Write-Host "Status:" $_.Exception.Response.StatusCode.value__
  $_.ErrorDetails.Message
}
```

**Narração:**
> "O serviço FastAPI carrega o modelo via URI canônica do MLflow na inicialização.
> O /predict retorna a classe e as duas probabilidades. Note que um payload
> inválido gera um erro 422 descritivo — validação feita pelo Pydantic com os
> 23 campos tipados."

**Resultado esperado:**
- `/health`: `{"status": "ok", "model_uri": "models:/m-4de1a2c47e7d40d9a679a40ba79c9c65"}`
- `/predict`: `{"prediction": 0, "probability_default": 0.2765, "probability_no_default": 0.7235}`
- payload inválido: HTTP 422 com 23 erros de validação

---

## Bloco 6 — Drift Report (45s)

**Tela:** Terminal PowerShell.

**Comandos a digitar:**
```powershell
uv run python scripts/run_drift_report.py

# Mostrar o relatório gerado
Get-Content reports/parte_6/drift_report.md
```

**Narração:**
> "O módulo de monitoring usa Kolmogorov-Smirnov para features contínuas e
> qui-quadrado para categóricas. Nesta simulação, train+val são os dados
> históricos e o test set representa dados novos de produção. Como os splits
> vêm do mesmo dataset estratificado, nenhum drift é detectado — o que
> confirma a homogeneidade dos dados. Em produção real, aplicaríamos o mesmo
> teste contra dados novos coletados periodicamente."

**Pontos a narrar:**
- alpha=0.05 configurável
- Plano de retreinamento documentado no relatório
- MLflow run criado com tag `stage=drift_report`

---

## Bloco 7 — GitHub Actions (45s)

**Tela:** Browser com GitHub Actions na aba Actions do repositório.

**O que mostrar:**
1. Aba "Actions" do repositório
2. Workflow `CI` → último run verde (confirmar que é o commit mais recente gravado)
3. Expandir job `lint` → mostrar ruff + black passando
4. Expandir job `test` → mostrar 117 testes passando
5. Mostrar o arquivo `.github/workflows/ci.yml`

**Comandos para mostrar o arquivo localmente:**
```powershell
Get-Content .github/workflows/ci.yml
```

**Narração:**
> "A CI roda automaticamente em push e pull request para main. O job de lint
> valida ruff e black. O job de test executa os 117 testes — incluindo os testes
> do módulo serving com mocks, que não precisam do mlruns local. O modelo final
> foi versionado no git via exceção no .gitignore, tornando o Docker build e
> o CI completamente reproduzíveis."

**Ponto de narrativa — CI vermelho histórico (transformar em demonstração):**
> "Vê-se aqui que o primeiro run do CI mostrou erro: o ruff não estava
> pinned nas dependências de dev. Esse foi exatamente o tipo de feedback
> que justifica ter CI: o erro foi pego em segundos, corrigido no
> commit seguinte, e o estado atual está verde."

**Nota:** Antes de gravar, confirmar que o último commit da branch `main` já
passou no CI — o check verde deve aparecer ao lado do commit no GitHub.

---

## Bloco 8 — Encerramento (30s)

**Tela:** Slide ou terminal com o repositório.

**Narração:**
> "O projeto demonstrou um pipeline completo de ML: da análise exploratória,
> passando pelo rastreamento de experimentos no MLflow, até a API de produção
> com FastAPI, containerização com Docker e CI/CD com GitHub Actions.
> O código está disponível no repositório, com testes e documentação em
> português. Obrigado!"

---

## Tomadas Problemáticas — Como Evitar

| Problema | Como Evitar |
|----------|-------------|
| Servidor não sobe (model not found) | Verificar `MODEL_URI` e `MLFLOW_TRACKING_URI`; confirmar que `mlruns/236.../models/m-4de.../` existe |
| `Invoke-RestMethod` retorna objeto sem formatação | Encadear `\| ConvertTo-Json` para visualização legível |
| MLflow UI não abre | `uv run mlflow ui --backend-store-uri ./mlruns --port 5000` |
| pytest muito lento no vídeo | Usar `uv run pytest -q --tb=no -x` para saída rápida |
| Terminal com encoding errado (caracteres especiais) | `$env:PYTHONIOENCODING="utf-8"` antes de rodar scripts |
| Docker build muito lento ao vivo | Mostrar um build já cacheado ou usar `docker compose up -d` direto |
| GitHub Actions ainda rodando ao gravar | Fazer push antes de gravar e aguardar o green check no commit |
| `ConvertTo-Json -Compress` quebrando no body do POST | Verificar se `$body` foi atribuído antes do `Invoke-RestMethod` |

---

## Checklist Pré-Gravação

- [ ] `uv run pytest -q` → 117 testes passando
- [ ] `uv run uvicorn credit_default.serving.app:app --port 8000` → sobe sem erro
- [ ] `Invoke-RestMethod -Uri http://localhost:8000/health` → responde com status ok
- [ ] `uv run mlflow ui --port 5000` → UI carrega com 25+ runs
- [ ] `uv run python scripts/run_drift_report.py` → termina sem erro
- [ ] GitHub Actions → verde no commit mais recente antes de gravar
- [ ] Arquivo `.github/workflows/ci.yml` existente
- [ ] `docker compose up -d` → container responde em `http://localhost:8000/health`
