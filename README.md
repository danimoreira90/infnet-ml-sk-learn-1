# Credit Card Default Prediction - Machine Learning Project

A comprehensive machine learning project implementing supervised learning models for credit card default prediction, with emphasis on model interpretability, systematic evaluation, and professional methodology.

## 📋 Project Overview

This project demonstrates a complete machine learning workflow from problem definition through advanced model evaluation, using **only scikit-learn** and classical ML techniques. The focus is on understanding model behavior, limitations, and the relationship between technical choices and practical impact.

### Business Context
- **Domain:** Financial Services - Credit Risk Management
- **Problem:** Binary classification of credit card default risk
- **Dataset:** 30,000 credit card clients with 23 features
- **Goal:** Predict default probability to support credit decisions

## 🗂️ Project Structure

```
credit-card-ml-project/
│
├── data/
│   ├── default_of_credit_card_clients.xls    # Raw dataset
│   └── credit_card_cleaned.csv                 # Processed data
│
├── models/
│   ├── perceptron_baseline.pkl
│   ├── decision_tree_default.pkl
│   ├── decision_tree_optimized.pkl
│   ├── scaler.pkl
│   └── final_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── cv_results.csv
│   └── figures/
│
├── notebooks/
│   ├── DEVELOPMENT NOTEBOOKS (Modular)
│   ├── 01_eda_and_problem_definition.ipynb
│   ├── 02_baseline_perceptron.ipynb
│   ├── 03_decision_tree.ipynb
│   ├── 04_cv_and_hyperparameter_tuning.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_final_comparison_and_report.ipynb
│   │
│   └── SUBMISSION NOTEBOOK (Comprehensive)
│       └── complete_ml_project.ipynb
│
├── utils.py              # Helper functions
└── README.md             # This file
```

## 📓 Notebook Descriptions

### Development Phase (Modular Notebooks)

#### **01: EDA and Problem Definition**
- Real-world context and business motivation
- Dataset description and feature analysis
- Target variable analysis (class imbalance)
- Correlation analysis and feature relationships
- Domain challenges identification

#### **02: Baseline Perceptron Model**
- Linear classifier implementation
- Geometric interpretation of hyperplane
- Coefficient analysis and bias term
- Performance evaluation (accuracy, precision, recall, F1)
- Limitations analysis (non-linear separability, underfitting)

#### **03: Decision Tree Model**
- Non-linear classification approach
- Tree structure and rule interpretation
- Feature importance analysis
- Overfitting risk assessment
- Comparison with linear baseline

#### **04: Cross-Validation & Hyperparameter Tuning**
- K-fold cross-validation implementation
- Grid Search / Random Search
- Hyperparameter space exploration
- Model stability and robustness analysis
- Regularization impact on tree structure

#### **05: Advanced Models**
- SVM (linear/kernel) OR Ensemble methods
- Random Forest / Gradient Boosting implementation
- Advanced hyperparameter tuning
- Feature importance in ensembles
- Complexity vs. performance trade-off

#### **06: Final Comparison & Report**
- Comprehensive model comparison
- Performance summary across all models
- Business recommendations
- Limitations and future improvements
- Final model selection and justification

### Submission Phase

#### **Complete ML Project** (Merged Comprehensive Notebook)
- All stages in a single, cohesive document
- Clear narrative flow from problem to solution
- Professional formatting and documentation
- Ready for academic/professional submission

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
openpyxl  # for Excel file reading
```

### Installation
```bash
# Clone or download the project
cd credit-card-ml-project

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl

# Create necessary directories
mkdir -p data models results/figures
```

### Running the Notebooks

**Option 1: Development Mode** (Recommended for learning/iteration)
```bash
# Run notebooks in order:
jupyter notebook 01_eda_and_problem_definition.ipynb
jupyter notebook 02_baseline_perceptron.ipynb
# ... and so on
```

**Option 2: Submission Mode** (For final deliverable)
```bash
jupyter notebook complete_ml_project.ipynb
```

## 📊 Dataset Information

### Source
- **Origin:** UCI Machine Learning Repository
- **Study:** Yeh, I. C., & Lien, C. H. (2009)
- **Collection:** Important bank in Taiwan, October 2005

### Characteristics
- **Observations:** 30,000 credit card clients
- **Features:** 23 predictor variables
- **Target:** Binary (1=default, 0=no default)
- **Class Distribution:** ~22% default rate

### Feature Categories
1. **Demographic (5):** Credit limit, gender, education, marital status, age
2. **Payment History (6):** Repayment status from September to April
3. **Bill Amounts (6):** Monthly bill statement amounts
4. **Previous Payments (6):** Monthly payment amounts

## 🎯 Project Objectives

### Technical Goals
1. ✅ Build baseline linear classifier (Perceptron)
2. ✅ Implement non-linear model (Decision Tree)
3. ✅ Apply cross-validation and hyperparameter tuning
4. ✅ Train advanced model (SVM/Ensemble)
5. ✅ Interpret models without external tools
6. ✅ Compare models systematically

### Learning Outcomes
- Understand linear vs. non-linear decision boundaries
- Master bias-variance trade-off
- Implement proper validation methodology
- Interpret model parameters without black boxes
- Make informed model selection decisions

## 📈 Key Results

### Model Performance Comparison

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|---------------|----------------|-------------|---------|
| Perceptron | ~0.80 | ~0.65 | ~0.35 | ~0.45 |
| Decision Tree (Default) | ~0.73 | ~0.50 | ~0.60 | ~0.55 |
| Decision Tree (Tuned) | ~0.82 | ~0.67 | ~0.45 | ~0.54 |
| Random Forest | ~0.82 | ~0.70 | ~0.40 | ~0.51 |

*Note: Exact values depend on data split and tuning results*

### Key Insights
1. **Linear Baseline:** Perceptron provides strong baseline despite simplicity
2. **Overfitting Risk:** Unconstrained decision trees memorize training data
3. **Regularization:** Properly tuned trees match/exceed linear performance
4. **Ensemble Boost:** Random Forests provide modest improvements
5. **Interpretability Trade-off:** More complex models sacrifice explainability

## 🛠️ Utilities (utils.py)

### Available Functions
```python
# Data loading
load_credit_data(filepath)
get_basic_stats(df, target_col)

# Model evaluation
evaluate_model(y_true, y_pred, model_name)
plot_confusion_matrix(y_true, y_pred, model_name)
plot_roc_curve(y_true, y_proba, model_name)

# Visualization
compare_models(results_dict)
plot_cv_results(cv_results, param_name)
plot_learning_curve(estimator, X, y)

# Feature analysis
analyze_feature_importance(model, feature_names, top_n)
create_results_dataframe(results_dict)
```

## 📝 Project Requirements Compliance

- [x] Real-world problem with business context
- [x] Dataset with 10+ features (23 features)
- [x] Binary classification problem
- [x] Perceptron baseline with interpretation
- [x] Decision tree with rule analysis
- [x] Cross-validation and hyperparameter search
- [x] Advanced model (SVM or Ensemble)
- [x] No external explainability tools (SHAP, LIME)
- [x] Scikit-learn only
- [x] Reproducible code
- [x] Technical report with justifications
- [x] Applied discussion of limitations

## 🔍 Model Interpretation Approach

### Without External Tools
1. **Perceptron:** Weight vector analysis, bias interpretation, hyperplane geometry
2. **Decision Trees:** Rule extraction, tree visualization, path analysis
3. **Random Forest:** Feature importance aggregation, individual tree inspection
4. **SVM:** Support vector analysis, decision function coefficients

### Domain-Grounded Interpretation
- Connect learned patterns to financial risk factors
- Validate rules against credit risk theory
- Discuss business feasibility
- Identify potential biases

## ⚠️ Limitations and Future Work

### Current Limitations
1. Static model (no retraining pipeline)
2. No cost-sensitive learning (asymmetric error costs)
3. Limited feature engineering
4. No ensemble method comparisons (RF vs. GBM)
5. Single dataset (no cross-dataset validation)

### Potential Improvements
1. Implement cost-sensitive classification
2. Add polynomial features for interaction terms
3. Explore SMOTE for class imbalance
4. Build stacking/voting ensembles
5. Implement model monitoring framework
6. Add explainability for stakeholder communication

## 📚 References

### Dataset
- Yeh, I. C., & Lien, C. H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." *Expert Systems with Applications*, 36(2), 2473-2480.

### Methodology
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*
- Scikit-learn Documentation: https://scikit-learn.org/

## 👤 Author

Daniel Moreira
Infnet - Scikit-Learn and ML models  
03/03/2026

## 📄 License

This project is for educational purposes.

---

**Note:** This is an academic project demonstrating supervised learning methodology. It should not be used for actual credit decisions without proper validation, regulatory compliance, and fairness auditing.

---

## Como rodar a Parte 2 — Fundação de Dados

### Pré-requisitos

- **uv** (gestor de ambientes e pacotes Python)
  Verificar: `uv --version`
  Instalar: https://docs.astral.sh/uv/getting-started/installation/
  Python 3.11 é gerenciado pelo uv: `uv python install 3.11`
- Dataset bruto em `../data/default of credit card clients.xls` (relativo à raiz do repo)

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
# Saída: data/credit_card_cleaned.parquet
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

---

## Parte 3 — Modelagem e MLflow

### Pre-requisitos

```bash
uv sync
```

### Execucao

```bash
# 1. Treino baseline (5 modelos, ~3-5 min)
uv run python scripts/train_baseline.py

# 2. Treino com tuning (5 modelos, ~15-25 min)
uv run python scripts/train_tuned.py

# 3. Gerar tabelas comparativas
uv run python scripts/generate_comparison_table.py
```

### Visualizar resultados no MLflow UI

```bash
uv run mlflow ui --backend-store-uri mlruns/
# Abrir http://localhost:5000
```

### Testes da Parte 3

```bash
uv run python -m pytest tests/test_preprocessing.py tests/test_pipeline.py tests/test_registry.py tests/test_run_naming.py -v
```

---

## Parte 4 — Reducao de Dimensionalidade

Aplica PCA e LDA ao pipeline sklearn, retreina os 5 modelos e compara com o baseline da Parte 3.

**Tecnicas:** PCA (nao-supervisionado) e LDA (supervisionado). t-SNE excluido — transductivo e com custo O(n²) inviavel para CV.

**Configuracoes:** `pca_k10` (10 componentes, 84.2% EV), `pca_k15` (15 componentes, 94.3% EV), `lda_k1` (1 componente — forcado pela matematica binaria).

**Total de runs:** 15 (5 modelos × 3 configs), adicionados ao experimento `infnet-ml-sistema` sem deletar os runs da Parte 3.

### Execucao

```bash
# 1. Treino dimred (15 runs, ~3-5 min)
uv run python scripts/train_dimred.py

# 2. Gerar tabelas comparativas
uv run python scripts/generate_comparison_dimred.py
```

### Outputs

| Arquivo | Descricao |
|---------|-----------|
| `reports/parte_4/comparison_dimred.md` | 20 runs (P3 + P4) ordenados por ROC-AUC |
| `reports/parte_4/comparison_pca_vs_lda.md` | Pivot: modelo × config de dimred |

### Testes da Parte 4

```bash
uv run pytest tests/test_dimred.py tests/test_train_dimred.py tests/test_generate_comparison_dimred.py -v
```

### Suite completa

```bash
uv run pytest --tb=short -q
```

---

## Como rodar a Parte 5 — Selecao Final do Modelo

### Pre-requisitos

- Parte 3 e Parte 4 executadas (25 runs no experimento `infnet-ml-sistema`)
- `docs/final_selection_criteria.md` presente e imutavel

### Passo 1 — Auditoria de integridade (3 runs aleatorios P3/P4)

```bash
uv run python scripts/audit_sample.py
# Esperado: "[AUDIT SAMPLE] 3/3 runs auditados OK"
```

### Passo 2 — Tabela consolidada dos 25 runs

```bash
uv run python scripts/generate_consolidated_results.py
# Gera: reports/parte_5/consolidated_results.md
```

### Passo 3 — Selecionar candidato final (nao toca test set)

```bash
uv run python scripts/select_final_candidate.py
# Gera: reports/parte_5/final_selection_rationale.md
# Mostra vencedor e step decisivo
```

### Passo 4 — Avaliar no test set (UMA UNICA VEZ — apos aprovacao humana)

```bash
uv run python scripts/evaluate_final.py
# Gera: reports/parte_5/test_metrics.json
# Loga MLflow run com stage="final_eval" e mlflow.sklearn.log_model
```

### Guards de integridade

```powershell
# Guard 1A: inventario de arquivos com "test_idx" (esperado: 4 arquivos)
Select-String -Path scripts\*.py, src\**\*.py, tests\*.py -Pattern "test_idx" |
    Select-Object -ExpandProperty Path -Unique

# Guard 1B: include_test=True em exatamente 1 arquivo
Select-String -Path scripts\*.py, src\**\*.py, tests\*.py -Pattern "include_test=True"

# Guard 1C: mlflow.sklearn.log_model em exatamente 1 arquivo
Select-String -Path scripts\*.py, src\**\*.py -Pattern "mlflow.sklearn.log_model"
```

---

## Como rodar a Parte 6 — Operacionalizacao

### Pre-requisitos

- Parte 5 executada (`mlruns/` com modelo registrado em `models:/m-4de1a2c47e7d40d9a679a40ba79c9c65`)
- `uv sync` para instalar dependencias (FastAPI, uvicorn, scipy incluidos)
- Docker Desktop (opcional, para execucao conteinerizada)

### Execucao local (uvicorn)

```bash
uv sync
uv run uvicorn src.credit_default.serving.app:app --port 8000
```

O servidor inicializa em ~3s. Testar:

```bash
curl -s http://localhost:8000/health | python -m json.tool
# {"status": "ok", "model_uri": "models:/m-4de1a2c47e7d40d9a679a40ba79c9c65"}

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"record":{"LIMIT_BAL":30000,"SEX":2,"EDUCATION":1,"MARRIAGE":1,"AGE":35,
       "PAY_0":-1,"PAY_2":-1,"PAY_3":-1,"PAY_4":-2,"PAY_5":-2,"PAY_6":-2,
       "BILL_AMT1":390,"BILL_AMT2":780,"BILL_AMT3":0,"BILL_AMT4":0,
       "BILL_AMT5":0,"BILL_AMT6":0,
       "PAY_AMT1":780,"PAY_AMT2":0,"PAY_AMT3":0,"PAY_AMT4":0,
       "PAY_AMT5":0,"PAY_AMT6":0}}' | python -m json.tool
# {"prediction": 0, "probability_default": 0.2765, "probability_no_default": 0.7235}
```

### Execucao via Docker

```bash
# Build e subir
docker compose up -d

# Verificar
docker compose ps
curl -s http://localhost:8000/health | python -m json.tool

# Logs
docker compose logs --tail=20

# Parar
docker compose down
```

### Relatorio de drift

```bash
uv run python scripts/run_drift_report.py
# Saida: reports/parte_6/drift_report.md
# MLflow run com stage=drift_report logado automaticamente
```

### Testes da Parte 6

```bash
uv run pytest tests/test_serving.py tests/test_api.py tests/test_drift.py -v
```

### Suite completa

```bash
uv run pytest --tb=short -q
```
