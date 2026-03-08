# Credit Card Default ML Project - Complete Implementation Guide

## 📦 Delivered Files

### ✅ Complete and Ready to Use
1. **utils.py** - All helper functions
2. **README.md** - Project documentation
3. **01_eda_and_problem_definition.ipynb** - Complete EDA
4. **02_baseline_perceptron.ipynb** - Complete baseline model
5. **03_decision_tree.ipynb** - Complete decision tree analysis

### 📝 Templates/Guides for Remaining Notebooks

## Notebook 04: Cross-Validation & Hyperparameter Tuning

###Structure:
```python
### Key Sections ###

## 1. Introduction
# - Why cross-validation (avoiding overfitting to single test set)
# - Hyperparameter tuning motivation
# - Grid Search vs. Random Search trade-offs

## 2. Setup
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from utils import *

## 3. Cross-Validation Implementation
# K-Fold CV (k=5 or 10)
from sklearn.model_selection import KFold, StratifiedKFold

# Use Stratified K-Fold to preserve class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate decision tree
dt = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(dt, X_train, y_train, cv=skf, scoring='f1')

print(f"CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

## 4. Hyperparameter Search Space
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20, 50, 100],
    'min_samples_leaf': [1, 5, 10, 20, 50],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

## 5. Grid Search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=skf,
    scoring='f1',  # or 'recall' if prioritizing default detection
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

## 6. Evaluate Optimized Model
best_dt = grid_search.best_estimator_
y_test_pred_optimized = best_dt.predict(X_test)

# Compare with unoptimized tree
evaluate_model(y_test, y_test_pred_optimized, "Optimized Decision Tree")

## 7. Analyze CV Results
cv_results_df = pd.DataFrame(grid_search.cv_results_)
# Plot performance vs. hyperparameters
# Analyze variance across folds
# Identify stable vs. unstable configurations

## 8. Tree Structure Comparison
print(f"Unoptimized tree depth: {dt_default.get_depth()}")
print(f"Optimized tree depth: {best_dt.get_depth()}")
print(f"Unoptimized tree leaves: {dt_default.get_n_leaves()}")
print(f"Optimized tree leaves: {best_dt.get_n_leaves()}")

## 9. Generalization Analysis
# Plot learning curves
# Analyze bias-variance trade-off
# Discussion on regularization impact

## 10. Save Results
```

### Key Analysis Points for Notebook 04:
- **Metric Variance:** How stable is performance across folds?
- **Overfitting Control:** Does regularization reduce train-test gap?
- **Hyperparameter Sensitivity:** Which parameters matter most?
- **Complexity Reduction:** How much simpler is the optimized tree?
- **Generalization:** Does CV score predict test performance?

---

## Notebook 05: Advanced Models (SVM or Ensembles)

### Option A: Random Forest (Recommended)

```python
### Key Sections ###

## 1. Introduction
# - Ensemble learning concept
# - Bootstrap aggregating (bagging)
# - Why Random Forest reduces variance
# - Expected performance improvements

## 2. Setup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

## 3. Random Forest - Default Configuration
rf_default = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_default.fit(X_train, y_train)
y_test_pred_rf = rf_default.predict(X_test)

evaluate_model(y_test, y_test_pred_rf, "Random Forest (Default)")

## 4. Hyperparameter Tuning
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions,
    n_iter=50,  # Try 50 random combinations
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
y_test_pred_rf_tuned = best_rf.predict(X_test)

## 5. Feature Importance Analysis
feature_importance = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Visualize top 15 features
analyze_feature_importance(best_rf, feature_names, top_n=15)

# Compare with single tree feature importance

## 6. Individual Tree Analysis
# Examine diversity among trees
from sklearn.tree import export_text

print("Sample tree rules from forest:")
for i in range(3):  # Show 3 random trees
    tree = best_rf.estimators_[i]
    print(f"\n--- Tree {i} (depth={tree.get_depth()}) ---")
    print(export_text(tree, feature_names=feature_names, max_depth=3))

## 7. Out-of-Bag (OOB) Score
# If bootstrap=True
if best_rf.bootstrap:
    rf_with_oob = RandomForestClassifier(
        **best_rf.get_params(),
        oob_score=True
    )
    rf_with_oob.fit(X_train, y_train)
    print(f"OOB Score: {rf_with_oob.oob_score_:.4f}")

## 8. Performance Analysis
# Precision-Recall trade-off
y_test_proba_rf = best_rf.predict_proba(X_test)[:, 1]
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba_rf)
# Plot PR curve

## 9. Complexity vs. Performance
# Compare:
# - Training time
# - Prediction time
# - Model size
# - Interpretability
# - Performance gain

## 10. Business Value Analysis
# What is the cost of false negatives vs. false positives?
# Optimize threshold for business metrics
# Calculate expected financial impact
```

### Option B: Support Vector Machine

```python
### Key Sections ###

## 1. Introduction
# - Maximum margin classification
# - Kernel trick for non-linearity
# - Support vectors concept

## 2. Linear SVM
from sklearn.svm import SVC

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)  # SVM needs scaling!

# Analyze support vectors
n_support = svm_linear.n_support_
print(f"Support vectors: {n_support} ({n_support.sum()}/{len(X_train)} = {n_support.sum()/len(X_train)*100:.1f}%)")

# Visualize decision boundary (2D projection)

## 3. Non-Linear SVM (RBF Kernel)
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid_svm = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid_svm,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_svm.fit(X_train_scaled, y_train)

## 4. Hyperparameter Impact Analysis
# C parameter: Regularization strength
# gamma parameter: RBF kernel width
# Trade-off between margin and classification accuracy

## 5. Performance Evaluation
# ... standard evaluation ...

## 6. Comparison with Tree-Based Models
# Linear SVM vs. Perceptron
# RBF SVM vs. Decision Tree/Random Forest
```

### Key Analysis Points for Notebook 05:
- **Performance Improvement:** Quantify gains over simpler models
- **Computational Cost:** Training/prediction time comparison
- **Interpretability Loss:** What do we sacrifice for performance?
- **Ensemble Wisdom:** How does averaging reduce variance?
- **Business Feasibility:** Can this model be deployed in production?

---

## Notebook 06: Final Comparison & Report

```python
### Key Sections ###

## 1. Executive Summary
# - Project recap
# - Business problem statement
# - Methodology overview
# - Key findings (3-5 bullets)

## 2. Comprehensive Model Comparison

### 2.1 Performance Metrics Table
results_final = pd.DataFrame({
    'Model': [
        'Perceptron',
        'Decision Tree (Default)',
        'Decision Tree (Optimized)',
        'Random Forest',
        'SVM Linear',  # if implemented
        'SVM RBF'      # if implemented
    ],
    'Test_Accuracy': [...],
    'Test_Precision': [...],
    'Test_Recall': [...],
    'Test_F1': [...],
    'Train_Time_sec': [...],
    'Predict_Time_sec': [...],
    'Model_Complexity': [...]  # qualitative: Low/Medium/High
})

### 2.2 Visual Comparison
# - Bar plots for all metrics
# - ROC curves overlaid
# - Precision-Recall curves overlaid
# - Learning curves comparison

### 2.3 Statistical Significance
# - McNemar's test for classifier comparison
# - Confidence intervals for metrics

## 3. Model Selection

### 3.1 Selection Criteria
# Define weighted scoring:
# - Performance (40%)
# - Interpretability (30%)
# - Computational efficiency (20%)
# - Robustness (10%)

### 3.2 Recommended Model
# Justify final selection based on:
# - Business requirements
# - Deployment constraints
# - Regulatory needs (explainability)
# - Performance trade-offs

## 4. Feature Importance Synthesis
# Compare feature importance across all models
# Identify consensus important features
# Discuss discrepancies

## 5. Business Recommendations

### 5.1 Model Deployment
# - Recommended model and configuration
# - Expected performance in production
# - Monitoring metrics
# - Retraining frequency

### 5.2 Decision Thresholds
# - Optimal threshold for business objectives
# - Cost-benefit analysis
# - Different strategies for different risk appetites

### 5.3 Action Items
# For predicted defaults:
# - Early intervention strategies
# - Credit limit adjustments
# - Collection prioritization

## 6. Limitations and Risks

### 6.1 Technical Limitations
# - Class imbalance handling
# - Feature engineering opportunities missed
# - Temporal dynamics not captured
# - No cost-sensitive learning

### 6.2 Business Risks
# - Model bias potential
# - Fair lending concerns
# - Regulatory compliance
# - Customer experience impact

### 6.3 Operational Constraints
# - Data quality dependencies
# - Real-time prediction requirements
# - Model interpretability for credit officers
# - Integration with existing systems

## 7. Future Improvements

### 7.1 Short-term (0-3 months)
# - Implement cost-sensitive learning
# - Add feature engineering
# - Ensemble method comparison
# - Threshold optimization

### 7.2 Medium-term (3-6 months)
# - Deep learning exploration
# - External data integration
# - Real-time scoring API
# - A/B testing framework

### 7.3 Long-term (6-12 months)
# - Causal inference modeling
# - Reinforcement learning for dynamic strategies
# - Fairness and bias mitigation
# - Explainable AI dashboard

## 8. Lessons Learned

### 8.1 Technical Insights
# - Importance of baseline models
# - Bias-variance trade-off in practice
# - Cross-validation necessity
# - Interpretability vs. performance

### 8.2 Domain Insights
# - Key default predictors
# - Feature interactions discovered
# - Business rule validation
# - Stakeholder communication importance

## 9. Conclusion
# - Summarize achievements
# - Restate recommendations
# - Call to action

## 10. Appendices
# - Detailed hyperparameter search results
# - Additional visualizations
# - Code repository structure
# - References
```

---

## Quick Start Instructions

### For Cursor IDE:

1. **Setup Project:**
```bash
# Create project structure
mkdir -p data models results/figures

# Place the credit card dataset in data/
# Copy all provided notebooks and utils.py to project root
```

2. **Install Dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl jupyter
```

3. **Run Development Notebooks (Recommended Path):**
```bash
# Start with completed notebooks
jupyter notebook 01_eda_and_problem_definition.ipynb
jupyter notebook 02_baseline_perceptron.ipynb
jupyter notebook 03_decision_tree.ipynb

# Then create and run notebooks 04, 05, 06 using templates above
```

4. **Follow This Workflow:**
   - Run each notebook sequentially
   - Each notebook saves models/results for the next
   - Use utils.py functions throughout
   - Generate figures for the final report

5. **Create Final Submission:**
   - Merge all 6 notebooks into one comprehensive notebook
   - Ensure narrative flows smoothly
   - Add executive summary at the top
   - Include all visualizations
   - Proofread for clarity and correctness

---

## Critical Implementation Notes

### 1. Data Persistence Between Notebooks
```python
# Always save models and results for next notebooks:

# In notebook 02:
pickle.dump(perceptron, open('../models/perceptron_baseline.pkl', 'wb'))
results_df.to_csv('../results/model_comparison.csv', index=False)

# In notebook 03:
results_df = pd.read_csv('../results/model_comparison.csv')
# Add new results...
results_df.to_csv('../results/model_comparison.csv', index=False)
```

### 2. Consistent Random Seeds
```python
# Use same random_state throughout:
np.random.seed(42)
train_test_split(..., random_state=42)
DecisionTreeClassifier(random_state=42)
```

### 3. Feature Scaling Consistency
```python
# Save scaler in notebook 02, reuse in 05 for SVM:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open('../models/scaler.pkl', 'wb'))

# In notebook 05:
scaler = pickle.load(open('../models/scaler.pkl', 'rb'))
X_test_scaled = scaler.transform(X_test)
```

### 4. Evaluation Consistency
```python
# Use same evaluation approach everywhere:
from utils import evaluate_model

# This ensures all models evaluated with identical metrics
train_metrics = evaluate_model(y_train, y_pred_train, "Model Name (Train)")
test_metrics = evaluate_model(y_test, y_pred_test, "Model Name (Test)")
```

---

## Grading Rubric Alignment

### ✅ Problem Definition & Data (Notebook 01)
- [x] Real-world context
- [x] ML justification
- [x] Domain challenges
- [x] Dataset ≥10 features
- [x] Binary classification

### ✅ Baseline Model (Notebook 02)
- [x] Perceptron implementation
- [x] Conceptual justification
- [x] Train-test split
- [x] All required metrics
- [x] Coefficient interpretation
- [x] Limitations discussion

### ✅ Decision Tree (Notebook 03)
- [x] Default parameters
- [x] Performance comparison
- [x] Metrics analysis
- [x] Rule interpretation
- [x] Overfitting analysis

### ⏳ Cross-Validation & Tuning (Notebook 04)
- [ ] CV implementation
- [ ] Hyperparameter search
- [ ] Grid/Random Search
- [ ] Optimization comparison
- [ ] Robustness analysis

### ⏳ Advanced Models (Notebook 05)
- [ ] SVM or Ensemble
- [ ] CV + hyperparameter search
- [ ] Comparison with previous
- [ ] Metrics evaluation
- [ ] Interpretation without tools

### ⏳ Final Report (Notebook 06)
- [ ] Complete comparison
- [ ] Justifications
- [ ] Critical discussion
- [ ] Real-world feasibility
- [ ] Future improvements

---

## Tips for Success

1. **Start Simple:** Run notebooks 01-03 first to understand the flow
2. **Iterative Development:** Don't try to perfect each notebook before moving on
3. **Use Utils:** Leverage provided functions for consistency
4. **Document Everything:** Add markdown cells explaining your reasoning
5. **Visualize Often:** Use plots to support your arguments
6. **Compare Continuously:** Always relate new results to previous models
7. **Think Business:** Connect technical findings to business implications
8. **Be Critical:** Discuss limitations, don't just celebrate successes

---

## Getting Help

If you encounter issues:

1. **Data Loading:** Ensure Excel file is in `../data/` relative to notebooks
2. **Import Errors:** Check that utils.py is in the parent directory or same folder
3. **Model Errors:** Verify sklearn version compatibility
4. **Performance Issues:** Random Forest can be slow; reduce n_estimators for testing
5. **Memory Issues:** Decision Tree visualization can be large; limit depth

---

## Final Checklist

Before submission:

- [ ] All 6 notebooks run without errors
- [ ] Models saved in `../models/`
- [ ] Results saved in `../results/`
- [ ] Figures generated and saved
- [ ] Code is well-commented
- [ ] Markdown explanations are clear
- [ ] Comprehensive notebook created (merged version)
- [ ] README.md is updated
- [ ] No external libraries beyond scikit-learn, pandas, numpy, matplotlib, seaborn
- [ ] Random seeds set for reproducibility
- [ ] Final report includes all required sections

---

**Good luck with your project! 🚀**
