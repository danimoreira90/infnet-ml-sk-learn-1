"""
Utility functions for Credit Card Default ML Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_credit_data(filepath):
    """
    Load and prepare the credit card default dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataset
    """
    # Read Excel file (header is in row 2, index 1)
    df = pd.read_excel(filepath, header=1)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df


def get_basic_stats(df, target_col='default_payment_next_month'):
    """
    Get basic statistics about the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    stats : dict
        Dictionary with dataset statistics
    """
    stats = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'missing_values': df.isnull().sum().sum(),
        'target_distribution': df[target_col].value_counts().to_dict(),
        'imbalance_ratio': df[target_col].value_counts()[0] / df[target_col].value_counts()[1]
    }
    return stats


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Comprehensive model evaluation with all required metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    Plot confusion matrix for model predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name='Model'):
    """
    Plot ROC curve for model predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities (for positive class)
    model_name : str
        Name of the model for title
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc


def compare_models(results_dict):
    """
    Compare multiple models using a bar plot.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics dict as values
        Format: {'Model1': {'accuracy': 0.85, 'precision': 0.80, ...}, ...}
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        axes[idx].bar(models, values, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_cv_results(cv_results, param_name, metric='mean_test_score'):
    """
    Plot cross-validation results for hyperparameter tuning.
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results from GridSearchCV or RandomizedSearchCV
    param_name : str
        Name of the parameter to plot
    metric : str
        Metric to plot (default: 'mean_test_score')
    """
    param_key = f'param_{param_name}'
    
    if param_key in cv_results:
        params = cv_results[param_key].data
        scores = cv_results[metric]
        
        plt.figure(figsize=(10, 6))
        plt.plot(params, scores, marker='o', linewidth=2, markersize=8)
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Cross-Validation: {metric.replace("_", " ").title()} vs {param_name.replace("_", " ").title()}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy'):
    """
    Plot learning curve to diagnose bias/variance.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation Score', color='red', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze and plot feature importance for tree-based and ensemble models.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue', alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print numerical values
    print(f"\nTop {top_n} Feature Importances:")
    print("-" * 50)
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")


def create_results_dataframe(results_dict):
    """
    Create a pandas DataFrame from results dictionary for easy comparison.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics dict as values
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with models as rows and metrics as columns
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df


def print_model_summary(model, model_name='Model'):
    """
    Print a summary of model parameters and configuration.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    model_name : str
        Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Configuration")
    print(f"{'='*60}")
    print(f"Model Type: {type(model).__name__}")
    print(f"\nParameters:")
    for param, value in model.get_params().items():
        print(f"  {param}: {value}")
    print(f"{'='*60}\n")
