"""
Visualization utilities for machine learning.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional


def plot_feature_importance(
    features: List[str],
    importance: np.ndarray,
    title: str = "Feature Importance",
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot feature importance.
    
    Args:
        features: List of feature names
        importance: Array of feature importance scores
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    importance_df = pd.DataFrame({'features': features, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    plt.barh(importance_df['features'], importance_df['importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    cm = pd.crosstab(y_true, y_pred, normalize='index')
    
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    train_scores: List[float],
    val_scores: List[float],
    title: str = "Learning Curve",
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot learning curve.
    
    Args:
        train_scores: List of training scores
        val_scores: List of validation scores
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Training Score')
    plt.plot(epochs, val_scores, 'r-', label='Validation Score')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
