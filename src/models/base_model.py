"""
Base model class for machine learning models.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the model.
        
        Args:
            params: Model parameters
        """
        self.params = params or {}
        self.model = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Model predictions
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
