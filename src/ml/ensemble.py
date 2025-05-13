"""
ML Ensemble for Edge Detection

This module implements an ensemble approach for combining predictions from
multiple models to make more robust trading decisions. The primary approach
is a median-probability combiner that reduces the impact of outlier predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging
import os
import json
import pickle

from ml.models import BaseModel, LightGBMModel, BidirectionalLSTMModel, OptionSurfaceCNNModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleMethod(ABC):
    """
    Abstract base class for ensemble methods.
    
    Defines the interface for combining predictions from multiple models.
    """
    
    @abstractmethod
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: List of prediction arrays from individual models
            
        Returns:
            Combined predictions
        """
        pass


class MeanProbabilityEnsemble(EnsembleMethod):
    """Ensemble method that averages probabilities from multiple models."""
    
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Average predictions from multiple models.
        
        Args:
            predictions: List of prediction arrays from individual models
            
        Returns:
            Averaged predictions
        """
        return np.mean(predictions, axis=0)


class MedianProbabilityEnsemble(EnsembleMethod):
    """
    Ensemble method that uses the median probability from multiple models.
    
    This approach is more robust to outlier predictions from individual models.
    """
    
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Calculate median predictions from multiple models.
        
        Args:
            predictions: List of prediction arrays from individual models
            
        Returns:
            Median predictions
        """
        return np.median(predictions, axis=0)


class WeightedEnsemble(EnsembleMethod):
    """
    Ensemble method that combines predictions using weighted average.
    
    Weights can be learned or specified manually based on model performance.
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize weighted ensemble with specified weights.
        
        Args:
            weights: List of weights for each model, should sum to 1.0
        """
        self.weights = weights
    
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Calculate weighted average of predictions from multiple models.
        
        Args:
            predictions: List of prediction arrays from individual models
            
        Returns:
            Weighted average predictions
        """
        if self.weights is None:
            # If no weights provided, use equal weights
            return np.mean(predictions, axis=0)
        
        if len(self.weights) != len(predictions):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(predictions)})")
        
        # Normalize weights to sum to 1
        normalized_weights = np.array(self.weights) / sum(self.weights)
        
        # Calculate weighted average
        return np.sum([w * p for w, p in zip(normalized_weights, predictions)], axis=0)


class ModelEnsemble:
    """
    Ensemble of models for edge detection.
    
    This class manages a collection of models and combines their predictions
    using a specified ensemble method.
    """
    
    def __init__(self, 
                models: List[BaseModel] = None, 
                ensemble_method: EnsembleMethod = None,
                threshold: float = 0.67,
                name: str = "edge_detection_ensemble"):
        """
        Initialize model ensemble.
        
        Args:
            models: List of models to include in ensemble
            ensemble_method: Method to combine predictions
            threshold: Probability threshold for positive prediction
            name: Name of the ensemble
        """
        self.models = models or []
        self.ensemble_method = ensemble_method or MedianProbabilityEnsemble()
        self.threshold = threshold
        self.name = name
        self.model_weights = None
    
    def add_model(self, model: BaseModel) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add
        """
        self.models.append(model)
    
    def set_ensemble_method(self, method: EnsembleMethod) -> None:
        """
        Set the ensemble method to use.
        
        Args:
            method: Ensemble method to use
        """
        self.ensemble_method = method
    
    def predict(self, X: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using all models in the ensemble.
        
        Args:
            X: Features to predict on. Can be:
               - Single numpy array for all models
               - Dictionary mapping model names to feature arrays
            
        Returns:
            Tuple of (ensemble_predictions, raw_predictions)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from each model
        if isinstance(X, dict):
            # Different features for each model
            raw_predictions = [
                model.predict(X[model.name]) 
                for model in self.models 
                if model.name in X
            ]
        else:
            # Same features for all models
            raw_predictions = [model.predict(X) for model in self.models]
        
        # Combine predictions
        ensemble_predictions = self.ensemble_method.combine(raw_predictions)
        
        return ensemble_predictions, np.array(raw_predictions)
    
    def predict_with_confidence(self, 
                              X: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence measures.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (binary_predictions, probabilities, confidence)
            - binary_predictions: 0/1 predictions based on threshold
            - probabilities: Raw probability predictions
            - confidence: Measure of ensemble agreement (std deviation across models)
        """
        ensemble_probs, raw_predictions = self.predict(X)
        
        # Calculate binary predictions based on threshold
        binary_predictions = (ensemble_probs >= self.threshold).astype(int)
        
        # Calculate standard deviation across models as a confidence measure
        # Lower std dev means higher agreement/confidence
        confidence = 1.0 - np.std(raw_predictions, axis=0)
        
        return binary_predictions, ensemble_probs, confidence
    
    def evaluate(self, X: Union[np.ndarray, Dict[str, np.ndarray]], y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance on validation data.
        
        Args:
            X: Features to predict on
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get ensemble predictions
        binary_preds, probs, confidence = self.predict_with_confidence(X)
        
        # Calculate accuracy
        accuracy = np.mean(binary_preds == y)
        
        # Calculate true positives, false positives, true negatives, false negatives
        tp = np.sum((binary_preds == 1) & (y == 1))
        fp = np.sum((binary_preds == 1) & (y == 0))
        tn = np.sum((binary_preds == 0) & (y == 0))
        fn = np.sum((binary_preds == 0) & (y == 1))
        
        # Calculate precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate average confidence
        avg_confidence = np.mean(confidence)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "avg_confidence": avg_confidence,
            "threshold": self.threshold
        }
    
    def evaluate_threshold_sweep(self, 
                               X: Union[np.ndarray, Dict[str, np.ndarray]], 
                               y: np.ndarray,
                               thresholds: List[float] = None) -> pd.DataFrame:
        """
        Evaluate ensemble performance across multiple thresholds.
        
        Args:
            X: Features to predict on
            y: True labels
            thresholds: List of thresholds to evaluate
            
        Returns:
            DataFrame with evaluation metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.5, 0.95, 10)
        
        # Get raw probabilities
        _, probs, _ = self.predict_with_confidence(X)
        
        # Evaluate each threshold
        results = []
        for threshold in thresholds:
            # Set threshold temporarily
            old_threshold = self.threshold
            self.threshold = threshold
            
            # Evaluate with this threshold
            metrics = self.evaluate(X, y)
            metrics["threshold"] = threshold
            results.append(metrics)
            
            # Restore original threshold
            self.threshold = old_threshold
        
        # Convert to DataFrame
        return pd.DataFrame(results)
    
    def get_optimal_threshold(self, X: Union[np.ndarray, Dict[str, np.ndarray]], y: np.ndarray, 
                            metric: str = "f1", thresholds: List[float] = None) -> float:
        """
        Find the optimal threshold based on a specified metric.
        
        Args:
            X: Features to predict on
            y: True labels
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
            thresholds: List of thresholds to evaluate
            
        Returns:
            Optimal threshold value
        """
        # Evaluate across thresholds
        results = self.evaluate_threshold_sweep(X, y, thresholds)
        
        # Find threshold with best metric
        best_idx = results[metric].argmax()
        best_threshold = results.iloc[best_idx]["threshold"]
        
        return best_threshold
    
    def optimize_weights(self, X: Union[np.ndarray, Dict[str, np.ndarray]], y: np.ndarray) -> List[float]:
        """
        Optimize model weights based on validation performance.
        
        This is a simple optimization that evaluates each model individually
        and sets weights proportional to their performance.
        
        Args:
            X: Features to predict on
            y: True labels
            
        Returns:
            List of optimized weights
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Evaluate each model individually
        model_metrics = []
        for i, model in enumerate(self.models):
            if isinstance(X, dict):
                if model.name not in X:
                    logger.warning(f"No features for model {model.name}, skipping")
                    model_metrics.append(0.0)
                    continue
                
                preds = model.predict(X[model.name])
            else:
                preds = model.predict(X)
            
            # Calculate accuracy
            binary_preds = (preds >= 0.5).astype(int)
            accuracy = np.mean(binary_preds == y)
            
            # Calculate F1 score
            tp = np.sum((binary_preds == 1) & (y == 1))
            fp = np.sum((binary_preds == 1) & (y == 0))
            fn = np.sum((binary_preds == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Use F1 score as the metric for weighting
            model_metrics.append(f1)
        
        # Convert to numpy array
        metrics = np.array(model_metrics)
        
        # If all metrics are 0, use equal weights
        if np.sum(metrics) == 0:
            weights = np.ones(len(self.models)) / len(self.models)
        else:
            # Set weights proportional to metrics
            weights = metrics / np.sum(metrics)
        
        # Update ensemble method if it's a weighted ensemble
        if isinstance(self.ensemble_method, WeightedEnsemble):
            self.ensemble_method = WeightedEnsemble(weights=weights)
        
        # Store weights
        self.model_weights = weights.tolist()
        
        return self.model_weights
    
    def save(self, directory: str) -> None:
        """
        Save ensemble configuration to disk.
        
        Args:
            directory: Directory to save ensemble to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            "name": self.name,
            "threshold": self.threshold,
            "model_names": [model.name for model in self.models],
            "ensemble_method": self.ensemble_method.__class__.__name__,
            "model_weights": self.model_weights
        }
        
        with open(os.path.join(directory, f"{self.name}_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save each model
        for model in self.models:
            model.save(directory)
            
        logger.info(f"Saved ensemble configuration to {directory}")
    
    def load(self, directory: str, load_models: bool = True) -> None:
        """
        Load ensemble configuration from disk.
        
        Args:
            directory: Directory to load ensemble from
            load_models: Whether to load the individual models
        """
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No ensemble configuration found at {config_path}")
        
        # Load ensemble configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.name = config["name"]
        self.threshold = config["threshold"]
        
        # Set ensemble method
        method_name = config.get("ensemble_method", "MedianProbabilityEnsemble")
        if method_name == "MedianProbabilityEnsemble":
            self.ensemble_method = MedianProbabilityEnsemble()
        elif method_name == "MeanProbabilityEnsemble":
            self.ensemble_method = MeanProbabilityEnsemble()
        elif method_name == "WeightedEnsemble":
            self.ensemble_method = WeightedEnsemble(weights=config.get("model_weights"))
        
        # Store model weights
        self.model_weights = config.get("model_weights")
        
        # Load models if requested
        if load_models:
            self.models = []
            for model_name in config["model_names"]:
                # Determine model type from metadata file
                metadata_path = os.path.join(directory, f"{model_name}_metadata.json")
                
                if not os.path.exists(metadata_path):
                    logger.warning(f"No metadata found for model {model_name}, skipping")
                    continue
                
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                model_type = metadata.get("model_type")
                
                # Create model of appropriate type
                if model_type == "LightGBMModel":
                    model = LightGBMModel(name=model_name)
                elif model_type == "BidirectionalLSTMModel":
                    model = BidirectionalLSTMModel(name=model_name)
                elif model_type == "OptionSurfaceCNNModel":
                    model = OptionSurfaceCNNModel(name=model_name)
                else:
                    logger.warning(f"Unknown model type {model_type} for {model_name}, skipping")
                    continue
                
                # Load model
                model.load(directory)
                
                # Add to ensemble
                self.models.append(model)
            
            logger.info(f"Loaded {len(self.models)} models for ensemble {self.name}")
        
        logger.info(f"Loaded ensemble configuration from {directory}")


def create_default_ensemble(model_directory: Optional[str] = None) -> ModelEnsemble:
    """
    Create a default ensemble with standard models for edge detection.
    
    Args:
        model_directory: Directory to load pre-trained models from
        
    Returns:
        Configured model ensemble
    """
    # Create the three model types
    lightgbm_model = LightGBMModel(name="edge_detection_gbm")
    lstm_model = BidirectionalLSTMModel(name="edge_detection_lstm")
    cnn_model = OptionSurfaceCNNModel(name="edge_detection_cnn")
    
    # Create ensemble with median probability combiner
    ensemble = ModelEnsemble(
        models=[lightgbm_model, lstm_model, cnn_model],
        ensemble_method=MedianProbabilityEnsemble(),
        threshold=0.67,  # Required threshold from spec
        name="edge_detection_ensemble"
    )
    
    # Load pre-trained models if directory provided
    if model_directory is not None and os.path.exists(model_directory):
        ensemble.load(model_directory)
    
    return ensemble