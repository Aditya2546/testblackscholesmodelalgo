"""
Stacked Ensemble ML Model for Options Trading Edge Detection

This module implements an ensemble approach combining:
1. LightGBM (Optuna-tuned) with class weighting
2. Bidirectional LSTM on price + IV surface grid
3. ConvNet on "surface image" (strike × tenor heat-map)

The ensemble uses median-probability voting to determine edge probability
and includes online partial-fit buffer for incremental learning.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
from river import drift
import joblib
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
import threading
from collections import deque

# Import local modules
from src.ml.models import LightGBMModel, BidirectionalLSTMModel, OptionSurfaceCNNModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model evaluation and tracking."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    log_loss: float = 0.0
    mcc: float = 0.0  # Matthews correlation coefficient
    timestamp: datetime = field(default_factory=datetime.now)
    trades_evaluated: int = 0
    win_count: int = 0
    loss_count: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate from evaluated trades."""
        if self.trades_evaluated == 0:
            return 0.0
        return self.win_count / self.trades_evaluated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc": self.auc,
            "log_loss": self.log_loss,
            "mcc": self.mcc,
            "timestamp": self.timestamp.isoformat(),
            "trades_evaluated": self.trades_evaluated,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformanceMetrics':
        """Create metrics object from dictionary."""
        metrics = cls(model_name=data["model_name"])
        metrics.accuracy = data.get("accuracy", 0.0)
        metrics.precision = data.get("precision", 0.0)
        metrics.recall = data.get("recall", 0.0)
        metrics.f1_score = data.get("f1_score", 0.0)
        metrics.auc = data.get("auc", 0.0)
        metrics.log_loss = data.get("log_loss", 0.0)
        metrics.mcc = data.get("mcc", 0.0)
        
        # Parse timestamp if provided
        if "timestamp" in data:
            metrics.timestamp = datetime.fromisoformat(data["timestamp"])
            
        metrics.trades_evaluated = data.get("trades_evaluated", 0)
        metrics.win_count = data.get("win_count", 0)
        metrics.loss_count = data.get("loss_count", 0)
        
        return metrics


class StackedEnsemble:
    """
    Stacked ensemble model combining LightGBM, BiLSTM, and CNN models.
    
    The ensemble uses median probability voting to determine the final
    prediction, with online learning capabilities through a partial-fit
    buffer that allows incremental model updates.
    """
    
    def __init__(self, 
                model_dir: str = "models",
                study_db_path: str = "study.sqlite",
                online_learning_buffer_size: int = 100,
                threshold: float = 0.67):
        """
        Initialize the stacked ensemble.
        
        Args:
            model_dir: Directory to save/load models
            study_db_path: Path to Optuna study database
            online_learning_buffer_size: Size of buffer for online learning
            threshold: Probability threshold for positive predictions
        """
        self.model_dir = model_dir
        self.study_db_path = study_db_path
        self.online_learning_buffer_size = online_learning_buffer_size
        self.threshold = threshold
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.lgbm_model = None
        self.lstm_model = None
        self.cnn_model = None
        
        # Online learning buffer
        self._buffer_X = deque(maxlen=online_learning_buffer_size)
        self._buffer_y = deque(maxlen=online_learning_buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Performance tracking
        self.metrics = ModelPerformanceMetrics(model_name="stacked_ensemble")
        self.individual_metrics = {}
        
        # Last prediction info for diagnostics
        self.last_prediction_time = 0.0
        self.last_prediction_probs = {}
        
        # Flag to track if models are loaded
        self.models_loaded = False
    
    def load_models(self) -> bool:
        """
        Load all models from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize LGBM model
            self.lgbm_model = LightGBMModel(name="edge_detection_lgbm")
            lgbm_path = os.path.join(self.model_dir, "lgbm")
            if os.path.exists(lgbm_path):
                self.lgbm_model.load(lgbm_path)
                logger.info(f"Loaded LGBM model from {lgbm_path}")
            else:
                logger.warning(f"LGBM model not found at {lgbm_path}")
                return False
            
            # Initialize LSTM model
            self.lstm_model = BidirectionalLSTMModel(name="edge_detection_lstm")
            lstm_path = os.path.join(self.model_dir, "lstm")
            if os.path.exists(lstm_path):
                self.lstm_model.load(lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
            else:
                logger.warning(f"LSTM model not found at {lstm_path}")
                return False
            
            # Initialize CNN model
            self.cnn_model = OptionSurfaceCNNModel(name="edge_detection_cnn")
            cnn_path = os.path.join(self.model_dir, "cnn")
            if os.path.exists(cnn_path):
                self.cnn_model.load(cnn_path)
                logger.info(f"Loaded CNN model from {cnn_path}")
            else:
                logger.warning(f"CNN model not found at {cnn_path}")
                return False
            
            # Load metrics if available
            metrics_path = os.path.join(self.model_dir, "ensemble_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics_data = json.load(f)
                    self.metrics = ModelPerformanceMetrics.from_dict(metrics_data["ensemble"])
                    
                    # Load individual model metrics
                    for model_name, model_metrics in metrics_data.get("individual", {}).items():
                        self.individual_metrics[model_name] = ModelPerformanceMetrics.from_dict(model_metrics)
            
            self.models_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def save_models(self) -> bool:
        """
        Save all models to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save LGBM model
            if self.lgbm_model:
                lgbm_path = os.path.join(self.model_dir, "lgbm")
                os.makedirs(lgbm_path, exist_ok=True)
                self.lgbm_model.save(lgbm_path)
                logger.info(f"Saved LGBM model to {lgbm_path}")
            
            # Save LSTM model
            if self.lstm_model:
                lstm_path = os.path.join(self.model_dir, "lstm")
                os.makedirs(lstm_path, exist_ok=True)
                self.lstm_model.save(lstm_path)
                logger.info(f"Saved LSTM model to {lstm_path}")
            
            # Save CNN model
            if self.cnn_model:
                cnn_path = os.path.join(self.model_dir, "cnn")
                os.makedirs(cnn_path, exist_ok=True)
                self.cnn_model.save(cnn_path)
                logger.info(f"Saved CNN model to {cnn_path}")
            
            # Save metrics
            metrics_path = os.path.join(self.model_dir, "ensemble_metrics.json")
            metrics_data = {
                "ensemble": self.metrics.to_dict(),
                "individual": {
                    name: metrics.to_dict() 
                    for name, metrics in self.individual_metrics.items()
                }
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def predict(self, 
               features: Dict[str, np.ndarray],
               return_individual_probs: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Generate ensemble prediction for edge probability.
        
        Args:
            features: Dictionary with feature arrays for each model:
                     - 'tabular': Tabular features for LGBM
                     - 'sequence': Sequential features for LSTM
                     - 'surface': Surface grid for CNN
                     - 'metadata' (optional): Additional features for CNN
            return_individual_probs: Whether to return individual model probabilities
            
        Returns:
            Edge probability or tuple of (edge probability, individual probabilities)
        """
        if not self.models_loaded:
            if not self.load_models():
                raise ValueError("Models not loaded and could not be loaded from disk")
        
        start_time = time.time()
        individual_probs = {}
        
        # Get LGBM prediction
        if 'tabular' in features and self.lgbm_model:
            lgbm_prob = self.lgbm_model.predict(features['tabular'])[0]
            individual_probs['lgbm'] = float(lgbm_prob)
        else:
            individual_probs['lgbm'] = 0.5
        
        # Get LSTM prediction
        if 'sequence' in features and self.lstm_model:
            lstm_prob = self.lstm_model.predict(features['sequence'])[0]
            individual_probs['lstm'] = float(lstm_prob)
        else:
            individual_probs['lstm'] = 0.5
        
        # Get CNN prediction
        if 'surface' in features and self.cnn_model:
            if 'metadata' in features:
                cnn_input = (features['surface'], features['metadata'])
            else:
                cnn_input = features['surface']
                
            cnn_prob = self.cnn_model.predict(cnn_input)[0]
            individual_probs['cnn'] = float(cnn_prob)
        else:
            individual_probs['cnn'] = 0.5
        
        # Calculate median probability
        probs = np.array(list(individual_probs.values()))
        edge_prob = float(np.median(probs))
        
        # Update diagnostics
        self.last_prediction_time = time.time() - start_time
        self.last_prediction_probs = individual_probs
        
        if return_individual_probs:
            return edge_prob, individual_probs
        
        return edge_prob
    
    def get_prediction_confidence(self, probs: Dict[str, float]) -> float:
        """
        Calculate confidence level based on agreement among models.
        
        Args:
            probs: Individual model probabilities
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Strategy 1: Standard deviation of predictions (inverted)
        # Lower stdev means higher agreement/confidence
        std_dev = np.std(list(probs.values()))
        inverse_std = 1.0 - min(std_dev, 0.5) / 0.5  # Normalize to 0-1
        
        # Strategy 2: Distance from decision boundary
        # Further from 0.5 means higher confidence
        median_prob = np.median(list(probs.values()))
        distance = abs(median_prob - 0.5) * 2  # Normalize to 0-1
        
        # Combine strategies
        confidence = (inverse_std + distance) / 2.0
        
        return min(confidence, 1.0)
    
    def add_to_buffer(self, X: Dict[str, np.ndarray], y: float) -> None:
        """
        Add sample to online learning buffer.
        
        Args:
            X: Feature dictionary with same structure as predict()
            y: Target (1.0 for win, 0.0 for loss)
        """
        with self._buffer_lock:
            self._buffer_X.append(X)
            self._buffer_y.append(y)
            
            # If buffer is full, trigger partial fit for tree-based models
            if len(self._buffer_X) >= self.online_learning_buffer_size:
                self._partial_fit_from_buffer()
    
    def _partial_fit_from_buffer(self) -> None:
        """Update tree-based models with buffered samples."""
        if len(self._buffer_X) == 0 or len(self._buffer_y) == 0:
            return
            
        with self._buffer_lock:
            try:
                # Only perform partial fit on tree-based models (LGBM)
                if self.lgbm_model and 'tabular' in self._buffer_X[0]:
                    # Extract tabular features from buffer
                    tabular_X = np.vstack([x['tabular'] for x in self._buffer_X if 'tabular' in x])
                    tabular_y = np.array([y for i, y in enumerate(self._buffer_y) 
                                         if i < len(self._buffer_X) and 'tabular' in self._buffer_X[i]])
                    
                    if len(tabular_X) > 0 and len(tabular_y) > 0:
                        # Partial fit LGBM model
                        update_info = self.lgbm_model.train(
                            tabular_X, tabular_y, 
                            num_boost_round=10,  # Small number of iterations
                            class_weight={0: 2.0, 1: 1.0}  # More weight on losses
                        )
                        logger.info(f"Performed partial fit on LGBM with {len(tabular_X)} samples")
                
                # Clear buffer after fitting
                self._buffer_X.clear()
                self._buffer_y.clear()
                
            except Exception as e:
                logger.error(f"Error during partial fit: {e}")
    
    def train(self, 
             train_data: Dict[str, Dict[str, np.ndarray]], 
             train_labels: np.ndarray,
             val_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
             val_labels: Optional[np.ndarray] = None,
             optimize_lgbm: bool = True) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            train_data: Dictionary mapping model names to feature dictionaries
                      for each model type
            train_labels: Target labels (1 for win, 0 for loss)
            val_data: Optional validation data in same format as train_data
            val_labels: Optional validation labels
            optimize_lgbm: Whether to run Optuna hyperparameter optimization for LGBM
            
        Returns:
            Dictionary with training results
        """
        results = {}
        
        # Train LGBM model
        if 'lgbm' in train_data and self.lgbm_model:
            if optimize_lgbm:
                lgbm_params = self._optimize_lgbm_params(
                    train_data['lgbm'], train_labels,
                    val_data['lgbm'] if val_data and 'lgbm' in val_data else None,
                    val_labels
                )
                self.lgbm_model.model_params.update(lgbm_params)
            
            lgbm_val_data = None
            if val_data and 'lgbm' in val_data:
                lgbm_val_data = (val_data['lgbm'], val_labels)
                
            # Use class weighting to address imbalance (assume more losses than wins)
            class_counts = np.bincount(train_labels.astype(int))
            if len(class_counts) > 1 and class_counts[0] > 0 and class_counts[1] > 0:
                class_weight = {
                    0: 1.0,  # Loss class
                    1: class_counts[0] / class_counts[1]  # Win class (weighted higher if less frequent)
                }
            else:
                class_weight = None
                
            lgbm_results = self.lgbm_model.train(
                train_data['lgbm'], train_labels,
                val_data=lgbm_val_data,
                class_weight=class_weight
            )
            results['lgbm'] = lgbm_results
        
        # Train LSTM model
        if 'lstm' in train_data and self.lstm_model:
            lstm_val_data = None
            if val_data and 'lstm' in val_data:
                lstm_val_data = (val_data['lstm'], val_labels)
                
            # Add class weighting for imbalance
            lstm_results = self.lstm_model.train(
                train_data['lstm'], train_labels,
                val_data=lstm_val_data,
                class_weight=class_weight  # Same as LGBM
            )
            results['lstm'] = lstm_results
        
        # Train CNN model
        if 'cnn' in train_data and self.cnn_model:
            cnn_val_data = None
            if val_data and 'cnn' in val_data:
                cnn_val_data = (val_data['cnn'], val_labels)
                
            # Add class weighting for imbalance
            cnn_results = self.cnn_model.train(
                train_data['cnn'], train_labels,
                val_data=cnn_val_data,
                class_weight=class_weight  # Same as LGBM
            )
            results['cnn'] = cnn_results
        
        # Save trained models
        self.save_models()
        
        return results 
    
    def _optimize_lgbm_params(self,
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Optimized parameters dictionary
        """
        def objective(trial):
            # Define the hyperparameters to optimize
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
                "verbose": -1
            }
            
            # Create dataset
            if X_val is not None and y_val is not None:
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Use class weighting to address imbalance
                class_counts = np.bincount(y_train.astype(int))
                if len(class_counts) > 1 and class_counts[0] > 0 and class_counts[1] > 0:
                    weight_ratio = class_counts[0] / class_counts[1]
                    class_weight = {0: 1.0, 1: weight_ratio}
                    params["class_weight"] = class_weight
                
                # Train with early stopping
                gbm = lgb.train(
                    params,
                    train_data,
                    valid_sets=[valid_data],
                    num_boost_round=1000,
                    early_stopping_rounds=50,
                    callbacks=[
                        lgb.callback.record_evaluation({}),
                        lgb.callback.early_stopping(50, verbose=False)
                    ]
                )
                
                # Return the validation score
                return gbm.best_score["valid_0"]["auc"]
            else:
                # No validation data, use cross-validation
                scores = lgb.cv(
                    params,
                    lgb.Dataset(X_train, label=y_train),
                    num_boost_round=500,
                    nfold=5,
                    stratified=True,
                    early_stopping_rounds=30,
                    seed=42,
                    callbacks=[lgb.callback.early_stopping(30, verbose=False)]
                )
                
                # Return the mean of the best CV scores
                return max(scores["auc-mean"])
        
        # Create or load the study
        try:
            if os.path.exists(self.study_db_path):
                # Load existing study
                study = optuna.load_study(
                    study_name="lightgbm_optimization",
                    storage=f"sqlite:///{self.study_db_path}"
                )
            else:
                # Create new study with MedianPruner
                study = optuna.create_study(
                    study_name="lightgbm_optimization",
                    storage=f"sqlite:///{self.study_db_path}",
                    direction="maximize",
                    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=50)
                )
                
            # Run optimization (50 trials by default, but can do up to 200 overnight)
            study.optimize(objective, n_trials=50, timeout=3600)  # 1 hour timeout
            
            # Get the best parameters
            best_params = study.best_params
            
            # Add fixed parameters
            best_params["objective"] = "binary"
            best_params["metric"] = "auc"
            best_params["verbose"] = -1
            
            logger.info(f"Optimized LightGBM parameters: {best_params}")
            logger.info(f"Best AUC score: {study.best_value:.4f}")
            
            return best_params
        except Exception as e:
            logger.error(f"Error during parameter optimization: {e}")
            
            # Return default parameters
            return {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1
            }
    
    def update_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Update performance metrics based on predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= self.threshold).astype(int)
        
        # Update ensemble metrics
        self.metrics.accuracy = accuracy_score(y_true, y_pred_binary)
        self.metrics.precision = precision_score(y_true, y_pred_binary, zero_division=0)
        self.metrics.recall = recall_score(y_true, y_pred_binary, zero_division=0)
        self.metrics.f1_score = f1_score(y_true, y_pred_binary, zero_division=0)
        self.metrics.auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        self.metrics.log_loss = log_loss(y_true, y_pred, eps=1e-7)
        self.metrics.mcc = matthews_corrcoef(y_true, y_pred_binary)
        
        # Update trade counts
        self.metrics.trades_evaluated += len(y_true)
        self.metrics.win_count += np.sum(y_true == 1)
        self.metrics.loss_count += np.sum(y_true == 0)
        
        # Update timestamp
        self.metrics.timestamp = datetime.now()
        
    def evaluate(self, X: Dict[str, Dict[str, np.ndarray]], y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate ensemble on test data.
        
        Args:
            X: Test features in the same format as train()
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions from each model and the ensemble
        predictions = {}
        ensemble_probs = []
        
        # Get LGBM predictions
        if 'lgbm' in X and self.lgbm_model:
            lgbm_pred = self.lgbm_model.predict(X['lgbm'])
            predictions['lgbm'] = lgbm_pred
            ensemble_probs.append(lgbm_pred)
        
        # Get LSTM predictions
        if 'lstm' in X and self.lstm_model:
            lstm_pred = self.lstm_model.predict(X['lstm'])
            predictions['lstm'] = lstm_pred
            ensemble_probs.append(lstm_pred)
        
        # Get CNN predictions
        if 'cnn' in X and self.cnn_model:
            if 'metadata' in X:
                cnn_pred = self.cnn_model.predict((X['cnn'], X['metadata']))
            else:
                cnn_pred = self.cnn_model.predict(X['cnn'])
            predictions['cnn'] = cnn_pred
            ensemble_probs.append(cnn_pred)
        
        # Calculate ensemble predictions (median probability)
        if ensemble_probs:
            ensemble_pred = np.median(ensemble_probs, axis=0)
        else:
            ensemble_pred = np.full(len(y), 0.5)
        
        # Update metrics
        self.update_metrics(y, ensemble_pred)
        
        # Calculate individual model metrics
        for model_name, model_pred in predictions.items():
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, matthews_corrcoef
            
            # Convert probabilities to binary predictions
            model_pred_binary = (model_pred >= self.threshold).astype(int)
            
            # Create or get metrics object
            if model_name not in self.individual_metrics:
                self.individual_metrics[model_name] = ModelPerformanceMetrics(model_name=model_name)
            
            metrics = self.individual_metrics[model_name]
            
            # Update metrics
            metrics.accuracy = accuracy_score(y, model_pred_binary)
            metrics.precision = precision_score(y, model_pred_binary, zero_division=0)
            metrics.recall = recall_score(y, model_pred_binary, zero_division=0)
            metrics.f1_score = f1_score(y, model_pred_binary, zero_division=0)
            metrics.auc = roc_auc_score(y, model_pred) if len(np.unique(y)) > 1 else 0.5
            metrics.log_loss = log_loss(y, model_pred, eps=1e-7)
            metrics.mcc = matthews_corrcoef(y, model_pred_binary)
            
            # Update trade counts
            metrics.trades_evaluated += len(y)
            metrics.win_count += np.sum(y == 1)
            metrics.loss_count += np.sum(y == 0)
            
            # Update timestamp
            metrics.timestamp = datetime.now()
        
        # Save updated metrics
        self.save_models()
        
        # Return evaluation results
        return {
            "ensemble": self.metrics.to_dict(),
            "individual": {
                name: metrics.to_dict()
                for name, metrics in self.individual_metrics.items()
            }
        }
    
    def predict_with_confidence(self, features: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Generate prediction with confidence score.
        
        Args:
            features: Feature dictionary as in predict()
            
        Returns:
            Tuple of (edge probability, confidence)
        """
        edge_prob, individual_probs = self.predict(features, return_individual_probs=True)
        confidence = self.get_prediction_confidence(individual_probs)
        
        return edge_prob, confidence
    
    def get_threshold_metrics(self, X: Dict[str, Dict[str, np.ndarray]], 
                            y: np.ndarray, 
                            thresholds: List[float]) -> pd.DataFrame:
        """
        Calculate metrics at different threshold values.
        
        Args:
            X: Test features
            y: Test labels
            thresholds: List of threshold values to evaluate
            
        Returns:
            DataFrame with metrics for each threshold
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Get ensemble predictions
        edge_probs = []
        
        # Get individual model predictions
        if 'lgbm' in X and self.lgbm_model:
            edge_probs.append(self.lgbm_model.predict(X['lgbm']))
            
        if 'lstm' in X and self.lstm_model:
            edge_probs.append(self.lstm_model.predict(X['lstm']))
            
        if 'cnn' in X and self.cnn_model:
            if 'metadata' in X:
                edge_probs.append(self.cnn_model.predict((X['cnn'], X['metadata'])))
            else:
                edge_probs.append(self.cnn_model.predict(X['cnn']))
        
        # Calculate median probability
        ensemble_probs = np.median(edge_probs, axis=0) if edge_probs else np.full(len(y), 0.5)
        
        # Calculate metrics for each threshold
        results = []
        for thresh in thresholds:
            y_pred = (ensemble_probs >= thresh).astype(int)
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            # Calculate expected value
            signal_rate = np.mean(y_pred)  # How often we predict a signal
            true_win_rate = np.sum((y == 1) & (y_pred == 1)) / np.sum(y_pred) if np.sum(y_pred) > 0 else 0
            
            # Calculate expectation assuming 1.0 R:R
            expectation = (2 * true_win_rate - 1) if true_win_rate > 0 else 0
            
            results.append({
                'threshold': thresh,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'signal_rate': signal_rate,
                'true_win_rate': true_win_rate,
                'expectation': expectation
            })
        
        return pd.DataFrame(results)


def create_default_ensemble(model_dir: str = "models", 
                           study_db_path: str = "study.sqlite") -> StackedEnsemble:
    """
    Create and initialize a default stacked ensemble.
    
    Args:
        model_dir: Directory to save/load models
        study_db_path: Path to Optuna study database
        
    Returns:
        Initialized StackedEnsemble
    """
    ensemble = StackedEnsemble(
        model_dir=model_dir,
        study_db_path=study_db_path,
        online_learning_buffer_size=100,
        threshold=0.67
    )
    
    # Try to load existing models
    if not ensemble.load_models():
        logger.warning("Could not load existing models, will need to train new ones")
    
    return ensemble 