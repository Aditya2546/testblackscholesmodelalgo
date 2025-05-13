"""
ML Models for Edge Detection

This module implements three model types for edge detection in options trading:
1. LightGBM gradient boosting tree
2. Bidirectional LSTM for sequential data
3. Convolutional Neural Network (CNN) for option surface visualization

All models are designed to predict trading edge (probability of profitable trade).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Bidirectional, Conv2D, MaxPooling2D, 
    Flatten, Dropout, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json
import pickle
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all edge detection models.
    
    Defines common interface and utility methods for all model types.
    """
    
    def __init__(self, model_params: Dict[str, Any] = None, name: str = "base_model"):
        """
        Initialize model with parameters.
        
        Args:
            model_params: Model-specific parameters
            name: Model name for saving/loading
        """
        self.model_params = model_params or {}
        self.name = name
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.feature_importance = None
    
    @abstractmethod
    def build_model(self, input_shape: Union[int, Tuple[int, ...]], **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input features
            **kwargs: Additional model parameters
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Model predictions
        """
        pass
    
    def save(self, directory: str) -> None:
        """
        Save model to disk.
        
        Args:
            directory: Directory to save model to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "model_type": self.__class__.__name__,
            "name": self.name,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "model_params": self.model_params
        }
        
        with open(os.path.join(directory, f"{self.name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved {self.name} metadata to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load model from disk.
        
        Args:
            directory: Directory to load model from
        """
        # Load metadata
        try:
            with open(os.path.join(directory, f"{self.name}_metadata.json"), "r") as f:
                metadata = json.load(f)
                
            self.name = metadata["name"]
            self.is_trained = metadata["is_trained"]
            self.feature_names = metadata["feature_names"]
            self.model_params.update(metadata["model_params"])
            
            logger.info(f"Loaded {self.name} metadata from {directory}")
        except FileNotFoundError:
            logger.warning(f"No metadata found for {self.name} in {directory}")


class LightGBMModel(BaseModel):
    """
    LightGBM gradient boosting tree model for edge detection.
    """
    
    def __init__(self, model_params: Dict[str, Any] = None, name: str = "lightgbm_model"):
        """
        Initialize LightGBM model with parameters.
        
        Args:
            model_params: LightGBM parameters
            name: Model name
        """
        super().__init__(model_params, name)
        
        # Set default LightGBM parameters if not provided
        if not model_params:
            self.model_params = {
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
    
    def build_model(self, input_shape: int, **kwargs) -> None:
        """
        Build LightGBM model with specified parameters.
        
        Args:
            input_shape: Number of features
            **kwargs: Additional model parameters
        """
        # LightGBM model is built during training
        self.feature_names = kwargs.get("feature_names", None)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train LightGBM model on provided data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters including:
                - feature_names: List of feature names
                - val_data: Tuple of (X_val, y_val) for validation
                - num_boost_round: Number of boosting iterations
                - early_stopping_rounds: Early stopping parameter
            
        Returns:
            Dictionary of training metrics
        """
        feature_names = kwargs.get("feature_names", self.feature_names)
        val_data = kwargs.get("val_data", None)
        num_boost_round = kwargs.get("num_boost_round", 1000)
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        else:
            # If no validation data provided, create it from training data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_dataset = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        # Create callback for capturing evaluation history
        history = {}
        
        def callback(env):
            """Callback to capture evaluation history."""
            if 'iteration' not in history:
                history['iteration'] = []
            history['iteration'].append(env.iteration)
            
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                if f"{data_name}_{eval_name}" not in history:
                    history[f"{data_name}_{eval_name}"] = []
                history[f"{data_name}_{eval_name}"].append(result)
        
        # Train model
        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_dataset],
            valid_names=['train', 'valid'],
            callbacks=[callback, lgb.early_stopping(early_stopping_rounds)],
            feature_name=feature_names
        )
        
        # Extract feature importance
        self.feature_importance = {
            name: importance for name, importance in zip(
                feature_names or [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importance()
            )
        }
        
        # Set model as trained
        self.is_trained = True
        self.feature_names = feature_names
        
        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "feature_importance": self.feature_importance,
            "history": history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using trained LightGBM model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted probability of positive class
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def save(self, directory: str) -> None:
        """
        Save LightGBM model to disk.
        
        Args:
            directory: Directory to save model to
        """
        super().save(directory)
        
        if self.is_trained and self.model is not None:
            # Save LightGBM model
            self.model.save_model(os.path.join(directory, f"{self.name}.txt"))
            
            # Save feature importance
            if self.feature_importance is not None:
                with open(os.path.join(directory, f"{self.name}_feature_importance.json"), "w") as f:
                    json.dump(self.feature_importance, f, indent=2)
                    
            logger.info(f"Saved {self.name} model to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load LightGBM model from disk.
        
        Args:
            directory: Directory to load model from
        """
        super().load(directory)
        
        model_path = os.path.join(directory, f"{self.name}.txt")
        if os.path.exists(model_path):
            self.model = lgb.Booster(model_file=model_path)
            self.is_trained = True
            
            # Load feature importance
            importance_path = os.path.join(directory, f"{self.name}_feature_importance.json")
            if os.path.exists(importance_path):
                with open(importance_path, "r") as f:
                    self.feature_importance = json.load(f)
                    
            logger.info(f"Loaded {self.name} model from {directory}")
        else:
            logger.warning(f"No model file found for {self.name} in {directory}")


class BidirectionalLSTMModel(BaseModel):
    """
    Bidirectional LSTM model for sequential edge detection.
    
    This model is designed to capture temporal patterns in market data
    using bidirectional LSTMs to process sequences of features.
    """
    
    def __init__(self, model_params: Dict[str, Any] = None, name: str = "lstm_model"):
        """
        Initialize Bidirectional LSTM model with parameters.
        
        Args:
            model_params: LSTM model parameters
            name: Model name
        """
        super().__init__(model_params, name)
        
        # Set default LSTM parameters if not provided
        if not model_params:
            self.model_params = {
                "lstm_units": [64, 32],
                "dense_units": [32, 16],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            }
    
    def build_model(self, input_shape: Tuple[int, int], **kwargs) -> None:
        """
        Build Bidirectional LSTM model with specified parameters.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, num_features)
            **kwargs: Additional model parameters
        """
        seq_length, n_features = input_shape
        
        # Create model
        inputs = Input(shape=(seq_length, n_features))
        x = inputs
        
        # Add LSTM layers
        lstm_units = self.model_params.get("lstm_units", [64, 32])
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # Return sequences for all except last
            x = Bidirectional(LSTM(units, return_sequences=return_sequences))(x)
            x = Dropout(self.model_params.get("dropout_rate", 0.2))(x)
        
        # Add dense layers
        dense_units = self.model_params.get("dense_units", [32, 16])
        for units in dense_units:
            x = Dense(units, activation="relu")(x)
            x = Dropout(self.model_params.get("dropout_rate", 0.2))(x)
        
        # Output layer
        outputs = Dense(1, activation="sigmoid")(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.model_params.get("learning_rate", 0.001)),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        
        # Save feature names if provided
        self.feature_names = kwargs.get("feature_names", None)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train Bidirectional LSTM model on provided data.
        
        Args:
            X: Training features (shape: [samples, seq_length, features])
            y: Training labels
            **kwargs: Additional training parameters including:
                - feature_names: List of feature names
                - val_data: Tuple of (X_val, y_val) for validation
                - batch_size: Batch size for training
                - epochs: Number of epochs to train for
                - model_directory: Directory to save best model to
            
        Returns:
            Dictionary of training metrics and history
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Extract training parameters
        feature_names = kwargs.get("feature_names", self.feature_names)
        val_data = kwargs.get("val_data", None)
        batch_size = kwargs.get("batch_size", self.model_params.get("batch_size", 32))
        epochs = kwargs.get("epochs", self.model_params.get("epochs", 100))
        model_directory = kwargs.get("model_directory", None)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        ]
        
        if model_directory:
            os.makedirs(model_directory, exist_ok=True)
            model_path = os.path.join(model_directory, f"{self.name}_best.h5")
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            )
        
        # Create validation data if not provided
        if val_data is None:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            val_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
            X_val, y_val = val_data
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Set model as trained
        self.is_trained = True
        self.feature_names = feature_names
        
        return {
            "history": history.history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using trained Bidirectional LSTM model.
        
        Args:
            X: Features to predict on (shape: [samples, seq_length, features])
            
        Returns:
            Predicted probability of positive class
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X).flatten()
    
    def save(self, directory: str) -> None:
        """
        Save Bidirectional LSTM model to disk.
        
        Args:
            directory: Directory to save model to
        """
        super().save(directory)
        
        if self.is_trained and self.model is not None:
            # Save Keras model
            self.model.save(os.path.join(directory, f"{self.name}.h5"))
            logger.info(f"Saved {self.name} model to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load Bidirectional LSTM model from disk.
        
        Args:
            directory: Directory to load model from
        """
        super().load(directory)
        
        model_path = os.path.join(directory, f"{self.name}.h5")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            logger.info(f"Loaded {self.name} model from {directory}")
        else:
            logger.warning(f"No model file found for {self.name} in {directory}")


class OptionSurfaceCNNModel(BaseModel):
    """
    CNN model for options surface visualization.
    
    This model treats the option implied volatility surface (strike x tenor grid)
    as a 2D image and applies convolutional layers to extract patterns.
    """
    
    def __init__(self, model_params: Dict[str, Any] = None, name: str = "cnn_model"):
        """
        Initialize Option Surface CNN model with parameters.
        
        Args:
            model_params: CNN model parameters
            name: Model name
        """
        super().__init__(model_params, name)
        
        # Set default CNN parameters if not provided
        if not model_params:
            self.model_params = {
                "conv_filters": [32, 64, 128],
                "conv_kernel_size": 3,
                "pool_size": 2,
                "dense_units": [128, 64],
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            }
    
    def build_model(self, input_shape: Tuple[int, int, int], 
                   additional_features_shape: Optional[int] = None, **kwargs) -> None:
        """
        Build CNN model with specified parameters.
        
        Args:
            input_shape: Shape of input option surface (height, width, channels)
            additional_features_shape: Shape of additional features to concatenate
            **kwargs: Additional model parameters
        """
        # CNN for the option surface image
        surface_input = Input(shape=input_shape, name="surface_input")
        x = surface_input
        
        # Add convolutional layers
        conv_filters = self.model_params.get("conv_filters", [32, 64, 128])
        conv_kernel_size = self.model_params.get("conv_kernel_size", 3)
        pool_size = self.model_params.get("pool_size", 2)
        
        for filters in conv_filters:
            x = Conv2D(filters, kernel_size=conv_kernel_size, activation="relu", padding="same")(x)
            x = MaxPooling2D(pool_size=pool_size)(x)
            x = Dropout(self.model_params.get("dropout_rate", 0.3))(x)
        
        # Flatten the output
        x = Flatten()(x)
        
        # If additional features are provided, create a separate input for them
        if additional_features_shape is not None:
            features_input = Input(shape=(additional_features_shape,), name="features_input")
            x = Concatenate()([x, features_input])
            inputs = [surface_input, features_input]
        else:
            inputs = surface_input
        
        # Add dense layers
        dense_units = self.model_params.get("dense_units", [128, 64])
        for units in dense_units:
            x = Dense(units, activation="relu")(x)
            x = Dropout(self.model_params.get("dropout_rate", 0.3))(x)
        
        # Output layer
        outputs = Dense(1, activation="sigmoid")(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.model_params.get("learning_rate", 0.001)),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        
        # Save feature names if provided
        self.feature_names = kwargs.get("feature_names", None)
    
    def train(self, X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
             y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train CNN model on provided data.
        
        Args:
            X: Training features, either:
               - Single numpy array with shape [samples, height, width, channels]
               - Tuple of (surface, features) for surface and additional features
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics and history
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Extract training parameters
        feature_names = kwargs.get("feature_names", self.feature_names)
        val_data = kwargs.get("val_data", None)
        batch_size = kwargs.get("batch_size", self.model_params.get("batch_size", 32))
        epochs = kwargs.get("epochs", self.model_params.get("epochs", 100))
        model_directory = kwargs.get("model_directory", None)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        ]
        
        if model_directory:
            os.makedirs(model_directory, exist_ok=True)
            model_path = os.path.join(model_directory, f"{self.name}_best.h5")
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            )
        
        # Create validation data if not provided
        if val_data is None:
            if isinstance(X, tuple):
                # Split both surface and features
                X_surface, X_features = X
                X_train_surface, X_val_surface, y_train, y_val = train_test_split(
                    X_surface, y, test_size=0.2, random_state=42
                )
                X_train_features, X_val_features, _, _ = train_test_split(
                    X_features, y, test_size=0.2, random_state=42
                )
                X_train = (X_train_surface, X_train_features)
                val_data = ((X_val_surface, X_val_features), y_val)
            else:
                # Just split the surface
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                val_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Set model as trained
        self.is_trained = True
        self.feature_names = feature_names
        
        return {
            "history": history.history
        }
    
    def predict(self, X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Generate predictions using trained CNN model.
        
        Args:
            X: Features to predict on, either:
               - Single numpy array with shape [samples, height, width, channels]
               - Tuple of (surface, features) for surface and additional features
            
        Returns:
            Predicted probability of positive class
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X).flatten()
    
    def save(self, directory: str) -> None:
        """
        Save CNN model to disk.
        
        Args:
            directory: Directory to save model to
        """
        super().save(directory)
        
        if self.is_trained and self.model is not None:
            # Save Keras model
            self.model.save(os.path.join(directory, f"{self.name}.h5"))
            logger.info(f"Saved {self.name} model to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load CNN model from disk.
        
        Args:
            directory: Directory to load model from
        """
        super().load(directory)
        
        model_path = os.path.join(directory, f"{self.name}.h5")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.is_trained = True
            logger.info(f"Loaded {self.name} model from {directory}")
        else:
            logger.warning(f"No model file found for {self.name} in {directory}")


# Helper functions for data preparation

def prepare_sequence_data(features: pd.DataFrame, 
                         target: pd.Series,
                         sequence_length: int = 120,
                         step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequence data for LSTM models.
    
    Args:
        features: DataFrame of features
        target: Series of target values
        sequence_length: Length of sequences to generate
        step: Step size for sliding window
        
    Returns:
        Tuple of (X, y) with X as sequence data and y as targets
    """
    # Convert to numpy arrays
    X = features.values
    y = target.values
    
    # Create sequences
    X_seq = []
    y_seq = []
    
    for i in range(0, len(X) - sequence_length, step):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


def create_surface_grid(options_data: pd.DataFrame,
                       strikes: List[float],
                       tenors: List[float],
                       field: str = "implied_volatility") -> np.ndarray:
    """
    Create a grid of option data suitable for CNN processing.
    
    Args:
        options_data: DataFrame of options data
        strikes: List of strike prices to include in grid
        tenors: List of tenors (days to expiration) to include in grid
        field: Field to grid (e.g., "implied_volatility", "delta", etc.)
        
    Returns:
        3D array with dimensions [timestamps, strikes, tenors]
    """
    if not all(col in options_data.columns for col in ["strike", "days_to_expiration", field]):
        raise ValueError(f"options_data must contain columns 'strike', 'days_to_expiration', and '{field}'")
    
    # Get unique timestamps
    timestamps = options_data.index.unique()
    
    # Create empty grid
    grid = np.zeros((len(timestamps), len(strikes), len(tenors)))
    
    # Fill grid
    for i, ts in enumerate(timestamps):
        ts_data = options_data.loc[ts]
        
        for j, strike in enumerate(strikes):
            for k, tenor in enumerate(tenors):
                # Find closest match in data
                strike_diff = np.abs(ts_data["strike"] - strike)
                tenor_diff = np.abs(ts_data["days_to_expiration"] - tenor)
                
                # Get combined distance
                distance = np.sqrt(strike_diff**2 + tenor_diff**2)
                
                if len(distance) > 0:
                    # Get value from closest match
                    closest_idx = distance.argmin()
                    grid[i, j, k] = ts_data.iloc[closest_idx][field]
    
    # Add channel dimension for CNN
    return grid.reshape(grid.shape[0], grid.shape[1], grid.shape[2], 1)