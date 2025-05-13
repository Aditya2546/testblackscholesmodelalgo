#!/usr/bin/env python
"""
Enhanced ML Model for Options Trading

This module provides an advanced machine learning model specifically designed
for options trading prediction with fast learning capabilities.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedOptionsMLModel:
    """
    Advanced machine learning model for options trading with fast learning capabilities.
    Supports multiple model types and provides online learning for rapid adaptation.
    """
    
    def __init__(self, model_type: str = "hybrid", model_path: str = None):
        """
        Initialize the enhanced ML model.
        
        Args:
            model_type: Type of model to use ('rf' for Random Forest, 
                        'lstm' for LSTM, 'hybrid' for combined approach)
            model_path: Path to pre-trained model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.models = {}
        self.feature_columns = None
        self.feature_importance = None
        self.online_buffer = {
            'X': [],
            'y': [],
            'weight': []
        }
        self.buffer_size = 1000  # Maximum buffer size
        self.min_samples_retrain = 50  # Minimum samples for retraining
        self.model_initialized = False
        self.scaler = None
        
        # Load required libraries
        try:
            # Core ML libraries
            import joblib
            self.joblib = joblib
            
            from sklearn.ensemble import RandomForestClassifier
            self.RandomForestClassifier = RandomForestClassifier
            
            from sklearn.preprocessing import StandardScaler
            self.StandardScaler = StandardScaler
            
            # Try to load advanced libraries
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential, load_model, save_model
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                
                # Store TensorFlow references
                self.tf = tf
                self.tf_keras_models = {
                    'Sequential': Sequential,
                    'load_model': load_model,
                    'save_model': save_model
                }
                self.tf_keras_layers = {
                    'LSTM': LSTM,
                    'Dense': Dense,
                    'Dropout': Dropout
                }
                self.has_tensorflow = True
                logger.info("TensorFlow successfully loaded")
            except ImportError:
                self.has_tensorflow = False
                if model_type in ['lstm', 'hybrid']:
                    logger.warning("TensorFlow not available. Install with: pip install tensorflow")
                    if model_type == 'lstm':
                        logger.warning("Falling back to RandomForest model")
                        self.model_type = 'rf'
            
            # Initialize models based on type
            self._initialize_models()
            
            # Load pre-trained model if provided
            if model_path:
                self.load_model(model_path)
                
        except ImportError as e:
            logger.error(f"Required libraries not available: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize the selected models"""
        # Base model (Random Forest) - always used for feature importance
        self.models['rf'] = self.RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        )
        
        # Initialize LSTM if requested and available
        if self.model_type in ['lstm', 'hybrid'] and self.has_tensorflow:
            self.models['lstm'] = self._create_lstm_model()
            
        # Initialize scaler for numerical stability
        self.scaler = self.StandardScaler()
        
        # Define default feature columns if not loaded from existing model
        if not self.feature_columns:
            self.feature_columns = [
                'option_type_C', 'option_type_P',
                'moneyness', 'relative_strike', 'days_to_expiry',
                'implied_volatility', 'delta', 'gamma', 'theta', 'vega',
                'bid_ask_spread', 'bid_ask_spread_percent', 
                'volume', 'open_interest', 'volume_oi_ratio'
            ]
        
        self.model_initialized = True
    
    def _create_lstm_model(self, input_dim: int = 15):
        """
        Create an LSTM model for time series prediction.
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Compiled LSTM model
        """
        model = self.tf_keras_models['Sequential']()
        
        # Add LSTM layer
        model.add(self.tf_keras_layers['LSTM'](
            64, 
            input_shape=(1, input_dim),
            return_sequences=True
        ))
        model.add(self.tf_keras_layers['Dropout'](0.2))
        
        # Add second LSTM layer
        model.add(self.tf_keras_layers['LSTM'](32, return_sequences=False))
        model.add(self.tf_keras_layers['Dropout'](0.2))
        
        # Add output layer
        model.add(self.tf_keras_layers['Dense'](1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare features for model input.
        
        Args:
            data: Input data (dictionary or DataFrame)
            
        Returns:
            DataFrame with prepared features
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0  # Default value
        
        # Reorder columns to match model expectations
        data = data[self.feature_columns]
        
        return data
    
    def add_training_data(self, features: Dict, outcome: Union[bool, float], weight: float = 1.0):
        """
        Add a single training example to the online learning buffer.
        
        Args:
            features: Feature dictionary for this example
            outcome: True/1.0 for profit, False/0.0 for loss, or probability value
            weight: Sample weight for importance (higher for more significant trades)
        """
        # Convert features to the expected format
        if isinstance(features, dict):
            # Prepare feature vector
            X = self.prepare_features(features)
            
            # Convert to values array
            X_values = X.values
            
            # Add to buffer
            self.online_buffer['X'].append(X_values[0])
            self.online_buffer['y'].append(1.0 if outcome else 0.0)
            self.online_buffer['weight'].append(weight)
            
            # Keep buffer under size limit (remove oldest first)
            if len(self.online_buffer['X']) > self.buffer_size:
                self.online_buffer['X'].pop(0)
                self.online_buffer['y'].pop(0)
                self.online_buffer['weight'].pop(0)
                
            # Retrain if buffer has enough samples
            if len(self.online_buffer['y']) >= self.min_samples_retrain:
                self.retrain_from_buffer()
    
    def add_batch_training_data(self, features_list: List[Dict], outcomes: List[Union[bool, float]], 
                               weights: Optional[List[float]] = None):
        """
        Add a batch of training examples to the online learning buffer.
        
        Args:
            features_list: List of feature dictionaries
            outcomes: List of outcomes (True/1.0 for profit, False/0.0 for loss)
            weights: Optional list of sample weights
        """
        if weights is None:
            weights = [1.0] * len(features_list)
            
        for i, (features, outcome) in enumerate(zip(features_list, outcomes)):
            weight = weights[i] if i < len(weights) else 1.0
            self.add_training_data(features, outcome, weight)
    
    def retrain_from_buffer(self, force: bool = False) -> bool:
        """
        Retrain models using the current online buffer data.
        
        Args:
            force: Force retraining even if buffer is small
            
        Returns:
            True if retraining was performed, False otherwise
        """
        if not self.model_initialized:
            logger.warning("Cannot retrain - models not initialized")
            return False
            
        # Check if we have enough data
        buffer_size = len(self.online_buffer['y'])
        if buffer_size < self.min_samples_retrain and not force:
            logger.info(f"Not enough data for retraining ({buffer_size}/{self.min_samples_retrain})")
            return False
            
        # Convert buffer to numpy arrays
        X = np.array(self.online_buffer['X'])
        y = np.array(self.online_buffer['y'])
        weights = np.array(self.online_buffer['weight'])
        
        try:
            # Scale features for better performance
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.models['rf'].fit(X_scaled, y, sample_weight=weights)
            
            # Update feature importance
            self.feature_importance = {
                feature: importance for feature, importance in 
                zip(self.feature_columns, self.models['rf'].feature_importances_)
            }
            
            # Train LSTM model if available
            if 'lstm' in self.models and self.has_tensorflow:
                # Reshape for LSTM [samples, time_steps, features]
                X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                
                # Train LSTM model
                self.models['lstm'].fit(
                    X_lstm, y,
                    sample_weight=weights,
                    epochs=10,
                    batch_size=32,
                    verbose=0
                )
            
            logger.info(f"Models retrained with {buffer_size} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False
    
    def predict(self, features: Union[Dict, pd.DataFrame]) -> float:
        """
        Predict the probability of profit for given option features.
        
        Args:
            features: Feature dictionary or DataFrame
            
        Returns:
            Probability of profit (0.0 to 1.0)
        """
        if not self.model_initialized:
            logger.warning("Cannot predict - models not initialized")
            return 0.5  # Default neutral prediction
            
        try:
            # Prepare features for prediction
            X = self.prepare_features(features)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from each model
            predictions = {}
            
            # Random Forest prediction
            if 'rf' in self.models:
                rf_pred = self.models['rf'].predict_proba(X_scaled)
                predictions['rf'] = rf_pred[0][1]  # Probability of class 1
            
            # LSTM prediction if available
            if 'lstm' in self.models and self.has_tensorflow:
                # Reshape for LSTM [samples, time_steps, features]
                X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                
                # Get prediction
                lstm_pred = self.models['lstm'].predict(X_lstm, verbose=0)
                predictions['lstm'] = lstm_pred[0][0]
            
            # Combine predictions based on model type
            if self.model_type == 'rf' or not self.has_tensorflow:
                return predictions['rf']
            elif self.model_type == 'lstm':
                return predictions['lstm']
            else:  # hybrid
                # Weighted average based on feature importance
                rf_weight = 0.6
                lstm_weight = 0.4
                return (rf_weight * predictions['rf'] + lstm_weight * predictions['lstm'])
                
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            return 0.5  # Default neutral prediction
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model (default is timestamped file in 'models' dir)
            
        Returns:
            Path where model was saved
        """
        if not self.model_initialized:
            logger.warning("Cannot save - models not initialized")
            return ""
            
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Generate path if not provided
            if not path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"models/options_model_{timestamp}"
            
            # Save Random Forest model
            self.joblib.dump(self.models['rf'], f"{path}_rf.joblib")
            
            # Save LSTM model if available
            if 'lstm' in self.models and self.has_tensorflow:
                self.tf_keras_models['save_model'](self.models['lstm'], f"{path}_lstm")
            
            # Save scaler
            self.joblib.dump(self.scaler, f"{path}_scaler.joblib")
            
            # Save feature columns and other metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Model saved to {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def load_model(self, path: str) -> bool:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load metadata first
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)
                
            # Set metadata attributes
            self.feature_columns = metadata['feature_columns']
            self.feature_importance = metadata['feature_importance']
            self.model_type = metadata['model_type']
            
            # Load Random Forest model
            self.models['rf'] = self.joblib.load(f"{path}_rf.joblib")
            
            # Load LSTM model if available
            if os.path.exists(f"{path}_lstm") and self.has_tensorflow:
                self.models['lstm'] = self.tf_keras_models['load_model'](f"{path}_lstm")
            
            # Load scaler
            self.scaler = self.joblib.load(f"{path}_scaler.joblib")
            
            self.model_initialized = True
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance rankings.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.feature_importance:
            return {}
            
        return dict(sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))


if __name__ == "__main__":
    # Test the enhanced ML model
    try:
        # Create model
        model = EnhancedOptionsMLModel(model_type="hybrid")
        
        # Create some synthetic training data
        import random
        
        # Generate 100 training examples
        features_list = []
        outcomes = []
        
        for i in range(100):
            # Generate random feature values
            option_type = random.choice(['call', 'put'])
            moneyness = random.uniform(0.8, 1.2)
            days_to_expiry = random.randint(7, 60)
            iv = random.uniform(0.1, 0.5)
            delta = random.uniform(0.3, 0.7) if option_type == 'call' else random.uniform(-0.7, -0.3)
            spread_pct = random.uniform(0.01, 0.1)
            volume = random.randint(100, 5000)
            oi = random.randint(500, 10000)
            
            # Create feature dict
            features = {
                'option_type_C': 1 if option_type == 'call' else 0,
                'option_type_P': 1 if option_type == 'put' else 0,
                'moneyness': moneyness,
                'relative_strike': moneyness - 1.0,
                'days_to_expiry': days_to_expiry,
                'implied_volatility': iv,
                'delta': abs(delta),
                'gamma': random.uniform(0.01, 0.05),
                'theta': random.uniform(-0.05, -0.01),
                'vega': random.uniform(0.05, 0.2),
                'bid_ask_spread': random.uniform(0.1, 1.0),
                'bid_ask_spread_percent': spread_pct,
                'volume': volume,
                'open_interest': oi,
                'volume_oi_ratio': volume / oi
            }
            
            # Add to list
            features_list.append(features)
            
            # Generate synthetic outcome (more likely to profit if near the money)
            prob = 0.7 - abs(moneyness - 1.0)  # Higher prob if moneyness close to 1.0
            outcomes.append(random.random() < prob)
        
        # Train the model
        model.add_batch_training_data(features_list, outcomes)
        
        # Test prediction
        test_feature = {
            'option_type_C': 1,
            'option_type_P': 0,
            'moneyness': 1.01,
            'relative_strike': 0.01,
            'days_to_expiry': 30,
            'implied_volatility': 0.25,
            'delta': 0.55,
            'gamma': 0.03,
            'theta': -0.02,
            'vega': 0.1,
            'bid_ask_spread': 0.5,
            'bid_ask_spread_percent': 0.03,
            'volume': 1200,
            'open_interest': 5500,
            'volume_oi_ratio': 0.22
        }
        
        prob = model.predict(test_feature)
        print(f"Probability of profit: {prob:.2f}")
        
        # Show feature importance
        importance = model.get_feature_importance()
        print("\nFeature Importance:")
        for feature, score in importance.items():
            print(f"{feature}: {score:.4f}")
        
        # Save model
        model.save_model()
        
    except Exception as e:
        print(f"Error testing ML model: {e}") 