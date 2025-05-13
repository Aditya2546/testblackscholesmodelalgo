#!/usr/bin/env python
"""
Options Trading Model Integrator

This script integrates the trained machine learning model with the options trading system.
It loads the trained model and provides functions to make predictions on new options data.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler
import glob
from datetime import datetime

# Import the existing market data integration module
from integrate_market_data import MarketDataIntegration
from market_data_api import MarketDataAPI

class EnhancedOptionsTrader:
    """
    Enhanced options trader that uses machine learning to improve trading decisions
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the enhanced options trader
        
        Args:
            model_path: Path to the trained model. If None, the latest model is used.
        """
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        self.market_data = MarketDataIntegration()
        
        # Feature columns used during training
        self.feature_cols = [
            'strike', 'underlying_price', 'days_to_expiry', 'implied_volatility',
            'delta', 'gamma', 'theta', 'vega', 'moneyness', 'relative_strike',
            'bid_ask_spread', 'bid_ask_spread_percent', 'intrinsic_value', 'time_value',
            'option_type_C', 'option_type_P'
        ]
        
        print(f"Enhanced Options Trader initialized with model: {model_path}")
    
    def _load_model(self, model_path: Optional[str] = None) -> object:
        """
        Load the trained model
        
        Args:
            model_path: Path to the trained model. If None, the latest model is used.
            
        Returns:
            Loaded model
        """
        if model_path is None:
            # Find the latest model
            models_dir = "models"
            model_files = glob.glob(f"{models_dir}/options_model_*.joblib")
            
            if not model_files:
                raise FileNotFoundError("No trained model found. Please run train_historical_model.py first.")
            
            # Sort by creation time (most recent first)
            model_path = sorted(model_files, key=os.path.getctime, reverse=True)[0]
            print(f"Using latest model: {model_path}")
        
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def prepare_features(self, option_data: Dict) -> pd.DataFrame:
        """
        Prepare features for model prediction
        
        Args:
            option_data: Dictionary containing option data
            
        Returns:
            DataFrame with features prepared for the model
        """
        # Create a DataFrame with a single row
        df = pd.DataFrame([option_data])
        
        # Calculate additional features
        if 'option_type' in df:
            df['option_type_C'] = (df['option_type'] == 'call').astype(int)
            df['option_type_P'] = (df['option_type'] == 'put').astype(int)
        else:
            df['option_type_C'] = 0
            df['option_type_P'] = 0
        
        # Calculate moneyness
        if 'strike' in df and 'underlying_price' in df:
            df['moneyness'] = df['strike'] / df['underlying_price']
            df['relative_strike'] = (df['strike'] - df['underlying_price']) / df['underlying_price']
        else:
            df['moneyness'] = 1.0
            df['relative_strike'] = 0.0
        
        # Calculate bid-ask spread
        if 'bid' in df and 'ask' in df and 'mid' in df:
            df['bid_ask_spread'] = df['ask'] - df['bid']
            df['bid_ask_spread_percent'] = df['bid_ask_spread'] / df['mid']
        else:
            df['bid_ask_spread'] = 0.0
            df['bid_ask_spread_percent'] = 0.0
        
        # Calculate time value and intrinsic value
        if 'mid' in df and 'underlying_price' in df and 'strike' in df:
            df['intrinsic_value'] = np.maximum(
                df['option_type_C'] * (df['underlying_price'] - df['strike']),
                df['option_type_P'] * (df['strike'] - df['underlying_price'])
            )
            df['intrinsic_value'] = df['intrinsic_value'].clip(lower=0)
            df['time_value'] = df['mid'] - df['intrinsic_value']
        else:
            df['intrinsic_value'] = 0.0
            df['time_value'] = 0.0
        
        # Ensure all required feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                print(f"Warning: Missing feature '{col}'. Using default value.")
                df[col] = 0.0
        
        # Select and reorder features
        features = df[self.feature_cols].copy()
        
        # Scale numerical features
        numerical_cols = [col for col in self.feature_cols if col not in ['option_type_C', 'option_type_P']]
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        return features
    
    def predict_profitability(self, option_data: Dict) -> Tuple[float, bool]:
        """
        Predict the profitability of an option trade
        
        Args:
            option_data: Dictionary containing option data
            
        Returns:
            Tuple of (probability of profitability, predicted profitable)
        """
        # Prepare features
        features = self.prepare_features(option_data)
        
        # Make prediction
        probability = self.model.predict_proba(features)[0, 1]
        prediction = self.model.predict(features)[0]
        
        return probability, bool(prediction)
    
    def get_enhanced_signals(self, symbols: List[str], expiration_days: int = 30, 
                            max_signals: int = 5, min_probability: float = 0.3, 
                            force_minimum: bool = True) -> List[Dict]:
        """
        Get enhanced trading signals using the trained model
        
        Args:
            symbols: List of symbols to get signals for
            expiration_days: Days to expiration to look for
            max_signals: Maximum number of signals to return
            min_probability: Minimum probability threshold for considering a signal profitable
            force_minimum: If True, will return at least some signals even if none meet the threshold
            
        Returns:
            List of enhanced trading signals
        """
        print(f"Getting enhanced signals for {symbols}...")
        
        # Get base signals from the market data integration
        base_signals = self.market_data.get_option_signals(symbols)
        
        # Enhanced signals will be stored here
        enhanced_signals = []
        
        # Process each base signal
        for signal in base_signals:
            # Convert to a dictionary for model input
            option_data = {
                'symbol': signal.symbol,
                'option_type': signal.option_type,
                'strike': signal.strike,
                'underlying_price': signal.underlying_price,
                'days_to_expiry': (datetime.strptime(signal.expiration, "%Y-%m-%d") - datetime.now()).days,
                'implied_volatility': 0.3,  # Default IV if not available
                'delta': signal.delta if hasattr(signal, 'delta') else 0.5,
                'gamma': signal.gamma if hasattr(signal, 'gamma') else 0.05,
                'theta': signal.theta if hasattr(signal, 'theta') else -0.1,
                'vega': signal.vega if hasattr(signal, 'vega') else 0.1,
                'bid': signal.entry_price_range[0],
                'ask': signal.entry_price_range[1],
                'mid': signal.current_price,
                'volume': getattr(signal, 'volume', 100),
                'open_interest': getattr(signal, 'open_interest', 500)
            }
            
            # Predict profitability
            profitability_prob, is_profitable = self.predict_profitability(option_data)
            
            # Add profitability information to the signal
            signal_dict = signal.__dict__.copy()
            signal_dict['ml_probability'] = profitability_prob
            signal_dict['ml_profitable'] = is_profitable
            signal_dict['ml_score'] = profitability_prob * signal.signal_strength
            
            # Add to enhanced signals if it meets the probability threshold
            if profitability_prob >= min_probability:
                enhanced_signals.append(signal_dict)
        
        # Sort by ML score (descending)
        enhanced_signals = sorted(enhanced_signals, key=lambda x: x['ml_score'], reverse=True)
        
        # If no signals meet the threshold but force_minimum is True, include the highest scoring signals
        if len(enhanced_signals) == 0 and force_minimum and base_signals:
            print("No signals met the probability threshold. Including highest scoring signals anyway...")
            for signal in base_signals:
                # Convert to a dictionary for model input
                option_data = {
                    'symbol': signal.symbol,
                    'option_type': signal.option_type,
                    'strike': signal.strike,
                    'underlying_price': signal.underlying_price,
                    'days_to_expiry': (datetime.strptime(signal.expiration, "%Y-%m-%d") - datetime.now()).days,
                    'implied_volatility': 0.3,
                    'delta': signal.delta if hasattr(signal, 'delta') else 0.5,
                    'gamma': signal.gamma if hasattr(signal, 'gamma') else 0.05,
                    'theta': signal.theta if hasattr(signal, 'theta') else -0.1,
                    'vega': signal.vega if hasattr(signal, 'vega') else 0.1,
                    'bid': signal.entry_price_range[0],
                    'ask': signal.entry_price_range[1],
                    'mid': signal.current_price,
                    'volume': getattr(signal, 'volume', 100),
                    'open_interest': getattr(signal, 'open_interest', 500)
                }
                
                profitability_prob, _ = self.predict_profitability(option_data)
                
                signal_dict = signal.__dict__.copy()
                signal_dict['ml_probability'] = profitability_prob
                signal_dict['ml_profitable'] = False  # Mark as not profitable but included
                signal_dict['ml_score'] = profitability_prob * signal.signal_strength
                
                enhanced_signals.append(signal_dict)
            
            # Sort by ML score (descending)
            enhanced_signals = sorted(enhanced_signals, key=lambda x: x['ml_score'], reverse=True)
        
        # Limit the number of signals
        return enhanced_signals[:max_signals]

def main():
    """Main function to demonstrate the enhanced options trader"""
    print("Enhanced Options Trader - Model Integration")
    print("=========================================")
    
    # Create enhanced trader
    trader = EnhancedOptionsTrader()
    
    # Test with some symbols
    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN"]
    
    # Get enhanced signals
    enhanced_signals = trader.get_enhanced_signals(symbols)
    
    # Display results
    print(f"\nFound {len(enhanced_signals)} enhanced trading signals:")
    for i, signal in enumerate(enhanced_signals):
        print(f"\n{i+1}. {signal['symbol']} {signal['option_type'].upper()} ${signal['strike']}")
        print(f"   ML Score: {signal['ml_score']:.2f} (Probability: {signal['ml_probability']*100:.1f}%)")
        print(f"   Signal Strength: {signal['signal_strength']:.2f}")
        print(f"   Current Price: ${signal['current_price']:.2f}")
        print(f"   Underlying Price: ${signal['underlying_price']:.2f}")
        print(f"   Expiration: {signal['expiration']}")
    
    print("\nModel integration complete!")

if __name__ == "__main__":
    main() 