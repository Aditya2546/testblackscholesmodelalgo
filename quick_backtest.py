#!/usr/bin/env python
"""
Quick ML Backtesting for Options Trading

A simplified script for running backtests to train the ML model on simulated trades.
This script skips the data collection phase and uses simulated data to speed up the training process.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import time
import logging
from datetime import datetime, timedelta
import random

# Import our custom modules
from train_historical_model import train_enhanced_model, evaluate_model, plot_results, save_model
from model_integrator import EnhancedOptionsTrader
from smart_trader import SmartTrader
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """
    Create necessary directories for training data and models
    """
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("training_data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/visualizations", exist_ok=True)

def run_backtests(num_sessions=5, trades_per_session=30):
    """
    Run backtests to generate trading data for ML training
    
    Args:
        num_sessions: Number of trading sessions to simulate
        trades_per_session: Number of trades per session
        
    Returns:
        List of trade data from the backtests
    """
    logger.info(f"Running {num_sessions} backtest sessions with {trades_per_session} trades each")
    
    # Create a smart trader for backtesting
    trader = SmartTrader()
    account_value = 1000.0
    
    # Run multiple sessions
    all_trades = []
    
    for i in range(num_sessions):
        logger.info(f"Running backtest session {i+1}/{num_sessions}")
        
        # Generate trades for this session
        session_trades = []
        for _ in range(trades_per_session):
            # Generate a random trade
            symbol = random.choice(Config.WATCHLIST)
            option_type = random.choice(['call', 'put'])
            strike = random.uniform(80, 120)  # Random strike price
            mid_price = random.uniform(1.0, 5.0)  # Random option price
            ml_probability = random.uniform(0.3, 0.7)  # Random ML probability
            
            # Create trade info
            trade_info = {
                'symbol': symbol,
                'option_type': option_type,
                'strike': strike,
                'mid': mid_price,
                'contracts': random.randint(1, 5),
                'ml_probability': ml_probability
            }
            
            # Simulate the trade
            trade_result = trader.simulate_trade(trade_info, account_value=account_value)
            
            # Add to session trades
            trade_result['session'] = i + 1
            session_trades.append(trade_result)
        
        # Add trades to the overall list
        all_trades.extend(session_trades)
        
        # Log session results
        win_count = sum(1 for t in session_trades if t.get('pnl', 0) > 0)
        win_rate = win_count / len(session_trades) if session_trades else 0
        total_profit = sum(t.get('pnl', 0) for t in session_trades)
        
        logger.info(f"Session {i+1} completed with {len(session_trades)} trades")
        logger.info(f"Win rate: {win_rate:.1%}, Total profit: ${total_profit:.2f}")
        
        # Wait briefly between sessions
        time.sleep(1)
    
    logger.info(f"Backtesting completed with {len(all_trades)} total trades")
    return all_trades

def extract_features(trades):
    """
    Extract features from trade data for ML training
    
    Args:
        trades: List of trade data from backtests
        
    Returns:
        Tuple of (features_df, targets_df)
    """
    logger.info("Extracting features from backtest trades")
    
    features = []
    targets = []
    
    for trade in trades:
        # Parse symbol to extract components
        symbol_parts = trade.get('symbol', '').split()
        base_symbol = symbol_parts[0] if len(symbol_parts) > 0 else ''
        
        # Basic trade info
        feature_dict = {
            # Skip symbol as it's not a numeric feature
            'option_type': trade.get('option_type', 'call'),
            'strike': float(symbol_parts[2].replace('$', '')) if len(symbol_parts) > 2 else 0,
            'underlying_price': random.uniform(80, 120),  # Simulate underlying price
            'days_to_expiry': random.randint(1, 30),  # Simulate days to expiry
            'implied_volatility': random.uniform(0.2, 0.5),  # Simulate IV
        }
        
        # Option Greeks (simulated)
        feature_dict['delta'] = random.uniform(0.3, 0.7) if feature_dict['option_type'] == 'call' else random.uniform(-0.7, -0.3)
        feature_dict['gamma'] = random.uniform(0.01, 0.1)
        feature_dict['theta'] = random.uniform(-0.05, -0.01)
        feature_dict['vega'] = random.uniform(0.05, 0.2)
        
        # Price info
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        
        # Calculate bid/ask from entry price
        bid = entry_price * 0.95
        ask = entry_price * 1.05
        
        # Calculate derived features
        feature_dict['moneyness'] = feature_dict['strike'] / feature_dict['underlying_price'] if feature_dict['underlying_price'] > 0 else 1.0
        feature_dict['relative_strike'] = (feature_dict['strike'] - feature_dict['underlying_price']) / feature_dict['underlying_price'] if feature_dict['underlying_price'] > 0 else 0.0
        feature_dict['bid_ask_spread'] = ask - bid
        feature_dict['bid_ask_spread_percent'] = (ask - bid) / entry_price if entry_price > 0 else 0.0
        
        # One-hot encoding for option type
        feature_dict['option_type_C'] = 1 if feature_dict['option_type'].upper() in ['C', 'CALL'] else 0
        feature_dict['option_type_P'] = 1 if feature_dict['option_type'].upper() in ['P', 'PUT'] else 0
        
        # Time value calculation (simulated)
        if feature_dict['option_type_C'] == 1:
            intrinsic = max(0, feature_dict['underlying_price'] - feature_dict['strike'])
        else:
            intrinsic = max(0, feature_dict['strike'] - feature_dict['underlying_price'])
        
        feature_dict['intrinsic_value'] = intrinsic
        feature_dict['time_value'] = entry_price - intrinsic if entry_price > intrinsic else 0
        
        # Set the target (whether the trade was profitable)
        target = 1 if trade.get('pnl', 0) > 0 else 0
        
        features.append(feature_dict)
        targets.append(target)
    
    # Convert to DataFrames
    features_df = pd.DataFrame(features)
    
    # Convert non-numeric columns and drop text columns
    if 'option_type' in features_df.columns:
        # Drop the text version after creating one-hot encoded columns
        features_df = features_df.drop(columns=['option_type'])
    
    # Ensure all features are numeric
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    features_df = features_df.dropna()
    
    # Create corresponding targets
    targets_df = pd.Series(targets[:len(features_df)], name='profitable')
    
    logger.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} trades")
    return features_df, targets_df

def save_training_data(features_df, targets_df):
    """
    Save the extracted features and targets to CSV files
    
    Args:
        features_df: DataFrame with features
        targets_df: Series with target values
    """
    # Create processed data directory
    processed_dir = "training_data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Split data into train and test sets (80/20 split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, targets_df, test_size=0.2, random_state=42
    )
    
    # Add target column to dataframes for saving
    train_df = X_train.copy()
    train_df['profitable'] = y_train
    
    test_df = X_test.copy()
    test_df['profitable'] = y_test
    
    # Save to CSV
    train_path = f"{processed_dir}/option_train_data.csv"
    test_path = f"{processed_dir}/option_test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Training data saved to {train_path}")
    logger.info(f"Test data saved to {test_path}")

def main():
    """
    Main function to run quick backtest and ML training
    """
    logger.info("Starting quick backtest and ML training")
    
    # Create necessary directories
    setup_directories()
    
    # Run backtests to generate trade data
    trades = run_backtests(num_sessions=Config.BACKTEST_PERIODS, trades_per_session=Config.TRADES_PER_PERIOD)
    
    if not trades:
        logger.error("No trades generated from backtests. Aborting.")
        return
    
    # Extract features from the trades
    features_df, targets_df = extract_features(trades)
    
    if features_df.empty:
        logger.error("Failed to extract valid features. Aborting.")
        return
    
    # Save the training data
    save_training_data(features_df, targets_df)
    
    # Split data for training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, targets_df, test_size=0.2, random_state=42
    )
    
    # Train the model
    logger.info("Training machine learning model on backtest data")
    logger.info(f"Feature columns: {X_train.columns.tolist()}")
    model_result = train_enhanced_model(X_train, y_train)
    
    # Evaluate the model
    evaluation = evaluate_model(model_result['model'], X_test, y_test)
    
    # Generate visualizations
    plot_results(model_result, evaluation, X_train, X_test, y_test)
    
    # Save the model
    model_path, metrics_path = save_model(model_result['model'], {
        'evaluation': evaluation,
        'feature_importances': model_result['feature_importances'].to_dict(),
        'best_params': model_result['best_params'],
        'cv_scores': model_result['cv_scores'].tolist()
    })
    
    logger.info(f"Model trained and saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Print final metrics
    logger.info("=== Model Performance ===")
    logger.info(f"Accuracy: {evaluation['accuracy']:.4f}")
    logger.info(f"Precision: {evaluation['precision']:.4f}")
    logger.info(f"Recall: {evaluation['recall']:.4f}")
    logger.info(f"F1 Score: {evaluation['f1']:.4f}")
    
    # Print most important features
    logger.info("=== Top Features ===")
    top_features = model_result['feature_importances'].sort_values('importance', ascending=False).head(5)
    for i, (feature, importance) in enumerate(zip(top_features.index, top_features['importance']), 1):
        logger.info(f"{i}. {feature}: {importance:.4f}")
    
    logger.info("Quick backtest and ML training complete!")

if __name__ == "__main__":
    main() 