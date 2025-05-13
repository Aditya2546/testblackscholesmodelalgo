#!/usr/bin/env python
"""
Test script for Enhanced Options Analyzer

This script demonstrates the features of the enhanced options analyzer, 
including near-the-money options detection, advanced ML model learning, 
and market data source failover.
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from tabulate import tabulate
import os
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from near_the_money_analyzer import NearTheMoneyAnalyzer
from enhanced_ml_model import EnhancedOptionsMLModel
from market_data_manager import MarketDataManager, AlpacaMarketData, YahooFinanceDataSource
from options_day_trader_sim import OptionSignal, Config

def print_option_table(signals: List, title: str = "Options Analysis"):
    """Print option signals in a formatted table"""
    if not signals:
        logger.warning("No signals to display")
        return
    
    # Convert to dictionary for tabulate
    rows = []
    for signal in signals:
        rows.append({
            "Symbol": signal.symbol,
            "Type": signal.option_type.upper(),
            "Strike": f"${signal.strike}",
            "Exp Date": signal.expiration.strftime("%Y-%m-%d"),
            "Price": f"${signal.current_price:.2f}",
            "Entry": f"${signal.entry_price_range[0]:.2f}-${signal.entry_price_range[1]:.2f}",
            "Target": f"${signal.target_price:.2f}",
            "Stop": f"${signal.stop_loss:.2f}",
            "Score": f"{signal.signal_strength:.2f}",
            "Volume": str(signal.volume),
            "OI": str(signal.open_interest),
            "IV": f"{signal.iv:.1f}%",
            "Delta": f"{signal.delta:.2f}"
        })
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(rows)
    
    # Print the table
    print(f"\n=== {title} ===")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print()

def simulate_trading(signals: List[OptionSignal], duration_days: int = 5):
    """
    Simulate trading the signals and generate synthetic outcomes for ML training.
    
    Args:
        signals: List of option signals
        duration_days: Number of days to simulate
        
    Returns:
        List of outcome data for ML training
    """
    print(f"\n=== Simulating Trading for {duration_days} Days ===")
    
    # Outcomes for ML training
    outcomes = []
    
    # Simulated win rate patterns
    win_rates = {
        'call': {
            'AAPL': 0.65, 'MSFT': 0.62, 'AMZN': 0.58, 'TSLA': 0.55,
            'NVDA': 0.60, 'META': 0.63, 'SPY': 0.59, 'QQQ': 0.61
        },
        'put': {
            'AAPL': 0.55, 'MSFT': 0.58, 'AMZN': 0.54, 'TSLA': 0.60,
            'NVDA': 0.57, 'META': 0.56, 'SPY': 0.62, 'QQQ': 0.60
        }
    }
    
    # Additional factors to adjust win probability
    def calculate_win_probability(signal):
        base_prob = win_rates.get(signal.option_type, {}).get(signal.symbol, 0.5)
        
        # Adjust based on delta (higher win rate for options with delta close to 0.5)
        delta_factor = 1.0 - abs(signal.delta - 0.5)
        
        # Adjust based on days to expiration (higher win rate for 20-40 DTE)
        days_to_exp = (signal.expiration - datetime.now().date()).days
        expiry_factor = 1.0 - (min(abs(days_to_exp - 30), 30) / 30)
        
        # Adjust based on IV (lower win rate for very high IV)
        iv_norm = signal.iv / 100 if signal.iv > 1 else signal.iv
        iv_factor = 1.0 - max(0, iv_norm - 0.3) / 0.7
        
        # Combine factors
        adjusted_prob = base_prob * (
            0.7 + 0.1 * delta_factor + 0.1 * expiry_factor + 0.1 * iv_factor
        )
        
        return min(0.95, max(0.05, adjusted_prob))
    
    # Simulate each signal
    for i, signal in enumerate(signals):
        win_prob = calculate_win_probability(signal)
        
        # Determine outcome (win or loss)
        import random
        is_win = random.random() < win_prob
        
        # Calculate profit/loss
        entry_price = (signal.entry_price_range[0] + signal.entry_price_range[1]) / 2
        
        if is_win:
            # Win scenario (hit target or partial profit)
            hit_target = random.random() < 0.7
            exit_price = signal.target_price if hit_target else (
                entry_price + (signal.target_price - entry_price) * random.uniform(0.3, 0.8)
            )
            outcome = "TARGET_HIT" if hit_target else "PARTIAL_PROFIT"
        else:
            # Loss scenario (hit stop or partial loss)
            hit_stop = random.random() < 0.6
            exit_price = signal.stop_loss if hit_stop else (
                entry_price - (entry_price - signal.stop_loss) * random.uniform(0.3, 0.9)
            )
            outcome = "STOP_LOSS" if hit_stop else "PARTIAL_LOSS"
        
        # Calculate P&L
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        # Print trade result
        print(f"Trade {i+1}: {signal.symbol} {signal.option_type.upper()} ${signal.strike}")
        print(f"  Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
        print(f"  Outcome: {outcome}, P&L: {'+'if pnl>0 else ''}{pnl:.2f} ({pnl_pct:.1f}%)")
        
        # Create feature dict for ML training
        days_to_exp = (signal.expiration - datetime.now().date()).days
        features = {
            'option_type_C': 1 if signal.option_type == 'call' else 0,
            'option_type_P': 1 if signal.option_type == 'put' else 0,
            'moneyness': signal.strike / signal.underlying_price,
            'relative_strike': (signal.strike - signal.underlying_price) / signal.underlying_price,
            'days_to_expiry': days_to_exp,
            'implied_volatility': signal.iv / 100 if signal.iv > 1 else signal.iv,
            'delta': signal.delta,
            'gamma': 0.03,  # Placeholder values for Greeks that might not be available
            'theta': -0.02,
            'vega': 0.1,
            'bid_ask_spread': signal.entry_price_range[1] - signal.entry_price_range[0],
            'bid_ask_spread_percent': (signal.entry_price_range[1] - signal.entry_price_range[0]) / signal.entry_price_range[1],
            'volume': signal.volume,
            'open_interest': signal.open_interest,
            'volume_oi_ratio': signal.volume / signal.open_interest if signal.open_interest > 0 else 0
        }
        
        # Add to training outcomes
        outcomes.append({
            'signal': signal,
            'features': features,
            'outcome': outcome,
            'profit': pnl,
            'profit_pct': pnl_pct,
            'win': is_win
        })
    
    # Print summary
    wins = sum(1 for o in outcomes if o['win'])
    win_rate = (wins / len(outcomes)) * 100 if outcomes else 0
    avg_profit = sum(o['profit'] for o in outcomes) / len(outcomes) if outcomes else 0
    
    print(f"\nSimulation Summary:")
    print(f"  Trades: {len(outcomes)}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Average P&L: {'+'if avg_profit>0 else ''}{avg_profit:.2f}")
    
    return outcomes

def test_market_data_manager():
    """Test the market data manager with failover capabilities"""
    print("\n=== Testing Market Data Manager ===")
    
    # Initialize data sources
    try:
        # Initialize Alpaca
        try:
            from alpaca_config import API_KEY, API_SECRET
            alpaca = AlpacaMarketData(API_KEY, API_SECRET)
        except ImportError:
            alpaca = AlpacaMarketData()  # Use env vars
        
        print(f"Alpaca initialized: {alpaca.__class__.__name__}")
        
        # Try to initialize Yahoo
        try:
            yahoo = YahooFinanceDataSource()
            print(f"Yahoo Finance initialized: {yahoo.__class__.__name__}")
        except Exception as e:
            yahoo = None
            print(f"Yahoo Finance not available: {e}")
        
        # Create manager with available sources
        data_sources = [source for source in [alpaca, yahoo] if source is not None]
        if len(data_sources) > 1:
            manager = MarketDataManager(
                primary_source=alpaca, 
                backup_sources=[source for source in data_sources if source != alpaca]
            )
            print(f"MarketDataManager initialized with {len(data_sources)} sources")
        else:
            manager = data_sources[0]
            print(f"Using single data source: {manager.__class__.__name__}")
        
        # Test with a few symbols
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        
        print("\nFetching current prices:")
        for symbol in symbols:
            try:
                start_time = time.time()
                price = manager.get_price(symbol)
                duration = time.time() - start_time
                print(f"  {symbol}: ${price:.2f} (fetched in {duration:.3f}s)")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        
        # Test failover by triggering an error on primary source
        if isinstance(manager, MarketDataManager) and len(manager.sources) > 1:
            print("\nTesting failover:")
            try:
                # First normal request
                active_source = manager.get_active_source()
                print(f"Active source: {active_source}")
                
                # Force failure on primary source (temporary monkey patch)
                original_get_price = manager.sources[0].get_price
                manager.sources[0].get_price = lambda x: 1/0  # Will raise ZeroDivisionError
                
                try:
                    # This should trigger failover
                    price = manager.get_price("SPY")
                    new_active_source = manager.get_active_source()
                    print(f"Failover successful. New active source: {new_active_source}")
                    print(f"SPY price from backup: ${price:.2f}")
                finally:
                    # Restore original method
                    manager.sources[0].get_price = original_get_price
            except Exception as e:
                print(f"Failover test failed: {e}")
                
        return manager
    except Exception as e:
        print(f"Error testing market data manager: {e}")
        return None

def test_enhanced_ml_model(outcomes=None):
    """Test the enhanced ML model with rapid learning"""
    print("\n=== Testing Enhanced ML Model ===")
    
    try:
        # Initialize model
        model = EnhancedOptionsMLModel(model_type="hybrid")
        print(f"ML model initialized: {model.__class__.__name__}")
        
        if not outcomes:
            # Create synthetic training data if no real outcomes provided
            print("Generating synthetic training data...")
            features_list = []
            outcome_list = []
            
            # Generate 100 samples
            for i in range(100):
                # Generate random features
                import random
                
                option_type = random.choice(['call', 'put'])
                moneyness = random.uniform(0.8, 1.2)
                days_to_expiry = random.randint(7, 60)
                iv = random.uniform(0.1, 0.5)
                delta = random.uniform(0.3, 0.7) if option_type == 'call' else random.uniform(-0.7, -0.3)
                spread_pct = random.uniform(0.01, 0.1)
                volume = random.randint(100, 5000)
                oi = random.randint(500, 10000)
                
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
                
                features_list.append(features)
                
                # Generate synthetic outcome (more likely to profit if near the money)
                prob = 0.7 - abs(moneyness - 1.0)  # Higher prob if moneyness close to 1.0
                outcome_list.append(random.random() < prob)
            
            print(f"Generated {len(features_list)} training samples")
        else:
            # Use real outcomes from simulation
            print(f"Using {len(outcomes)} real outcomes for training")
            features_list = [o['features'] for o in outcomes]
            outcome_list = [o['win'] for o in outcomes]
        
        # Train the model
        start_time = time.time()
        model.add_batch_training_data(features_list, outcome_list)
        duration = time.time() - start_time
        print(f"Model trained in {duration:.3f}s")
        
        # Test predictions
        test_cases = [
            {
                'name': "ATM AAPL Call 30 DTE",
                'features': {
                    'option_type_C': 1, 'option_type_P': 0,
                    'moneyness': 1.0, 'relative_strike': 0.0,
                    'days_to_expiry': 30, 'implied_volatility': 0.3,
                    'delta': 0.5, 'gamma': 0.03, 'theta': -0.02, 'vega': 0.1,
                    'bid_ask_spread': 0.5, 'bid_ask_spread_percent': 0.03,
                    'volume': 1000, 'open_interest': 5000, 'volume_oi_ratio': 0.2
                }
            },
            {
                'name': "Deep ITM SPY Put 7 DTE",
                'features': {
                    'option_type_C': 0, 'option_type_P': 1,
                    'moneyness': 0.85, 'relative_strike': -0.15,
                    'days_to_expiry': 7, 'implied_volatility': 0.45,
                    'delta': 0.8, 'gamma': 0.02, 'theta': -0.03, 'vega': 0.05,
                    'bid_ask_spread': 0.8, 'bid_ask_spread_percent': 0.05,
                    'volume': 800, 'open_interest': 3000, 'volume_oi_ratio': 0.27
                }
            },
            {
                'name': "Far OTM NVDA Call 60 DTE",
                'features': {
                    'option_type_C': 1, 'option_type_P': 0,
                    'moneyness': 1.2, 'relative_strike': 0.2,
                    'days_to_expiry': 60, 'implied_volatility': 0.6,
                    'delta': 0.25, 'gamma': 0.01, 'theta': -0.01, 'vega': 0.15,
                    'bid_ask_spread': 0.3, 'bid_ask_spread_percent': 0.06,
                    'volume': 500, 'open_interest': 2000, 'volume_oi_ratio': 0.25
                }
            }
        ]
        
        print("\nModel Predictions:")
        for case in test_cases:
            prob = model.predict(case['features'])
            print(f"  {case['name']}: {prob:.2f} probability")
        
        # Feature importance
        importance = model.get_feature_importance()
        if importance:
            print("\nFeature Importance:")
            for feature, score in list(importance.items())[:5]:  # Show top 5
                print(f"  {feature}: {score:.4f}")
        
        # Save model
        model_path = model.save_model()
        if model_path:
            print(f"Model saved to: {model_path}")
        
        return model
    except Exception as e:
        print(f"Error testing enhanced ML model: {e}")
        return None

def main():
    """Main test function"""
    print("=== Enhanced Options Analyzer Test ===")
    
    # Test market data manager first
    market_data = test_market_data_manager()
    
    # Initialize near-the-money analyzer
    analyzer = NearTheMoneyAnalyzer(market_data=market_data)
    print(f"\nNear-the-money analyzer initialized: {analyzer.__class__.__name__}")
    
    # Test analyzer with some symbols
    test_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
    
    print("\nFetching option signals...")
    signals = analyzer.get_option_signals(test_symbols, max_per_symbol=2)
    
    # Display results
    print_option_table(signals, "Top Near-Money Options Signals")
    
    # Simulate trading to generate outcomes
    outcomes = simulate_trading(signals, duration_days=5)
    
    # Test enhanced ML model
    ml_model = test_enhanced_ml_model(outcomes)
    
    # Test retrieval of specific strike ranges
    print("\n=== Testing Specific Strike Range Analysis ===")
    symbol = "AAPL"
    
    try:
        # Get current price
        price = analyzer.market_data.get_price(symbol)
        print(f"{symbol} current price: ${price}")
        
        # Generate strikes around current price
        strikes = analyzer._generate_strikes_around_price(price)
        print(f"Testing strikes: {strikes}")
        
        # Get options for these strikes
        expiration = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
        print(f"Target expiration: {expiration}")
        
        options = analyzer.get_options_by_strike_range(symbol, strikes, expiration)
        
        # Print summary of found options
        call_count = len(options["calls"])
        put_count = len(options["puts"])
        print(f"Found {call_count} calls and {put_count} puts")
        
        # Generate signals from these options
        specific_signals = []
        for option_type in ["calls", "puts"]:
            for strike_str, option_data in options[option_type].items():
                try:
                    strike = float(strike_str)
                    
                    # Score this option
                    score = analyzer.score_option(option_data, price, option_type[:-1])  # Remove 's' from calls/puts
                    
                    # Skip low scores
                    if score < 0.2:
                        continue
                        
                    # Create signal object
                    mid_price = (option_data["bid"] + option_data["ask"]) / 2
                    expiry_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                    
                    signal = OptionSignal(
                        symbol=symbol,
                        option_type=option_type[:-1],  # Remove 's' from calls/puts
                        strike=strike,
                        expiration=expiry_date,
                        current_price=mid_price,
                        underlying_price=price,
                        entry_price_range=(option_data["bid"], option_data["ask"]),
                        stop_loss=mid_price * 0.9,
                        target_price=mid_price * 1.2,
                        signal_strength=score,
                        volume=option_data.get("volume", 0),
                        open_interest=option_data.get("open_interest", 0),
                        iv=option_data.get("iv", 30.0),
                        delta=option_data.get("delta", 0.5)
                    )
                    
                    specific_signals.append(signal)
                except Exception as e:
                    print(f"Error creating signal for {strike_str}: {e}")
        
        # Print specific strike signals
        print_option_table(specific_signals, f"{symbol} Specific Strike Analysis")
        
        # Feed ML model more data if we found new signals
        if specific_signals and ml_model:
            print("\n=== Continuous Learning with New Signals ===")
            
            # Simulate trading these specific signals
            new_outcomes = simulate_trading(specific_signals, duration_days=3)
            
            # Update ML model with new data
            features_list = [o['features'] for o in new_outcomes]
            outcome_list = [o['win'] for o in new_outcomes]
            
            start_time = time.time()
            ml_model.add_batch_training_data(features_list, outcome_list)
            duration = time.time() - start_time
            print(f"Model updated with {len(new_outcomes)} samples in {duration:.3f}s")
            
            # Show updated predictions
            if new_outcomes:
                print("\nUpdated Model Predictions:")
                for i, outcome in enumerate(new_outcomes[:3]):  # Show first 3
                    new_prob = ml_model.predict(outcome['features'])
                    signal = outcome['signal']
                    print(f"  {signal.symbol} {signal.option_type.upper()} ${signal.strike}: {new_prob:.2f} probability")
    except Exception as e:
        print(f"Error testing specific strike range: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 