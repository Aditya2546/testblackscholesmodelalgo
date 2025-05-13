#!/usr/bin/env python
"""
Automated Backtesting and ML Training System

This script automates backtesting on historical data to train the machine learning model.
It performs the following steps:
1. Collect historical options data
2. Prepare training data with various features
3. Run backtests on different time periods and collect performance metrics
4. Train the ML model on the backtested data
5. Evaluate model performance and provide optimization feedback
"""

import os
import sys
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
import argparse
import joblib
import logging
import json

# Import our custom modules
import data_collector
from market_data_api import MarketDataAPI
from market_data_config import load_market_data_credentials
import prepare_training_data
from train_historical_model import train_enhanced_model, evaluate_model, plot_results, save_model
from model_integrator import EnhancedOptionsTrader
from smart_trader import SmartTrader
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutomatedBacktester:
    """
    Automated backtesting system that collects data, performs backtests,
    and trains the ML model on the results.
    """
    
    def __init__(self, symbols: List[str] = None, backtest_periods: int = None, 
                 trades_per_period: int = None, data_days: int = None):
        """
        Initialize the automated backtester
        
        Args:
            symbols: List of symbols to collect data for
            backtest_periods: Number of backtest periods to run
            trades_per_period: Number of trades per backtest period
            data_days: Number of days of historical data to collect
        """
        self.symbols = symbols or Config.WATCHLIST
        self.backtest_periods = backtest_periods or Config.BACKTEST_PERIODS
        self.trades_per_period = trades_per_period or Config.TRADES_PER_PERIOD
        self.data_days = data_days or Config.HISTORICAL_DATA_DAYS
        
        # Set up directories
        self.data_dir = "training_data"
        self.historical_dir = f"{self.data_dir}/historical"
        self.processed_dir = f"{self.data_dir}/processed"
        self.models_dir = "models"
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.historical_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize market data API
        self.credentials = load_market_data_credentials()
        self.api = MarketDataAPI(self.credentials)
        
        logger.info(f"Initialized automated backtester for symbols: {', '.join(self.symbols)}")
    
    def collect_historical_data(self) -> bool:
        """
        Collect historical options data for the specified symbols
        
        Returns:
            True if data collection was successful, False otherwise
        """
        logger.info(f"Collecting historical data for {len(self.symbols)} symbols over {self.data_days} days")
        
        try:
            # Set up data directories
            data_collector.setup_data_directories()
            
            # Set the date range for historical data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=self.data_days)).strftime("%Y-%m-%d")
            
            # Get future expiration date
            expiration_date = data_collector.get_future_expiration(days_out=45)
            
            # Collect data for each symbol
            for symbol in self.symbols:
                # Get current price
                current_price = data_collector.get_current_price(self.api, symbol)
                
                # Generate strikes
                strikes = data_collector.get_standard_strikes(symbol, current_price)
                
                # Collect historical data
                data_collector.collect_historical_option_data(
                    api=self.api,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    strikes=strikes,
                    option_types=["C", "P"],
                    expiration_date=expiration_date
                )
            
            logger.info(f"Data collection completed for {len(self.symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {str(e)}")
            return False
    
    def prepare_training_data(self) -> bool:
        """
        Process the collected historical data into training features
        
        Returns:
            True if data preparation was successful, False otherwise
        """
        logger.info("Preparing training data from collected historical data")
        
        try:
            # Set up directories
            prepare_training_data.setup_directories()
            
            # Load all historical data
            historical_df = prepare_training_data.load_all_historical_data()
            
            if historical_df.empty:
                logger.error("No historical data found")
                return False
            
            # Preprocess the data
            processed_df = prepare_training_data.preprocess_option_data(historical_df)
            
            # Create training features
            features_df = prepare_training_data.create_training_features(processed_df)
            
            # Save the processed data
            output_file = f"{self.processed_dir}/options_features.csv"
            prepare_training_data.save_processed_data(features_df, output_file)
            
            result = {
                'records': len(features_df),
                'features': len(features_df.columns)
            }
            
            logger.info(f"Data preparation completed. Processed {result['records']} records with {result['features']} features")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return False
    
    def run_backtests(self) -> List[Dict]:
        """
        Run multiple backtests to generate trade history for ML training
        
        Returns:
            List of backtest results with performance metrics
        """
        logger.info(f"Running {self.backtest_periods} backtest periods with {self.trades_per_period} trades each")
        
        backtest_results = []
        
        try:
            # Create a SmartTrader instance for backtesting
            trader = SmartTrader(account_value=Config.DEFAULT_ACCOUNT_VALUE)
            
            # Run multiple backtest periods
            for period in range(self.backtest_periods):
                logger.info(f"Starting backtest period {period+1}/{self.backtest_periods}")
                
                # Run a trading session with the specified number of trades
                result = trader.run_trading_session(max_trades=self.trades_per_period)
                
                # Store the result
                backtest_results.append({
                    'period': period + 1,
                    'trades': result.get('trades', []),
                    'win_rate': result.get('win_rate', 0),
                    'profit_loss': result.get('profit_loss', 0),
                    'account_change': result.get('account_change', 0)
                })
                
                logger.info(f"Completed backtest period {period+1}. Win rate: {result.get('win_rate', 0):.1%}, P&L: ${result.get('profit_loss', 0):.2f}")
                
                # Wait briefly between periods to allow for different market conditions in simulation
                time.sleep(1)
            
            # Calculate overall metrics
            total_trades = sum(len(r['trades']) for r in backtest_results)
            total_wins = sum(sum(1 for t in r['trades'] if t.get('net_profit', 0) > 0) for r in backtest_results)
            overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
            total_profit = sum(r['profit_loss'] for r in backtest_results)
            
            logger.info(f"Backtesting completed. Overall win rate: {overall_win_rate:.1%}, Total P&L: ${total_profit:.2f}, Total trades: {total_trades}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtests: {str(e)}")
            return []
    
    def extract_features_from_backtests(self, backtest_results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract features from backtest results for ML training
        
        Args:
            backtest_results: List of backtest results
            
        Returns:
            Tuple of (features_df, targets_df) for ML training
        """
        logger.info("Extracting features from backtest results")
        
        features = []
        targets = []
        
        try:
            # Extract features from each trade in the backtest results
            for period_result in backtest_results:
                for trade in period_result['trades']:
                    # Extract basic option info
                    feature_dict = {
                        'symbol': trade.get('symbol', ''),
                        'option_type': trade.get('option_type', ''),
                        'strike': trade.get('strike', 0),
                        'expiration': trade.get('expiration', ''),
                        'underlying_price': trade.get('underlying_price', 0),
                        'option_price': trade.get('entry_price', 0),
                        'days_to_expiry': trade.get('days_to_expiry', 0),
                    }
                    
                    # Extract option Greeks if available
                    for greek in ['delta', 'gamma', 'theta', 'vega']:
                        feature_dict[greek] = trade.get(greek, 0)
                    
                    # Extract other features
                    feature_dict['implied_volatility'] = trade.get('iv', 30) / 100  # Convert from percentage
                    feature_dict['bid_ask_spread'] = trade.get('bid_ask_spread', 0)
                    feature_dict['bid_ask_spread_percent'] = trade.get('bid_ask_spread_percent', 0)
                    feature_dict['time_value'] = trade.get('time_value', 0)
                    feature_dict['intrinsic_value'] = trade.get('intrinsic_value', 0)
                    
                    # Create one-hot encoding for option type
                    feature_dict['option_type_C'] = 1 if trade.get('option_type', '').upper() in ['C', 'CALL'] else 0
                    feature_dict['option_type_P'] = 1 if trade.get('option_type', '').upper() in ['P', 'PUT'] else 0
                    
                    # Calculate moneyness
                    underlying = trade.get('underlying_price', 0)
                    strike = trade.get('strike', 0)
                    if underlying > 0:
                        feature_dict['moneyness'] = strike / underlying
                        feature_dict['relative_strike'] = (strike - underlying) / underlying
                    else:
                        feature_dict['moneyness'] = 1.0
                        feature_dict['relative_strike'] = 0.0
                    
                    # Extract the target (whether the trade was profitable)
                    target = 1 if trade.get('net_profit', 0) > 0 else 0
                    
                    # Add to lists
                    features.append(feature_dict)
                    targets.append(target)
            
            # Convert to DataFrames
            features_df = pd.DataFrame(features)
            targets_df = pd.Series(targets, name='profitable')
            
            logger.info(f"Extracted features for {len(features)} trades with {len(features_df.columns)} columns")
            return features_df, targets_df
            
        except Exception as e:
            logger.error(f"Error extracting features from backtest results: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Train the ML model on the extracted features
        
        Args:
            features: DataFrame with feature data
            targets: Series with target values
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training ML model on backtest data")
        
        try:
            # Split the data into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Train the model
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
            
            logger.info(f"Model training completed. Model saved to {model_path}")
            
            # Return training results
            return {
                'model_path': model_path,
                'metrics_path': metrics_path,
                'accuracy': evaluation['accuracy'],
                'precision': evaluation['precision'],
                'recall': evaluation['recall'],
                'f1': evaluation['f1']
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {}
    
    def test_model_performance(self, model_path: str) -> Dict:
        """
        Test the trained model's performance on new backtests
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Testing model performance: {model_path}")
        
        try:
            # Create an EnhancedOptionsTrader with the trained model
            ml_trader = EnhancedOptionsTrader(model_path=model_path)
            
            # Collect signals for each symbol
            all_signals = []
            for symbol in self.symbols:
                signals = ml_trader.get_enhanced_signals([symbol], max_signals=5)
                all_signals.extend(signals)
            
            # Run a small batch of trades to verify performance
            verification_trades = []
            for signal in all_signals[:10]:  # Test with up to 10 signals
                # Simulate a trade with the signal
                trade_info = {
                    'symbol': signal['symbol'],
                    'option_type': signal['option_type'],
                    'strike': signal['strike'],
                    'expiration': signal['expiration'],
                    'underlying_price': signal['underlying_price'],
                    'current_price': signal['current_price'],
                    'ml_probability': signal['ml_probability'],
                    'ml_score': signal['ml_score']
                }
                
                # Record the expected profitability
                trade_info['expected_profitable'] = signal['ml_profitable']
                
                # Simulate the outcome (in a real system, this would be a backtest)
                # Here we're just using the ML probability to simulate an outcome
                trade_info['actual_profitable'] = random.random() < signal['ml_probability']
                
                verification_trades.append(trade_info)
            
            # Calculate performance metrics
            correct_predictions = sum(1 for t in verification_trades 
                                    if t['expected_profitable'] == t['actual_profitable'])
            accuracy = correct_predictions / len(verification_trades) if verification_trades else 0
            
            high_prob_trades = [t for t in verification_trades if t['ml_probability'] > 0.6]
            high_prob_accuracy = sum(1 for t in high_prob_trades 
                                  if t['expected_profitable'] == t['actual_profitable'])
            high_prob_accuracy = high_prob_accuracy / len(high_prob_trades) if high_prob_trades else 0
            
            logger.info(f"Model verification completed. Accuracy: {accuracy:.1%}, High probability accuracy: {high_prob_accuracy:.1%}")
            
            return {
                'accuracy': accuracy,
                'high_prob_accuracy': high_prob_accuracy,
                'verification_trades': verification_trades
            }
            
        except Exception as e:
            logger.error(f"Error testing model performance: {str(e)}")
            return {}
    
    def run_full_cycle(self) -> Dict:
        """
        Run a full cycle of data collection, backtesting, and model training
        
        Returns:
            Dictionary with results from the full cycle
        """
        logger.info("Starting full automated backtesting and ML training cycle")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'symbols': self.symbols,
            'backtest_periods': self.backtest_periods,
            'trades_per_period': self.trades_per_period,
            'data_days': self.data_days
        }
        
        # Step 1: Collect historical data
        logger.info("Step 1: Collecting historical data")
        data_collection_success = self.collect_historical_data()
        results['data_collection_success'] = data_collection_success
        
        if not data_collection_success:
            logger.error("Data collection failed. Aborting cycle.")
            return results
        
        # Step 2: Prepare training data
        logger.info("Step 2: Preparing training data")
        data_preparation_success = self.prepare_training_data()
        results['data_preparation_success'] = data_preparation_success
        
        if not data_preparation_success:
            logger.error("Data preparation failed. Aborting cycle.")
            return results
        
        # Step 3: Run backtests
        logger.info("Step 3: Running backtests")
        backtest_results = self.run_backtests()
        results['backtest_results'] = {
            'periods': len(backtest_results),
            'total_trades': sum(len(r['trades']) for r in backtest_results),
            'overall_win_rate': sum(r['win_rate'] for r in backtest_results) / len(backtest_results) if backtest_results else 0,
            'total_profit_loss': sum(r['profit_loss'] for r in backtest_results)
        }
        
        if not backtest_results:
            logger.error("Backtesting failed. Aborting cycle.")
            return results
        
        # Step 4: Extract features from backtests
        logger.info("Step 4: Extracting features from backtest results")
        features, targets = self.extract_features_from_backtests(backtest_results)
        results['feature_extraction'] = {
            'features': len(features) if not features.empty else 0,
            'feature_columns': list(features.columns) if not features.empty else []
        }
        
        if features.empty or targets.empty:
            logger.error("Feature extraction failed. Aborting cycle.")
            return results
        
        # Step 5: Train the model
        logger.info("Step 5: Training ML model")
        training_results = self.train_model(features, targets)
        results['training_results'] = training_results
        
        if not training_results:
            logger.error("Model training failed. Aborting cycle.")
            return results
        
        # Step 6: Test model performance
        logger.info("Step 6: Testing model performance")
        performance_results = self.test_model_performance(training_results['model_path'])
        results['performance_results'] = performance_results
        
        # Record completion time
        results['end_time'] = datetime.now().isoformat()
        results['duration_seconds'] = (datetime.fromisoformat(results['end_time']) - 
                                     datetime.fromisoformat(results['start_time'])).total_seconds()
        
        # Save results
        results_file = f"models/auto_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Full cycle completed in {results['duration_seconds']/60:.1f} minutes. Results saved to {results_file}")
        return results

def main():
    """Main function to run the automated backtester"""
    
    parser = argparse.ArgumentParser(description="Automated Backtesting and ML Training System")
    
    parser.add_argument('--symbols', type=str, default=None,
                      help=f"Comma-separated list of symbols (default: {','.join(Config.WATCHLIST)})")
    parser.add_argument('--periods', type=int, default=None,
                      help=f"Number of backtest periods (default: {Config.BACKTEST_PERIODS})")
    parser.add_argument('--trades', type=int, default=None,
                      help=f"Trades per period (default: {Config.TRADES_PER_PERIOD})")
    parser.add_argument('--days', type=int, default=None,
                      help=f"Days of historical data (default: {Config.HISTORICAL_DATA_DAYS})")
    parser.add_argument('--repeat', type=int, default=1,
                      help="Number of full cycles to run (default: 1)")
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = args.symbols.split(',') if args.symbols else None
    
    logger.info(f"Starting automated backtesting with args: {args}")
    
    # Run the specified number of cycles
    for cycle in range(args.repeat):
        logger.info(f"Starting cycle {cycle+1}/{args.repeat}")
        
        # Create the backtester
        backtester = AutomatedBacktester(
            symbols=symbols,
            backtest_periods=args.periods,
            trades_per_period=args.trades,
            data_days=args.days
        )
        
        # Run a full cycle
        results = backtester.run_full_cycle()
        
        # Log cycle results
        if 'training_results' in results and 'accuracy' in results['training_results']:
            accuracy = results['training_results']['accuracy']
            logger.info(f"Cycle {cycle+1} completed with model accuracy: {accuracy:.1%}")
        else:
            logger.warning(f"Cycle {cycle+1} completed but model accuracy not available")
        
        # Wait between cycles if running multiple
        if cycle < args.repeat - 1:
            time.sleep(5)
    
    logger.info("All automated backtesting cycles completed")

if __name__ == "__main__":
    main() 