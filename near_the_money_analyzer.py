#!/usr/bin/env python
"""
Near-the-Money Options Analyzer

This module focuses on analyzing options near the current market price,
with emphasis on volume, open interest, and liquidity metrics.
It's designed to work with the Alpaca API to retrieve real-time options data.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import requests
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from market_data_manager import MarketDataManager, AlpacaMarketData, YahooFinanceDataSource
from options_day_trader_sim import OptionSignal, Config

class NearTheMoneyAnalyzer:
    """
    Analyzes options near the current market price to find optimal trading opportunities.
    Uses Black-Scholes pricing and enhanced ML model predictions where available.
    """
    
    def __init__(self, market_data=None, config: Dict = None):
        """
        Initialize the near-the-money analyzer.
        
        Args:
            market_data: Market data provider (AlpacaMarketData or MarketDataManager)
            config: Configuration dictionary
        """
        # Initialize market data provider with fallback capability
        if market_data is None:
            # Try to set up market data manager with available sources
            try:
                # Initialize Alpaca as primary
                try:
                    from alpaca_config import API_KEY, API_SECRET
                    alpaca = AlpacaMarketData(API_KEY, API_SECRET)
                except ImportError:
                    alpaca = AlpacaMarketData()  # Use env vars
                
                # Try to initialize Yahoo as backup
                yahoo = None
                try:
                    yahoo = YahooFinanceDataSource()
                except ImportError:
                    logger.warning("YahooFinanceDataSource not available, install with: pip install yfinance")
                
                # Create manager with available sources
                data_sources = [source for source in [alpaca, yahoo] if source is not None]
                if len(data_sources) > 1:
                    self.market_data = MarketDataManager(
                        primary_source=alpaca, 
                        backup_sources=[source for source in data_sources if source != alpaca]
                    )
                    logger.info(f"Using MarketDataManager with {len(data_sources)} sources")
                else:
                    self.market_data = alpaca
                    logger.info("Using AlpacaMarketData as the only available source")
            except Exception as e:
                logger.error(f"Error initializing market data: {e}")
                raise
        else:
            self.market_data = market_data
        
        # Set configuration
        self.config = config or {
            "moneyness_range": 0.10,  # Default ±10% from current price
            "min_volume": 50,         # Minimum acceptable volume
            "min_open_interest": 100, # Minimum acceptable open interest
            "max_spread_pct": 0.10,   # Maximum bid-ask spread as percentage
            "days_to_expiry_min": 5,  # Minimum days to expiration
            "days_to_expiry_max": 45, # Maximum days to expiration
            "target_profit_pct": 0.20, # Target profit percentage
            "stop_loss_pct": 0.10,    # Stop loss percentage
            "volume_weight": 0.25,    # Weight for volume in scoring
            "open_interest_weight": 0.20, # Weight for open interest in scoring
            "liquidity_weight": 0.25, # Weight for liquidity (bid-ask spread)
            "delta_weight": 0.15,     # Weight for delta (proximity to 0.50)
            "iv_weight": 0.15         # Weight for implied volatility
        }
        
        # Load ML model if available
        self.ml_model = None
        self.ml_features = None
        self.load_ml_model()
    
    def load_ml_model(self):
        """Load the latest ML model for enhanced option predictions"""
        try:
            import joblib
            import glob
            
            # Find the latest model file
            model_files = glob.glob("models/options_model_*.joblib")
            if model_files:
                latest_model = max(model_files)
                logger.info(f"Using model: {latest_model}")
                self.ml_model = joblib.load(latest_model)
                
                # Also load feature definitions if available
                try:
                    feature_file = latest_model.replace('.joblib', '_features.json')
                    if os.path.exists(feature_file):
                        with open(feature_file, 'r') as f:
                            self.ml_features = json.load(f)
                            logger.info(f"Loaded feature definitions from {feature_file}")
                except Exception as e:
                    logger.warning(f"Could not load ML feature definitions: {e}")
            else:
                logger.warning("No ML model found. Will use conventional scoring only.")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
    
    def get_near_the_money_options(self, symbol: str, price: float = None) -> Dict:
        """
        Get options near the current market price.
        
        Args:
            symbol: Stock symbol
            price: Current stock price (will fetch if not provided)
            
        Returns:
            Dictionary with calls and puts near the money
        """
        logger.info(f"Analyzing near-the-money options for {symbol}")
        
        # Get current price if not provided
        if price is None:
            price = self.market_data.get_price(symbol)
            logger.info(f"Retrieved {symbol} price: ${price}")
        
        # Calculate price range for near-the-money options
        moneyness_range = self.config["moneyness_range"]
        min_strike = price * (1 - moneyness_range)
        max_strike = price * (1 + moneyness_range)
        
        # Get options chain
        options_chain = self.market_data.get_options_chain(symbol)
        
        # Filter for near-the-money options
        near_money_options = {
            "calls": {},
            "puts": {}
        }
        
        # Get today's date for expiry filtering
        today = datetime.now().date()
        min_expiry = today + timedelta(days=self.config["days_to_expiry_min"])
        max_expiry = today + timedelta(days=self.config["days_to_expiry_max"])
        
        # Process each expiration date
        for exp_date, exp_data in options_chain.items():
            # Convert to datetime for comparison
            expiry_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
            
            # Skip if outside our expiry range
            if expiry_date < min_expiry or expiry_date > max_expiry:
                continue
            
            # Process calls
            for strike_str, call_data in exp_data["calls"].items():
                strike = float(strike_str)
                
                # Check if near the money
                if min_strike <= strike <= max_strike:
                    # Check minimum requirements
                    if (call_data.get("volume", 0) >= self.config["min_volume"] and
                        call_data.get("open_interest", 0) >= self.config["min_open_interest"]):
                        
                        # Calculate spread percentage
                        bid = call_data.get("bid", 0)
                        ask = call_data.get("ask", 0)
                        if bid > 0 and ask > 0:
                            spread_pct = (ask - bid) / ask
                            
                            # Only include if spread is acceptable
                            if spread_pct <= self.config["max_spread_pct"]:
                                # Store this option
                                if exp_date not in near_money_options["calls"]:
                                    near_money_options["calls"][exp_date] = {}
                                near_money_options["calls"][exp_date][strike_str] = call_data
            
            # Process puts
            for strike_str, put_data in exp_data["puts"].items():
                strike = float(strike_str)
                
                # Check if near the money
                if min_strike <= strike <= max_strike:
                    # Check minimum requirements
                    if (put_data.get("volume", 0) >= self.config["min_volume"] and
                        put_data.get("open_interest", 0) >= self.config["min_open_interest"]):
                        
                        # Calculate spread percentage
                        bid = put_data.get("bid", 0)
                        ask = put_data.get("ask", 0)
                        if bid > 0 and ask > 0:
                            spread_pct = (ask - bid) / ask
                            
                            # Only include if spread is acceptable
                            if spread_pct <= self.config["max_spread_pct"]:
                                # Store this option
                                if exp_date not in near_money_options["puts"]:
                                    near_money_options["puts"][exp_date] = {}
                                near_money_options["puts"][exp_date][strike_str] = put_data
        
        return near_money_options
    
    def score_option(self, option_data: Dict, price: float, option_type: str) -> float:
        """
        Score an option based on various factors.
        
        Args:
            option_data: Option data dictionary
            price: Current stock price
            option_type: 'call' or 'put'
            
        Returns:
            Score between 0 and 1
        """
        # Get values with defaults
        volume = option_data.get("volume", 0)
        open_interest = option_data.get("open_interest", 0)
        bid = option_data.get("bid", 0)
        ask = option_data.get("ask", 0)
        delta = abs(option_data.get("delta", 0.5))
        iv = option_data.get("iv", 30.0)
        strike = float(option_data.get("strike", price))
        
        # Calculate metrics
        spread_pct = (ask - bid) / ask if ask > 0 else 1.0
        moneyness = abs(1 - (strike / price))
        
        # Volume score (higher is better)
        volume_score = min(1.0, volume / 1000)
        
        # Open interest score (higher is better)
        oi_score = min(1.0, open_interest / 5000)
        
        # Liquidity score (lower spread is better)
        liquidity_score = 1.0 - spread_pct
        
        # Delta score (closer to 0.5 is better)
        delta_score = 1.0 - abs(delta - 0.5)
        
        # Implied volatility score (moderate IV is better)
        # We prefer options with IV between 20-40%
        iv_norm = iv / 100.0 if iv > 1.0 else iv  # Normalize to 0-1 range
        iv_score = 1.0 - abs(iv_norm - 0.3)
        
        # Calculate combined score
        score = (
            volume_score * self.config["volume_weight"] +
            oi_score * self.config["open_interest_weight"] +
            liquidity_score * self.config["liquidity_weight"] +
            delta_score * self.config["delta_weight"] +
            iv_score * self.config["iv_weight"]
        )
        
        # Apply ML model if available
        if self.ml_model:
            try:
                # Calculate days to expiry if available
                days_to_expiry = option_data.get('days_to_expiry', 30)
                if "expiration" in option_data:
                    exp_date = datetime.strptime(option_data["expiration"], "%Y-%m-%d").date()
                    today = datetime.now().date()
                    days_to_expiry = (exp_date - today).days
                
                # Prepare features for ML model
                features = pd.DataFrame([{
                    'option_type_C': 1 if option_type == 'call' else 0,
                    'option_type_P': 1 if option_type == 'put' else 0,
                    'moneyness': strike / price,
                    'relative_strike': (strike - price) / price,
                    'days_to_expiry': days_to_expiry,
                    'implied_volatility': iv_norm,
                    'delta': delta,
                    'gamma': option_data.get('gamma', 0.01),
                    'theta': option_data.get('theta', -0.01),
                    'vega': option_data.get('vega', 0.1),
                    'bid_ask_spread': ask - bid,
                    'bid_ask_spread_percent': spread_pct,
                    'volume': volume,
                    'open_interest': open_interest,
                    'volume_oi_ratio': volume / open_interest if open_interest > 0 else 0
                }])
                
                # If we have custom feature definitions, use those
                if self.ml_features and 'columns' in self.ml_features:
                    # Ensure all required features are present
                    for col in self.ml_features['columns']:
                        if col not in features.columns:
                            features[col] = 0  # Default value
                    
                    # Reorder columns to match model expectations
                    features = features[self.ml_features['columns']]
                
                # Get ML prediction (probability of being profitable)
                ml_prob = self.ml_model.predict_proba(features)[0][1]
                
                # Combine conventional score with ML score (equal weight)
                score = 0.5 * score + 0.5 * ml_prob
            except Exception as e:
                logger.warning(f"Error applying ML model: {e}")
        
        return score
    
    def get_option_signals(self, symbols: List[str], max_per_symbol: int = 2) -> List[OptionSignal]:
        """
        Get option trading signals for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            max_per_symbol: Maximum signals per symbol
            
        Returns:
            List of OptionSignal objects
        """
        signals = []
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}")
            
            try:
                # Get current price
                price = self.market_data.get_price(symbol)
                logger.info(f"Updated {symbol} price to ${price} based on market data")
                
                # Get near-the-money options
                near_money_options = self.get_near_the_money_options(symbol, price)
                
                # Score and rank calls
                call_candidates = []
                for exp_date, strikes in near_money_options["calls"].items():
                    expiry_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
                    for strike_str, call_data in strikes.items():
                        strike = float(strike_str)
                        score = self.score_option(call_data, price, "call")
                        call_candidates.append({
                            "symbol": symbol,
                            "option_type": "call",
                            "strike": strike,
                            "expiration": expiry_date,
                            "data": call_data,
                            "score": score
                        })
                
                # Score and rank puts
                put_candidates = []
                for exp_date, strikes in near_money_options["puts"].items():
                    expiry_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
                    for strike_str, put_data in strikes.items():
                        strike = float(strike_str)
                        score = self.score_option(put_data, price, "put")
                        put_candidates.append({
                            "symbol": symbol,
                            "option_type": "put",
                            "strike": strike,
                            "expiration": expiry_date,
                            "data": put_data,
                            "score": score
                        })
                
                # Sort by score (highest first)
                call_candidates.sort(key=lambda x: x["score"], reverse=True)
                put_candidates.sort(key=lambda x: x["score"], reverse=True)
                
                # Take top signals
                top_calls = call_candidates[:max_per_symbol]
                top_puts = put_candidates[:max_per_symbol]
                
                # Convert to OptionSignal objects
                for candidate in top_calls + top_puts:
                    if candidate["score"] > 0.35:  # Minimum score threshold
                        option_data = candidate["data"]
                        mid_price = (option_data["bid"] + option_data["ask"]) / 2
                        entry_low = option_data["bid"]
                        entry_high = min(mid_price * 1.02, option_data["ask"])
                        
                        signal = OptionSignal(
                            symbol=candidate["symbol"],
                            option_type=candidate["option_type"],
                            strike=candidate["strike"],
                            expiration=candidate["expiration"],
                            current_price=mid_price,
                            underlying_price=price,
                            entry_price_range=(entry_low, entry_high),
                            stop_loss=mid_price * (1 - self.config["stop_loss_pct"]),
                            target_price=mid_price * (1 + self.config["target_profit_pct"]),
                            signal_strength=candidate["score"],
                            volume=option_data.get("volume", 0),
                            open_interest=option_data.get("open_interest", 0),
                            iv=option_data.get("iv", 30.0),
                            delta=option_data.get("delta", 0.5)
                        )
                        
                        signals.append(signal)
                        logger.info(f"Added signal for {symbol} {candidate['option_type'].upper()} ${candidate['strike']}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return signals
    
    def get_options_by_strike_range(self, symbol: str, strikes: List[float], expiration: str = None) -> Dict:
        """
        Get options for specific strikes near the current price.
        
        Args:
            symbol: Stock symbol
            strikes: List of strike prices
            expiration: Target expiration date (optional)
            
        Returns:
            Dictionary with calls and puts for those strikes
        """
        # Get current price
        price = self.market_data.get_price(symbol)
        logger.info(f"Updated {symbol} price to ${price} based on market data")
        
        # Generate strikes if not provided (default to 5 strikes around current price)
        if not strikes:
            strikes = self._generate_strikes_around_price(price)
        
        # Get options chain
        options_chain = self.market_data.get_options_chain(symbol)
        
        # Find expiration if not provided
        if expiration is None:
            # Find closest monthly expiration
            today = datetime.now().date()
            best_exp = None
            best_days = float('inf')
            
            for exp_date in options_chain.keys():
                exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d").date()
                days_to_exp = (exp_datetime - today).days
                
                # Look for expiration 30-60 days out
                if 20 <= days_to_exp <= 60:
                    if abs(days_to_exp - 45) < best_days:
                        best_days = abs(days_to_exp - 45)
                        best_exp = exp_date
            
            if best_exp:
                expiration = best_exp
            else:
                # Use first available expiration if no good match
                expiration = list(options_chain.keys())[0] if options_chain else None
        
        # Initialize results
        results = {
            "calls": {},
            "puts": {}
        }
        
        # Filter for our target expiration
        if expiration in options_chain:
            exp_data = options_chain[expiration]
            
            # Get data for our target strikes
            for strike in strikes:
                strike_str = str(strike)
                
                # Check calls
                if strike_str in exp_data["calls"]:
                    logger.info(f"Checking {symbol}{expiration}C{strike_str.zfill(8)}...")
                    results["calls"][strike_str] = exp_data["calls"][strike_str]
                
                # Check puts
                if strike_str in exp_data["puts"]:
                    logger.info(f"Checking {symbol}{expiration}P{strike_str.zfill(8)}...")
                    results["puts"][strike_str] = exp_data["puts"][strike_str]
        
        return results
    
    def _generate_strikes_around_price(self, price: float, num_strikes: int = 5) -> List[float]:
        """
        Generate a list of strikes centered around a price.
        
        Args:
            price: Current price
            num_strikes: Number of strikes to generate
            
        Returns:
            List of strike prices
        """
        # Round to nearest 5
        base_strike = round(price / 5) * 5
        
        # Generate strikes (2 below, current, 2 above)
        half_count = num_strikes // 2
        if num_strikes % 2 == 1:
            # Odd number of strikes
            strikes = [base_strike + (i - half_count) * 5 for i in range(num_strikes)]
        else:
            # Even number of strikes
            strikes = [base_strike + (i - half_count) * 5 + 2.5 for i in range(num_strikes)]
        
        return [max(5.0, strike) for strike in strikes]  # Ensure no strikes below $5
    
    def add_training_data(self, outcome_data):
        """
        Add training data to improve ML model.
        
        Args:
            outcome_data: Dictionary with trade outcomes for learning
        """
        if not self.ml_model:
            logger.warning("Cannot add training data - no ML model available")
            return False
            
        try:
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            import os
            from datetime import datetime
            
            # Create training data from outcomes
            X_new = []
            y_new = []
            
            for outcome in outcome_data:
                # Extract features
                features = outcome.get('features', {})
                if not features:
                    continue
                    
                # Extract outcome (1 for profit, 0 for loss)
                result = 1 if outcome.get('profit', 0) > 0 else 0
                
                # Add to training data
                X_new.append(list(features.values()))
                y_new.append(result)
                
            if not X_new:
                logger.warning("No valid training data provided")
                return False
                
            # Convert to numpy arrays
            X_new = np.array(X_new)
            y_new = np.array(y_new)
            
            # If model is not RandomForestClassifier, create a new one
            if not isinstance(self.ml_model, RandomForestClassifier):
                self.ml_model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42
                )
                
            # Partial fit with new data
            self.ml_model.fit(X_new, y_new)
            
            # Save updated model
            os.makedirs("models", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/options_model_{timestamp}.joblib"
            joblib.dump(self.ml_model, model_path)
            
            # Also save feature names if available
            if self.ml_features:
                feature_path = f"models/options_model_{timestamp}_features.json"
                with open(feature_path, 'w') as f:
                    json.dump(self.ml_features, f)
            
            logger.info(f"ML model updated and saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating ML model: {e}")
            return False


if __name__ == "__main__":
    # Test the analyzer
    try:
        from alpaca_config import API_KEY, API_SECRET
        
        # Create analyzer (will auto-initialize market data)
        analyzer = NearTheMoneyAnalyzer()
        
        # Test with a few symbols
        test_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
        
        # Generate estimated strikes for each symbol
        for symbol in test_symbols:
            try:
                # Get current price
                price = analyzer.market_data.get_price(symbol)
                print(f"Processing {symbol} with price: ${price}")
                
                # Generate strikes around current price
                strikes = analyzer._generate_strikes_around_price(price)
                print(f"Generated strikes for {symbol}: {strikes}")
                
                # Get options for these strikes (closest expiration ~45 days out)
                options = analyzer.get_options_by_strike_range(symbol, strikes)
                
                # Check if we found viable options and score them
                for option_type in ["calls", "puts"]:
                    for strike_str, option_data in options[option_type].items():
                        # Score this option
                        score = analyzer.score_option(option_data, price, option_type[:-1])  # Remove 's' from calls/puts
                        
                        # If score is good, print details
                        if score > 0.35:
                            mid_price = (option_data["bid"] + option_data["ask"]) / 2
                            print(f"Added signal for {symbol} {option_type[:-1].upper()} ${strike_str}")
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
    except Exception as e:
        print(f"Error running analyzer test: {e}") 