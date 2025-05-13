#!/usr/bin/env python
"""
Options Trader System

This module integrates all components of the options trading system,
providing a complete pipeline from market data acquisition to signal generation
and trade execution.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from market_data_manager import MarketDataManager, AlpacaMarketData, YahooFinanceDataSource
from near_the_money_analyzer import NearTheMoneyAnalyzer
from enhanced_ml_model import EnhancedOptionsMLModel
from black_scholes import BlackScholes
from options_day_trader_sim import OptionSignal, Config

class OptionsTraderSystem:
    """
    Integrated options trading system that combines market data,
    analysis, pricing, and execution components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the options trader system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # System components
        self.market_data = None
        self.analyzer = None
        self.ml_model = None
        self.risk_manager = None
        self.trading_history = []
        self.positions = []
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Options Trader System initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load system configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "market_data": {
                "primary_source": "alpaca",
                "backup_sources": ["yahoo"],
                "retry_count": 3,
                "retry_delay": 2
            },
            "analyzer": {
                "moneyness_range": 0.10,
                "min_volume": 50,
                "min_open_interest": 100,
                "max_spread_pct": 0.10,
                "days_to_expiry_min": 5,
                "days_to_expiry_max": 45,
                "target_profit_pct": 0.20,
                "stop_loss_pct": 0.10
            },
            "ml_model": {
                "model_type": "hybrid",
                "auto_update": True,
                "update_frequency": 24  # hours
            },
            "risk_management": {
                "max_position_size_pct": 0.05,  # % of account
                "max_total_exposure_pct": 0.25,  # % of account
                "max_per_symbol_exposure_pct": 0.10,  # % of account
                "kelly_fraction": 0.25,  # 1/4 Kelly criterion
                "max_delta_exposure": 50000,  # Total delta exposure
                "max_gamma_exposure": 5000,  # Total gamma exposure
                "max_vega_exposure": 10000,  # Total vega exposure
                "max_theta_exposure": -2000  # Max negative theta
            },
            "account": {
                "initial_balance": 100000,
                "commission_per_contract": 0.65,
                "slippage_model": "percent",
                "slippage_value": 0.01  # 1% slippage
            },
            "watchlist": [
                "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"
            ],
            "signals": {
                "min_score": 0.35,
                "max_signals_per_symbol": 2,
                "max_total_signals": 20
            },
            "execution": {
                "trading_hours_start": "09:30",
                "trading_hours_end": "16:00",
                "timezone": "America/New_York"
            }
        }
        
        # Load from file if provided
        config = default_config
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Update default config with loaded values
                for section in loaded_config:
                    if section in config:
                        config[section].update(loaded_config[section])
                    else:
                        config[section] = loaded_config[section]
                        
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return config
    
    def _initialize_components(self):
        """Initialize system components"""
        # Initialize market data manager
        self._initialize_market_data()
        
        # Initialize analyzer
        self._initialize_analyzer()
        
        # Initialize ML model
        self._initialize_ml_model()
        
        # Initialize risk manager (currently handled internally)
        self._initialize_risk_manager()
    
    def _initialize_market_data(self):
        """Initialize market data component"""
        try:
            # Initialize Alpaca if configured
            if self.config["market_data"]["primary_source"] == "alpaca":
                try:
                    from alpaca_config import API_KEY, API_SECRET
                    alpaca = AlpacaMarketData(API_KEY, API_SECRET)
                except ImportError:
                    alpaca = AlpacaMarketData()  # Use env vars
                
                primary_source = alpaca
            else:
                primary_source = None
                
            # Initialize backup sources
            backup_sources = []
            if "yahoo" in self.config["market_data"]["backup_sources"]:
                try:
                    yahoo = YahooFinanceDataSource()
                    backup_sources.append(yahoo)
                except Exception as e:
                    logger.warning(f"Could not initialize Yahoo Finance: {e}")
            
            # Create market data manager
            if primary_source or backup_sources:
                if len(backup_sources) > 0 or (primary_source and len(backup_sources) == 0):
                    self.market_data = MarketDataManager(
                        primary_source=primary_source,
                        backup_sources=backup_sources
                    )
                    logger.info(f"Market data manager initialized with {len(backup_sources) + 1} sources")
                else:
                    self.market_data = backup_sources[0]
                    logger.info(f"Using single market data source: {self.market_data.__class__.__name__}")
            else:
                raise ValueError("No market data sources available")
                
        except Exception as e:
            logger.error(f"Error initializing market data: {e}")
            raise
    
    def _initialize_analyzer(self):
        """Initialize near-the-money analyzer"""
        try:
            # Create analyzer with market data and config
            analyzer_config = self.config["analyzer"]
            self.analyzer = NearTheMoneyAnalyzer(
                market_data=self.market_data,
                config=analyzer_config
            )
            logger.info("Near-the-money analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            raise
    
    def _initialize_ml_model(self):
        """Initialize ML model"""
        try:
            # Create ML model
            ml_config = self.config["ml_model"]
            self.ml_model = EnhancedOptionsMLModel(
                model_type=ml_config["model_type"]
            )
            logger.info(f"ML model initialized with type: {ml_config['model_type']}")
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            logger.warning("Continuing without ML model")
    
    def _initialize_risk_manager(self):
        """Initialize risk management component"""
        # Currently risk management is handled internally
        logger.info("Risk management initialized")
    
    def generate_signals(self) -> List[OptionSignal]:
        """
        Generate trading signals from the watchlist.
        
        Returns:
            List of option signals
        """
        logger.info("Generating trading signals...")
        
        # Get watchlist
        watchlist = self.config["watchlist"]
        max_per_symbol = self.config["signals"]["max_signals_per_symbol"]
        
        # Generate signals
        signals = self.analyzer.get_option_signals(watchlist, max_per_symbol)
        
        # Filter by minimum score
        min_score = self.config["signals"]["min_score"]
        signals = [s for s in signals if s.signal_strength >= min_score]
        
        # Limit total number of signals
        max_signals = self.config["signals"]["max_total_signals"]
        if len(signals) > max_signals:
            # Sort by signal strength (highest first)
            signals.sort(key=lambda s: s.signal_strength, reverse=True)
            signals = signals[:max_signals]
        
        logger.info(f"Generated {len(signals)} signals")
        return signals
    
    def analyze_signals(self, signals: List[OptionSignal]) -> pd.DataFrame:
        """
        Analyze signals to add Black-Scholes calculations and ML predictions.
        
        Args:
            signals: List of option signals
            
        Returns:
            DataFrame with enhanced signal analysis
        """
        if not signals:
            return pd.DataFrame()
        
        # Create result DataFrame
        results = []
        
        for signal in signals:
            # Calculate days to expiry
            days_to_expiry = (signal.expiration - datetime.now().date()).days
            T = BlackScholes.days_to_years(days_to_expiry)
            
            # Set up calculation inputs
            S = signal.underlying_price
            K = signal.strike
            r = 0.05  # Assumed risk-free rate
            sigma = signal.iv / 100 if signal.iv > 1 else signal.iv  # Convert to decimal
            option_type = signal.option_type
            
            # Calculate Black-Scholes price and Greeks
            try:
                bs_results = BlackScholes.price_and_greeks(S, K, T, r, sigma, option_type)
                
                # Create result row
                row = {
                    "symbol": signal.symbol,
                    "option_type": signal.option_type,
                    "strike": signal.strike,
                    "expiration": signal.expiration,
                    "days_to_expiry": days_to_expiry,
                    "underlying_price": signal.underlying_price,
                    "market_price": signal.current_price,
                    "bs_price": bs_results["price"][0],
                    "price_difference": signal.current_price - bs_results["price"][0],
                    "delta": bs_results["delta"][0],
                    "gamma": bs_results["gamma"][0],
                    "theta": bs_results["theta"][0],
                    "vega": bs_results["vega"][0],
                    "iv": signal.iv,
                    "volume": signal.volume,
                    "open_interest": signal.open_interest,
                    "bid": signal.entry_price_range[0],
                    "ask": signal.entry_price_range[1],
                    "signal_strength": signal.signal_strength
                }
                
                # Add ML prediction if available
                if self.ml_model:
                    try:
                        features = {
                            'option_type_C': 1 if signal.option_type == 'call' else 0,
                            'option_type_P': 1 if signal.option_type == 'put' else 0,
                            'moneyness': signal.strike / signal.underlying_price,
                            'relative_strike': (signal.strike - signal.underlying_price) / signal.underlying_price,
                            'days_to_expiry': days_to_expiry,
                            'implied_volatility': sigma,
                            'delta': abs(signal.delta),
                            'gamma': bs_results["gamma"][0],
                            'theta': bs_results["theta"][0],
                            'vega': bs_results["vega"][0],
                            'bid_ask_spread': signal.entry_price_range[1] - signal.entry_price_range[0],
                            'bid_ask_spread_percent': (signal.entry_price_range[1] - signal.entry_price_range[0]) / signal.entry_price_range[1],
                            'volume': signal.volume,
                            'open_interest': signal.open_interest,
                            'volume_oi_ratio': signal.volume / signal.open_interest if signal.open_interest > 0 else 0
                        }
                        
                        ml_prediction = self.ml_model.predict(features)
                        row["ml_prediction"] = ml_prediction
                    except Exception as e:
                        logger.warning(f"Error getting ML prediction: {e}")
                        row["ml_prediction"] = None
                
                results.append(row)
            except Exception as e:
                logger.warning(f"Error analyzing signal {signal.symbol} {signal.option_type} {signal.strike}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def rank_trades(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """
        Rank trades by combining signal strength, ML prediction, and risk metrics.
        
        Args:
            analysis: DataFrame with signal analysis
            
        Returns:
            DataFrame with ranked trades
        """
        if analysis.empty:
            return analysis
        
        # Calculate a combined score
        if "ml_prediction" in analysis.columns:
            # If ML prediction available, combine with signal strength
            analysis["combined_score"] = 0.5 * analysis["signal_strength"] + 0.5 * analysis["ml_prediction"]
        else:
            # Otherwise just use signal strength
            analysis["combined_score"] = analysis["signal_strength"]
        
        # Adjust score for mispricing (higher weight for underpriced options)
        analysis["price_diff_pct"] = analysis["price_difference"] / analysis["bs_price"]
        
        # Higher score for underpriced options (negative price difference)
        underpriced_mask = analysis["price_diff_pct"] < 0
        analysis.loc[underpriced_mask, "combined_score"] = analysis.loc[underpriced_mask, "combined_score"] * (1 + abs(analysis.loc[underpriced_mask, "price_diff_pct"]))
        
        # Lower score for overpriced options
        overpriced_mask = analysis["price_diff_pct"] > 0
        analysis.loc[overpriced_mask, "combined_score"] = analysis.loc[overpriced_mask, "combined_score"] * (1 - analysis.loc[overpriced_mask, "price_diff_pct"] * 0.5)
        
        # Sort by combined score (descending)
        ranked = analysis.sort_values("combined_score", ascending=False)
        
        return ranked
    
    def apply_risk_limits(self, ranked_trades: pd.DataFrame, account_value: float) -> pd.DataFrame:
        """
        Apply risk limits to filter trades.
        
        Args:
            ranked_trades: DataFrame with ranked trades
            account_value: Current account value
            
        Returns:
            DataFrame with trades that pass risk limits
        """
        if ranked_trades.empty:
            return ranked_trades
        
        # Get risk parameters
        risk_config = self.config["risk_management"]
        max_position_size = account_value * risk_config["max_position_size_pct"]
        max_total_exposure = account_value * risk_config["max_total_exposure_pct"]
        max_per_symbol_exposure = account_value * risk_config["max_per_symbol_exposure_pct"]
        
        # Filter out trades that exceed position size limit
        filtered = ranked_trades.copy()
        
        # Calculate position size (100 shares per contract)
        contract_size = 100
        filtered["notional_value"] = filtered["underlying_price"] * contract_size
        
        # Filter out trades that exceed max position size
        filtered = filtered[filtered["notional_value"] <= max_position_size]
        
        # Sort again
        filtered = filtered.sort_values("combined_score", ascending=False)
        
        # Calculate cumulative exposure
        filtered["cumulative_exposure"] = filtered["notional_value"].cumsum()
        
        # Filter by total exposure
        filtered = filtered[filtered["cumulative_exposure"] <= max_total_exposure]
        
        # Filter by per-symbol exposure (group by symbol)
        symbol_exposure = filtered.groupby("symbol")["notional_value"].sum()
        symbols_over_limit = symbol_exposure[symbol_exposure > max_per_symbol_exposure].index
        
        # For symbols over the limit, keep only the best trades up to the limit
        final_filtered = []
        for symbol in filtered["symbol"].unique():
            symbol_trades = filtered[filtered["symbol"] == symbol]
            
            if symbol in symbols_over_limit:
                # Keep only trades up to the limit
                symbol_trades = symbol_trades.sort_values("combined_score", ascending=False)
                cumulative = symbol_trades["notional_value"].cumsum()
                symbol_trades = symbol_trades[cumulative <= max_per_symbol_exposure]
            
            final_filtered.append(symbol_trades)
        
        # Combine all filtered trades
        if final_filtered:
            result = pd.concat(final_filtered)
            result = result.sort_values("combined_score", ascending=False)
        else:
            result = pd.DataFrame()
        
        return result
    
    def size_positions(self, filtered_trades: pd.DataFrame, account_value: float) -> pd.DataFrame:
        """
        Determine optimal position size using Kelly criterion.
        
        Args:
            filtered_trades: DataFrame with filtered trades
            account_value: Current account value
            
        Returns:
            DataFrame with position sizing
        """
        if filtered_trades.empty:
            return filtered_trades
        
        # Get Kelly fraction
        kelly_fraction = self.config["risk_management"]["kelly_fraction"]
        
        # Calculate Kelly position sizes
        sized = filtered_trades.copy()
        
        # Use either ML prediction or signal strength as win probability
        if "ml_prediction" in sized.columns:
            win_prob = sized["ml_prediction"]
        else:
            # Scale signal_strength to be a probability (assuming it's 0-1)
            win_prob = sized["signal_strength"]
        
        # Calculate potential payoffs
        # Assume profit target is 20% and stop loss is 10%
        profit_pct = self.config["analyzer"]["target_profit_pct"]
        loss_pct = self.config["analyzer"]["stop_loss_pct"]
        
        # Kelly formula: f* = (p/q) * (b/a) where:
        # p = win probability, q = loss probability (1-p)
        # b = profit on win, a = loss on loss
        # Simplifies to: f* = (p*(1+b) - 1)/b
        sized["kelly_pct"] = (win_prob * (1 + profit_pct) - 1) / profit_pct
        
        # Apply Kelly fraction and clip to reasonable values
        sized["kelly_pct"] = sized["kelly_pct"] * kelly_fraction
        sized["kelly_pct"] = sized["kelly_pct"].clip(0, 0.25)  # Max 25% per trade
        
        # Calculate dollar amount and contracts
        sized["position_size"] = sized["kelly_pct"] * account_value
        sized["max_contracts"] = (sized["position_size"] / sized["notional_value"]).astype(int)
        sized["max_contracts"] = sized["max_contracts"].clip(1, 10)  # At least 1, at most 10 contracts
        
        # Calculate actual position size
        sized["actual_position_size"] = sized["max_contracts"] * sized["notional_value"]
        
        return sized
    
    def get_trading_signals(self, account_value: Optional[float] = None) -> pd.DataFrame:
        """
        Get complete trading signals with analysis and position sizing.
        
        Args:
            account_value: Current account value (defaults to config value)
            
        Returns:
            DataFrame with complete trading signals
        """
        # Use default account value if not provided
        if account_value is None:
            account_value = self.config["account"]["initial_balance"]
        
        # Generate initial signals
        signals = self.generate_signals()
        
        # Analyze signals
        analysis = self.analyze_signals(signals)
        
        # Rank trades
        ranked = self.rank_trades(analysis)
        
        # Apply risk limits
        filtered = self.apply_risk_limits(ranked, account_value)
        
        # Size positions
        sized = self.size_positions(filtered, account_value)
        
        return sized
    
    def update_ml_model(self, trades: List[Dict]):
        """
        Update ML model with trade outcomes.
        
        Args:
            trades: List of completed trades with outcomes
        """
        if not self.ml_model or not trades:
            return
            
        try:
            # Extract features and outcomes
            features_list = []
            outcomes = []
            
            for trade in trades:
                if "features" in trade and "win" in trade:
                    features_list.append(trade["features"])
                    outcomes.append(trade["win"])
            
            if features_list:
                # Update model
                self.ml_model.add_batch_training_data(features_list, outcomes)
                
                # Save updated model
                self.ml_model.save_model()
                
                logger.info(f"ML model updated with {len(features_list)} new training examples")
        except Exception as e:
            logger.error(f"Error updating ML model: {e}")
    
    def run_trading_cycle(self):
        """Run a complete trading cycle"""
        logger.info("Starting trading cycle")
        
        try:
            # Get trading signals
            signals_df = self.get_trading_signals()
            
            if signals_df.empty:
                logger.warning("No valid trading signals found")
                return
            
            # Log top signals
            top_n = min(5, len(signals_df))
            logger.info(f"Top {top_n} trading signals:")
            for i, row in signals_df.head(top_n).iterrows():
                logger.info(f"{row['symbol']} {row['option_type'].upper()} ${row['strike']} "
                          f"(Score: {row['combined_score']:.2f}, Contracts: {row['max_contracts']})")
            
            # In a real system, this would execute trades
            # For now, we just simulate and track
            
            # Add to trading history
            for i, row in signals_df.iterrows():
                trade = {
                    "timestamp": datetime.now(),
                    "symbol": row["symbol"],
                    "option_type": row["option_type"],
                    "strike": row["strike"],
                    "expiration": row["expiration"],
                    "entry_price": row["market_price"],
                    "quantity": row["max_contracts"],
                    "position_size": row["actual_position_size"],
                    "signal_strength": row["signal_strength"],
                    "ml_prediction": row.get("ml_prediction", None),
                    "combined_score": row["combined_score"],
                    "status": "pending"
                }
                self.trading_history.append(trade)
            
            logger.info(f"Added {len(signals_df)} trades to history")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def simulate_trading_session(self, days: int = 5):
        """
        Simulate a trading session over multiple days.
        
        Args:
            days: Number of days to simulate
        """
        logger.info(f"Simulating {days} days of trading")
        
        account_value = self.config["account"]["initial_balance"]
        completed_trades = []
        
        for day in range(1, days + 1):
            logger.info(f"=== Day {day} ===")
            logger.info(f"Account value: ${account_value:.2f}")
            
            # Run trading cycle
            self.run_trading_cycle()
            
            # Simulate trade outcomes
            new_trades = [t for t in self.trading_history if t["status"] == "pending"]
            
            for trade in new_trades:
                # Simulate outcome
                import random
                
                # Win probability based on combined score or ML prediction
                if trade.get("ml_prediction"):
                    win_prob = trade["ml_prediction"]
                else:
                    win_prob = trade["combined_score"]
                
                # Determine outcome
                is_win = random.random() < win_prob
                
                # Calculate P&L
                if is_win:
                    # Win scenario
                    profit_pct = self.config["analyzer"]["target_profit_pct"]
                    pnl = trade["entry_price"] * profit_pct * 100 * trade["quantity"]  # 100 shares per contract
                else:
                    # Loss scenario
                    loss_pct = self.config["analyzer"]["stop_loss_pct"]
                    pnl = -trade["entry_price"] * loss_pct * 100 * trade["quantity"]
                
                # Update account value
                account_value += pnl
                
                # Update trade record
                trade["status"] = "completed"
                trade["exit_date"] = datetime.now() + timedelta(days=random.randint(1, 3))
                trade["pnl"] = pnl
                trade["win"] = is_win
                
                # Create features for ML training
                days_to_expiry = (trade["expiration"] - datetime.now().date()).days
                
                trade["features"] = {
                    'option_type_C': 1 if trade["option_type"] == 'call' else 0,
                    'option_type_P': 1 if trade["option_type"] == 'put' else 0,
                    'moneyness': float(trade["strike"]) / float(trade.get("underlying_price", 100)),
                    'relative_strike': (float(trade["strike"]) - float(trade.get("underlying_price", 100))) / float(trade.get("underlying_price", 100)),
                    'days_to_expiry': days_to_expiry,
                    'implied_volatility': float(trade.get("iv", 0.3)),
                    'delta': abs(float(trade.get("delta", 0.5))),
                    'gamma': float(trade.get("gamma", 0.03)),
                    'theta': float(trade.get("theta", -0.02)),
                    'vega': float(trade.get("vega", 0.1)),
                    'bid_ask_spread': float(trade.get("ask", 0)) - float(trade.get("bid", 0)),
                    'bid_ask_spread_percent': (float(trade.get("ask", 1)) - float(trade.get("bid", 0))) / float(trade.get("ask", 1)),
                    'volume': float(trade.get("volume", 1000)),
                    'open_interest': float(trade.get("open_interest", 5000)),
                    'volume_oi_ratio': float(trade.get("volume", 1000)) / float(trade.get("open_interest", 5000))
                }
                
                completed_trades.append(trade)
            
            # Update ML model with completed trades
            if completed_trades and self.ml_model and self.config["ml_model"]["auto_update"]:
                self.update_ml_model(completed_trades)
                completed_trades = []
            
            # Print daily summary
            day_trades = [t for t in self.trading_history if t["status"] == "completed" and 
                          t["exit_date"].date() == (datetime.now() + timedelta(days=day)).date()]
            
            if day_trades:
                day_pnl = sum(t["pnl"] for t in day_trades)
                win_count = sum(1 for t in day_trades if t["win"])
                win_rate = win_count / len(day_trades) * 100
                
                logger.info(f"Day {day} Results:")
                logger.info(f"  Trades: {len(day_trades)}")
                logger.info(f"  Win Rate: {win_rate:.1f}%")
                logger.info(f"  P&L: ${day_pnl:.2f}")
            else:
                logger.info("No trades completed today")
        
        # Calculate overall results
        completed = [t for t in self.trading_history if t["status"] == "completed"]
        if completed:
            total_pnl = sum(t["pnl"] for t in completed)
            win_count = sum(1 for t in completed if t["win"])
            win_rate = win_count / len(completed) * 100
            
            logger.info(f"\n=== Simulation Summary ===")
            logger.info(f"Starting Account: ${self.config['account']['initial_balance']:.2f}")
            logger.info(f"Ending Account: ${account_value:.2f}")
            logger.info(f"Total P&L: ${total_pnl:.2f} ({total_pnl / self.config['account']['initial_balance'] * 100:.1f}%)")
            logger.info(f"Total Trades: {len(completed)}")
            logger.info(f"Win Rate: {win_rate:.1f}%")
        else:
            logger.info("No trades completed during simulation")

# Example usage
if __name__ == "__main__":
    # Create trader system
    trader = OptionsTraderSystem()
    
    # Run simulation
    trader.simulate_trading_session(days=5) 