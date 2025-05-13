#!/usr/bin/env python
"""
Options Trading System Example

This script demonstrates the usage of the options trading system
with a simple example workflow.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.market_data import OptionChain, Quote, OptionQuote, OptionRight
from src.strategy.signal_generator import SignalManager
from src.strategy.signal import SignalType
from src.pricing.model_interface import ModelFactory
from src.risk.position_sizing import KellyPositionSizer
from src.execution.order_manager import OrderManager, TimeInForce, OrderType
from src.backtester.backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_option_chain() -> OptionChain:
    """Create a sample option chain for demonstration."""
    # Create an option chain for SPY
    chain = OptionChain(
        underlying_symbol="SPY",
        timestamp=datetime.now()
    )
    
    # Current price
    underlying_price = 450.0
    
    # Create options at different strikes
    expiration = datetime.now() + timedelta(days=30)
    
    for strike in range(430, 471, 5):
        # Calculate option prices using approximate Black-Scholes
        # This is just for demonstration
        strike_price = float(strike)
        time_to_expiry = 30.0 / 365.0
        volatility = 0.20
        
        # Simple approximation
        atm_price = underlying_price * volatility * np.sqrt(time_to_expiry) / 2.5
        
        # Calls decay as strike increases
        call_price = max(0.1, atm_price - 0.15 * abs(strike_price - underlying_price))
        # Puts increase as strike decreases
        put_price = max(0.1, atm_price - 0.15 * abs(strike_price - underlying_price))
        
        # Create implied volatility skew
        if strike_price < underlying_price:
            # Higher IV for OTM puts
            put_iv = volatility * (1.0 + 0.05 * (underlying_price - strike_price) / underlying_price)
            call_iv = volatility
        else:
            # Higher IV for OTM calls
            call_iv = volatility * (1.0 + 0.02 * (strike_price - underlying_price) / underlying_price)
            put_iv = volatility
        
        # Create bid/ask spread
        call_bid = call_price * 0.95
        call_ask = call_price * 1.05
        put_bid = put_price * 0.95
        put_ask = put_price * 1.05
        
        # Create call option
        call_quote = OptionQuote(
            timestamp=datetime.now(),
            bid_price=call_bid,
            ask_price=call_ask,
            bid_size=10,
            ask_size=10,
            exchange="MOCK",
            underlying_symbol="SPY",
            expiration=expiration,
            strike=strike_price,
            right=OptionRight.CALL,
            last_price=call_price,
            volume=100,
            open_interest=1000,
            implied_volatility=call_iv,
            delta=0.5,
            gamma=0.01,
            theta=-0.01,
            vega=0.1,
            rho=0.01
        )
        
        # Create put option
        put_quote = OptionQuote(
            timestamp=datetime.now(),
            bid_price=put_bid,
            ask_price=put_ask,
            bid_size=10,
            ask_size=10,
            exchange="MOCK",
            underlying_symbol="SPY",
            expiration=expiration,
            strike=strike_price,
            right=OptionRight.PUT,
            last_price=put_price,
            volume=100,
            open_interest=1000,
            implied_volatility=put_iv,
            delta=-0.5,
            gamma=0.01,
            theta=-0.01,
            vega=0.1,
            rho=-0.01
        )
        
        # Add to chain
        chain.add_option(call_quote)
        chain.add_option(put_quote)
    
    return chain


def generate_signals_example() -> None:
    """Example of generating signals from an option chain."""
    logger.info("Running signal generation example")
    
    # Create sample data
    chain = create_sample_option_chain()
    underlying_price = 450.0
    
    # Create signal manager
    signal_manager = SignalManager(
        risk_free_rate=0.05,
        min_edge=5.0,
        min_confidence=0.6
    )
    
    # Generate signals
    signals = signal_manager.generate_all_signals(chain, underlying_price)
    
    # Show results
    logger.info(f"Generated {len(signals)} signals")
    
    for i, signal in enumerate(signals[:5]):  # Show top 5
        logger.info(f"Signal {i+1}: {signal}")


def backtest_example() -> None:
    """Example of running a backtest."""
    logger.info("Running backtest example")
    
    # Create sample data for backtesting
    # In a real system, this would load historical data
    
    # Create synthetic price data
    days = 30
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Forward order
    
    prices = pd.DataFrame({
        'time': dates,
        'open': np.linspace(440, 460, days) + np.random.normal(0, 2, days),
        'high': np.linspace(440, 460, days) + np.random.normal(0, 4, days),
        'low': np.linspace(440, 460, days) - np.random.normal(0, 4, days),
        'close': np.linspace(440, 460, days) + np.random.normal(0, 2, days),
        'bid': np.linspace(440, 460, days) - 0.05,
        'ask': np.linspace(440, 460, days) + 0.05,
        'volume': np.random.randint(1000, 10000, days)
    })
    
    # Create a signal generator
    from src.strategy.spreads import SpreadSignalGenerator
    signal_gen = SpreadSignalGenerator()
    
    # Create a backtest
    backtest = Backtester(
        strategy_generator=signal_gen,
        initial_capital=100000.0
    )
    
    # Load market data
    market_data = {
        'SPY': prices
    }
    
    # Create option chains for each day
    option_chains = {}
    for date in dates:
        current_price = float(prices[prices['time'] == date]['close'].iloc[0])
        chain = create_sample_option_chain()
        # Adjust chain timestamp
        chain.timestamp = date
        
        # Add to option chains
        if date not in option_chains:
            option_chains[date] = {}
        option_chains[date]['SPY'] = chain
    
    # Load data into backtest
    backtest.load_market_data(market_data, option_chains)
    
    # Run backtest
    results = backtest.run()
    
    # Show results
    logger.info(f"Backtest results:")
    logger.info(f"Initial capital: ${results.initial_capital:,.2f}")
    logger.info(f"Final capital: ${results.final_capital:,.2f}")
    logger.info(f"Total return: {results.total_return:.2f}%")
    logger.info(f"Annualized return: {results.annualized_return:.2f}%")
    logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max drawdown: {results.max_drawdown:.2f}%")
    logger.info(f"Win rate: {results.performance_metrics.get('win_rate', 0):.2f}%")
    logger.info(f"Total trades: {results.performance_metrics.get('total_trades', 0)}")


def main() -> None:
    """Main entry point."""
    logger.info("Options Trading System Example")
    
    # Run signal generation example
    generate_signals_example()
    
    # Run backtest example
    backtest_example()


if __name__ == "__main__":
    main() 