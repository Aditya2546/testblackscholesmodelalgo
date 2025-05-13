#!/usr/bin/env python
"""
Test script for Near-the-Money Options Analyzer

This script demonstrates the use of the NearTheMoneyAnalyzer to find
and evaluate options trading opportunities close to current market prices.
"""

import logging
import sys
import pandas as pd
import datetime
from tabulate import tabulate
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the analyzer
from near_the_money_analyzer import NearTheMoneyAnalyzer
from alpaca_market_data import AlpacaMarketData

# Import configuration
from alpaca_config import API_KEY, API_SECRET

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

def test_near_money_analyzer():
    """Test the NearTheMoneyAnalyzer"""
    # Create market data provider
    market_data = AlpacaMarketData(API_KEY, API_SECRET)
    
    # Create analyzer
    analyzer = NearTheMoneyAnalyzer(market_data)
    
    # Test symbols
    test_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
    
    # Get option signals
    signals = analyzer.get_option_signals(test_symbols, max_per_symbol=2)
    
    # Print results
    print_option_table(signals, "Top Near-Money Options Signals")
    
    # Test with specific expiration and strikes
    print("\nTesting specific strikes and expiration...")
    symbol = "AAPL"
    expiration = (datetime.datetime.now() + datetime.timedelta(days=45)).strftime("%Y-%m-%d")
    
    # Get current price
    price = market_data.get_price(symbol)
    print(f"{symbol} current price: ${price}")
    
    # Generate strikes around current price
    strikes = analyzer._generate_strikes_around_price(price)
    print(f"Testing strikes: {strikes}")
    
    # Get options data
    options = analyzer.get_options_by_strike_range(symbol, strikes, expiration)
    
    # Create signals for display
    specific_signals = []
    for option_type in ["calls", "puts"]:
        for strike_str, option_data in options[option_type].items():
            # Score this option
            score = analyzer.score_option(option_data, price, option_type[:-1])
            if score > 0.0:  # Include all for testing
                mid_price = (option_data["bid"] + option_data["ask"]) / 2
                signal = None
                try:
                    from options_day_trader_sim import OptionSignal
                    strike = float(strike_str)
                    expiry_date = datetime.datetime.strptime(expiration, "%Y-%m-%d").date()
                    
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
                    logger.error(f"Error creating signal: {e}")
    
    print_option_table(specific_signals, f"{symbol} Options Analysis for {expiration}")

if __name__ == "__main__":
    test_near_money_analyzer() 