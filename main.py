#!/usr/bin/env python
"""
Main entry point for the Black-Scholes Options Trading System
"""

import sys
import os
from market_data_config import load_market_data_credentials
from integrate_market_data import MarketDataIntegration

def main():
    """
    Main entry point for the application.
    
    This function loads credentials, initializes the Market Data integration,
    and generates trading signals based on current market data.
    """
    print("Black-Scholes Options Trading System")
    print("====================================")
    
    # Load Market Data API credentials
    try:
        credentials = load_market_data_credentials()
        print(f"Market Data API token loaded successfully")
    except Exception as e:
        print(f"Error loading Market Data API credentials: {e}")
        sys.exit(1)
    
    # Initialize MarketDataIntegration
    try:
        integration = MarketDataIntegration()
    except Exception as e:
        print(f"Error initializing Market Data integration: {e}")
        sys.exit(1)
    
    # Define symbols to scan
    symbols = ["SPY", "AAPL", "MSFT", "QQQ", "AMZN", "GOOG", "TSLA", "META"]
    
    # Generate trading signals
    try:
        print(f"Scanning {len(symbols)} symbols for trading signals...")
        signals = integration.get_option_signals(symbols, max_signals=15)
        
        if signals:
            print(f"\nFound {len(signals)} trading signals:")
            for i, signal in enumerate(signals):
                print(f"\nSignal {i+1}:")
                print(f"  {signal.symbol} {signal.option_type.upper()} ${signal.strike} expiring {signal.expiration}")
                print(f"  Current price: ${signal.current_price:.2f}")
                print(f"  Entry range: ${signal.entry_price_range[0]:.2f} - ${signal.entry_price_range[1]:.2f}")
                print(f"  Stop loss: ${signal.stop_loss:.2f}")
                print(f"  Target price: ${signal.target_price:.2f}")
                print(f"  Signal strength: {signal.signal_strength:.2f}")
                print(f"  Implied volatility: {signal.implied_volatility:.2f}")
                print(f"  Greeks: Delta {signal.delta:.2f}, Gamma {signal.gamma:.2f}, Theta {signal.theta:.2f}, Vega {signal.vega:.2f}")
                print(f"  Underlying price: ${signal.underlying_price:.2f}")
                print(f"  Days to expiry: {signal.time_to_expiry:.0f}")
                print(f"  Premium to strike ratio: {signal.premium_to_strike_ratio:.4f}")
        else:
            print("\nNo trading signals found meeting our criteria.")
    except Exception as e:
        print(f"Error generating trading signals: {e}")
        sys.exit(1)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 