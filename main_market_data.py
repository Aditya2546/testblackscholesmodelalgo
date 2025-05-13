#!/usr/bin/env python
"""
Main entry point for the Black-Scholes Options Trading System using Market Data API
"""

import sys
import os
from market_data_config import load_market_data_credentials
from integrate_market_data import MarketDataIntegration

def main():
    """
    Main entry point for the application.
    
    This function loads credentials, initializes the Market Data API integration,
    and generates trading signals based on current market data.
    """
    print("Black-Scholes Options Trading System - Market Data API")
    print("=====================================================")
    
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
        print(f"Error initializing Market Data API integration: {e}")
        sys.exit(1)
    
    # Start with a smaller set of popular symbols for testing
    # Full set: ["SPY", "AAPL", "MSFT", "QQQ", "AMZN", "GOOG", "TSLA", "META"]
    symbols = ["SPY", "AAPL", "QQQ"]
    
    # Generate trading signals
    try:
        print(f"Scanning {len(symbols)} symbols for trading signals...")
        signals = integration.get_option_signals(symbols, max_signals=5)
        
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
            
        # Demonstrate the option data retrieval capability
        print("\nExample option data for AAPL:")
        
        # Calculate a near-term expiration (third Friday of next month)
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        # Use a near-term expiration date (30-45 days out)
        expiry_date = today + timedelta(days=40)
        # Adjust to the nearest Friday
        while expiry_date.weekday() != 4:  # 4 is Friday
            expiry_date += timedelta(days=1)
            
        expiration_date = expiry_date.strftime("%Y-%m-%d")
        
        print(f"Using expiration date: {expiration_date}")
        
        # Create an option symbol for a sample AAPL option
        option_symbol = integration.api.format_option_symbol(
            ticker="AAPL",
            expiration_date=expiration_date,
            strike_price=210.0,  # Using a more current price estimate
            option_type="C"
        )
        
        print(f"Fetching data for option: {option_symbol}")
        
        # Get and display the option data
        option_data = integration.get_option_data(option_symbol)
        if option_data and option_data.get("s") in ["ok", "error"]:
            print(f"  Symbol: {option_symbol}")
            
            # Extract data, handling both array and direct formats
            extracted_data = integration.extract_option_data(option_data)
            
            print(f"  Bid: ${extracted_data.get('bid', 'N/A')}")
            print(f"  Ask: ${extracted_data.get('ask', 'N/A')}")
            print(f"  Last: ${extracted_data.get('last', 'N/A')}")
            print(f"  IV: {extracted_data.get('iv', 'N/A')}")
            print(f"  Delta: {extracted_data.get('delta', 'N/A')}")
            print(f"  Open Interest: {extracted_data.get('open_interest', 'N/A')}")
            print(f"  Volume: {extracted_data.get('volume', 'N/A')}")
            
            if "underlyingPrice" in extracted_data or "underlying_price" in extracted_data:
                underlying = extracted_data.get("underlyingPrice", extracted_data.get("underlying_price", "N/A"))
                print(f"  Underlying Price: ${underlying}")
        else:
            print("  Unable to fetch option data")
            if option_data and "errmsg" in option_data:
                print(f"  Error: {option_data['errmsg']}")
        
    except Exception as e:
        print(f"Error generating trading signals: {e}")
        sys.exit(1)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 