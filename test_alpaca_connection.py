#!/usr/bin/env python
"""
Test script for Alpaca Markets API connection

This script tests the connection to Alpaca Markets API and retrieves some basic market data.
"""

import os
import sys
from alpaca_market_data import AlpacaMarketData
from alpaca_config import load_credentials

def main():
    """Test Alpaca market data integration"""
    
    # Load credentials from config file
    load_credentials()
    
    # Initialize Alpaca market data provider
    alpaca = AlpacaMarketData()
    
    # Print API URLs
    print("API URLs being used:")
    print(f"  Base URL: {alpaca.base_url}")
    print(f"  Options Base URL: {alpaca.options_base_url}")
    print(f"  Options Contracts URL: {alpaca.options_contracts_url}")
    
    # Test basic functionality
    test_symbols = ["SPY", "AAPL", "MSFT"]
    
    for symbol in test_symbols:
        print(f"\nTesting with {symbol}:")
        
        # Get current price
        price = alpaca.get_price(symbol)
        print(f"  Current price: ${price:.2f}")
        
        # Get technical indicators
        indicators = alpaca.get_technical_indicators(symbol)
        print(f"  RSI: {indicators['rsi']:.2f}")
        print(f"  1-day price change: {indicators['price_change_1d']*100:.2f}%")
        
        # Get options chain (this might be large)
        print(f"  Fetching options chain...")
        chain = alpaca.get_options_chain(symbol)
        
        if chain:
            print(f"  Found options data for {len(chain)} expiration dates")
            
            # Print a sample of the first expiration date
            first_exp = next(iter(chain))
            print(f"  Sample for expiration {first_exp}:")
            
            # Print first call and put
            if chain[first_exp]["calls"]:
                first_call_strike = next(iter(chain[first_exp]["calls"]))
                call = chain[first_exp]["calls"][first_call_strike]
                print(f"    Call @{first_call_strike}: Bid=${call['bid']:.2f}, Ask=${call['ask']:.2f}, IV={call['iv']:.1f}%")
            
            if chain[first_exp]["puts"]:
                first_put_strike = next(iter(chain[first_exp]["puts"]))
                put = chain[first_exp]["puts"][first_put_strike]
                print(f"    Put @{first_put_strike}: Bid=${put['bid']:.2f}, Ask=${put['ask']:.2f}, IV={put['iv']:.1f}%")
        else:
            print("  No options chain data available")
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 