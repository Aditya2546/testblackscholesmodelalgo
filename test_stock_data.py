#!/usr/bin/env python
"""
Test script for retrieving basic stock data from Alpaca Markets API
"""

from alpaca_market_data import AlpacaMarketData
from alpaca_config import load_credentials
import time

def main():
    # Load credentials
    load_credentials()
    
    # Initialize Alpaca market data provider
    alpaca = AlpacaMarketData()
    
    # Test symbols
    symbols = ["SPY", "AAPL", "MSFT", "AMZN", "GOOG", "TSLA"]
    
    print("Testing basic stock data retrieval from Alpaca API:")
    print("="*60)
    
    for symbol in symbols:
        print(f"\nFetching data for {symbol}:")
        
        # Get current price
        price = alpaca.get_price(symbol)
        print(f"  Current price: ${price:.2f}")
        
        # Get technical indicators
        indicators = alpaca.get_technical_indicators(symbol)
        print(f"  RSI: {indicators['rsi']:.2f}")
        print(f"  1-day price change: {indicators['price_change_1d']*100:.2f}%")
        
        # Add a small delay between requests to avoid rate limiting
        time.sleep(0.5)
    
    print("\nTest completed.")

if __name__ == "__main__":
    main() 