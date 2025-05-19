#!/usr/bin/env python
"""
Test Alpaca API Connection

This script tests the connection to Alpaca API and displays
account information to verify that everything is working correctly.
"""

import os
import sys
import requests
import json
from datetime import datetime, timedelta

# Import configuration
try:
    from alpaca_config import load_credentials, ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL
    load_credentials()
except ImportError:
    print("Error: Could not import alpaca_config.py")
    print("Make sure the file exists in the current directory.")
    sys.exit(1)

def test_alpaca_connection():
    """Test connection to Alpaca API and show account details."""
    
    # Get credentials from environment (these should be set by load_credentials())
    api_key = os.environ.get("ALPACA_API_KEY", ALPACA_API_KEY)
    api_secret = os.environ.get("ALPACA_API_SECRET", ALPACA_API_SECRET)
    
    # API URLs
    paper_trading_url = "https://paper-api.alpaca.markets/v2"
    data_api_url = "https://data.alpaca.markets/v2"
    
    # Print header
    print("=" * 60)
    print("ALPACA API CONNECTION TEST")
    print("=" * 60)
    
    # Set up headers for authentication
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json"
    }
    
    # Mask API key for display
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "Not set"
    
    # Test Trading API connection
    print(f"Testing connection to Alpaca Trading API at {paper_trading_url}")
    print(f"API Key: {masked_key}")
    
    # Test account endpoint
    account_url = f"{paper_trading_url}/account"
    print(f"Requesting: {account_url}")
    
    try:
        response = requests.get(account_url, headers=headers)
        
        if response.status_code == 200:
            print("✅ Successfully connected to Alpaca Trading API")
            account_data = response.json()
            print(f"Account ID: {account_data.get('id')}")
            print(f"Account Status: {account_data.get('status')}")
            print(f"Buying Power: ${float(account_data.get('buying_power', 0)):.2f}")
            print(f"Portfolio Value: ${float(account_data.get('portfolio_value', 0)):.2f}")
            print(f"Cash: ${float(account_data.get('cash', 0)):.2f}")
        else:
            print(f"❌ Error connecting to Alpaca Trading API: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Exception when connecting to Alpaca Trading API: {e}")
    
    print("\n" + "-" * 60 + "\n")
    
    # Test Market Data API connection
    print(f"Testing connection to Alpaca Data API at {data_api_url}")
    
    # Test with a simple market data request (latest quote for SPY)
    symbol = "SPY"
    quote_url = f"{data_api_url}/stocks/{symbol}/quotes/latest"
    print(f"Requesting latest quote for {symbol}: {quote_url}")
    
    try:
        response = requests.get(quote_url, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Successfully connected to Alpaca Data API")
            quote_data = response.json()
            if "quote" in quote_data:
                quote = quote_data["quote"]
                bid_price = quote.get("bp", 0)
                ask_price = quote.get("ap", 0)
                print(f"Latest {symbol} Quote:")
                print(f"  Bid: ${bid_price}")
                print(f"  Ask: ${ask_price}")
                print(f"  Spread: ${ask_price - bid_price:.2f}")
                print(f"  Time: {quote.get('t')}")
            else:
                print(f"No quote data available for {symbol}")
        else:
            print(f"❌ Error fetching market data: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Exception when fetching market data: {e}")
    
    print("\n" + "-" * 60 + "\n")
    
    # Test options data with v1beta1 API
    options_api_url = "https://data.alpaca.markets/v1beta1"
    print(f"Testing connection to Alpaca Options API at {options_api_url}")
    
    # Get options exchange codes as a simple test
    exchange_url = f"{options_api_url}/options/meta/exchanges"
    print(f"Requesting options exchange codes: {exchange_url}")
    
    try:
        response = requests.get(exchange_url, headers=headers)
        
        if response.status_code == 200:
            print("✅ Successfully connected to Alpaca Options API")
            exchange_data = response.json()
            print("Options Exchange Codes:")
            for code, exchange in exchange_data.items():
                print(f"  {code}: {exchange}")
        else:
            print(f"❌ Error fetching options exchange data: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Exception when fetching options exchange data: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed. See above for results and any error messages.")

if __name__ == "__main__":
    test_alpaca_connection() 