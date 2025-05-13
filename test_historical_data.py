#!/usr/bin/env python
"""
Test script for Alpaca Historical Market Data API

This script follows the Alpaca documentation to test retrieving historical market data.
"""

import requests
import json
import datetime
from datetime import datetime, timedelta
from alpaca_config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_DATA_URL

def main():
    """Test Alpaca Historical Market Data API"""
    
    # Set up headers with API keys
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    }
    
    print("Testing Alpaca Historical Market Data API...")
    print("="*60)
    
    # Test Stock Bars API
    print("\nTesting Stock Bars API:")
    try:
        # Create parameters for stock bars request
        params = {
            "symbols": "AAPL,MSFT,SPY",
            "timeframe": "1Day",
            "start": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
            "end": datetime.now().strftime("%Y-%m-%d"),
            "limit": 10,
            "adjustment": "all"
        }
        
        # Make the request to Bars API
        bars_url = f"{ALPACA_DATA_URL}/stocks/bars"
        bars_response = requests.get(bars_url, headers=headers, params=params)
        
        print(f"  Status code: {bars_response.status_code}")
        print(f"  X-Request-ID: {bars_response.headers.get('X-Request-ID', 'N/A')}")
        
        if bars_response.status_code == 200:
            bars_data = bars_response.json()
            print(f"  Retrieved data for {len(bars_data.get('bars', {}))} symbols")
            
            # Print a sample of the data
            for symbol, bars in bars_data.get('bars', {}).items():
                print(f"  {symbol}: {len(bars)} bars")
                if bars:
                    print(f"    Sample bar: {json.dumps(bars[0], indent=2)[:100]}...")
                break  # Just show the first symbol
        else:
            print(f"  Error: {bars_response.text}")
    except Exception as e:
        print(f"  Exception testing Stock Bars API: {e}")
    
    # Test Historical Options Data API
    print("\nTesting Historical Options Data API:")
    try:
        # First get available options contracts for a symbol
        contracts_url = "https://api.alpaca.markets/v2/options/contracts"
        contracts_params = {
            "underlying_symbols": "AAPL",
            "expiration_date_gte": datetime.now().strftime("%Y-%m-%d"),
            "expiration_date_lte": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "limit": 5
        }
        
        contracts_response = requests.get(contracts_url, headers=headers, params=contracts_params)
        
        print(f"  Options Contracts Status: {contracts_response.status_code}")
        
        if contracts_response.status_code == 200:
            contracts_data = contracts_response.json()
            contracts = contracts_data.get('option_contracts', [])
            print(f"  Found {len(contracts)} option contracts")
            
            if contracts:
                # Get the first contract's symbol to use for historical data
                contract_symbol = contracts[0]['symbol']
                print(f"  Using contract: {contract_symbol}")
                
                # Now get historical options data
                options_bars_url = f"{ALPACA_DATA_URL}/options/bars"
                options_params = {
                    "symbols": contract_symbol,
                    "timeframe": "1Day",
                    "start": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d"),
                    "limit": 10
                }
                
                options_bars_response = requests.get(options_bars_url, headers=headers, params=options_params)
                
                print(f"  Options Bars Status: {options_bars_response.status_code}")
                print(f"  X-Request-ID: {options_bars_response.headers.get('X-Request-ID', 'N/A')}")
                
                if options_bars_response.status_code == 200:
                    options_data = options_bars_response.json()
                    print(f"  Retrieved options data: {json.dumps(options_data, indent=2)[:200]}...")
                else:
                    print(f"  Error getting options bars: {options_bars_response.text}")
        else:
            print(f"  Error getting options contracts: {contracts_response.text}")
    except Exception as e:
        print(f"  Exception testing Options Data API: {e}")
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 