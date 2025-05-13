#!/usr/bin/env python
"""
Test script for Alpaca Markets Options API

This script tests the connection to Alpaca Markets Options API and retrieves options data.
"""

import os
import requests
import json
from datetime import datetime
import sys

def main():
    """Test Alpaca Options API directly"""
    
    # Credentials
    api_key = "AKTJN56J3HTI2KWEEZ8A"
    api_secret = "9uhVQTGFRzSRBkk1RWI0ovhBGWg3UDH2u7woInaX"
    
    # Set up headers
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    
    # Base URL for options API
    options_base_url = "https://paper-api.alpaca.markets/v2/options"
    
    # Test symbols
    symbols = ["SPY", "AAPL", "MSFT"]
    
    print("Testing Alpaca Options API directly...")
    print("====================================")
    
    # First test: Account Information
    print("\nTesting Account API:")
    try:
        response = requests.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers=headers
        )
        
        if response.status_code == 200:
            account = response.json()
            print(f"  Account ID: {account.get('id')}")
            print(f"  Account Status: {account.get('status')}")
            print(f"  Equity: ${float(account.get('equity', '0')):.2f}")
            print(f"  Options Trading Level: {account.get('options_level', 'Unknown')}")
        else:
            print(f"  Error getting account info: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Exception while testing account API: {e}")
    
    # Second test: Get available options contracts
    for symbol in symbols:
        print(f"\nTesting options contracts for {symbol}:")
        
        try:
            # Query the contracts endpoint
            response = requests.get(
                f"{options_base_url}/contracts",
                headers=headers,
                params={
                    "underlying_symbols": symbol,
                    "limit": 10  # Limit to 10 contracts for the test
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get("option_contracts", [])
                
                if contracts:
                    print(f"  Found {len(contracts)} option contracts")
                    
                    # Show details for the first contract
                    first_contract = contracts[0]
                    print(f"  Sample contract: {first_contract.get('symbol')}")
                    print(f"    Type: {first_contract.get('type')}")
                    print(f"    Strike: ${first_contract.get('strike_price')}")
                    print(f"    Expiration: {first_contract.get('expiration_date')}")
                    
                    # Try to get pricing for this contract
                    contract_symbol = first_contract.get('symbol')
                    print(f"\n  Getting pricing data for {contract_symbol}:")
                    
                    quote_response = requests.get(
                        f"https://data.alpaca.markets/v2/stocks/options/{contract_symbol}/snapshot",
                        headers=headers
                    )
                    
                    if quote_response.status_code == 200:
                        quote_data = quote_response.json()
                        if 'snapshot' in quote_data:
                            snapshot = quote_data['snapshot']
                            print(f"    Bid: ${snapshot.get('bid_price', 'N/A')}")
                            print(f"    Ask: ${snapshot.get('ask_price', 'N/A')}")
                            print(f"    Last: ${snapshot.get('last_price', 'N/A')}")
                            print(f"    IV: {snapshot.get('implied_volatility', 'N/A')}")
                        else:
                            print(f"    No snapshot data available")
                    else:
                        print(f"    Error getting pricing: {quote_response.status_code} - {quote_response.text}")
                else:
                    print(f"  No option contracts found")
            else:
                print(f"  Error fetching contracts: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Exception while testing options API for {symbol}: {e}")
    
    print("\nAPI tests completed!")

if __name__ == "__main__":
    main() 