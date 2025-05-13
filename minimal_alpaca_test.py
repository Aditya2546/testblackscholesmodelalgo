#!/usr/bin/env python
"""
Minimal test script for Alpaca API authentication
"""

import requests

def main():
    """Test basic Alpaca API authentication"""
    
    # Credentials
    api_key = "AKTJN56J3HTI2KWEEZ8A"
    api_secret = "9uhVQTGFRzSRBkk1RWI0ovhBGWg3UDH2u7woInaX"
    
    # Set up headers
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    
    # Test endpoints
    endpoints = [
        "https://paper-api.alpaca.markets/v2/account",
        "https://api.alpaca.markets/v2/account",
        "https://paper-api.alpaca.markets/v2/options/contracts?underlying_symbols=SPY&limit=5",
        "https://api.alpaca.markets/v2/options/contracts?underlying_symbols=SPY&limit=5",
        "https://data.alpaca.markets/v2/stocks/SPY/trades/latest"
    ]
    
    print("Testing Alpaca API with minimal configuration...")
    print("==============================================")
    
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        try:
            response = requests.get(endpoint, headers=headers)
            
            print(f"  Status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}..." if len(response.text) > 200 else f"  Response: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 