#!/usr/bin/env python
"""
Test script for Alpaca Options Contracts API

This script tests fetching options contracts data from Alpaca.
"""

import requests
import json
from datetime import datetime, timedelta

def main():
    """Test Alpaca Options Contracts API"""
    
    # Credentials
    api_key = "AKTJN56J3HTI2KWEEZ8A"
    api_secret = "9uhVQTGFRzSRBkk1RWI0ovhBGWg3UDH2u7woInaX"
    
    # Set up headers
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    
    # URLs from documentation
    contracts_url = "https://api.alpaca.markets/v2/options/contracts"
    
    # Test symbols
    symbols = ["SPY", "AAPL", "MSFT", "AMZN", "NVDA"]
    
    print("Testing Alpaca Options Contracts API...")
    print("======================================")
    
    # Set up expiration date range based on documentation
    today = datetime.now().date()
    next_week = today + timedelta(days=7)
    next_month = today + timedelta(days=30)
    
    # Try different queries
    queries = [
        {"name": "Default Query", "params": {"underlying_symbols": symbols[0]}},
        {"name": "With Expiration Range", "params": {
            "underlying_symbols": symbols[0],
            "expiration_date_gte": today.isoformat(),
            "expiration_date_lte": next_week.isoformat()
        }},
        {"name": "With Type Filter", "params": {
            "underlying_symbols": symbols[0],
            "type": "call"
        }},
        {"name": "With Multiple Symbols", "params": {
            "underlying_symbols": ",".join(symbols[:3])
        }}
    ]
    
    for query in queries:
        print(f"\nTesting: {query['name']}")
        try:
            response = requests.get(
                contracts_url,
                headers=headers,
                params=query['params']
            )
            
            print(f"  Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get("option_contracts", [])
                print(f"  Found {len(contracts)} contracts")
                
                if contracts:
                    # Print details of the first contract
                    first = contracts[0]
                    print(f"  Sample contract: {first.get('symbol')}")
                    print(f"    Underlying: {first.get('underlying_symbol')}")
                    print(f"    Type: {first.get('type')}")
                    print(f"    Strike: ${first.get('strike_price')}")
                    print(f"    Expiration: {first.get('expiration_date')}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    # Try to get a specific contract
    if symbols:
        print("\nTesting specific contract:")
        try:
            # First get a contract ID
            response = requests.get(
                contracts_url,
                headers=headers,
                params={"underlying_symbols": symbols[0], "limit": 1}
            )
            
            if response.status_code == 200:
                data = response.json()
                contracts = data.get("option_contracts", [])
                
                if contracts:
                    contract_id = contracts[0].get("id")
                    contract_symbol = contracts[0].get("symbol")
                    
                    print(f"  Using contract: {contract_symbol}")
                    
                    # Get the specific contract by ID
                    detailed_response = requests.get(
                        f"{contracts_url}/{contract_id}",
                        headers=headers
                    )
                    
                    print(f"  Status code: {detailed_response.status_code}")
                    
                    if detailed_response.status_code == 200:
                        contract_data = detailed_response.json()
                        print(f"  Contract details: {json.dumps(contract_data, indent=2)[:200]}...")
                    else:
                        print(f"  Error: {detailed_response.text}")
            else:
                print(f"  Error getting contracts: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 