#!/usr/bin/env python
"""
Test script for accessing options data using live Alpaca API credentials
"""

import requests
import datetime
import time

# Live API credentials 
LIVE_API_KEY = "AK33WBZVLKJQ101CH97O"
LIVE_API_SECRET = "p085ImoKibKPI3fjhuvf18dVhAwSURehH6y2DsRv"
LIVE_API_BASE_URL = "https://api.alpaca.markets/v2"
LIVE_DATA_BASE_URL = "https://data.alpaca.markets/v2"

def main():
    print("Testing Alpaca Live API for Options Data Access")
    print("="*60)
    
    # Set up authentication headers
    headers = {
        "APCA-API-KEY-ID": LIVE_API_KEY,
        "APCA-API-SECRET-KEY": LIVE_API_SECRET
    }
    
    # First check account status
    try:
        print("\nChecking account status:")
        account_url = f"{LIVE_API_BASE_URL}/account"
        account_response = requests.get(account_url, headers=headers)
        
        if account_response.status_code == 200:
            account_data = account_response.json()
            print(f"Account status: {account_data.get('status', 'Unknown')}")
            print(f"Account type: {account_data.get('account_type', 'Unknown')}")
        else:
            print(f"Error retrieving account data: {account_response.status_code} - {account_response.text}")
    except Exception as e:
        print(f"Exception while checking account status: {e}")
    
    # Test different data endpoints
    print("\nTesting available data endpoints:")
    
    endpoints = [
        ("Stock market data", f"{LIVE_DATA_BASE_URL}/stocks/AAPL/trades/latest"),
        ("Stock quotes", f"{LIVE_DATA_BASE_URL}/stocks/AAPL/quotes/latest"),
        ("Crypto data", f"{LIVE_DATA_BASE_URL}/crypto/BTCUSD/trades/latest"),
        ("Options expirations", f"{LIVE_DATA_BASE_URL}/stocks/options/SPY/expirations"),
        ("News", f"{LIVE_DATA_BASE_URL}/news")
    ]
    
    for desc, url in endpoints:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                print(f"  ✅ {desc}: Available")
                if desc == "Options expirations" and response.status_code == 200:
                    exp_data = response.json()
                    print(f"    Found {len(exp_data)} expiration dates for SPY")
                    if exp_data:
                        print(f"    Sample expiration dates: {exp_data[:3]}")
            else:
                print(f"  ❌ {desc}: Not available ({response.status_code} - {response.text})")
        except Exception as e:
            print(f"  ❌ {desc}: Error ({e})")
            
    # If options expirations are available, test getting a chain
    options_base_url = f"{LIVE_DATA_BASE_URL}/stocks/options"
    test_symbols = ["SPY", "AAPL"]
    
    print("\nTesting options chain retrieval:")
    for symbol in test_symbols:
        print(f"\nChecking options data for {symbol}:")
        
        # Get expirations
        try:
            expirations_url = f"{options_base_url}/{symbol}/expirations"
            expirations_response = requests.get(expirations_url, headers=headers)
            
            if expirations_response.status_code == 200:
                expirations = expirations_response.json()
                print(f"  Found {len(expirations)} expiration dates")
                
                # Take the first expiration
                if expirations:
                    first_expiry = expirations[0]
                    print(f"  Checking strikes for expiration {first_expiry}")
                    
                    # Get strikes for this expiration
                    strikes_url = f"{options_base_url}/{symbol}/strikes"
                    strikes_response = requests.get(
                        strikes_url, 
                        headers=headers,
                        params={"expiration": first_expiry}
                    )
                    
                    if strikes_response.status_code == 200:
                        strikes = strikes_response.json()
                        print(f"  Found {len(strikes)} strike prices")
                        
                        # Take the middle strike
                        if strikes:
                            mid_strike = strikes[len(strikes)//2]
                            print(f"  Retrieving snapshot for {symbol} {first_expiry} strike {mid_strike}")
                            
                            # Get call and put options
                            for option_type in ["call", "put"]:
                                snapshot_url = f"{options_base_url}/{symbol}/snapshot"
                                snapshot_response = requests.get(
                                    snapshot_url,
                                    headers=headers,
                                    params={
                                        "expiration": first_expiry,
                                        "strike": mid_strike,
                                        "type": option_type
                                    }
                                )
                                
                                if snapshot_response.status_code == 200:
                                    snapshot_data = snapshot_response.json()
                                    print(f"  ✅ Successfully retrieved {option_type} option data:")
                                    print(f"    Bid: ${snapshot_data.get('snapshot', {}).get('bid_price', 'N/A')}")
                                    print(f"    Ask: ${snapshot_data.get('snapshot', {}).get('ask_price', 'N/A')}")
                                    print(f"    IV: {snapshot_data.get('snapshot', {}).get('implied_volatility', 'N/A')}")
                                else:
                                    print(f"  ❌ Error retrieving {option_type} option: {snapshot_response.status_code} - {snapshot_response.text}")
                        else:
                            print("  No strike prices found")
                    else:
                        print(f"  Error retrieving strikes: {strikes_response.status_code} - {strikes_response.text}")
                else:
                    print("  No expiration dates found")
            else:
                print(f"  Error retrieving expirations: {expirations_response.status_code} - {expirations_response.text}")
        except Exception as e:
            print(f"  Exception while checking {symbol}: {e}")
            
        # Add a small delay between requests
        time.sleep(1)
    
    print("\nLive API test completed.")

if __name__ == "__main__":
    main() 