#!/usr/bin/env python
"""
Check Alpaca account status and active subscriptions
"""

import requests
from alpaca_config import load_credentials, ALPACA_API_KEY, ALPACA_API_SECRET

def main():
    # Load credentials
    load_credentials()
    
    # Set up authentication headers
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    }
    
    # Base URLs
    trading_base_url = "https://paper-api.alpaca.markets/v2"
    data_base_url = "https://data.alpaca.markets/v2"
    
    print("Checking Alpaca account status:")
    print("="*50)
    
    # Check account status
    try:
        account_url = f"{trading_base_url}/account"
        account_response = requests.get(account_url, headers=headers)
        
        if account_response.status_code == 200:
            account_data = account_response.json()
            print(f"Account status: {account_data.get('status', 'Unknown')}")
            print(f"Account type: {account_data.get('account_type', 'Unknown')}")
            print(f"Trading blocked: {account_data.get('trading_blocked', 'Unknown')}")
            print(f"Account blocked: {account_data.get('account_blocked', 'Unknown')}")
        else:
            print(f"Error retrieving account data: {account_response.status_code} - {account_response.text}")
    except Exception as e:
        print(f"Exception while checking account status: {e}")
    
    # Check for available endpoints to determine subscription level
    print("\nTesting available data endpoints:")
    
    endpoints = [
        ("Stock market data", f"{data_base_url}/stocks/AAPL/trades/latest"),
        ("Stock quotes", f"{data_base_url}/stocks/AAPL/quotes/latest"),
        ("Crypto data", f"{data_base_url}/crypto/BTCUSD/trades/latest"),
        ("Options expirations", f"{data_base_url}/stocks/options/SPY/expirations"),
        ("News", f"{data_base_url}/news")
    ]
    
    for desc, url in endpoints:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                print(f"  ✅ {desc}: Available")
            else:
                print(f"  ❌ {desc}: Not available ({response.status_code} - {response.text})")
        except Exception as e:
            print(f"  ❌ {desc}: Error ({e})")
    
    # Check subscription status if there's an endpoint for it
    try:
        subscription_url = f"{trading_base_url}/account/configurations"
        subscription_response = requests.get(subscription_url, headers=headers)
        
        print("\nAccount configuration and subscription info:")
        if subscription_response.status_code == 200:
            subscription_data = subscription_response.json()
            print(f"  Configuration data: {subscription_data}")
        else:
            print(f"  Could not retrieve subscription info: {subscription_response.status_code} - {subscription_response.text}")
    except Exception as e:
        print(f"  Exception while checking subscription: {e}")
    
    print("\nCheck completed. For full access to options data, you may need to:")
    print("1. Ensure you have an active paid subscription that includes options market data")
    print("2. Contact Alpaca support to confirm your data access is properly configured")
    print("3. Check if your API keys are correctly linked to your subscription")

if __name__ == "__main__":
    main() 