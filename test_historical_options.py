#!/usr/bin/env python
"""
Test script for retrieving historical options data from Alpaca Markets API
"""

import os
import datetime
import requests
import pandas as pd
from alpaca_config import load_credentials, ALPACA_API_KEY, ALPACA_API_SECRET

def main():
    # Load credentials
    load_credentials()
    
    # Set up authentication headers
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    }
    
    # Options API base URL
    options_base_url = "https://data.alpaca.markets/v2/stocks/options"
    
    # Test symbols
    symbols = ["SPY", "AAPL", "MSFT"]
    
    # Set date range for historical data (5 days ago to yesterday)
    end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=5)
    
    print("Testing historical options data retrieval from Alpaca API:")
    print("="*60)
    
    for symbol in symbols:
        print(f"\nChecking historical options data for {symbol}:")
        
        # First, try to get available options expirations
        try:
            expirations_url = f"{options_base_url}/{symbol}/expirations"
            expirations_response = requests.get(expirations_url, headers=headers)
            
            if expirations_response.status_code == 200:
                expirations = expirations_response.json()
                print(f"  Found {len(expirations)} expiration dates")
                
                # Take the first expiration date
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
                            print(f"  Checking historical data for strike price {mid_strike}")
                            
                            # Try to get historical data for a call option
                            historical_url = f"{options_base_url}/snapshots"
                            historical_response = requests.get(
                                historical_url,
                                headers=headers,
                                params={
                                    "symbols": f"{symbol}#{first_expiry}#{mid_strike}#C",
                                    "start": start_date.isoformat(),
                                    "end": end_date.isoformat(),
                                    "limit": 100
                                }
                            )
                            
                            if historical_response.status_code == 200:
                                historical_data = historical_response.json()
                                print(f"  Successfully retrieved historical options data")
                                print(f"  Data points: {len(historical_data) if isinstance(historical_data, list) else 'N/A'}")
                                print(f"  Sample data: {historical_data[:1] if isinstance(historical_data, list) else historical_data}")
                            else:
                                print(f"  Error retrieving historical data: {historical_response.status_code} - {historical_response.text}")
                        else:
                            print("  No strike prices found")
                    else:
                        print(f"  Error retrieving strikes: {strikes_response.status_code} - {strikes_response.text}")
                else:
                    print("  No expiration dates found")
            else:
                print(f"  Error retrieving expirations: {expirations_response.status_code} - {expirations_response.text}")
                
                # Try a different approach for historical data - directly query a common option
                print("  Trying to retrieve historical data for a typical option contract...")
                
                # For SPY, let's try a common ATM option from a recent expiration
                today = datetime.datetime.now().date()
                next_month = today.replace(day=1) + datetime.timedelta(days=32)
                next_month = next_month.replace(day=1) - datetime.timedelta(days=1)
                
                # Format as YYYY-MM-DD
                expiry_str = next_month.strftime("%Y-%m-%d")
                
                # For strike, use a rounded value based on current price
                current_price_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
                price_response = requests.get(current_price_url, headers=headers)
                
                if price_response.status_code == 200 and 'trade' in price_response.json():
                    current_price = price_response.json()['trade']['p']
                    # Round to nearest 5 for SPY, 10 for others
                    round_base = 5 if symbol == "SPY" else 10
                    strike = round(current_price / round_base) * round_base
                    
                    print(f"  Trying option: {symbol} with strike {strike} expiring {expiry_str}")
                    
                    # Try both call and put
                    for option_type in ["C", "P"]:
                        symbol_str = f"{symbol}#{expiry_str}#{strike}#{option_type}"
                        
                        historical_url = f"{options_base_url}/snapshots"
                        historical_response = requests.get(
                            historical_url,
                            headers=headers,
                            params={
                                "symbols": symbol_str,
                            }
                        )
                        
                        if historical_response.status_code == 200:
                            data = historical_response.json()
                            print(f"  Successfully retrieved data for {option_type}alls")
                            print(f"  Sample data: {data}")
                            break
                        else:
                            print(f"  Error retrieving {option_type}alls data: {historical_response.status_code} - {historical_response.text}")
                else:
                    print(f"  Could not retrieve current price for {symbol}")
                        
        except Exception as e:
            print(f"  Exception while checking {symbol}: {e}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main() 