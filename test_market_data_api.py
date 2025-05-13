#!/usr/bin/env python
"""
Test script for Market Data API integration
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from market_data_api import MarketDataAPI
from market_data_config import load_market_data_credentials

def main():
    """
    Main function to test Market Data API connectivity and usage
    """
    print("Testing Market Data API for Options Data")
    print("========================================")
    
    # Load Market Data API credentials
    try:
        credentials = load_market_data_credentials()
        print(f"Market Data API token loaded successfully")
    except Exception as e:
        print(f"Error loading Market Data API credentials: {e}")
        sys.exit(1)
    
    # Initialize the Market Data API client
    api = MarketDataAPI(api_token=credentials["token"])

    # Test 1: Get data for SPY options (very liquid)
    test_spy_option(api)
    
    # Test 2: Get data for AAPL options (popular stock)
    test_apple_option(api)
    
    # Test 3: Get data for QQQ options (popular ETF)
    test_qqq_option(api)
    
    print("\nDone!")

def test_spy_option(api):
    """Test fetching SPY option data"""
    print("\n--- SPY Option Test ---")
    
    # SPY is currently around $550-560, so use a reasonable strike
    # Use a date about 1-2 months in the future
    today = datetime.now()
    expiry_date = today + timedelta(days=45)
    # Round to the nearest Friday
    while expiry_date.weekday() != 4:  # 4 is Friday
        expiry_date += timedelta(days=1)
    
    expiration_date = expiry_date.strftime("%Y-%m-%d")
    print(f"Using SPY with expiration: {expiration_date}")
    
    # Use ATM option
    strike_price = 550.0  # Adjust based on current SPY price if needed
    option_type = "C"  # Call option
    
    try:
        # Format the option symbol
        option_symbol = api.format_option_symbol(
            ticker="SPY",
            expiration_date=expiration_date,
            strike_price=strike_price,
            option_type=option_type
        )
        print(f"SPY option symbol: {option_symbol}")
        
        # Get real-time option quote
        print("Fetching SPY option data...")
        quote = api.get_option_quote(option_symbol)
        
        # Display the data in a readable format
        display_option_data(quote, "SPY")
        
    except Exception as e:
        print(f"Error in SPY option test: {e}")

def test_apple_option(api):
    """Test fetching AAPL option data"""
    print("\n--- AAPL Option Test ---")
    
    # AAPL is currently around $200-210, so use a reasonable strike
    # Use a date about 1-2 months in the future
    today = datetime.now()
    expiry_date = today + timedelta(days=45)
    # Round to the nearest Friday
    while expiry_date.weekday() != 4:  # 4 is Friday
        expiry_date += timedelta(days=1)
    
    expiration_date = expiry_date.strftime("%Y-%m-%d")
    print(f"Using AAPL with expiration: {expiration_date}")
    
    # Use ATM option
    strike_price = 210.0  # Adjust based on current AAPL price if needed
    option_type = "C"  # Call option
    
    try:
        # Format the option symbol
        option_symbol = api.format_option_symbol(
            ticker="AAPL",
            expiration_date=expiration_date,
            strike_price=strike_price,
            option_type=option_type
        )
        print(f"AAPL option symbol: {option_symbol}")
        
        # Get real-time option quote
        print("Fetching AAPL option data...")
        quote = api.get_option_quote(option_symbol)
        
        # Display the data in a readable format
        display_option_data(quote, "AAPL")
        
    except Exception as e:
        print(f"Error in AAPL option test: {e}")

def test_qqq_option(api):
    """Test fetching QQQ option data"""
    print("\n--- QQQ Option Test ---")
    
    # QQQ is currently around $450-460, so use a reasonable strike
    # Use a date about 1-2 months in the future
    today = datetime.now()
    expiry_date = today + timedelta(days=45)
    # Round to the nearest Friday
    while expiry_date.weekday() != 4:  # 4 is Friday
        expiry_date += timedelta(days=1)
    
    expiration_date = expiry_date.strftime("%Y-%m-%d")
    print(f"Using QQQ with expiration: {expiration_date}")
    
    # Use ATM option
    strike_price = 450.0  # Adjust based on current QQQ price if needed
    option_type = "C"  # Call option
    
    try:
        # Format the option symbol
        option_symbol = api.format_option_symbol(
            ticker="QQQ",
            expiration_date=expiration_date,
            strike_price=strike_price,
            option_type=option_type
        )
        print(f"QQQ option symbol: {option_symbol}")
        
        # Get real-time option quote
        print("Fetching QQQ option data...")
        quote = api.get_option_quote(option_symbol)
        
        # Display the data in a readable format
        display_option_data(quote, "QQQ")
        
    except Exception as e:
        print(f"Error in QQQ option test: {e}")

def display_option_data(quote, symbol):
    """Display option data in a readable format"""
    if not quote:
        print(f"No data received for {symbol}")
        return
        
    if quote.get("s") not in ["ok", "error"]:
        print(f"Error receiving data for {symbol}: {quote.get('errmsg', 'Unknown error')}")
        return
    
    # Check for array-based response format
    if "bid" in quote and isinstance(quote["bid"], list):
        if len(quote["bid"]) > 0:
            print(f"\n{symbol} option data (array format):")
            print(f"  Bid: ${quote['bid'][0]}")
            print(f"  Ask: ${quote['ask'][0]}")
            print(f"  Last: ${quote['last'][0] if 'last' in quote and len(quote['last']) > 0 else 'N/A'}")
            print(f"  Volume: {quote['volume'][0] if 'volume' in quote and len(quote['volume']) > 0 else 'N/A'}")
            print(f"  Open Interest: {quote['openInterest'][0] if 'openInterest' in quote and len(quote['openInterest']) > 0 else 'N/A'}")
            print(f"  Implied Volatility: {quote['iv'][0] if 'iv' in quote and len(quote['iv']) > 0 else 'N/A'}")
            print(f"  Underlying Price: ${quote['underlyingPrice'][0] if 'underlyingPrice' in quote and len(quote['underlyingPrice']) > 0 else 'N/A'}")
            
            # Print Greeks if available
            if 'delta' in quote and len(quote['delta']) > 0:
                print(f"  Delta: {quote['delta'][0]}")
            if 'gamma' in quote and len(quote['gamma']) > 0:
                print(f"  Gamma: {quote['gamma'][0]}")
            if 'theta' in quote and len(quote['theta']) > 0:
                print(f"  Theta: {quote['theta'][0]}")
            if 'vega' in quote and len(quote['vega']) > 0:
                print(f"  Vega: {quote['vega'][0]}")
        else:
            print(f"Empty data arrays for {symbol}")
    elif "bid" in quote:
        # Object-based response format
        print(f"\n{symbol} option data (object format):")
        print(f"  Bid: ${quote.get('bid', 'N/A')}")
        print(f"  Ask: ${quote.get('ask', 'N/A')}")
        print(f"  Last: ${quote.get('last', 'N/A')}")
        print(f"  Volume: {quote.get('volume', 'N/A')}")
        print(f"  Open Interest: {quote.get('open_interest', 'N/A')}")
        print(f"  Implied Volatility: {quote.get('iv', 'N/A')}")
        print(f"  Underlying Price: ${quote.get('underlying_price', 'N/A')}")
        
        # Print Greeks if available
        if 'delta' in quote:
            print(f"  Delta: {quote.get('delta', 'N/A')}")
        if 'gamma' in quote:
            print(f"  Gamma: {quote.get('gamma', 'N/A')}")
        if 'theta' in quote:
            print(f"  Theta: {quote.get('theta', 'N/A')}")
        if 'vega' in quote:
            print(f"  Vega: {quote.get('vega', 'N/A')}")
    else:
        print(f"No quote data available for {symbol}")

if __name__ == "__main__":
    main() 