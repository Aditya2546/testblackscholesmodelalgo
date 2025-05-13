#!/usr/bin/env python
"""
Data Collector for Options Model Training
Uses Market Data API to collect historical and real-time options data for model training
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from market_data_config import load_market_data_credentials
from market_data_api import MarketDataAPI

# Directory for storing collected data
DATA_DIR = "training_data"
HISTORICAL_DIR = f"{DATA_DIR}/historical"
REALTIME_DIR = f"{DATA_DIR}/realtime"

def setup_data_directories():
    """Create directories for storing collected data"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORICAL_DIR, exist_ok=True)
    os.makedirs(REALTIME_DIR, exist_ok=True)
    print(f"Data directories set up: {DATA_DIR}, {HISTORICAL_DIR}, {REALTIME_DIR}")

def collect_realtime_option_data(api: MarketDataAPI, symbol: str, expiration_date: str, strikes: List[float], option_types: List[str]):
    """
    Collect real-time option data for a specific symbol, expiration, and strikes
    
    Args:
        api: MarketDataAPI instance
        symbol: The ticker symbol (e.g., "AAPL")
        expiration_date: Option expiration date (YYYY-MM-DD)
        strikes: List of strike prices
        option_types: List of option types ("C", "P")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{REALTIME_DIR}/{symbol}_{timestamp}.csv"
    
    data_rows = []
    for strike in strikes:
        for option_type in option_types:
            try:
                # Format option symbol
                option_symbol = api.format_option_symbol(
                    ticker=symbol,
                    expiration_date=expiration_date,
                    strike_price=strike,
                    option_type=option_type
                )
                
                print(f"Collecting data for {option_symbol}...")
                
                # Get option data
                option_data = api.get_option_quote(option_symbol)
                
                if option_data and option_data.get("s") in ["ok", "error"]:
                    # Extract data from potentially array-based response
                    data_row = {}
                    
                    # Basic option info
                    data_row["symbol"] = symbol
                    data_row["expiration_date"] = expiration_date
                    data_row["strike"] = strike
                    data_row["option_type"] = option_type
                    data_row["option_symbol"] = option_symbol
                    data_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extract data based on response format
                    if "bid" in option_data and isinstance(option_data["bid"], list) and len(option_data["bid"]) > 0:
                        # Array format
                        data_row["bid"] = option_data["bid"][0]
                        data_row["ask"] = option_data["ask"][0] if "ask" in option_data and len(option_data["ask"]) > 0 else None
                        data_row["last"] = option_data["last"][0] if "last" in option_data and len(option_data["last"]) > 0 else None
                        data_row["volume"] = option_data["volume"][0] if "volume" in option_data and len(option_data["volume"]) > 0 else None
                        data_row["open_interest"] = option_data["openInterest"][0] if "openInterest" in option_data and len(option_data["openInterest"]) > 0 else None
                        data_row["iv"] = option_data["iv"][0] if "iv" in option_data and len(option_data["iv"]) > 0 else None
                        data_row["delta"] = option_data["delta"][0] if "delta" in option_data and len(option_data["delta"]) > 0 else None
                        data_row["gamma"] = option_data["gamma"][0] if "gamma" in option_data and len(option_data["gamma"]) > 0 else None
                        data_row["theta"] = option_data["theta"][0] if "theta" in option_data and len(option_data["theta"]) > 0 else None
                        data_row["vega"] = option_data["vega"][0] if "vega" in option_data and len(option_data["vega"]) > 0 else None
                        data_row["underlying_price"] = option_data["underlyingPrice"][0] if "underlyingPrice" in option_data and len(option_data["underlyingPrice"]) > 0 else None
                    else:
                        # Object format
                        data_row["bid"] = option_data.get("bid")
                        data_row["ask"] = option_data.get("ask")
                        data_row["last"] = option_data.get("last")
                        data_row["volume"] = option_data.get("volume")
                        data_row["open_interest"] = option_data.get("open_interest")
                        data_row["iv"] = option_data.get("iv")
                        data_row["delta"] = option_data.get("delta")
                        data_row["gamma"] = option_data.get("gamma")
                        data_row["theta"] = option_data.get("theta")
                        data_row["vega"] = option_data.get("vega")
                        data_row["underlying_price"] = option_data.get("underlying_price")
                    
                    data_rows.append(data_row)
                else:
                    print(f"  No data available for {option_symbol}")
            
            except Exception as e:
                print(f"Error collecting data for {symbol} {strike} {option_type}: {e}")
    
    # Save data to CSV if any rows collected
    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"Saved {len(data_rows)} data points to {filename}")
    else:
        print(f"No data collected for {symbol}")

def collect_historical_option_data(api: MarketDataAPI, symbol: str, start_date: str, end_date: str, strikes: List[float], option_types: List[str], expiration_date: str):
    """
    Collect historical option data for a specific symbol, date range, and strikes
    
    Args:
        api: MarketDataAPI instance
        symbol: The ticker symbol (e.g., "AAPL")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strikes: List of strike prices
        option_types: List of option types ("C", "P")
        expiration_date: Option expiration date (YYYY-MM-DD)
    """
    filename = f"{HISTORICAL_DIR}/{symbol}_{start_date}_to_{end_date}.csv"
    
    data_rows = []
    for strike in strikes:
        for option_type in option_types:
            try:
                # Format option symbol
                option_symbol = api.format_option_symbol(
                    ticker=symbol,
                    expiration_date=expiration_date,
                    strike_price=strike,
                    option_type=option_type
                )
                
                print(f"Collecting historical data for {option_symbol}...")
                
                # Get option data
                option_df = api.get_option_quote_as_dataframe(
                    option_symbol=option_symbol,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not option_df.empty:
                    # Add additional columns
                    option_df["symbol"] = symbol
                    option_df["expiration_date"] = expiration_date
                    option_df["strike"] = strike
                    option_df["option_type"] = option_type
                    option_df["option_symbol"] = option_symbol
                    
                    # Collect all data rows
                    for _, row in option_df.iterrows():
                        data_rows.append(row.to_dict())
                    
                    print(f"  Collected {len(option_df)} historical data points")
                else:
                    print(f"  No historical data available for {option_symbol}")
            
            except Exception as e:
                print(f"Error collecting historical data for {symbol} {strike} {option_type}: {e}")
    
    # Save data to CSV if any rows collected
    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"Saved {len(data_rows)} historical data points to {filename}")
    else:
        print(f"No historical data collected for {symbol}")

def get_standard_strikes(symbol: str, current_price: float, count: int = 5) -> List[float]:
    """
    Generate standard strike prices for options based on the current price
    
    Args:
        symbol: The ticker symbol
        current_price: The current price of the underlying asset
        count: Number of strikes to generate (count per side from ATM)
        
    Returns:
        List of standard strike prices
    """
    # Define strike intervals based on stock price
    if symbol in ["SPY", "QQQ"]:
        # SPY and QQQ options typically have $5 strike intervals
        interval = 5.0
    elif current_price >= 200:
        # Higher priced stocks often have $5 or $10 strike intervals
        interval = 5.0
    elif current_price >= 100:
        # Medium priced stocks typically have $5 strike intervals
        interval = 5.0
    elif current_price >= 50:
        # Lower-medium priced stocks might have $2.5 strike intervals
        interval = 2.5
    else:
        # Lower priced stocks often have $1 or $2.5 strike intervals
        interval = 1.0
    
    # Round to nearest interval
    base = round(current_price / interval) * interval
    
    # Generate strikes around the base
    strikes = [base]
    for i in range(1, count + 1):
        strikes.append(base + i * interval)
        strikes.append(base - i * interval)
    
    # Sort strikes
    strikes.sort()
    return strikes

def get_current_price(api: MarketDataAPI, symbol: str) -> float:
    """
    Get the current price of a symbol using an option as a proxy
    
    Args:
        api: MarketDataAPI instance
        symbol: The ticker symbol
        
    Returns:
        Current price of the symbol (or an estimation)
    """
    # Default price estimates
    default_prices = {
        "SPY": 550.0,
        "QQQ": 450.0,
        "AAPL": 210.0,
        "MSFT": 420.0,
        "AMZN": 185.0,
        "GOOG": 180.0,
        "TSLA": 190.0,
        "META": 480.0
    }
    
    try:
        # Get a near-term ATM option
        today = datetime.now()
        expiry = today + timedelta(days=30)
        # Adjust to nearest Friday
        while expiry.weekday() != 4:  # 4 is Friday
            expiry += timedelta(days=1)
        expiration_date = expiry.strftime("%Y-%m-%d")
        
        # Use the default price as a starting point
        default_price = default_prices.get(symbol, 100.0)
        
        # Format an ATM option
        option_symbol = api.format_option_symbol(
            ticker=symbol,
            expiration_date=expiration_date,
            strike_price=default_price,
            option_type="C"
        )
        
        # Get option data
        option_data = api.get_option_quote(option_symbol)
        
        if option_data and option_data.get("s") in ["ok", "error"]:
            # Extract underlying price
            if "underlyingPrice" in option_data and isinstance(option_data["underlyingPrice"], list) and len(option_data["underlyingPrice"]) > 0:
                return option_data["underlyingPrice"][0]
            elif "underlying_price" in option_data:
                return option_data["underlying_price"]
        
        return default_prices.get(symbol, 100.0)
    
    except Exception as e:
        print(f"Error getting current price for {symbol}: {e}")
        return default_prices.get(symbol, 100.0)

def get_future_expiration(days_out: int = 45) -> str:
    """Get a future expiration date adjusted to third Friday"""
    today = datetime.now()
    future = today + timedelta(days=days_out)
    
    # Get to the next month
    if days_out > 30:
        month = future.month + 1 if future.month < 12 else 1
        year = future.year + 1 if future.month == 12 else future.year
        future = future.replace(year=year, month=month, day=1)
    
    # Find third Friday
    day = 1
    first_day = datetime(future.year, future.month, day)
    
    # Find the first Friday
    while first_day.weekday() != 4:  # 4 is Friday
        day += 1
        first_day = datetime(future.year, future.month, day)
    
    # Add two weeks to get the third Friday
    third_friday = first_day + timedelta(days=14)
    
    return third_friday.strftime("%Y-%m-%d")

def main():
    """Main entry point for the data collection system"""
    print("Options Data Collection for Model Training")
    print("=========================================")
    
    # Setup data directories
    setup_data_directories()
    
    # Load Market Data API credentials
    try:
        credentials = load_market_data_credentials()
        print("Market Data API token loaded successfully")
    except Exception as e:
        print(f"Error loading Market Data API credentials: {e}")
        sys.exit(1)
    
    # Initialize MarketDataAPI
    try:
        api = MarketDataAPI(api_token=credentials["token"])
    except Exception as e:
        print(f"Error initializing Market Data API: {e}")
        sys.exit(1)
    
    # Define symbols to scan
    symbols = ["SPY", "QQQ", "AAPL"]
    option_types = ["C", "P"]  # Both calls and puts
    
    # Set collection parameters
    collection_interval = 60 * 60  # Collect data hourly
    
    try:
        print("Starting data collection. Press Ctrl+C to exit.")
        while True:
            collection_time = datetime.now()
            print(f"\n{collection_time.strftime('%Y-%m-%d %H:%M:%S')} - Starting data collection round")
            
            for symbol in symbols:
                try:
                    # Get current price
                    current_price = get_current_price(api, symbol)
                    print(f"{symbol} current price: ${current_price:.2f}")
                    
                    # Generate strikes around current price
                    strikes = get_standard_strikes(symbol, current_price)
                    print(f"Using strikes: {strikes}")
                    
                    # Get future expiration date (for active options)
                    expiration_date = get_future_expiration()
                    print(f"Using expiration date: {expiration_date}")
                    
                    # Collect real-time data
                    collect_realtime_option_data(api, symbol, expiration_date, strikes, option_types)
                    
                    # Only collect historical data once per day (at the start of the day)
                    if collection_time.hour == 9 and collection_time.minute < 30:
                        # Historical data for the past 30 days
                        end_date = collection_time.strftime("%Y-%m-%d")
                        start_date = (collection_time - timedelta(days=30)).strftime("%Y-%m-%d")
                        
                        # Use a past expiration for historical data
                        past_expiration = (collection_time - timedelta(days=15)).strftime("%Y-%m-%d")
                        
                        collect_historical_option_data(api, symbol, start_date, end_date, strikes, option_types, past_expiration)
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
            
            # Sleep until next collection
            sleep_seconds = collection_interval - (datetime.now() - collection_time).total_seconds()
            if sleep_seconds > 0:
                print(f"Waiting {sleep_seconds:.0f} seconds until next collection...")
                time.sleep(sleep_seconds)
    
    except KeyboardInterrupt:
        print("\nData collection stopped by user.")
    
    print("\nDone!")

def run_once():
    """Run a single data collection without continuous monitoring"""
    print("Options Data Collection - Single Run")
    print("==================================")
    
    # Setup data directories
    setup_data_directories()
    
    # Load Market Data API credentials
    try:
        credentials = load_market_data_credentials()
        print("Market Data API token loaded successfully")
    except Exception as e:
        print(f"Error loading Market Data API credentials: {e}")
        sys.exit(1)
    
    # Initialize MarketDataAPI
    try:
        api = MarketDataAPI(api_token=credentials["token"])
    except Exception as e:
        print(f"Error initializing Market Data API: {e}")
        sys.exit(1)
    
    # Define symbols to scan - smaller set for faster testing
    symbols = ["SPY", "AAPL"]
    option_types = ["C", "P"]
    
    for symbol in symbols:
        try:
            # Get current price
            current_price = get_current_price(api, symbol)
            print(f"{symbol} current price: ${current_price:.2f}")
            
            # Generate strikes around current price
            strikes = get_standard_strikes(symbol, current_price)
            print(f"Using strikes: {strikes}")
            
            # Get future expiration date (for active options)
            expiration_date = get_future_expiration()
            print(f"Using expiration date: {expiration_date}")
            
            # Collect real-time data
            collect_realtime_option_data(api, symbol, expiration_date, strikes, option_types)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once()
    else:
        main() 