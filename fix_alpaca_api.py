#!/usr/bin/env python
"""
Utility module to fix Alpaca API issues

This module provides utility functions to make proper API calls to Alpaca's
options endpoints, fixing issues with feed parameter and NoneType contract errors.
"""

import requests
import json
import datetime
from typing import Dict, List, Optional, Union, Any
import os
import sys

# Try to import local config if available
try:
    from alpaca_config import ALPACA_API_KEY, ALPACA_API_SECRET, load_credentials
    
    # Call load_credentials to set environment variables
    try:
        load_credentials()
    except:
        pass
    
    # Use imported credentials or environment variables
    API_KEY = os.environ.get("ALPACA_API_KEY", ALPACA_API_KEY)
    API_SECRET = os.environ.get("ALPACA_API_SECRET", ALPACA_API_SECRET)
except ImportError:
    # Fall back to environment variables
    API_KEY = os.environ.get("ALPACA_API_KEY", "")
    API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
    
    if not API_KEY or not API_SECRET:
        print("Warning: No Alpaca API credentials found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")

# Base URLs for Alpaca APIs
TRADING_BASE_URL = "https://api.alpaca.markets/v2"
DATA_BASE_URL = "https://data.alpaca.markets/v2"
DATA_BETA_URL = "https://data.alpaca.markets/v1beta1"

# Headers for authentication
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Accept": "application/json"
}

def get_options_contracts(underlying_symbols: Union[str, List[str]], 
                         expiration_date_gte: Optional[str] = None,
                         expiration_date_lte: Optional[str] = None,
                         strike_price_gte: Optional[float] = None,
                         strike_price_lte: Optional[float] = None,
                         option_type: Optional[str] = None) -> List[Dict]:
    """
    Get available options contracts for specified symbols with various filters.
    
    Args:
        underlying_symbols: Stock symbol(s) (e.g., 'AAPL' or ['AAPL', 'MSFT'])
        expiration_date_gte: Filter for expiration date >= this date (YYYY-MM-DD)
        expiration_date_lte: Filter for expiration date <= this date (YYYY-MM-DD)
        strike_price_gte: Filter for strike price >= this value
        strike_price_lte: Filter for strike price <= this value
        option_type: Filter by option type ('call' or 'put')
    
    Returns:
        List of option contract dictionaries
    """
    url = f"{TRADING_BASE_URL}/options/contracts"
    
    # Convert list to comma-separated string if needed
    if isinstance(underlying_symbols, list):
        underlying_symbols = ','.join(underlying_symbols)
    
    # Build parameters
    params = {"underlying_symbols": underlying_symbols}
    
    # Add optional filters
    if expiration_date_gte:
        params["expiration_date_gte"] = expiration_date_gte
    
    if expiration_date_lte:
        params["expiration_date_lte"] = expiration_date_lte
    
    if strike_price_gte is not None:
        # Convert to string to avoid NoneType error
        params["strike_price_gte"] = str(strike_price_gte)
    
    if strike_price_lte is not None:
        # Convert to string to avoid NoneType error
        params["strike_price_lte"] = str(strike_price_lte)
    
    if option_type:
        params["type"] = option_type
    
    # Make the request
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching options contracts: {response.status_code} - {response.text}")
        return []
    
    data = response.json()
    return data.get("option_contracts", [])

def get_options_bars(symbols: Union[str, List[str]],
                     timeframe: str = "1Day",
                     start: Optional[str] = None,
                     end: Optional[str] = None,
                     limit: int = 1000) -> Dict[str, List[Dict]]:
    """
    Get historical bars data for options contracts.
    Note: No feed parameter needed as all options bars come from OPRA.
    
    Args:
        symbols: Option contract symbol(s) (e.g., 'AAPL240621C00200000')
        timeframe: Time frame for bars ('1Day', '1Hour', etc.)
        start: Start date/time in ISO format
        end: End date/time in ISO format
        limit: Maximum number of bars to return
    
    Returns:
        Dictionary with bars data keyed by symbol
    """
    url = f"{DATA_BASE_URL}/options/bars"
    
    # Convert list to comma-separated string if needed
    if isinstance(symbols, list):
        symbols = ','.join(symbols)
    
    # Build parameters
    params = {
        "symbols": symbols,
        "timeframe": timeframe,
        "limit": limit
    }
    
    # Add optional parameters
    if start:
        params["start"] = start
    
    if end:
        params["end"] = end
    
    # Note: Do NOT add feed parameter for options bars
    
    # Make the request
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching options bars: {response.status_code} - {response.text}")
        return {}
    
    return response.json().get("bars", {})

def get_option_chain(underlying_symbol: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get option chain data for a symbol using the v1beta1 endpoint.
    
    Args:
        underlying_symbol: Stock symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary with option chain data
    """
    url = f"{DATA_BETA_URL}/options/snapshots/{underlying_symbol}"
    
    # Make the request with no feed parameter
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Error fetching option chain: {response.status_code} - {response.text}")
        return {}
    
    return response.json()

def patch_request_for_options_bars(params: Dict) -> Dict:
    """
    Patch request parameters for options bars to ensure compatibility.
    Removes feed parameter if present as it's not needed for options bars.
    
    Args:
        params: Original request parameters
        
    Returns:
        Patched parameters dictionary
    """
    # Make a copy to avoid modifying the original
    patched_params = params.copy()
    
    # Remove feed parameter if it exists
    if 'feed' in patched_params:
        del patched_params['feed']
        print("Warning: 'feed' parameter removed from options bars request (not needed)")
    
    return patched_params

def patch_request_for_options_contracts(params: Dict) -> Dict:
    """
    Patch request parameters for options contracts to ensure compatibility.
    Converts numeric values to strings to avoid NoneType errors.
    
    Args:
        params: Original request parameters
        
    Returns:
        Patched parameters dictionary
    """
    # Make a copy to avoid modifying the original
    patched_params = params.copy()
    
    # Convert numeric parameters to strings
    numeric_params = ['strike_price', 'strike_price_gte', 'strike_price_lte']
    
    for param in numeric_params:
        if param in patched_params and patched_params[param] is not None:
            patched_params[param] = str(patched_params[param])
    
    return patched_params

def patch_options_api_call(url: str, params: Dict) -> Dict:
    """
    Apply the appropriate patch based on the API endpoint.
    
    Args:
        url: API endpoint URL
        params: Request parameters
        
    Returns:
        Patched parameters dictionary
    """
    # Make a copy to avoid modifying the original
    patched_params = params.copy()
    
    # Identify the endpoint and apply the appropriate patch
    if '/options/bars' in url:
        patched_params = patch_request_for_options_bars(patched_params)
    elif '/options/contracts' in url:
        patched_params = patch_request_for_options_contracts(patched_params)
    elif '/options/snapshots' in url:
        # Remove feed parameter if it exists for snapshots endpoint
        if 'feed' in patched_params:
            del patched_params['feed']
    
    return patched_params

def monkey_patch_requests_get():
    """
    Monkey patch the requests.get function to automatically fix Alpaca API calls.
    This should be called at the start of your script to apply the fixes globally.
    
    Note: Use with caution as it affects all requests.get calls in your application.
    """
    original_get = requests.get
    
    def patched_get(url, **kwargs):
        # Only patch Alpaca API calls
        if 'alpaca.markets' in url and ('/options/' in url or '/v1beta1/' in url):
            if 'params' in kwargs:
                # Apply appropriate patches to parameters
                kwargs['params'] = patch_options_api_call(url, kwargs['params'])
                
                # Debug output for API calls
                if os.environ.get('DEBUG_ALPACA_API') == '1':
                    print(f"DEBUG: Patched API call to {url}")
                    print(f"       Params: {kwargs['params']}")
            
            # Make sure we have Accept header for JSON
            if 'headers' in kwargs and kwargs['headers'] is not None:
                if 'Accept' not in kwargs['headers']:
                    kwargs['headers']['Accept'] = 'application/json'
            
            # Make sure the verify parameter is not passed
            if 'verify' in kwargs:
                del kwargs['verify']
                
        # Call the original function with the modified kwargs
        response = original_get(url, **kwargs)
        
        # Additional post-processing for specific Alpaca endpoints to handle any response issues
        if response.status_code == 200 and 'alpaca.markets' in url and '/options/snapshots' in url:
            try:
                # Try to get the JSON data
                data = response.json()
                
                # Check if the response has a next_page_token
                if 'next_page_token' in data:
                    # Don't try to parse this token - just leave it as is
                    # Some code might try to convert it to an int and fail
                    pass
                
                # Return the modified response with our changes
                return response
            except Exception as e:
                # If any error occurs in our post-processing, log it but return the original response
                print(f"Warning: Error processing Alpaca API response: {str(e)}")
                return response
                
        return response
    
    # Replace the original function with our patched version
    requests.get = patched_get
    
    print("✓ Monkey patched requests.get to fix Alpaca API issues")
    print("  - Removed feed parameter from options endpoints")
    print("  - Converted numeric values to strings in options contracts")
    print("  - Added Accept header for JSON responses")
    print("  - Removed verify parameter from requests")

def main():
    """Test the API fixes"""
    from datetime import datetime, timedelta
    
    # Get current date and date range
    today = datetime.now().date()
    thirty_days_ahead = today + timedelta(days=30)
    
    # Test getting options contracts
    print("Testing options contracts API...")
    contracts = get_options_contracts(
        underlying_symbols="SPY",
        expiration_date_gte=today.isoformat(),
        expiration_date_lte=thirty_days_ahead.isoformat(),
        strike_price_gte=400,  # Now properly handled as string
        strike_price_lte=500   # Now properly handled as string
    )
    
    print(f"Found {len(contracts)} option contracts")
    if contracts:
        contract_symbol = contracts[0]["symbol"]
        print(f"Sample contract: {contract_symbol}")
        
        # Test getting historical bars data
        print("\nTesting options bars API...")
        start_date = (today - timedelta(days=5)).isoformat()
        end_date = today.isoformat()
        
        bars = get_options_bars(
            symbols=contract_symbol,
            timeframe="1Day",
            start=start_date,
            end=end_date
        )
        
        if contract_symbol in bars:
            print(f"Found {len(bars[contract_symbol])} bars for {contract_symbol}")
        else:
            print(f"No bars found for {contract_symbol}")
        
        # Test getting option chain
        print("\nTesting option chain API...")
        chain = get_option_chain(underlying_symbol="SPY")
        print(f"Option chain data retrieved: {bool(chain)}")

    # Test the monkey patch
    print("\nTesting the monkey patch...")
    original_params = {
        "symbols": "AAPL240621C00200000",
        "timeframe": "1Day",
        "feed": "indicative"  # This should be removed by the patch
    }
    
    patched_params = patch_request_for_options_bars(original_params)
    print(f"Original params: {original_params}")
    print(f"Patched params: {patched_params}")
    
    original_contract_params = {
        "underlying_symbols": "AAPL",
        "strike_price_gte": 150
    }
    
    patched_contract_params = patch_request_for_options_contracts(original_contract_params)
    print(f"Original contract params: {original_contract_params}")
    print(f"Patched contract params: {patched_contract_params}")

if __name__ == "__main__":
    main() 