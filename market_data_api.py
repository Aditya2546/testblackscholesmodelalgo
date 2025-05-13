#!/usr/bin/env python
"""
Market Data API Integration for Options Data
"""

import os
import json
import requests
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

class MarketDataAPI:
    """
    Class to interact with the Market Data API for options data
    """
    
    BASE_URL = "https://api.marketdata.app/v1/options"
    
    def __init__(self, api_token: str = None):
        """
        Initialize the Market Data API client
        
        Args:
            api_token: The API token for Market Data API
        """
        self.api_token = api_token or os.environ.get("MARKET_DATA_API_TOKEN")
        if not self.api_token:
            raise ValueError("Market Data API token is required")
    
    def get_option_quote(self, 
                         option_symbol: str, 
                         date: str = None,
                         from_date: str = None,
                         to_date: str = None,
                         columns: List[str] = None) -> Dict:
        """
        Get option quote data from Market Data API
        
        Args:
            option_symbol: The OCC option symbol (e.g., AAPL250117C00150000)
            date: Single date for historical data (YYYY-MM-DD)
            from_date: Start date for historical range (YYYY-MM-DD)
            to_date: End date for historical range (YYYY-MM-DD)
            columns: Specific data columns to request
            
        Returns:
            Dict containing the option quote data
        """
        endpoint = f"{self.BASE_URL}/quotes/{option_symbol}/"
        
        params = {}
        if self.api_token:
            params["token"] = self.api_token
            
        if date:
            params["date"] = date
        elif from_date and to_date:
            params["from"] = from_date
            params["to"] = to_date
            
        if columns:
            params["columns"] = ",".join(columns)
        
        response = requests.get(endpoint, params=params)
        
        # The API can return 203 status code with data
        if response.status_code not in [200, 203]:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()
    
    def get_option_quote_as_dataframe(self, 
                                     option_symbol: str, 
                                     date: str = None,
                                     from_date: str = None,
                                     to_date: str = None,
                                     columns: List[str] = None) -> pd.DataFrame:
        """
        Get option quote data from Market Data API as a pandas DataFrame
        
        Args:
            option_symbol: The OCC option symbol (e.g., AAPL250117C00150000)
            date: Single date for historical data (YYYY-MM-DD)
            from_date: Start date for historical range (YYYY-MM-DD)
            to_date: End date for historical range (YYYY-MM-DD)
            columns: Specific data columns to request
            
        Returns:
            DataFrame containing the option quote data
        """
        data = self.get_option_quote(option_symbol, date, from_date, to_date, columns)
        
        if "s" in data and data["s"] in ["ok", "error"]:  # API sometimes returns "error" with data
            # Handle array-based response (typical for quotes)
            if "bid" in data and isinstance(data["bid"], list):
                df = pd.DataFrame()
                
                # Identify available columns in the response
                array_columns = [k for k, v in data.items() if isinstance(v, list) and len(v) > 0]
                
                # Map API field names to DataFrame column names
                field_mapping = {
                    "t": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "bid": "bid",
                    "ask": "ask",
                    "mid": "mid",
                    "last": "last",
                    "underlyingPrice": "underlying_price",
                    "openInterest": "open_interest",
                    "iv": "iv",
                    "delta": "delta",
                    "gamma": "gamma",
                    "theta": "theta",
                    "vega": "vega",
                    "rho": "rho",
                }
                
                # Build DataFrame from available columns
                for api_field, df_column in field_mapping.items():
                    if api_field in array_columns:
                        df[df_column] = data[api_field]
                
                # Convert timestamps to datetime if present
                if "timestamp" in df.columns and len(df["timestamp"]) > 0:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                
                return df
            # Handle single object response
            elif "o" in data:
                # Single day quote
                return pd.DataFrame([data["o"]])
            # Handle time series response
            elif "t" in data:
                # Time series quotes
                df = pd.DataFrame({
                    "timestamp": data.get("t", []),
                    "open": data.get("o", []),
                    "high": data.get("h", []),
                    "low": data.get("l", []),
                    "close": data.get("c", []),
                    "volume": data.get("v", []),
                    "bid": data.get("bid", []),
                    "ask": data.get("ask", []),
                    "mid": data.get("mid", []),
                    "last": data.get("last", []),
                    "underlying_price": data.get("underlying_price", []),
                    "iv": data.get("iv", []),
                    "delta": data.get("delta", []),
                    "gamma": data.get("gamma", []),
                    "theta": data.get("theta", []),
                    "vega": data.get("vega", []),
                    "rho": data.get("rho", []),
                    "open_interest": data.get("open_interest", []),
                })
                
                # Convert timestamps to datetime if present
                if "timestamp" in df.columns and len(df["timestamp"]) > 0:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                
                return df
        
        return pd.DataFrame()  # Return empty DataFrame if data format is unexpected
    
    def format_option_symbol(self, 
                            ticker: str, 
                            expiration_date: str, 
                            strike_price: float, 
                            option_type: str) -> str:
        """
        Format an option symbol in OCC format
        
        Args:
            ticker: The stock ticker symbol (e.g., AAPL)
            expiration_date: The option expiration date (YYYY-MM-DD)
            strike_price: The option strike price
            option_type: The option type ('C' for call, 'P' for put)
            
        Returns:
            The formatted OCC option symbol
        """
        # Convert expiration date to required format (YYMMDD)
        exp_date = datetime.datetime.strptime(expiration_date, "%Y-%m-%d")
        exp_str = exp_date.strftime("%y%m%d")
        
        # Format strike price with 8 digits, padded with leading zeros
        strike_str = f"{int(strike_price * 1000):08d}"
        
        # Option type must be 'C' or 'P'
        if option_type.upper() not in ['C', 'P']:
            raise ValueError("Option type must be 'C' for call or 'P' for put")
        
        # Create the OCC symbol
        return f"{ticker.upper()}{exp_str}{option_type.upper()}{strike_str}" 