#!/usr/bin/env python
"""
Alpaca Market Data Provider

This module connects to the Alpaca Markets API to provide real market data
for stocks and options to the trading system.
"""

import requests
import pandas as pd
import numpy as np
import datetime
import os
import time
import base64
from typing import Dict, List, Tuple, Optional
import json
import math
import fix_alpaca_options  # This automatically applies all fixes

class AlpacaMarketData:
    """
    Provides real market data using the Alpaca Markets API.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, base_url: str = None):
        """
        Initialize the Alpaca market data provider with API credentials.
        
        Args:
            api_key: Alpaca API key (default: from environment variable)
            api_secret: Alpaca API secret (default: from environment variable)
            base_url: Alpaca API base URL (default: from environment variable or paper trading endpoint)
        """
        # Get API credentials from environment variables if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        self.base_url = base_url or os.environ.get("ALPACA_API_BASE_URL", "https://api.alpaca.markets/v2")
        
        # Market Data API endpoints
        self.data_url = "https://data.alpaca.markets/v2"
        self.data_beta_url = "https://data.alpaca.markets/v1beta1"
        self.options_contracts_url = "https://api.alpaca.markets/v2/options/contracts"
        
        # Standard Alpaca API authentication headers
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json"
        }
        
        # Cache for price data to reduce API calls
        self.price_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # Cache duration in seconds
        
        print(f"Alpaca Market Data Provider initialized")
        print(f"Testing connection to Alpaca API...")
        
        try:
            # Test connection by fetching account info
            response = requests.get(f"{self.base_url}/account", headers=self.headers)
            if response.status_code == 200:
                print(f"Connection successful: Alpaca API connected.")
            else:
                print(f"Error connecting to Alpaca API: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception while connecting to Alpaca API: {e}")
    
    def get_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Current price as a float
        """
        # Check cache first
        current_time = time.time()
        if symbol in self.price_cache and current_time - self.cache_expiry.get(symbol, 0) < self.cache_duration:
            return self.price_cache[symbol]
        
        # If not in cache or expired, fetch from API
        try:
            # Try to get the latest trade first
            params = {}  # Empty params dict, no verify parameter
            response = requests.get(
                f"{self.data_url}/stocks/{symbol}/trades/latest",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200 and 'trade' in response.json():
                # Use last trade price
                price = response.json()['trade']['p']
                
                # Update cache
                self.price_cache[symbol] = price
                self.cache_expiry[symbol] = current_time
                
                return price
            else:
                # Fall back to quotes if trades don't work
                response = requests.get(
                    f"{self.data_url}/stocks/{symbol}/quotes/latest",
                    headers=self.headers,
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Use midpoint of bid/ask as the price
                    if 'quote' in data and data['quote']:
                        price = (data['quote']['ap'] + data['quote']['bp']) / 2
                        
                        # Update cache
                        self.price_cache[symbol] = price
                        self.cache_expiry[symbol] = current_time
                        
                        return price
                
                print(f"Error getting price data for {symbol}: {response.status_code} - {response.text}")
                return 0.0
                
        except Exception as e:
            print(f"Exception while getting price for {symbol}: {e}")
            return 0.0
    
    def get_options_chain(self, symbol: str) -> Dict:
        """
        Get the options chain for a symbol using the Alpaca Options API.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with options chain data organized by expiration date
        """
        try:
            # Get current date
            today = datetime.datetime.now().date()
            next_month = today + datetime.timedelta(days=30)
            
            # Use the v1beta1 option chain endpoint (snapshots endpoint)
            url = f"{self.data_beta_url}/options/snapshots/{symbol}"
            
            print(f"  Requesting options chain for {symbol} from URL: {url}")
            
            # Make the request with no feed parameter (fix_alpaca_options should handle this)
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Error fetching options chain: {response.status_code} - {response.text}")
                # Try the contracts endpoint as a fallback
                return self._get_options_chain_from_contracts(symbol, today, next_month)
            
            # Process the v1beta1 options snapshots response
            snapshots_data = response.json()
            
            # Check for next_page_token and handle gracefully if present
            if 'next_page_token' in snapshots_data:
                # Don't try to parse the token as int, just log that we've found it
                # The token is a base64 encoded string and not meant to be parsed as an integer
                print(f"Found next_page_token in response: {snapshots_data['next_page_token'][:5]}...")
                
            # Check if we have snapshot data
            if 'snapshots' in snapshots_data:
                snapshots_data = snapshots_data['snapshots']
            elif not snapshots_data:
                print(f"No options snapshots data found for {symbol}")
                # Try the contracts endpoint as a fallback
                return self._get_options_chain_from_contracts(symbol, today, next_month)
            
            # Organize by expiration date
            options_chain = {}
            
            # Get the current price for the underlying
            current_price = self.get_price(symbol)
            
            # Process each contract in snapshots
            for contract_symbol, snapshot in snapshots_data.items():
                # Extract contract details from the symbol (e.g., AAPL250517C00190000)
                # Format: Symbol + Expiration (YYMMDD) + Type (C/P) + Strike (8 digits with leading zeros)
                try:
                    # Parse option symbol to get expiration, type, and strike
                    if len(contract_symbol) >= 15:  # Basic validation
                        exp_yymmdd = contract_symbol[-15:-9]  # Extract date portion YYMMDD
                        option_type = contract_symbol[-9:-8].lower()  # Extract C or P
                        strike_str = contract_symbol[-8:]  # Extract strike with padding
                        
                        # Convert to standard format
                        year = int("20" + exp_yymmdd[0:2])
                        month = int(exp_yymmdd[2:4])
                        day = int(exp_yymmdd[4:6])
                        
                        # Create expiration date in YYYY-MM-DD format
                        exp_date = f"{year}-{month:02d}-{day:02d}"
                        
                        # Convert strike to float (remove padding and divide by 1000)
                        strike = float(strike_str) / 1000
                        
                        # Skip if expiration is more than a month away or if no valid data
                        exp_datetime = datetime.datetime.strptime(exp_date, "%Y-%m-%d").date()
                        if exp_datetime > next_month:
                            continue
                            
                        # Initialize chain structure for this expiration if not exists
                        if exp_date not in options_chain:
                            options_chain[exp_date] = {
                                "calls": {},
                                "puts": {}
                            }
                        
                        # Get quote data
                        quote_data = snapshot.get('quote', {})
                        trade_data = snapshot.get('trade', {})
                        greeks_data = snapshot.get('greeks', {})
                        
                        # Skip if no quote data
                        if not quote_data:
                            continue
                        
                        # Prepare contract data
                        contract_data = {
                            "symbol": contract_symbol,
                            "bid": quote_data.get('bp', 0),
                            "ask": quote_data.get('ap', 0),
                            "last": trade_data.get('p', 0) if trade_data else (quote_data.get('bp', 0) + quote_data.get('ap', 0)) / 2,
                            "volume": trade_data.get('s', 0) if trade_data else 0,
                            "open_interest": 0,  # Not provided in the snapshot
                            "iv": greeks_data.get('implied_volatility', 30.0),
                            "delta": greeks_data.get('delta', 0.5 if option_type == 'c' else -0.5),
                            "gamma": greeks_data.get('gamma', 0.01),
                            "theta": greeks_data.get('theta', -0.01),
                            "vega": greeks_data.get('vega', 0.1),
                            "rho": greeks_data.get('rho', 0.01),
                            "strike": strike,
                            "underlying_price": current_price
                        }
                        
                        # Add to the appropriate section (calls or puts)
                        if option_type == 'c':
                            options_chain[exp_date]["calls"][str(strike)] = contract_data
                        elif option_type == 'p':
                            options_chain[exp_date]["puts"][str(strike)] = contract_data
                except Exception as e:
                    print(f"Error processing contract {contract_symbol}: {e}")
                    continue
            
            # If we got a valid options chain, return it
            if options_chain and any(len(chain["calls"]) > 0 or len(chain["puts"]) > 0 for chain in options_chain.values()):
                return options_chain
            
            # If we didn't get valid data, try the contracts endpoint
            return self._get_options_chain_from_contracts(symbol, today, next_month)
                
        except Exception as e:
            print(f"Error processing contract next_page_token: {e}")
            # Try the contracts endpoint as a fallback
            return self._get_options_chain_from_contracts(symbol, today, next_month)
    
    def _get_options_chain_from_contracts(self, symbol: str, start_date, end_date) -> Dict:
        """
        Alternative method to get options chain using the contracts endpoint.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for expiration range
            end_date: End date for expiration range
            
        Returns:
            Dictionary with options chain data
        """
        try:
            # Format the options contracts URL with query parameters
            url = f"{self.options_contracts_url}"
            
            # Fetch options contracts for the symbol, for both calls and puts
            params = {
                "underlying_symbols": symbol,
                "expiration_date_gte": start_date.isoformat(),
                "expiration_date_lte": end_date.isoformat(),
                "limit": 100  # Max limit to get a good sample of contracts
            }
            
            print(f"  Requesting URL: {url} with params: {params}")
            
            # Fetch options contracts
            response = requests.get(
                url,
                headers=self.headers,
                params=params
            )
            
            if response.status_code != 200:
                print(f"Error fetching options contracts: {response.status_code} - {response.text}")
                return {}
                
            # Process the response to get the options chain
            contract_data = response.json()
            
            if 'option_contracts' not in contract_data or not contract_data['option_contracts']:
                print(f"No options contracts found for {symbol}")
                return {}
                
            options_chain = {}
            
            # Get the current price for the underlier
            current_price = self.get_price(symbol)
            
            # Process each contract
            for contract in contract_data['option_contracts']:
                # Get expiration date
                exp_date = contract['expiration_date']
                
                # Initialize chain structure for this expiration if not exists
                if exp_date not in options_chain:
                    options_chain[exp_date] = {
                        "calls": {},
                        "puts": {}
                    }
                
                # Get contract details
                contract_type = contract['type'].lower()  # 'call' or 'put'
                strike = float(contract['strike_price'])
                contract_symbol = contract['symbol']
                
                # We need to get actual quote and trade data for this contract
                try:
                    # Get snapshot for this contract
                    snapshot_url = f"{self.data_beta_url}/options/snapshots"
                    snapshot_params = {"symbols": contract_symbol}
                    snapshot_response = requests.get(
                        snapshot_url,
                        headers=self.headers,
                        params=snapshot_params
                    )
                    
                    if snapshot_response.status_code != 200:
                        # Use default values if we can't get snapshot data
                        contract_data = {
                            "symbol": contract_symbol,
                            "bid": 0,
                            "ask": 0,
                            "last": 0,
                            "volume": 0,
                            "open_interest": 0,
                            "iv": 30.0,  # Default IV
                            "delta": 0.5 if contract_type == 'call' else -0.5,
                            "gamma": 0.01,
                            "theta": -0.01,
                            "vega": 0.1,
                            "rho": 0.01,
                            "strike": strike,
                            "underlying_price": current_price
                        }
                    else:
                        snapshot_data = snapshot_response.json()
                        contract_snapshot = snapshot_data.get(contract_symbol, {})
                        
                        # Get quote and trade data
                        quote_data = contract_snapshot.get('quote', {})
                        trade_data = contract_snapshot.get('trade', {})
                        greeks_data = contract_snapshot.get('greeks', {})
                        
                        contract_data = {
                            "symbol": contract_symbol,
                            "bid": quote_data.get('bp', 0),
                            "ask": quote_data.get('ap', 0),
                            "last": trade_data.get('p', 0) if trade_data else (quote_data.get('bp', 0) + quote_data.get('ap', 0)) / 2,
                            "volume": trade_data.get('s', 0) if trade_data else 0,
                            "open_interest": 0,  # Not provided in the snapshot
                            "iv": greeks_data.get('implied_volatility', 30.0),
                            "delta": greeks_data.get('delta', 0.5 if contract_type == 'call' else -0.5),
                            "gamma": greeks_data.get('gamma', 0.01),
                            "theta": greeks_data.get('theta', -0.01),
                            "vega": greeks_data.get('vega', 0.1),
                            "rho": greeks_data.get('rho', 0.01),
                            "strike": strike,
                            "underlying_price": current_price
                        }
                    
                    # Add to the appropriate section (calls or puts)
                    if contract_type == 'call':
                        options_chain[exp_date]["calls"][str(strike)] = contract_data
                    elif contract_type == 'put':
                        options_chain[exp_date]["puts"][str(strike)] = contract_data
                        
                except Exception as e:
                    print(f"Error getting snapshot for {contract_symbol}: {e}")
                    continue
            
            return options_chain
                
        except Exception as e:
            print(f"Exception in _get_options_chain_from_contracts for {symbol}: {e}")
            return {}
    
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1Day") -> Dict:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Time frame for the bars (e.g., '1Day', '1Hour', '5Min')
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Map timeframe format to match Alpaca's expected format
            timeframe_map = {
                "1D": "1Day",
                "1Day": "1Day",
                "1H": "1Hour",
                "1Hour": "1Hour",
                "5m": "5Min",
                "5min": "5Min",
                "5Min": "5Min",
                "1m": "1Min",
                "1min": "1Min",
                "1Min": "1Min",
            }
            
            # Use the provided timeframe or map it if needed
            alpaca_timeframe = timeframe_map.get(timeframe, timeframe)
            
            # Format the URL for stock bars
            url = f"{self.data_url}/stocks/bars"
            
            params = {
                'symbols': symbol,
                'timeframe': alpaca_timeframe,
                'start': start_date,
                'end': end_date,
                'limit': 1000,
                'adjustment': 'raw'
            }
            
            print(f"Requesting historical prices for {symbol} from {start_date} to {end_date}")
            print(f"URL: {url}")
            print(f"Params: {params}")
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching historical prices: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data or 'bars' not in data or symbol not in data['bars']:
                print(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            bars = data['bars'][symbol]
            df = pd.DataFrame(bars)
            
            # Convert timestamp to datetime and set as index
            if 't' in df.columns:
                df['timestamp'] = pd.to_datetime(df['t'])
                df.set_index('timestamp', inplace=True)
            
            # Rename columns to match expected format
            column_map = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Ensure we have all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Missing column {col} in historical data")
            
            # Add missing columns if needed
            if 'adjusted_close' not in df.columns and 'close' in df.columns:
                df['adjusted_close'] = df['close']
                
            # Make column names match what the application expects
            df.columns = [col.lower() for col in df.columns]
            
            # Make first letter uppercase to match what the simulator is expecting
            final_columns = {}
            for col in df.columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    final_columns[col] = col.capitalize()
            
            if final_columns:
                df.rename(columns=final_columns, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Exception while getting historical prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str) -> Dict:
        """
        Calculate technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with technical indicators (RSI, MACD, etc.)
        """
        try:
            # Get current price
            current_price = self.get_price(symbol)
            
            # Try to get yesterday's price from snapshot
            yesterday_price = current_price  # Default to current price
            
            try:
                # Use the snapshot endpoint which is often available on all tiers
                response = requests.get(
                    f"https://data.alpaca.markets/v2/stocks/{symbol}/snapshot",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'dailyBar' in data:
                        yesterday_price = data['dailyBar']['c']
                    elif 'prevDailyBar' in data:
                        yesterday_price = data['prevDailyBar']['c']
            except Exception as e:
                print(f"Couldn't fetch snapshot for {symbol}, using defaults: {e}")
            
            # Calculate 1-day price change
            price_change_1d = (current_price / yesterday_price - 1) if yesterday_price > 0 else 0
            
            # For RSI and other indicators that need historical data,
            # we'll use a default value since historical data might not be available
            # on the current subscription level
            rsi = 50  # Neutral RSI value as default
            
            # Try to get more historical data if available for better calculations
            try:
                # Attempt to get a small amount of recent data - may not work on all subscription tiers
                end = datetime.datetime.now()
                start = end - datetime.timedelta(days=14)  # 14 days for RSI calculation
                
                response = requests.get(
                    f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                    headers=self.headers,
                    params={
                        "timeframe": "1Day",
                        "start": start.strftime("%Y-%m-%d"),
                        "end": end.strftime("%Y-%m-%d"),
                        "limit": 14
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and data.get('bars') and len(data['bars']) > 2:
                        # Convert to DataFrame for calculations
                        bars = data['bars']
                        prices = pd.DataFrame(bars)
                        prices['timestamp'] = pd.to_datetime(prices['t'])
                        prices.set_index('timestamp', inplace=True)
                        prices = prices.sort_index()
                        
                        # Calculate RSI
                        delta = prices['c'].diff()
                        gains = delta.where(delta > 0, 0).rolling(window=14).mean()
                        losses = -delta.where(delta < 0, 0).rolling(window=14).mean()
                        
                        if losses.iloc[-1] == 0:
                            rsi = 100
                        else:
                            rs = gains.iloc[-1] / losses.iloc[-1]
                            rsi = 100 - (100 / (1 + rs))
                        
                        # Recalculate 1-day price change from actual data
                        if len(prices) >= 2:
                            price_change_1d = (prices['c'].iloc[-1] / prices['c'].iloc[-2] - 1)
            except Exception as e:
                print(f"Couldn't calculate RSI for {symbol} using historical data: {e}")
            
            # Return technical indicators
            return {
                "rsi": rsi,
                "price_change_1d": price_change_1d
            }
            
        except Exception as e:
            print(f"Exception while calculating indicators for {symbol}: {e}")
            return {"rsi": 50, "price_change_1d": 0}
    
    def get_option_signal(self, symbol: str, option_type: str, strike: float, expiration: str) -> Optional[Dict]:
        """
        Get detailed information for a specific option contract.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            option_type: 'call' or 'put'
            strike: Option strike price
            expiration: Expiration date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with option details or None if not available
        """
        try:
            # First, try to find the contract ID by querying with filters
            contract_params = {
                "underlying_symbols": symbol,
                "expiration_date": expiration,
                "strike_price": str(strike),
                "type": option_type
            }
            
            response = requests.get(
                self.options_contracts_url,
                headers=self.headers,
                params=contract_params
            )
            
            if response.status_code != 200:
                print(f"Error fetching option contract: {response.status_code} - {response.text}")
                return None
                
            data = response.json()
            
            if not data or not data.get('option_contracts') or len(data['option_contracts']) == 0:
                print(f"No option contract found for {symbol} {expiration} {strike} {option_type}")
                return None
                
            # Get the first matching contract
            contract = data['option_contracts'][0]
            contract_id = contract['id']
            contract_symbol = contract['symbol']
            
            # Get detailed contract data
            detailed_response = requests.get(
                f"{self.options_contracts_url}/{contract_id}",
                headers=self.headers
            )
            
            if detailed_response.status_code != 200:
                print(f"Error fetching detailed contract data: {detailed_response.status_code} - {detailed_response.text}")
                return None
                
            detailed_contract = detailed_response.json()
            
            # Get underlying price
            underlying_price = self.get_price(symbol)
            
            # Try to get market data (bid/ask) from the snapshot endpoint
            market_data_response = requests.get(
                f"https://data.alpaca.markets/v2/stocks/options/{contract_symbol}/snapshot",
                headers=self.headers
            )
            
            # Initialize with default values
            bid = 0
            ask = 0
            last = 0
            iv = 0.3  # Default IV (30%)
            delta = 0.5 if option_type == 'call' else -0.5  # Default delta
            gamma = 0.01  # Default gamma
            theta = -0.01  # Default theta
            vega = 0.1  # Default vega
            
            # Update with market data if available
            if market_data_response.status_code == 200:
                market_data = market_data_response.json()
                if 'snapshot' in market_data:
                    snapshot = market_data['snapshot']
                    bid = snapshot.get('bid_price', 0)
                    ask = snapshot.get('ask_price', 0)
                    last = snapshot.get('last_price', 0)
                    iv = snapshot.get('implied_volatility', 0.3)
                    delta = snapshot.get('delta', delta)
                    gamma = snapshot.get('gamma', gamma)
                    theta = snapshot.get('theta', theta)
                    vega = snapshot.get('vega', vega)
            else:
                # If no market data, use close price from contract data if available
                if detailed_contract.get('close_price'):
                    price = float(detailed_contract.get('close_price', 0))
                    bid = price * 0.98  # Estimated
                    ask = price * 1.02  # Estimated
                    last = price
            
            # Calculate mid price
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            
            # Use a default if all price sources failed
            if mid <= 0:
                # Estimate a reasonable option price based on proximity to strike
                # This is very approximate
                moneyness = underlying_price / strike - 1
                time_to_expiry = (datetime.datetime.strptime(expiration, "%Y-%m-%d").date() - datetime.datetime.now().date()).days / 365.0
                
                # Very basic price estimate
                if option_type == 'call':
                    mid = max(0, underlying_price * 0.05 * (1 + moneyness) * time_to_expiry)
                else:
                    mid = max(0, underlying_price * 0.05 * (1 - moneyness) * time_to_expiry)
            
            # Convert to our system's signal format
            return {
                "symbol": symbol,
                "option_type": option_type,
                "strike": strike,
                "expiration": expiration,
                "current_price": mid,
                "underlying_price": underlying_price,
                "entry_price_range": (bid, ask),
                "stop_loss": mid * 0.85,  # Default 15% stop loss
                "target_price": mid * 1.15,  # Default 15% profit target
                "signal_strength": 0.7,  # Default value, will be calculated by pattern recognition
                "volume": int(detailed_contract.get('volume', 0)),
                "open_interest": int(detailed_contract.get('open_interest', 0)),
                "iv": iv * 100,  # Convert to percentage
                "delta": delta,
                "bid": bid,
                "ask": ask,
                "mid": mid
            }
            
        except Exception as e:
            print(f"Exception while getting option signal: {e}")
            return None

if __name__ == "__main__":
    # Test the market data provider
    alpaca = AlpacaMarketData()
    
    # Test getting price
    price = alpaca.get_price("AAPL")
    print(f"AAPL price: ${price:.2f}")
    
    # Test getting technical indicators
    indicators = alpaca.get_technical_indicators("AAPL")
    print(f"AAPL RSI: {indicators['rsi']:.2f}, 1-day change: {indicators['price_change_1d']*100:.2f}%")
    
    # Test getting options chain
    chain = alpaca.get_options_chain("AAPL")
    if chain:
        print(f"Found options data for {len(chain)} expiration dates")
        for exp in chain:
            print(f"Expiration {exp}: {len(chain[exp]['calls'])} calls, {len(chain[exp]['puts'])} puts")
    else:
        print("No options chain data found") 