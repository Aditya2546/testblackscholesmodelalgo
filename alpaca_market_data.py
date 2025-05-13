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
        
        # Market Data API endpoint - correct URL according to documentation
        self.data_url = "https://data.alpaca.markets/v2"
        self.options_base_url = "https://data.alpaca.markets/v2"
        self.options_contracts_url = "https://api.alpaca.markets/v2/options/contracts"
        
        # Standard Alpaca Trading API authentication headers
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        # Set up options API headers with API key auth - this is the preferred method for newer endpoints
        self.options_headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
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
            # Try to get the latest trade first since this appears to work on all subscription levels
            response = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                headers=self.headers
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
                    f"{self.base_url}/stocks/{symbol}/quotes/latest",
                    headers=self.headers
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
            
            # Format the options contracts URL with query parameters
            # The correct endpoint is /v2/options/contracts according to the docs
            url = f"{self.options_contracts_url}"
            
            # Fetch options contracts for the symbol, for both calls and puts
            params = {
                "underlying_symbols": symbol,
                "expiration_date_gte": today.isoformat(),
                "expiration_date_lte": next_month.isoformat(),
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
                exp_key = exp_date
                
                # Initialize chain structure for this expiration if not exists
                if exp_key not in options_chain:
                    options_chain[exp_key] = {
                        "calls": {},
                        "puts": {}
                    }
                
                # Get contract details
                contract_type = contract['type'].lower()  # 'call' or 'put'
                strike = float(contract['strike_price'])
                contract_symbol = contract['symbol']
                contract_id = contract['id']
                
                # Skip if strike price is too far from current price (±20%)
                min_strike = current_price * 0.8
                max_strike = current_price * 1.2
                if not (min_strike <= strike <= max_strike):
                    continue
                
                # Try to get market data for this contract (might not be available)
                try:
                    # Get the detailed contract data
                    contract_response = requests.get(
                        f"{self.options_contracts_url}/{contract_id}",
                        headers=self.headers
                    )
                    
                    if contract_response.status_code != 200:
                        continue
                    
                    detailed_contract = contract_response.json()
                    
                    # Create basic contract data with available information
                    contract_data = {
                        "bid": 0,  # Will try to get from market data
                        "ask": 0,  # Will try to get from market data
                        "last": 0,  # Will try to get from market data
                        "volume": 0,
                        "open_interest": int(detailed_contract.get('open_interest', 0)),
                        "iv": 30.0,  # Default IV %
                        "delta": 0.5 if contract_type == 'call' else -0.5,  # Default delta
                        "gamma": 0.01,  # Default gamma
                        "theta": -0.01,  # Default theta
                        "vega": 0.1    # Default vega
                    }
                    
                    # Try to get real-time market data from stock options endpoint if available
                    try:
                        market_data_response = requests.get(
                            f"{self.options_base_url}/stocks/options/{contract_symbol}/snapshot",
                            headers=self.headers
                        )
                        
                        if market_data_response.status_code == 200:
                            market_data = market_data_response.json()
                            if 'snapshot' in market_data:
                                snapshot = market_data['snapshot']
                                
                                # Update with real market data
                                contract_data.update({
                                    "bid": snapshot.get('bid_price', 0),
                                    "ask": snapshot.get('ask_price', 0),
                                    "last": snapshot.get('last_price', 0),
                                    "volume": snapshot.get('volume', 0),
                                    "iv": snapshot.get('implied_volatility', 0.3) * 100,  # Convert to percentage
                                    "delta": snapshot.get('delta', contract_data['delta']),
                                    "gamma": snapshot.get('gamma', contract_data['gamma']),
                                    "theta": snapshot.get('theta', contract_data['theta']),
                                    "vega": snapshot.get('vega', contract_data['vega'])
                                })
                    except Exception as e:
                        # If market data isn't available, we'll use the basic info
                        print(f"Couldn't get market data for {contract_symbol}: {e}")
                        
                        # Use some estimated values based on close price from contract data
                        if detailed_contract.get('close_price'):
                            price = float(detailed_contract.get('close_price', 0))
                            contract_data.update({
                                "bid": price * 0.98,  # Estimated bid (2% below last)
                                "ask": price * 1.02,  # Estimated ask (2% above last)
                                "last": price
                            })
                    
                    # Add to the chain
                    options_chain[exp_key][contract_type + "s"][str(strike)] = contract_data
                    
                except Exception as e:
                    print(f"Error processing contract {contract_symbol}: {e}")
            
            return options_chain
            
        except Exception as e:
            print(f"Exception while getting options chain for {symbol}: {e}")
            return {}
    
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
                f"{self.options_base_url}/stocks/options/{contract_symbol}/snapshot",
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