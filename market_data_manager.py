#!/usr/bin/env python
"""
Market Data Manager

This module provides a unified interface for market data access with automatic
failover between different data sources.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import data source modules
from alpaca_market_data import AlpacaMarketData

# We'll define abstract base class to ensure consistent interface 
class MarketDataSource:
    """Base class for market data sources"""
    
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        raise NotImplementedError
    
    def get_options_chain(self, symbol: str) -> Dict:
        """Get options chain for a symbol"""
        raise NotImplementedError
        
    def get_option_quote(self, symbol: str, expiration: str, strike: float, option_type: str) -> Dict:
        """Get quote for a specific option contract"""
        raise NotImplementedError
        
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> Dict:
        """Get historical price data"""
        raise NotImplementedError
        
    def check_connection(self) -> bool:
        """Check if connection to the data source is working"""
        raise NotImplementedError


class MarketDataManager:
    """
    Manages multiple market data sources with automatic failover.
    Provides a unified interface regardless of the underlying data source.
    """
    
    def __init__(self, primary_source: Optional[MarketDataSource] = None, 
                 backup_sources: List[MarketDataSource] = None):
        """
        Initialize the market data manager.
        
        Args:
            primary_source: Primary market data source
            backup_sources: List of backup data sources in priority order
        """
        self.sources = []
        self.current_source_index = 0
        self.retry_count = 3
        self.retry_delay = 2  # seconds
        self.last_failover_time = None
        self.failover_cooldown = 300  # seconds (5 minutes)
        
        # Initialize with Alpaca as primary source if none provided
        if primary_source is None:
            try:
                logger.info("Initializing Alpaca as primary data source")
                # Try to import from alpaca_config first
                try:
                    from alpaca_config import API_KEY, API_SECRET
                    primary_source = AlpacaMarketData(API_KEY, API_SECRET)
                except ImportError:
                    # Fall back to environment variables
                    primary_source = AlpacaMarketData()
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca data source: {e}")
                primary_source = None
        
        # Add primary source if available
        if primary_source:
            self.sources.append(primary_source)
            
        # Add backup sources
        if backup_sources:
            self.sources.extend(backup_sources)
            
        # Verify we have at least one data source
        if not self.sources:
            raise ValueError("No market data sources available")
            
        # Initial check of data sources
        self._check_sources()
        
        logger.info(f"Market Data Manager initialized with {len(self.sources)} data sources")

    def _check_sources(self) -> bool:
        """
        Check all data sources and rearrange them by availability.
        
        Returns:
            True if at least one source is available, False otherwise
        """
        working_sources = []
        failed_sources = []
        
        for source in self.sources:
            try:
                if source.check_connection():
                    working_sources.append(source)
                else:
                    failed_sources.append(source)
            except Exception as e:
                logger.warning(f"Error checking data source: {e}")
                failed_sources.append(source)
                
        # Rearrange sources to put working ones first
        self.sources = working_sources + failed_sources
        self.current_source_index = 0
        
        return len(working_sources) > 0
    
    def _call_with_failover(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method with automatic failover if it fails.
        
        Args:
            method_name: Name of the method to call
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            Result of the method call
            
        Raises:
            Exception: If all data sources fail
        """
        # Try current source first
        start_idx = self.current_source_index
        
        # Try each source up to retry_count times
        for attempt in range(self.retry_count):
            idx = start_idx
            
            # Try each source
            for _ in range(len(self.sources)):
                source = self.sources[idx]
                
                try:
                    method = getattr(source, method_name)
                    result = method(*args, **kwargs)
                    
                    # If successful, update current source index
                    if idx != self.current_source_index:
                        logger.info(f"Switched to data source {idx}")
                        self.current_source_index = idx
                        self.last_failover_time = datetime.now()
                        
                    return result
                    
                except Exception as e:
                    logger.warning(f"Data source {idx} failed for {method_name}: {e}")
                    # Try next source
                    idx = (idx + 1) % len(self.sources)
            
            # All sources failed, wait and try again
            if attempt < self.retry_count - 1:
                logger.warning(f"All data sources failed, retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                
        # Check if we should try to recover primary source
        now = datetime.now()
        if (self.last_failover_time and 
            (now - self.last_failover_time).total_seconds() > self.failover_cooldown):
            logger.info("Trying to recover primary data source")
            self._check_sources()
            
        # All attempts failed
        raise Exception(f"All data sources failed for {method_name} after {self.retry_count} attempts")
    
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        return self._call_with_failover("get_price", symbol)
    
    def get_options_chain(self, symbol: str) -> Dict:
        """Get options chain for a symbol"""
        return self._call_with_failover("get_options_chain", symbol)
        
    def get_option_quote(self, symbol: str, expiration: str, strike: float, option_type: str) -> Dict:
        """Get quote for a specific option contract"""
        return self._call_with_failover("get_option_quote", symbol, expiration, strike, option_type)
        
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> Dict:
        """Get historical price data"""
        return self._call_with_failover("get_historical_prices", symbol, start_date, end_date, timeframe)
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return [type(source).__name__ for source in self.sources]
    
    def get_active_source(self) -> str:
        """Get name of currently active data source"""
        return type(self.sources[self.current_source_index]).__name__


# Example alternative data source implementation
class YahooFinanceDataSource(MarketDataSource):
    """Yahoo Finance data source implementation"""
    
    def __init__(self):
        """Initialize Yahoo Finance data source"""
        self.connected = False
        try:
            import yfinance as yf
            self.yf = yf
            self.connected = True
            logger.info("Yahoo Finance data source initialized")
        except ImportError:
            logger.warning("yfinance package not available. Install with: pip install yfinance")
            
    def check_connection(self) -> bool:
        """Check if connection to Yahoo Finance is working"""
        if not self.connected:
            return False
            
        try:
            # Try to fetch a simple quote as a test
            test = self.yf.Ticker("SPY").info
            return "symbol" in test
        except Exception as e:
            logger.warning(f"Yahoo Finance connection check failed: {e}")
            return False
            
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if not self.connected:
            raise Exception("Yahoo Finance data source not connected")
            
        ticker = self.yf.Ticker(symbol)
        data = ticker.history(period="1d")
        
        if data.empty:
            raise Exception(f"No price data found for {symbol}")
            
        return float(data["Close"].iloc[-1])
        
    def get_options_chain(self, symbol: str) -> Dict:
        """Get options chain for a symbol"""
        if not self.connected:
            raise Exception("Yahoo Finance data source not connected")
            
        ticker = self.yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            raise Exception(f"No options data found for {symbol}")
            
        result = {}
        
        # Get current price
        price = self.get_price(symbol)
        
        # Define near-the-money range (±10%)
        min_strike = price * 0.9
        max_strike = price * 1.1
        
        # Get options for each expiration
        for expiry in expirations:
            # Convert Yahoo date format to ISO
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").strftime("%Y-%m-%d")
            
            calls = ticker.option_chain(expiry).calls
            puts = ticker.option_chain(expiry).puts
            
            # Filter for near-the-money options
            calls = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)]
            puts = puts[(puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)]
            
            result[exp_date] = {
                "calls": {},
                "puts": {}
            }
            
            # Format call options
            for _, row in calls.iterrows():
                strike = str(row["strike"])
                result[exp_date]["calls"][strike] = {
                    "bid": float(row["bid"]),
                    "ask": float(row["ask"]),
                    "last": float(row["lastPrice"]),
                    "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                    "open_interest": int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0,
                    "iv": float(row["impliedVolatility"]) * 100,  # Convert to percentage
                    "delta": 0.5,  # Yahoo doesn't provide Greeks, using approximation
                    "gamma": 0.01,
                    "theta": -0.01,
                    "vega": 0.1,
                    "strike": float(row["strike"]),
                    "underlying_price": price
                }
                
            # Format put options
            for _, row in puts.iterrows():
                strike = str(row["strike"])
                result[exp_date]["puts"][strike] = {
                    "bid": float(row["bid"]),
                    "ask": float(row["ask"]),
                    "last": float(row["lastPrice"]),
                    "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                    "open_interest": int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0,
                    "iv": float(row["impliedVolatility"]) * 100,  # Convert to percentage
                    "delta": -0.5,  # Yahoo doesn't provide Greeks, using approximation
                    "gamma": 0.01,
                    "theta": -0.01,
                    "vega": 0.1,
                    "strike": float(row["strike"]),
                    "underlying_price": price
                }
                
        return result
        
    def get_option_quote(self, symbol: str, expiration: str, strike: float, option_type: str) -> Dict:
        """Get quote for a specific option contract"""
        if not self.connected:
            raise Exception("Yahoo Finance data source not connected")
            
        # Get the options chain
        chain = self.get_options_chain(symbol)
        
        # Check if expiration exists
        if expiration not in chain:
            raise Exception(f"Expiration {expiration} not found for {symbol}")
            
        # Check option type
        if option_type.lower() not in ["call", "put"]:
            raise Exception(f"Invalid option type: {option_type}")
            
        # Get calls or puts
        option_dict = chain[expiration]["calls"] if option_type.lower() == "call" else chain[expiration]["puts"]
        
        # Convert strike to string for lookup
        strike_str = str(strike)
        
        # Check if strike exists
        if strike_str not in option_dict:
            raise Exception(f"Strike {strike} not found for {symbol} {expiration} {option_type}")
            
        return option_dict[strike_str]
        
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> Dict:
        """Get historical price data"""
        if not self.connected:
            raise Exception("Yahoo Finance data source not connected")
            
        # Convert timeframe to Yahoo format
        timeframe_map = {
            "1D": "1d",
            "1H": "1h",
            "5m": "5m",
            "1m": "1m"
        }
        
        period = timeframe_map.get(timeframe, "1d")
        
        # Get historical data
        ticker = self.yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=period)
        
        if data.empty:
            raise Exception(f"No historical data found for {symbol}")
            
        # Convert to desired format
        result = {
            "symbol": symbol,
            "data": []
        }
        
        for idx, row in data.iterrows():
            result["data"].append({
                "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
            
        return result


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Create market data manager with Alpaca as primary and Yahoo as backup
    try:
        # Try to import Alpaca credentials
        from alpaca_config import API_KEY, API_SECRET
        alpaca = AlpacaMarketData(API_KEY, API_SECRET)
        
        # Initialize Yahoo Finance if available
        yahoo = None
        try:
            yahoo = YahooFinanceDataSource()
        except:
            logger.warning("Yahoo Finance data source not available")
            
        # Create manager with available sources
        data_sources = [source for source in [alpaca, yahoo] if source is not None]
        manager = MarketDataManager(primary_source=alpaca, backup_sources=data_sources[1:] if len(data_sources) > 1 else None)
        
        # Test the manager
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        
        print("Testing Market Data Manager...")
        print(f"Available sources: {manager.get_available_sources()}")
        print(f"Active source: {manager.get_active_source()}")
        
        for symbol in symbols:
            try:
                price = manager.get_price(symbol)
                print(f"{symbol} price: ${price:.2f}")
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                
        # Test options data for AAPL
        try:
            options = manager.get_options_chain("AAPL")
            print("\nAAPL Options Chain:")
            for expiry, chain in options.items():
                calls_count = len(chain["calls"])
                puts_count = len(chain["puts"])
                print(f"  {expiry}: {calls_count} calls, {puts_count} puts")
        except Exception as e:
            print(f"Error getting options chain: {e}")
            
    except Exception as e:
        print(f"Error initializing Market Data Manager: {e}") 