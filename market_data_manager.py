#!/usr/bin/env python
"""
Market Data Manager

This module provides a unified interface for market data access with
Alpaca as the data provider.
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
    Manages market data source with error handling and retry logic.
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
        Call a method with automatic retry if it fails.
        
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
    
    def get_active_source(self) -> str:
        """Get the name of the currently active data source"""
        if self.current_source_index < len(self.sources):
            return self.sources[self.current_source_index].__class__.__name__
        return "Unknown"

def get_market_data_manager():
    """
    Factory function to create a market data manager with Alpaca.
    
    Returns:
        MarketDataManager instance
    """
    # Initialize Alpaca data source
    try:
        logger.info("Initializing Alpaca market data source")
        alpaca = AlpacaMarketData()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca market data source: {e}")
        alpaca = None
    
    # Add available data sources to the list
    data_sources = [source for source in [alpaca] if source is not None]
    
    if not data_sources:
        raise Exception("No market data sources available")
    
    # Create market data manager with available sources
    return MarketDataManager(data_sources[0], data_sources[1:] if len(data_sources) > 1 else [])

if __name__ == "__main__":
    # Test the market data manager
    manager = get_market_data_manager()
    print(f"Active data source: {manager.get_active_source()}")
    
    # Test basic functionality
    symbols = ["SPY", "AAPL", "MSFT"]
    for symbol in symbols:
        try:
            price = manager.get_price(symbol)
            print(f"{symbol} price: ${price:.2f}")
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}") 