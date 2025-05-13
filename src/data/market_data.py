"""
Market data acquisition and processing module.

Provides interfaces for handling real-time and historical market data 
with high-performance streaming capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import numpy as np
import pandas as pd


class AssetType(Enum):
    """Supported asset types for market data."""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    OPTION_ON_FUTURE = "option_on_future"
    INDEX = "index"
    ETF = "etf"


class OptionRight(Enum):
    """Option contract right (call or put)."""
    CALL = "call"
    PUT = "put"


class BarPeriod(Enum):
    """Supported bar periods for time series data."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


@dataclass
class Asset:
    """Base class for financial assets."""
    symbol: str
    asset_type: AssetType
    exchange: str = ""


@dataclass
class Equity(Asset):
    """Equity (stock) asset."""
    def __init__(self, symbol: str, exchange: str = ""):
        super().__init__(symbol=symbol, asset_type=AssetType.EQUITY, exchange=exchange)


class Option(Asset):
    """Option contract asset."""
    
    def __init__(self, symbol: str, underlying_symbol: str, expiration: datetime, 
                strike: float, right: OptionRight, exchange: str = ""):
        super().__init__(symbol=symbol, asset_type=AssetType.OPTION, exchange=exchange)
        self.underlying_symbol = underlying_symbol
        self.expiration = expiration
        self.strike = strike
        self.right = right
        
    @classmethod
    def create_from_symbol(cls, symbol: str, exchange: str = "") -> "Option":
        """Create an Option instance by parsing an OCC-style option symbol."""
        # Implement OCC symbol parsing (e.g., AAPL230616C00150000)
        # This is a simplified example, a real implementation would be more robust
        if len(symbol) < 15:
            raise ValueError(f"Invalid option symbol: {symbol}")
        
        underlying = symbol[:6].strip()
        year = int("20" + symbol[6:8])
        month = int(symbol[8:10])
        day = int(symbol[10:12])
        right = OptionRight.CALL if symbol[12] == "C" else OptionRight.PUT
        strike = float(symbol[13:]) / 1000.0
        
        expiration = datetime(year, month, day)
        
        return cls(
            symbol=symbol,
            underlying_symbol=underlying,
            expiration=expiration,
            strike=strike,
            right=right,
            exchange=exchange
        )


@dataclass
class MarketDataBar:
    """A single bar of market data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    trades: int = 0
    vwap: Optional[float] = None
    open_interest: Optional[int] = None
    
    @property
    def midpoint(self) -> float:
        """Calculate the midpoint price of the bar."""
        return (self.high + self.low) / 2.0


@dataclass
class Quote:
    """Market quote with bid/ask prices and sizes."""
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    exchange: str = ""
    
    @property
    def midpoint(self) -> float:
        """Calculate the midpoint of the bid/ask spread."""
        return (self.bid_price + self.ask_price) / 2.0
    
    @property 
    def spread(self) -> float:
        """Calculate the bid/ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_percent(self) -> float:
        """Calculate the spread as a percentage of the midpoint."""
        return (self.spread / self.midpoint) * 100.0 if self.midpoint > 0 else 0.0


@dataclass
class Trade:
    """Market trade with execution details."""
    timestamp: datetime
    price: float
    size: int
    exchange: str = ""
    trade_id: str = ""
    
    
@dataclass
class OptionChain:
    """Complete option chain for an underlying asset."""
    underlying_symbol: str
    timestamp: datetime
    options: Dict[Tuple[datetime, float, OptionRight], "OptionQuote"] = field(default_factory=dict)
    
    def get_options_by_expiration(self, expiration: datetime) -> Dict[Tuple[float, OptionRight], "OptionQuote"]:
        """Get all options for a specific expiration date."""
        return {(strike, right): quote 
                for (exp, strike, right), quote in self.options.items() 
                if exp == expiration}
    
    def get_options_by_strike(self, strike: float) -> Dict[Tuple[datetime, OptionRight], "OptionQuote"]:
        """Get all options for a specific strike price."""
        return {(exp, right): quote 
                for (exp, strike_, right), quote in self.options.items() 
                if strike_ == strike}
    
    def get_options_by_right(self, right: OptionRight) -> Dict[Tuple[datetime, float], "OptionQuote"]:
        """Get all options of a specific right (calls or puts)."""
        return {(exp, strike): quote 
                for (exp, strike, right_), quote in self.options.items() 
                if right_ == right}
    
    def add_option(self, option: "OptionQuote") -> None:
        """Add an option to the chain."""
        key = (option.expiration, option.strike, option.right)
        self.options[key] = option
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the option chain to a pandas DataFrame."""
        data = []
        for (expiration, strike, right), quote in self.options.items():
            data.append({
                'underlying': self.underlying_symbol,
                'expiration': expiration,
                'strike': strike,
                'right': right.value,
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'last': quote.last_price,
                'volume': quote.volume,
                'open_interest': quote.open_interest,
                'implied_volatility': quote.implied_volatility,
                'delta': quote.delta,
                'gamma': quote.gamma,
                'theta': quote.theta,
                'vega': quote.vega,
                'rho': quote.rho
            })
        return pd.DataFrame(data)


class OptionQuote(Quote):
    """Market data for an option contract with Greeks and implied volatility."""
    
    def __init__(self, timestamp: datetime, bid_price: float, ask_price: float, 
                bid_size: int, ask_size: int, underlying_symbol: str, 
                expiration: datetime, strike: float, right: OptionRight,
                exchange: str = "", last_price: float = 0.0, volume: int = 0, 
                open_interest: int = 0, implied_volatility: float = 0.0,
                delta: float = 0.0, gamma: float = 0.0, theta: float = 0.0, 
                vega: float = 0.0, rho: float = 0.0):
        """Initialize an option quote with all necessary parameters."""
        super().__init__(timestamp=timestamp, bid_price=bid_price, ask_price=ask_price,
                       bid_size=bid_size, ask_size=ask_size, exchange=exchange)
        self.underlying_symbol = underlying_symbol
        self.expiration = expiration
        self.strike = strike
        self.right = right
        self.last_price = last_price
        self.volume = volume
        self.open_interest = open_interest
        self.implied_volatility = implied_volatility
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho


class MarketDataSource(ABC):
    """Abstract base class for market data sources."""
    
    @abstractmethod
    def subscribe_quote(self, asset: Asset, callback: Callable[[Quote], None]) -> None:
        """
        Subscribe to quotes for an asset.
        
        Args:
            asset: Asset to subscribe to
            callback: Callback function to receive quote updates
        """
        pass
    
    @abstractmethod
    def subscribe_trade(self, asset: Asset, callback: Callable[[Trade], None]) -> None:
        """
        Subscribe to trades for an asset.
        
        Args:
            asset: Asset to subscribe to
            callback: Callback function to receive trade updates
        """
        pass
    
    @abstractmethod
    def subscribe_bar(self, asset: Asset, period: BarPeriod, 
                     callback: Callable[[MarketDataBar], None]) -> None:
        """
        Subscribe to bars for an asset.
        
        Args:
            asset: Asset to subscribe to
            period: Bar period
            callback: Callback function to receive bar updates
        """
        pass
    
    @abstractmethod
    def subscribe_option_chain(self, symbol: str, 
                              callback: Callable[[OptionChain], None]) -> None:
        """
        Subscribe to an option chain for an underlying symbol.
        
        Args:
            symbol: Underlying symbol
            callback: Callback function to receive option chain updates
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, asset: Asset) -> None:
        """
        Unsubscribe from updates for an asset.
        
        Args:
            asset: Asset to unsubscribe from
        """
        pass
    
    @abstractmethod
    def get_last_quote(self, asset: Asset) -> Quote:
        """
        Get the last quote for an asset.
        
        Args:
            asset: Asset to get the quote for
            
        Returns:
            Last quote for the asset
        """
        pass
    
    @abstractmethod
    def get_option_chain(self, symbol: str) -> OptionChain:
        """
        Get the complete option chain for an underlying symbol.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            Option chain for the underlying symbol
        """
        pass
    
    @abstractmethod
    def get_historical_bars(self, asset: Asset, period: BarPeriod, 
                           start_time: datetime, end_time: datetime) -> List[MarketDataBar]:
        """
        Get historical bars for an asset.
        
        Args:
            asset: Asset to get historical data for
            period: Bar period
            start_time: Start time of the historical data
            end_time: End time of the historical data
            
        Returns:
            List of historical bars
        """
        pass
    
    @abstractmethod
    def get_historical_quotes(self, asset: Asset, start_time: datetime, 
                             end_time: datetime) -> List[Quote]:
        """
        Get historical quotes for an asset.
        
        Args:
            asset: Asset to get historical data for
            start_time: Start time of the historical data
            end_time: End time of the historical data
            
        Returns:
            List of historical quotes
        """
        pass


class MarketDataException(Exception):
    """Exception raised for market data errors."""
    pass 