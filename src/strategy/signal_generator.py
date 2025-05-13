"""
Signal generator module for options trading.

Integrates different signal generation algorithms and provides a
unified interface for generating trading signals.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd

from .signal import Signal, SignalType, SignalGenerator, MispricingSignalGenerator, VolatilitySkewSignalGenerator
from .spreads import SpreadSignalGenerator, SpreadType
from ..data.market_data import OptionChain, OptionQuote, OptionRight, Asset, Option

# Configure logging
logger = logging.getLogger(__name__)


class SignalManager:
    """
    Manager for multiple signal generation algorithms.
    
    Orchestrates and combines signals from different algorithms
    to generate consolidated trading signals.
    """
    
    def __init__(self, risk_free_rate: float = 0.05, min_edge: float = 5.0,
                min_confidence: float = 0.7):
        """
        Initialize signal manager.
        
        Args:
            risk_free_rate: Risk-free interest rate used for pricing models
            min_edge: Minimum edge percentage required for signal generation
            min_confidence: Minimum confidence level required for signal generation
        """
        self.risk_free_rate = risk_free_rate
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        
        # Initialize signal generators
        self.mispricing_generator = MispricingSignalGenerator(
            min_edge_pct=min_edge,
            min_confidence=min_confidence
        )
        
        self.vol_skew_generator = VolatilitySkewSignalGenerator(
            min_skew_deviation=min_edge,
            min_confidence=min_confidence
        )
        
        self.spread_generator = SpreadSignalGenerator(
            min_profit_prob=min_confidence,
            min_risk_reward=1.5
        )
        
        # Signal history
        self.signal_history: List[Signal] = []
        
        # Signal filters
        self.symbol_filter: List[str] = []
        self.type_filter: List[SignalType] = []
        self.max_signals_per_run = 10
    
    def set_filters(self, symbols: Optional[List[str]] = None,
                   signal_types: Optional[List[SignalType]] = None,
                   max_signals: int = 10) -> None:
        """
        Configure signal filtering parameters.
        
        Args:
            symbols: List of symbols to filter by (None for all)
            signal_types: List of signal types to include (None for all)
            max_signals: Maximum number of signals to return per run
        """
        self.symbol_filter = symbols or []
        self.type_filter = signal_types or []
        self.max_signals_per_run = max_signals
    
    def generate_mispricing_signals(self, chain: OptionChain, 
                                  underlying_price: float) -> List[Signal]:
        """
        Generate mispricing signals.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            
        Returns:
            List of mispricing signals
        """
        try:
            return self.mispricing_generator.generate_signals(
                chain=chain,
                underlying_price=underlying_price,
                risk_free_rate=self.risk_free_rate
            )
        except Exception as e:
            logger.error(f"Error generating mispricing signals: {e}")
            return []
    
    def generate_vol_skew_signals(self, chain: OptionChain) -> List[Signal]:
        """
        Generate volatility skew signals.
        
        Args:
            chain: Option chain to analyze
            
        Returns:
            List of volatility skew signals
        """
        try:
            return self.vol_skew_generator.generate_signals(chain=chain)
        except Exception as e:
            logger.error(f"Error generating volatility skew signals: {e}")
            return []
    
    def generate_spread_signals(self, chain: OptionChain, 
                              underlying_price: float) -> List[Signal]:
        """
        Generate option spread signals.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            
        Returns:
            List of spread strategy signals
        """
        try:
            return self.spread_generator.generate_signals(
                chain=chain,
                underlying_price=underlying_price
            )
        except Exception as e:
            logger.error(f"Error generating spread signals: {e}")
            return []
    
    def generate_all_signals(self, chain: OptionChain, 
                           underlying_price: float) -> List[Signal]:
        """
        Generate all types of trading signals.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            
        Returns:
            Combined list of signals from all generators
        """
        signals = []
        
        # Run all signal generators
        mispricing_signals = self.generate_mispricing_signals(chain, underlying_price)
        vol_skew_signals = self.generate_vol_skew_signals(chain)
        spread_signals = self.generate_spread_signals(chain, underlying_price)
        
        # Log stats
        logger.info(f"Generated {len(mispricing_signals)} mispricing signals")
        logger.info(f"Generated {len(vol_skew_signals)} volatility skew signals")
        logger.info(f"Generated {len(spread_signals)} spread signals")
        
        # Combine all signals
        signals.extend(mispricing_signals)
        signals.extend(vol_skew_signals)
        signals.extend(spread_signals)
        
        # Apply filters
        signals = self._apply_filters(signals)
        
        # Add to history
        for signal in signals:
            signal.timestamp = datetime.now()
            self.signal_history.append(signal)
        
        return signals
    
    def _apply_filters(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply filters to signal list.
        
        Args:
            signals: List of signals to filter
            
        Returns:
            Filtered list of signals
        """
        # Apply symbol filter
        if self.symbol_filter:
            signals = [s for s in signals if s.symbol in self.symbol_filter]
        
        # Apply type filter
        if self.type_filter:
            signals = [s for s in signals if s.signal_type in self.type_filter]
        
        # Sort by expected edge
        signals.sort(key=lambda s: s.expected_edge, reverse=True)
        
        # Limit to max signals
        return signals[:self.max_signals_per_run]
    
    def get_signal_history(self, days: int = 30, 
                         symbol: Optional[str] = None,
                         signal_type: Optional[SignalType] = None) -> List[Signal]:
        """
        Get historical signals with optional filtering.
        
        Args:
            days: Number of days of history to return
            symbol: Filter by symbol (None for all)
            signal_type: Filter by signal type (None for all)
            
        Returns:
            Filtered list of historical signals
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter by time
        signals = [s for s in self.signal_history if s.timestamp >= cutoff_time]
        
        # Apply additional filters
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        return signals
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history = []


class CustomSignalGenerator(SignalGenerator):
    """
    Custom signal generator for specific trading strategies.
    
    Implements custom logic for generating signals based on
    specific market conditions or trading rules.
    """
    
    def __init__(self, name: str, signal_function: Callable):
        """
        Initialize custom signal generator.
        
        Args:
            name: Name of the signal generator
            signal_function: Function that generates signals
        """
        self.name = name
        self.signal_function = signal_function
    
    def generate_signals(self, *args, **kwargs) -> List[Signal]:
        """
        Generate trading signals using the custom function.
        
        Returns:
            List of trading signals
        """
        try:
            return self.signal_function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in custom signal generator '{self.name}': {e}")
            return []


class EarningsSignalGenerator(SignalGenerator):
    """
    Signal generator for earnings-related options trades.
    
    Identifies opportunities around earnings announcements when
    implied volatility typically increases.
    """
    
    def __init__(self, earnings_data: Dict[str, List[datetime]],
                days_before: int = 5, min_iv_percentile: float = 0.7):
        """
        Initialize earnings signal generator.
        
        Args:
            earnings_data: Dictionary mapping symbols to earnings dates
            days_before: Days before earnings to start looking for signals
            min_iv_percentile: Minimum IV percentile for signal generation
        """
        self.earnings_data = earnings_data
        self.days_before = days_before
        self.min_iv_percentile = min_iv_percentile
    
    def generate_signals(self, chain: OptionChain, underlying_price: float) -> List[Signal]:
        """
        Generate earnings-related trading signals.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            
        Returns:
            List of earnings-related signals
        """
        signals = []
        symbol = chain.underlying_symbol
        
        # Check if we have earnings data for this symbol
        if symbol not in self.earnings_data:
            return []
        
        # Get next earnings date
        earnings_dates = self.earnings_data[symbol]
        upcoming_earnings = [d for d in earnings_dates if d > datetime.now()]
        
        if not upcoming_earnings:
            return []
        
        next_earnings = min(upcoming_earnings)
        days_to_earnings = (next_earnings - datetime.now()).days
        
        # Check if within our window
        if days_to_earnings > self.days_before:
            return []
        
        # Find an ATM straddle around earnings
        # Get options chain expirations sorted by date
        expirations = sorted(set(exp for exp, _, _ in chain.options.keys()))
        
        # Find the first expiration after earnings
        post_earnings_exp = None
        for exp in expirations:
            if exp > next_earnings:
                post_earnings_exp = exp
                break
        
        if not post_earnings_exp:
            return []
        
        # Get options for this expiration
        options = chain.get_options_by_expiration(post_earnings_exp)
        
        # Find closest strike to current price
        closest_strike = min(set(strike for strike, _ in options.keys()), 
                           key=lambda x: abs(x - underlying_price))
        
        # Get call and put at this strike
        call = None
        put = None
        
        for (strike, right), quote in options.items():
            if strike == closest_strike:
                if right == OptionRight.CALL:
                    call = quote
                else:
                    put = quote
        
        if not call or not put:
            return []
        
        # Check IV percentile
        # In a real system, would have historical IV for comparison
        # Using a simplified approach here
        iv_percentile = 0.8  # Placeholder value
        
        if iv_percentile >= self.min_iv_percentile:
            # IV is high, look to sell volatility
            signal = Signal(
                timestamp=datetime.now(),
                signal_type=SignalType.EARNINGS,
                symbol=symbol,
                direction=-1,  # Short (selling volatility)
                expiration=post_earnings_exp,
                strike=closest_strike,
                implied_vol=(call.implied_volatility + put.implied_volatility) / 2,
                expected_edge=10.0,  # Simplified edge calculation
                confidence=0.7,
                legs=[
                    {
                        "expiration": post_earnings_exp,
                        "strike": closest_strike,
                        "right": OptionRight.CALL,
                        "quantity": -1,
                        "price": call.ask_price
                    },
                    {
                        "expiration": post_earnings_exp,
                        "strike": closest_strike,
                        "right": OptionRight.PUT,
                        "quantity": -1,
                        "price": put.ask_price
                    }
                ]
            )
            
            signals.append(signal)
        
        return signals
    

class SignalFactory:
    """
    Factory for creating and registering signal generators.
    
    Manages different signal generation strategies and provides
    a unified interface for using them.
    """
    
    _generators: Dict[str, SignalGenerator] = {}
    
    @classmethod
    def register_generator(cls, name: str, generator: SignalGenerator) -> None:
        """
        Register a signal generator.
        
        Args:
            name: Name of the generator
            generator: Signal generator instance
        """
        cls._generators[name] = generator
        logger.info(f"Registered signal generator: {name}")
    
    @classmethod
    def get_generator(cls, name: str) -> SignalGenerator:
        """
        Get a registered signal generator.
        
        Args:
            name: Name of the generator
            
        Returns:
            Signal generator instance
            
        Raises:
            ValueError: If generator not found
        """
        if name not in cls._generators:
            raise ValueError(f"Signal generator not found: {name}")
            
        return cls._generators[name]
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """
        List all registered generators.
        
        Returns:
            List of generator names
        """
        return list(cls._generators.keys())
    
    @classmethod
    def create_generator(cls, name: str, generator_type: str, **kwargs) -> SignalGenerator:
        """
        Create and register a new generator.
        
        Args:
            name: Name for the new generator
            generator_type: Type of generator to create
            **kwargs: Arguments for the generator
            
        Returns:
            Created signal generator
            
        Raises:
            ValueError: If generator type not supported
        """
        if generator_type == "mispricing":
            generator = MispricingSignalGenerator(**kwargs)
        elif generator_type == "vol_skew":
            generator = VolatilitySkewSignalGenerator(**kwargs)
        elif generator_type == "spread":
            generator = SpreadSignalGenerator(**kwargs)
        elif generator_type == "earnings":
            generator = EarningsSignalGenerator(**kwargs)
        elif generator_type == "custom":
            if "signal_function" not in kwargs:
                raise ValueError("Custom generator requires 'signal_function'")
            generator = CustomSignalGenerator(name, kwargs["signal_function"])
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        cls.register_generator(name, generator)
        return generator


# Register default generators
SignalFactory.register_generator("mispricing", MispricingSignalGenerator())
SignalFactory.register_generator("vol_skew", VolatilitySkewSignalGenerator())
SignalFactory.register_generator("spread", SpreadSignalGenerator()) 