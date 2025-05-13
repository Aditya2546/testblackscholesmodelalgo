"""
Signal generation module for identifying trading opportunities.

Provides algorithms for detecting mispricing and volatility edge in options.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set

import numpy as np
import pandas as pd

from ..data.market_data import OptionChain, OptionQuote, OptionRight, Asset, Option
from ..pricing.model_interface import OptionModel, ModelFactory


class SignalType(Enum):
    """Types of trading signals."""
    MISPRICING = "mispricing"  # Model vs. market price divergence
    VOL_SKEW = "vol_skew"      # Volatility skew opportunity
    TERM_STRUCTURE = "term_structure"  # Term structure opportunity
    EARNINGS = "earnings"      # Earnings-related opportunity
    SPREAD = "spread"          # Multi-leg spread opportunity
    LIQUIDITY = "liquidity"    # Liquidity-based opportunity


@dataclass
class Signal:
    """Trading signal with opportunity details."""
    timestamp: datetime
    signal_type: SignalType
    symbol: str
    direction: int  # 1 for long, -1 for short
    expiration: Optional[datetime] = None
    strike: Optional[float] = None
    right: Optional[OptionRight] = None
    market_price: Optional[float] = None
    model_price: Optional[float] = None
    implied_vol: Optional[float] = None
    model_vol: Optional[float] = None
    expected_edge: float = 0.0  # Expected edge in percentage
    expected_pnl: float = 0.0   # Expected P&L in dollars
    confidence: float = 0.0     # Signal confidence (0.0-1.0)
    legs: List[Dict[str, Any]] = field(default_factory=list)  # For multi-leg strategies
    
    def __str__(self) -> str:
        """String representation of the signal."""
        if self.right:
            option_type = "CALL" if self.right == OptionRight.CALL else "PUT"
            return (f"{self.signal_type.value.upper()} {self.direction:+d} "
                   f"{self.symbol} {self.strike} {option_type} {self.expiration.strftime('%Y-%m-%d')} "
                   f"Edge: {self.expected_edge:.2f}% Conf: {self.confidence:.2f}")
        else:
            return (f"{self.signal_type.value.upper()} {self.direction:+d} "
                   f"{self.symbol} Edge: {self.expected_edge:.2f}% Conf: {self.confidence:.2f}")


class SignalGenerator(ABC):
    """Base class for signal generation algorithms."""
    
    @abstractmethod
    def generate_signals(self, *args, **kwargs) -> List[Signal]:
        """
        Generate trading signals based on market conditions.
        
        Returns:
            List of trading signals
        """
        pass


class MispricingSignalGenerator(SignalGenerator):
    """
    Signal generator that identifies options mispricing.
    
    Compares market prices to model prices and generates signals
    when significant discrepancies are detected.
    """
    
    def __init__(self, model_name: str = "black_scholes", min_edge_pct: float = 5.0, 
                min_confidence: float = 0.7):
        """
        Initialize MispricingSignalGenerator.
        
        Args:
            model_name: Name of the pricing model to use
            min_edge_pct: Minimum edge percentage to generate a signal
            min_confidence: Minimum confidence level to generate a signal
        """
        self.model = ModelFactory.create_model(model_name)
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence
    
    def _calculate_model_prices(self, chain: OptionChain, underlying_price: float, 
                               risk_free_rate: float) -> Dict[Tuple[datetime, float, OptionRight], float]:
        """
        Calculate model prices for all options in a chain.
        
        Args:
            chain: Option chain to price
            underlying_price: Price of the underlying asset
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Dictionary mapping option key to model price
        """
        model_prices = {}
        
        for (expiration, strike, right), quote in chain.options.items():
            # Calculate time to expiration in years
            now = datetime.now()
            dte = (expiration - now).total_seconds() / (365.25 * 24 * 60 * 60)
            
            # Skip options with no market data
            if quote.bid_price <= 0 or quote.ask_price <= 0:
                continue
                
            # Calculate mid price
            market_mid = quote.midpoint
            
            # Calculate model price using implied volatility if available
            if quote.implied_volatility > 0:
                model_price = self.model.price(
                    S=underlying_price,
                    K=strike,
                    T=max(dte, 1e-8),  # Avoid numerical issues with very short-dated options
                    r=risk_free_rate,
                    sigma=quote.implied_volatility,
                    option_type="call" if right == OptionRight.CALL else "put"
                )
                model_prices[(expiration, strike, right)] = model_price
            
        return model_prices
    
    def generate_signals(self, chain: OptionChain, underlying_price: float, 
                        risk_free_rate: float = 0.05) -> List[Signal]:
        """
        Generate mispricing signals from an option chain.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Price of the underlying asset
            risk_free_rate: Risk-free interest rate
            
        Returns:
            List of trading signals
        """
        signals = []
        now = datetime.now()
        
        # Calculate model prices
        model_prices = self._calculate_model_prices(chain, underlying_price, risk_free_rate)
        
        # Analyze each option for mispricing
        for (expiration, strike, right), quote in chain.options.items():
            # Skip options with no model price
            if (expiration, strike, right) not in model_prices:
                continue
            
            # Get model price and market mid price
            model_price = model_prices[(expiration, strike, right)]
            market_mid = quote.midpoint
            
            # Calculate edge
            edge_pct = abs(model_price - market_mid) / market_mid * 100
            
            # Determine if model price is higher or lower than market
            direction = 1 if model_price > market_mid else -1  # 1 for long, -1 for short
            
            # Calculate time to expiration in days
            dte = (expiration - now).days
            
            # Calculate confidence based on liquidity and time to expiration
            # Higher confidence for liquid options and longer-dated options
            liquidity_factor = min(1.0, (quote.bid_size + quote.ask_size) / 100)
            time_factor = min(1.0, dte / 30)  # Higher confidence for options with at least 30 days
            confidence = 0.7 * liquidity_factor + 0.3 * time_factor
            
            # If edge and confidence meet thresholds, generate signal
            if edge_pct >= self.min_edge_pct and confidence >= self.min_confidence:
                # Calculate expected P&L
                contract_size = 100  # Standard options contract size
                expected_pnl = direction * (model_price - market_mid) * contract_size
                
                signal = Signal(
                    timestamp=now,
                    signal_type=SignalType.MISPRICING,
                    symbol=chain.underlying_symbol,
                    direction=direction,
                    expiration=expiration,
                    strike=strike,
                    right=right,
                    market_price=market_mid,
                    model_price=model_price,
                    implied_vol=quote.implied_volatility,
                    expected_edge=edge_pct,
                    expected_pnl=expected_pnl,
                    confidence=confidence
                )
                
                signals.append(signal)
        
        # Sort signals by edge descending
        signals.sort(key=lambda s: s.expected_edge, reverse=True)
        
        return signals


class VolatilitySkewSignalGenerator(SignalGenerator):
    """
    Signal generator that identifies volatility skew opportunities.
    
    Analyzes the volatility skew (IV vs. strike) to identify
    options with relative value opportunities.
    """
    
    def __init__(self, min_skew_deviation: float = 2.0, min_confidence: float = 0.7):
        """
        Initialize VolatilitySkewSignalGenerator.
        
        Args:
            min_skew_deviation: Minimum deviation from normal skew to generate a signal
            min_confidence: Minimum confidence level to generate a signal
        """
        self.min_skew_deviation = min_skew_deviation
        self.min_confidence = min_confidence
    
    def _calculate_skew_curve(self, chain: OptionChain, expiration: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the volatility skew curve for a specific expiration.
        
        Args:
            chain: Option chain to analyze
            expiration: Expiration date to analyze
            
        Returns:
            Tuple of (normalized strikes, implied volatilities)
        """
        # Get options for this expiration
        options = chain.get_options_by_expiration(expiration)
        
        # Extract strike and IV data
        strikes = []
        ivs = []
        
        for (strike, right), quote in options.items():
            # Only use puts for downside skew and calls for upside skew
            if (right == OptionRight.PUT and strike < 100) or (right == OptionRight.CALL and strike >= 100):
                # Skip options with no IV
                if quote.implied_volatility <= 0:
                    continue
                
                strikes.append(strike)
                ivs.append(quote.implied_volatility)
        
        if not strikes:
            return np.array([]), np.array([])
        
        # Sort by strike
        idx = np.argsort(strikes)
        strikes = np.array(strikes)[idx]
        ivs = np.array(ivs)[idx]
        
        # Normalize strikes to ATM
        atm_idx = np.argmin(np.abs(strikes - 100))
        atm_strike = strikes[atm_idx]
        norm_strikes = strikes / atm_strike
        
        return norm_strikes, ivs
    
    def _fit_skew_model(self, norm_strikes: np.ndarray, ivs: np.ndarray) -> Callable:
        """
        Fit a model to the volatility skew.
        
        Args:
            norm_strikes: Normalized strikes
            ivs: Implied volatilities
            
        Returns:
            Function that predicts IV given a normalized strike
        """
        # Use quadratic polynomial fit
        coeffs = np.polyfit(norm_strikes, ivs, 2)
        
        def skew_model(x):
            return coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        
        return skew_model
    
    def generate_signals(self, chain: OptionChain) -> List[Signal]:
        """
        Generate volatility skew signals from an option chain.
        
        Args:
            chain: Option chain to analyze
            
        Returns:
            List of trading signals
        """
        signals = []
        now = datetime.now()
        
        # Get unique expirations
        expirations = set()
        for (expiration, _, _) in chain.options:
            expirations.add(expiration)
        
        # Analyze each expiration
        for expiration in expirations:
            # Calculate time to expiration in days
            dte = (expiration - now).days
            
            # Skip very short-dated options
            if dte < 5:
                continue
            
            # Calculate skew curve
            norm_strikes, ivs = self._calculate_skew_curve(chain, expiration)
            
            if len(norm_strikes) < 5:  # Need at least 5 points for reliable fit
                continue
            
            # Fit skew model
            skew_model = self._fit_skew_model(norm_strikes, ivs)
            
            # Check each option against the model
            options = chain.get_options_by_expiration(expiration)
            for (strike, right), quote in options.items():
                # Skip options with no IV
                if quote.implied_volatility <= 0:
                    continue
                
                # Normalize strike
                atm_strike = 100  # Placeholder for ATM strike
                norm_strike = strike / atm_strike
                
                # Calculate model IV
                model_iv = skew_model(norm_strike)
                
                # Calculate deviation
                iv_deviation = (quote.implied_volatility - model_iv) / model_iv * 100
                
                # Determine direction
                direction = -1 if iv_deviation > 0 else 1  # Sell overpriced IV, buy underpriced IV
                
                # Calculate confidence
                liquidity_factor = min(1.0, (quote.bid_size + quote.ask_size) / 100)
                time_factor = min(1.0, dte / 30)
                confidence = 0.6 * liquidity_factor + 0.4 * time_factor
                
                # If deviation and confidence meet thresholds, generate signal
                if abs(iv_deviation) >= self.min_skew_deviation and confidence >= self.min_confidence:
                    signal = Signal(
                        timestamp=now,
                        signal_type=SignalType.VOL_SKEW,
                        symbol=chain.underlying_symbol,
                        direction=direction,
                        expiration=expiration,
                        strike=strike,
                        right=right,
                        market_price=quote.midpoint,
                        implied_vol=quote.implied_volatility,
                        model_vol=model_iv,
                        expected_edge=abs(iv_deviation),
                        confidence=confidence
                    )
                    
                    signals.append(signal)
        
        # Sort signals by edge descending
        signals.sort(key=lambda s: s.expected_edge, reverse=True)
        
        return signals 