"""
Greeks Cap Module for Risk Management

This module implements live risk checks for options portfolio Greek exposures:
|Δ|≤$1M, |Γ|≤$250k, |Vega|≤$50k, |Θ|≤$25k/day

Provides functionality to check if a proposed trade would exceed these limits
and adjust quantities if needed to stay within risk tolerances.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GreeksLimits:
    """Risk limits for Greek exposures."""
    max_delta: float = 1000000.0  # $1M total delta exposure
    max_gamma: float = 250000.0   # $250k total gamma exposure
    max_vega: float = 50000.0     # $50k total vega exposure
    max_theta: float = 25000.0    # $25k total theta decay per day
    
    # Additional limits for specific sectors or symbols
    sector_delta_multipliers: Dict[str, float] = field(default_factory=dict)
    symbol_delta_multipliers: Dict[str, float] = field(default_factory=dict)


@dataclass
class GreeksExposure:
    """Current Greek exposures for the portfolio."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    
    # Exposures by sector
    sector_delta: Dict[str, float] = field(default_factory=dict)
    sector_gamma: Dict[str, float] = field(default_factory=dict)
    sector_vega: Dict[str, float] = field(default_factory=dict)
    sector_theta: Dict[str, float] = field(default_factory=dict)
    
    # Exposures by symbol
    symbol_delta: Dict[str, float] = field(default_factory=dict)
    symbol_gamma: Dict[str, float] = field(default_factory=dict)
    symbol_vega: Dict[str, float] = field(default_factory=dict)
    symbol_theta: Dict[str, float] = field(default_factory=dict)


class GreeksCap:
    """
    Greek exposure manager ensuring portfolio stays within risk limits.
    
    Monitors delta, gamma, vega, and theta exposures and provides
    checks against configurable limits.
    """
    
    def __init__(self, limits: Optional[GreeksLimits] = None):
        """
        Initialize Greeks cap manager.
        
        Args:
            limits: Risk limits (default to standard limits if None)
        """
        self.limits = limits or GreeksLimits()
        self.exposures = GreeksExposure()
        
        # Track all positions for accurate calculations
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Last update time
        self.last_update_time = datetime.now()
    
    def check_trade_limits(self, 
                          symbol: str,
                          contract_id: str,
                          quantity: int,
                          greeks: Dict[str, float],
                          price: float = 0.0,
                          sector: Optional[str] = None,
                          contract_multiplier: float = 100.0) -> Tuple[bool, str, int]:
        """
        Check if a trade would exceed Greek risk limits.
        
        Args:
            symbol: Trading symbol
            contract_id: Option contract ID
            quantity: Proposed quantity
            greeks: Option Greeks (delta, gamma, vega, theta)
            price: Option price
            sector: Stock sector
            contract_multiplier: Contract multiplier
            
        Returns:
            Tuple of (passes_limit, reason, adjusted_quantity)
        """
        # Calculate contract exposure
        delta = greeks.get('delta', 0.0) * quantity * contract_multiplier
        gamma = greeks.get('gamma', 0.0) * quantity * contract_multiplier
        vega = greeks.get('vega', 0.0) * quantity * contract_multiplier
        theta = greeks.get('theta', 0.0) * quantity * contract_multiplier
        
        # Calculate new total exposures
        new_delta = self.exposures.delta + delta
        new_gamma = self.exposures.gamma + gamma
        new_vega = self.exposures.vega + vega
        new_theta = self.exposures.theta + theta
        
        # Check against limits
        if abs(new_delta) > self.limits.max_delta:
            reason = f"Delta limit exceeded: {abs(new_delta):.0f} > {self.limits.max_delta:.0f}"
            adjusted_quantity = self._adjust_quantity_for_delta(
                symbol, quantity, greeks.get('delta', 0.0), contract_multiplier
            )
            return False, reason, adjusted_quantity
            
        if abs(new_gamma) > self.limits.max_gamma:
            reason = f"Gamma limit exceeded: {abs(new_gamma):.0f} > {self.limits.max_gamma:.0f}"
            adjusted_quantity = self._adjust_quantity_for_greek(
                quantity, self.limits.max_gamma, self.exposures.gamma, 
                greeks.get('gamma', 0.0), contract_multiplier
            )
            return False, reason, adjusted_quantity
            
        if abs(new_vega) > self.limits.max_vega:
            reason = f"Vega limit exceeded: {abs(new_vega):.0f} > {self.limits.max_vega:.0f}"
            adjusted_quantity = self._adjust_quantity_for_greek(
                quantity, self.limits.max_vega, self.exposures.vega, 
                greeks.get('vega', 0.0), contract_multiplier
            )
            return False, reason, adjusted_quantity
            
        if abs(new_theta) > self.limits.max_theta:
            reason = f"Theta limit exceeded: {abs(new_theta):.0f} > {self.limits.max_theta:.0f}"
            adjusted_quantity = self._adjust_quantity_for_greek(
                quantity, self.limits.max_theta, self.exposures.theta, 
                greeks.get('theta', 0.0), contract_multiplier
            )
            return False, reason, adjusted_quantity
        
        # Check sector-specific limits if applicable
        if sector and sector in self.limits.sector_delta_multipliers:
            sector_delta_limit = self.limits.max_delta * self.limits.sector_delta_multipliers[sector]
            
            # Calculate new sector delta
            current_sector_delta = self.exposures.sector_delta.get(sector, 0.0)
            new_sector_delta = current_sector_delta + delta
            
            if abs(new_sector_delta) > sector_delta_limit:
                reason = (
                    f"Sector delta limit exceeded for {sector}: "
                    f"{abs(new_sector_delta):.0f} > {sector_delta_limit:.0f}"
                )
                adjusted_quantity = self._adjust_quantity_for_greek(
                    quantity, sector_delta_limit, current_sector_delta, 
                    greeks.get('delta', 0.0), contract_multiplier
                )
                return False, reason, adjusted_quantity
        
        # All checks passed
        return True, "", quantity
    
    def _adjust_quantity_for_delta(self, 
                                 symbol: str,
                                 quantity: int,
                                 delta: float,
                                 contract_multiplier: float) -> int:
        """
        Adjust quantity to stay within delta limits.
        
        Args:
            symbol: Trading symbol
            quantity: Original quantity
            delta: Delta per contract
            contract_multiplier: Contract multiplier
            
        Returns:
            Adjusted quantity
        """
        # Special handling for symbol-specific limits
        if symbol in self.limits.symbol_delta_multipliers:
            delta_limit = self.limits.max_delta * self.limits.symbol_delta_multipliers[symbol]
        else:
            delta_limit = self.limits.max_delta
        
        return self._adjust_quantity_for_greek(
            quantity, delta_limit, self.exposures.delta, delta, contract_multiplier
        )
    
    def _adjust_quantity_for_greek(self, 
                                 quantity: int,
                                 limit: float,
                                 current_exposure: float,
                                 greek_per_contract: float,
                                 contract_multiplier: float) -> int:
        """
        Adjust quantity to stay within a Greek limit.
        
        Args:
            quantity: Original quantity
            limit: Greek limit
            current_exposure: Current exposure
            greek_per_contract: Greek per contract
            contract_multiplier: Contract multiplier
            
        Returns:
            Adjusted quantity
        """
        # If Greek is zero, can't adjust
        if greek_per_contract == 0.0:
            return 0
        
        # Calculate available capacity
        available_exposure = limit - abs(current_exposure)
        
        # If same sign as current exposure, need to ensure we don't exceed limit
        unit_exposure = greek_per_contract * contract_multiplier
        
        if available_exposure <= 0:
            # Already at limit
            return 0
        
        # Calculate max quantity
        max_quantity = int(available_exposure / abs(unit_exposure))
        
        # Return adjusted quantity
        return min(quantity, max_quantity)
    
    def update_exposures(self, 
                       symbol: str,
                       contract_id: str,
                       quantity: int,
                       greeks: Dict[str, float],
                       sector: Optional[str] = None,
                       contract_multiplier: float = 100.0) -> None:
        """
        Update portfolio exposures after a trade.
        
        Args:
            symbol: Trading symbol
            contract_id: Option contract ID
            quantity: Trade quantity
            greeks: Option Greeks (delta, gamma, vega, theta)
            sector: Stock sector
            contract_multiplier: Contract multiplier
        """
        # Calculate contract exposure
        delta = greeks.get('delta', 0.0) * quantity * contract_multiplier
        gamma = greeks.get('gamma', 0.0) * quantity * contract_multiplier
        vega = greeks.get('vega', 0.0) * quantity * contract_multiplier
        theta = greeks.get('theta', 0.0) * quantity * contract_multiplier
        
        # Update total exposures
        self.exposures.delta += delta
        self.exposures.gamma += gamma
        self.exposures.vega += vega
        self.exposures.theta += theta
        
        # Update sector exposures
        if sector:
            self.exposures.sector_delta[sector] = self.exposures.sector_delta.get(sector, 0.0) + delta
            self.exposures.sector_gamma[sector] = self.exposures.sector_gamma.get(sector, 0.0) + gamma
            self.exposures.sector_vega[sector] = self.exposures.sector_vega.get(sector, 0.0) + vega
            self.exposures.sector_theta[sector] = self.exposures.sector_theta.get(sector, 0.0) + theta
        
        # Update symbol exposures
        self.exposures.symbol_delta[symbol] = self.exposures.symbol_delta.get(symbol, 0.0) + delta
        self.exposures.symbol_gamma[symbol] = self.exposures.symbol_gamma.get(symbol, 0.0) + gamma
        self.exposures.symbol_vega[symbol] = self.exposures.symbol_vega.get(symbol, 0.0) + vega
        self.exposures.symbol_theta[symbol] = self.exposures.symbol_theta.get(symbol, 0.0) + theta
        
        # Store position
        self.positions[contract_id] = {
            'symbol': symbol,
            'quantity': quantity,
            'greeks': greeks.copy(),
            'sector': sector,
            'contract_multiplier': contract_multiplier,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
        
        # Log significant exposures
        if abs(self.exposures.delta) > self.limits.max_delta * 0.8:
            logger.warning(
                f"Delta exposure at {abs(self.exposures.delta) / self.limits.max_delta:.1%} of limit"
            )
        
        if abs(self.exposures.gamma) > self.limits.max_gamma * 0.8:
            logger.warning(
                f"Gamma exposure at {abs(self.exposures.gamma) / self.limits.max_gamma:.1%} of limit"
            )
        
        if abs(self.exposures.vega) > self.limits.max_vega * 0.8:
            logger.warning(
                f"Vega exposure at {abs(self.exposures.vega) / self.limits.max_vega:.1%} of limit"
            )
        
        if abs(self.exposures.theta) > self.limits.max_theta * 0.8:
            logger.warning(
                f"Theta exposure at {abs(self.exposures.theta) / self.limits.max_theta:.1%} of limit"
            )
    
    def remove_position(self, contract_id: str) -> bool:
        """
        Remove a position and update exposures.
        
        Args:
            contract_id: Option contract ID
            
        Returns:
            True if position was found and removed, False otherwise
        """
        if contract_id not in self.positions:
            logger.warning(f"Position not found for contract {contract_id}")
            return False
        
        # Get position details
        position = self.positions[contract_id]
        
        # Update total exposures (subtract)
        self.exposures.delta -= position['delta']
        self.exposures.gamma -= position['gamma']
        self.exposures.vega -= position['vega']
        self.exposures.theta -= position['theta']
        
        # Update sector exposures if applicable
        sector = position.get('sector')
        if sector:
            self.exposures.sector_delta[sector] -= position['delta']
            self.exposures.sector_gamma[sector] -= position['gamma']
            self.exposures.sector_vega[sector] -= position['vega']
            self.exposures.sector_theta[sector] -= position['theta']
            
            # Remove sector entry if zero
            if self.exposures.sector_delta[sector] == 0.0:
                del self.exposures.sector_delta[sector]
            if self.exposures.sector_gamma[sector] == 0.0:
                del self.exposures.sector_gamma[sector]
            if self.exposures.sector_vega[sector] == 0.0:
                del self.exposures.sector_vega[sector]
            if self.exposures.sector_theta[sector] == 0.0:
                del self.exposures.sector_theta[sector]
        
        # Update symbol exposures
        symbol = position['symbol']
        self.exposures.symbol_delta[symbol] -= position['delta']
        self.exposures.symbol_gamma[symbol] -= position['gamma']
        self.exposures.symbol_vega[symbol] -= position['vega']
        self.exposures.symbol_theta[symbol] -= position['theta']
        
        # Remove symbol entry if zero
        if self.exposures.symbol_delta[symbol] == 0.0:
            del self.exposures.symbol_delta[symbol]
        if self.exposures.symbol_gamma[symbol] == 0.0:
            del self.exposures.symbol_gamma[symbol]
        if self.exposures.symbol_vega[symbol] == 0.0:
            del self.exposures.symbol_vega[symbol]
        if self.exposures.symbol_theta[symbol] == 0.0:
            del self.exposures.symbol_theta[symbol]
        
        # Remove position
        del self.positions[contract_id]
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current Greek metrics and limits.
        
        Returns:
            Dictionary with metrics
        """
        return {
            "exposures": {
                "delta": self.exposures.delta,
                "gamma": self.exposures.gamma,
                "vega": self.exposures.vega,
                "theta": self.exposures.theta,
                "delta_pct": abs(self.exposures.delta) / self.limits.max_delta,
                "gamma_pct": abs(self.exposures.gamma) / self.limits.max_gamma,
                "vega_pct": abs(self.exposures.vega) / self.limits.max_vega,
                "theta_pct": abs(self.exposures.theta) / self.limits.max_theta
            },
            "limits": {
                "max_delta": self.limits.max_delta,
                "max_gamma": self.limits.max_gamma,
                "max_vega": self.limits.max_vega,
                "max_theta": self.limits.max_theta
            },
            "positions_count": len(self.positions),
            "sector_exposures": {
                "delta": self.exposures.sector_delta,
                "gamma": self.exposures.sector_gamma,
                "vega": self.exposures.sector_vega,
                "theta": self.exposures.sector_theta
            },
            "symbol_exposures": {
                "delta": self.exposures.symbol_delta,
                "gamma": self.exposures.symbol_gamma,
                "vega": self.exposures.symbol_vega,
                "theta": self.exposures.symbol_theta
            }
        }


# Global instance for convenience
_greeks_cap = None

def get_greeks_cap(limits: Optional[GreeksLimits] = None) -> GreeksCap:
    """
    Get the global Greeks cap instance.
    
    Args:
        limits: Risk limits (for initialization)
        
    Returns:
        Global GreeksCap instance
    """
    global _greeks_cap
    
    if _greeks_cap is None:
        _greeks_cap = GreeksCap(limits=limits)
    
    return _greeks_cap 