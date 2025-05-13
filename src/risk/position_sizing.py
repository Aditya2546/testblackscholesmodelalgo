"""
Position sizing module using Kelly criterion and risk limits.

Calculates optimal position sizes based on edge, probability,
and portfolio risk constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Set, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..strategy.signal import Signal


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    signal: Signal
    quantity: int
    notional_value: float
    max_loss: float
    kelly_fraction: float
    reason: str = ""


@dataclass
class RiskLimits:
    """Portfolio risk limits for various metrics."""
    max_position_size_pct: float = 0.05  # Max position size as percent of portfolio
    max_total_delta: float = 1000000.0   # Max delta in dollars
    max_total_gamma: float = 250000.0    # Max gamma in dollars
    max_total_vega: float = 50000.0      # Max vega in dollars
    max_total_theta: float = 50000.0     # Max theta in dollars
    max_sector_exposure_pct: float = 0.2 # Max exposure to any sector
    max_single_name_exposure_pct: float = 0.1  # Max exposure to any single name
    max_short_delta_pct: float = 0.5     # Max short delta as percent of portfolio
    max_long_delta_pct: float = 0.5      # Max long delta as percent of portfolio


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    
    Defines the interface for position sizers that determine
    optimal trade size based on various factors.
    """
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal) -> PositionSizeResult:
        """
        Calculate optimal position size for a given signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size result
        """
        pass


class KellyPositionSizer(PositionSizer):
    """
    Position sizer using Kelly criterion with risk limits.
    
    Implements fractional Kelly sizing that respects portfolio-level
    risk limits for optimal capital allocation.
    """
    
    def __init__(self, portfolio_value: float, risk_limits: RiskLimits = None, 
                kelly_fraction: float = 0.25):
        """
        Initialize position sizer.
        
        Args:
            portfolio_value: Total portfolio value in dollars
            risk_limits: Risk limits for the portfolio
            kelly_fraction: Fraction of full Kelly criterion to use (0.0-1.0)
        """
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits or RiskLimits()
        self.kelly_fraction = kelly_fraction
        
        # Current portfolio risk usage
        self.current_delta = 0.0
        self.current_gamma = 0.0
        self.current_vega = 0.0
        self.current_theta = 0.0
        self.current_exposures: Dict[str, float] = {}
        
    def _calculate_kelly_size(self, signal: Signal) -> Tuple[float, float]:
        """
        Calculate optimal Kelly position size.
        
        Args:
            signal: Trading signal
            
        Returns:
            Tuple of (kelly fraction, max loss per contract)
        """
        # Extract signal properties
        edge = signal.expected_edge / 100.0  # Convert to decimal
        win_probability = signal.confidence
        
        if len(signal.legs) > 0:
            # Multi-leg strategy
            # Use the spread max loss per contract
            max_loss_per_contract = 0.0
            for leg in signal.legs:
                leg_price = leg.get("price", 0)
                leg_quantity = leg.get("quantity", 0)
                max_loss_per_contract += leg_price * leg_quantity * 100
            
            # Ensure max loss is positive
            max_loss_per_contract = abs(max_loss_per_contract)
        else:
            # Single option
            if signal.market_price:
                max_loss_per_contract = signal.market_price * 100
            else:
                # Default to a conservative value if no market price
                max_loss_per_contract = 500.0
        
        # Calculate Kelly fraction (f*)
        if win_probability <= 0 or win_probability >= 1:
            return 0.0, max_loss_per_contract
        
        # Kelly formula: f* = (p(b+1) - 1) / b where p is win probability and b is win/loss ratio
        win_amount = edge * max_loss_per_contract
        loss_amount = max_loss_per_contract
        
        # Win/loss ratio
        if loss_amount == 0:
            return 0.0, max_loss_per_contract
            
        b = win_amount / loss_amount
        
        # Calculate Kelly fraction
        kelly = (win_probability * (b + 1) - 1) / b
        
        # Apply Kelly fraction
        kelly *= self.kelly_fraction
        
        # Ensure Kelly fraction is between 0 and 1
        kelly = max(0.0, min(1.0, kelly))
        
        return kelly, max_loss_per_contract
    
    def _check_greek_limits(self, signal: Signal, quantity: int) -> Tuple[bool, str]:
        """
        Check if a position would violate Greek risk limits.
        
        Args:
            signal: Trading signal
            quantity: Proposed position quantity
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # If no Greeks provided, assume within limits
        if not hasattr(signal, 'delta') or not signal.delta:
            return True, ""
        
        # For spread strategies, sum up Greeks across legs
        delta = 0.0
        gamma = 0.0
        vega = 0.0
        theta = 0.0
        
        if len(signal.legs) > 0:
            # Multi-leg strategy
            for leg in signal.legs:
                leg_quantity = leg.get("quantity", 0)
                leg_greeks = leg.get("greeks", {})
                
                delta += leg_greeks.get("delta", 0) * leg_quantity * 100
                gamma += leg_greeks.get("gamma", 0) * leg_quantity * 100
                vega += leg_greeks.get("vega", 0) * leg_quantity * 100
                theta += leg_greeks.get("theta", 0) * leg_quantity * 100
        else:
            # Single option
            delta = signal.delta * quantity * 100
            gamma = getattr(signal, 'gamma', 0) * quantity * 100
            vega = getattr(signal, 'vega', 0) * quantity * 100
            theta = getattr(signal, 'theta', 0) * quantity * 100
        
        # Check if adding this position would exceed Greek limits
        new_delta = self.current_delta + delta
        new_gamma = self.current_gamma + gamma
        new_vega = self.current_vega + vega
        new_theta = self.current_theta + theta
        
        if abs(new_delta) > self.risk_limits.max_total_delta:
            return False, f"Delta limit exceeded: {new_delta:.2f} > {self.risk_limits.max_total_delta:.2f}"
            
        if abs(new_gamma) > self.risk_limits.max_total_gamma:
            return False, f"Gamma limit exceeded: {new_gamma:.2f} > {self.risk_limits.max_total_gamma:.2f}"
            
        if abs(new_vega) > self.risk_limits.max_total_vega:
            return False, f"Vega limit exceeded: {new_vega:.2f} > {self.risk_limits.max_total_vega:.2f}"
            
        if abs(new_theta) > self.risk_limits.max_total_theta:
            return False, f"Theta limit exceeded: {new_theta:.2f} > {self.risk_limits.max_total_theta:.2f}"
        
        return True, ""
    
    def _check_exposure_limits(self, signal: Signal, notional_value: float) -> Tuple[bool, str]:
        """
        Check if a position would violate exposure limits.
        
        Args:
            signal: Trading signal
            notional_value: Notional value of the position
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        symbol = signal.symbol
        
        # Check single name exposure limit
        current_exposure = self.current_exposures.get(symbol, 0.0)
        new_exposure = current_exposure + notional_value
        max_single_name = self.portfolio_value * self.risk_limits.max_single_name_exposure_pct
        
        if new_exposure > max_single_name:
            return False, f"Single name exposure limit exceeded: {new_exposure:.2f} > {max_single_name:.2f}"
        
        # Check position size limit
        max_position_size = self.portfolio_value * self.risk_limits.max_position_size_pct
        
        if notional_value > max_position_size:
            return False, f"Position size limit exceeded: {notional_value:.2f} > {max_position_size:.2f}"
        
        return True, ""
    
    def calculate_position_size(self, signal: Signal) -> PositionSizeResult:
        """
        Calculate optimal position size based on Kelly criterion and risk limits.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size result
        """
        # Calculate Kelly fraction and max loss
        kelly_fraction, max_loss_per_contract = self._calculate_kelly_size(signal)
        
        # Calculate optimal position size using Kelly
        kelly_amount = kelly_fraction * self.portfolio_value
        
        # Calculate maximum number of contracts based on Kelly
        max_contracts = int(kelly_amount / max_loss_per_contract)
        
        # Ensure at least 1 contract if Kelly suggests a position
        if max_contracts == 0 and kelly_fraction > 0:
            max_contracts = 1
        
        # Calculate notional value
        if len(signal.legs) > 0:
            # For multi-leg strategies, use max loss as a proxy for notional
            notional_value = max_contracts * max_loss_per_contract
        else:
            # For single options, use option price * 100 * number of contracts
            price = signal.market_price or max_loss_per_contract / 100
            notional_value = max_contracts * price * 100
        
        # Check Greek limits
        greek_allowed, greek_reason = self._check_greek_limits(signal, max_contracts)
        
        if not greek_allowed:
            # If Greek limits are violated, reduce position size
            # Simple approach: cut position size in half until constraints are satisfied
            reduced_contracts = max_contracts
            while reduced_contracts > 0:
                reduced_contracts = reduced_contracts // 2
                greek_allowed, greek_reason = self._check_greek_limits(signal, reduced_contracts)
                
                if greek_allowed:
                    max_contracts = reduced_contracts
                    notional_value = reduced_contracts * max_loss_per_contract
                    break
            
            if not greek_allowed:
                # If even a single contract violates limits, reject the trade
                return PositionSizeResult(
                    signal=signal,
                    quantity=0,
                    notional_value=0.0,
                    max_loss=0.0,
                    kelly_fraction=kelly_fraction,
                    reason=greek_reason
                )
        
        # Check exposure limits
        exposure_allowed, exposure_reason = self._check_exposure_limits(signal, notional_value)
        
        if not exposure_allowed:
            # If exposure limits are violated, reduce position size
            # Simple approach: cut position size in half until constraints are satisfied
            reduced_contracts = max_contracts
            while reduced_contracts > 0:
                reduced_contracts = reduced_contracts // 2
                
                if len(signal.legs) > 0:
                    reduced_notional = reduced_contracts * max_loss_per_contract
                else:
                    price = signal.market_price or max_loss_per_contract / 100
                    reduced_notional = reduced_contracts * price * 100
                
                exposure_allowed, exposure_reason = self._check_exposure_limits(signal, reduced_notional)
                
                if exposure_allowed:
                    max_contracts = reduced_contracts
                    notional_value = reduced_notional
                    break
            
            if not exposure_allowed:
                # If even a single contract violates limits, reject the trade
                return PositionSizeResult(
                    signal=signal,
                    quantity=0,
                    notional_value=0.0,
                    max_loss=0.0,
                    kelly_fraction=kelly_fraction,
                    reason=exposure_reason
                )
        
        # Calculate maximum loss for the position
        max_loss = max_contracts * max_loss_per_contract
        
        return PositionSizeResult(
            signal=signal,
            quantity=max_contracts,
            notional_value=notional_value,
            max_loss=max_loss,
            kelly_fraction=kelly_fraction,
            reason="Position sized within risk limits"
        )
    
    def update_portfolio_risk(self, signal: Signal, quantity: int) -> None:
        """
        Update portfolio risk metrics after adding a position.
        
        Args:
            signal: Trading signal
            quantity: Position quantity
        """
        symbol = signal.symbol
        
        # Update exposure
        if len(signal.legs) > 0:
            # For multi-leg strategies, use max loss as a proxy for notional
            _, max_loss_per_contract = self._calculate_kelly_size(signal)
            notional_value = quantity * max_loss_per_contract
        else:
            # For single options, use option price * 100 * number of contracts
            price = signal.market_price or 0.0
            notional_value = quantity * price * 100
        
        self.current_exposures[symbol] = self.current_exposures.get(symbol, 0.0) + notional_value
        
        # Update Greeks
        delta = 0.0
        gamma = 0.0
        vega = 0.0
        theta = 0.0
        
        if len(signal.legs) > 0:
            # Multi-leg strategy
            for leg in signal.legs:
                leg_quantity = leg.get("quantity", 0)
                leg_greeks = leg.get("greeks", {})
                
                delta += leg_greeks.get("delta", 0) * leg_quantity * 100
                gamma += leg_greeks.get("gamma", 0) * leg_quantity * 100
                vega += leg_greeks.get("vega", 0) * leg_quantity * 100
                theta += leg_greeks.get("theta", 0) * leg_quantity * 100
        else:
            # Single option
            delta = getattr(signal, 'delta', 0) * quantity * 100
            gamma = getattr(signal, 'gamma', 0) * quantity * 100
            vega = getattr(signal, 'vega', 0) * quantity * 100
            theta = getattr(signal, 'theta', 0) * quantity * 100
        
        # Scale by quantity
        delta *= quantity
        gamma *= quantity
        vega *= quantity
        theta *= quantity
        
        # Update portfolio Greeks
        self.current_delta += delta
        self.current_gamma += gamma
        self.current_vega += vega
        self.current_theta += theta


class StressTestPositionSizer(KellyPositionSizer):
    """
    Position sizer that incorporates stress testing and CVaR constraints.
    
    Extends KellyPositionSizer with Monte Carlo simulation to ensure
    positions meet risk constraints under stress scenarios.
    """
    
    def __init__(self, portfolio_value: float, risk_limits: RiskLimits = None,
                kelly_fraction: float = 0.25, max_cvar: float = 0.05,
                confidence_level: float = 0.99, num_scenarios: int = 10000):
        """
        Initialize stress test position sizer.
        
        Args:
            portfolio_value: Total portfolio value in dollars
            risk_limits: Risk limits for the portfolio
            kelly_fraction: Fraction of full Kelly criterion to use (0.0-1.0)
            max_cvar: Maximum Conditional Value at Risk (CVaR) as fraction of portfolio
            confidence_level: Confidence level for CVaR calculation
            num_scenarios: Number of Monte Carlo scenarios to simulate
        """
        super().__init__(portfolio_value, risk_limits, kelly_fraction)
        self.max_cvar = max_cvar
        self.confidence_level = confidence_level
        self.num_scenarios = num_scenarios
    
    def _simulate_scenarios(self, signal: Signal, quantity: int) -> np.ndarray:
        """
        Run Monte Carlo simulation for a potential position.
        
        Args:
            signal: Trading signal
            quantity: Position quantity
            
        Returns:
            Array of P&L outcomes across scenarios
        """
        # Extract parameters for simulation
        if len(signal.legs) > 0:
            # Multi-leg simulation is complex - simplified approach
            # In a real system, would model correlations between legs
            
            # Generate random price scenarios
            spot_price = 100.0  # Normalized spot price
            vol = 0.3  # Typical implied volatility
            
            # Time to expiration in years
            if signal.expiration:
                days_to_expiry = max(1, (signal.expiration - datetime.now()).days)
                t = days_to_expiry / 365.0
            else:
                t = 0.1  # Default 1 month
            
            # Generate log-normal price distribution
            np.random.seed(42)  # For reproducibility
            scenarios = spot_price * np.exp(
                (0 - 0.5 * vol**2) * t + vol * np.sqrt(t) * np.random.normal(0, 1, self.num_scenarios)
            )
            
            # Calculate P&L for each scenario
            pnl = np.zeros(self.num_scenarios)
            
            # Very simplified P&L calculation
            # In reality, would price each leg properly at each scenario price
            for i, scenario_price in enumerate(scenarios):
                scenario_pnl = 0.0
                
                # Calculate payoff at expiration (simplified)
                for leg in signal.legs:
                    leg_strike = leg.get("strike", 0)
                    leg_right = leg.get("right", OptionRight.CALL)
                    leg_quantity = leg.get("quantity", 0)
                    leg_price = leg.get("price", 0)
                    
                    # Simplified payoff calculation
                    if leg_right == OptionRight.CALL:
                        payoff = max(0, scenario_price - leg_strike)
                    else:
                        payoff = max(0, leg_strike - scenario_price)
                    
                    # P&L = payoff - premium
                    leg_pnl = (payoff - leg_price) * leg_quantity * 100
                    scenario_pnl += leg_pnl
                
                pnl[i] = scenario_pnl * quantity
            
            return pnl
        else:
            # Single option simulation
            # Generate random price scenarios
            spot_price = 100.0  # Normalized spot price
            vol = getattr(signal, 'implied_vol', 0.3)
            
            # Time to expiration in years
            if signal.expiration:
                days_to_expiry = max(1, (signal.expiration - datetime.now()).days)
                t = days_to_expiry / 365.0
            else:
                t = 0.1  # Default 1 month
            
            # Generate log-normal price distribution
            np.random.seed(42)  # For reproducibility
            scenarios = spot_price * np.exp(
                (0 - 0.5 * vol**2) * t + vol * np.sqrt(t) * np.random.normal(0, 1, self.num_scenarios)
            )
            
            # Calculate P&L for each scenario
            pnl = np.zeros(self.num_scenarios)
            
            for i, scenario_price in enumerate(scenarios):
                # Simplified payoff calculation
                if signal.right == OptionRight.CALL:
                    payoff = max(0, scenario_price - signal.strike)
                else:
                    payoff = max(0, signal.strike - scenario_price)
                
                # P&L = payoff - premium
                option_pnl = (payoff - signal.market_price) * 100
                pnl[i] = option_pnl * quantity
            
            return pnl
    
    def _calculate_cvar(self, pnl: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        Args:
            pnl: Array of P&L outcomes
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Sort P&L outcomes ascending
        sorted_pnl = np.sort(pnl)
        
        # Calculate index for VaR
        var_index = int(self.num_scenarios * (1 - self.confidence_level))
        
        # Get VaR
        var = -sorted_pnl[var_index]
        
        # Calculate CVaR (expected shortfall)
        # Average of the worst (1-confidence_level)% of outcomes
        cvar = -np.mean(sorted_pnl[:var_index + 1])
        
        return var, cvar
    
    def calculate_position_size(self, signal: Signal) -> PositionSizeResult:
        """
        Calculate position size with stress testing.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size result
        """
        # First calculate position size based on Kelly criterion
        result = super().calculate_position_size(signal)
        
        if result.quantity == 0:
            # If Kelly sizing already rejected the trade, no need for stress testing
            return result
        
        # Run stress test
        pnl_scenarios = self._simulate_scenarios(signal, result.quantity)
        var, cvar = self._calculate_cvar(pnl_scenarios)
        
        # Check if CVaR exceeds limit
        max_cvar_dollars = self.portfolio_value * self.max_cvar
        
        if cvar > max_cvar_dollars:
            # If CVaR is too high, reduce position size
            # Simple approach: cut position size in half until CVaR is acceptable
            reduced_contracts = result.quantity
            while reduced_contracts > 0:
                reduced_contracts = reduced_contracts // 2
                
                if reduced_contracts == 0:
                    # If even a single contract violates CVaR, reject the trade
                    return PositionSizeResult(
                        signal=signal,
                        quantity=0,
                        notional_value=0.0,
                        max_loss=0.0,
                        kelly_fraction=result.kelly_fraction,
                        reason=f"CVaR ({cvar:.2f}) exceeds maximum ({max_cvar_dollars:.2f})"
                    )
                
                # Recalculate CVaR with reduced position
                pnl_scenarios = self._simulate_scenarios(signal, reduced_contracts)
                var, cvar = self._calculate_cvar(pnl_scenarios)
                
                if cvar <= max_cvar_dollars:
                    # Update result with reduced position
                    adjusted_notional = result.notional_value * (reduced_contracts / result.quantity)
                    adjusted_max_loss = result.max_loss * (reduced_contracts / result.quantity)
                    
                    return PositionSizeResult(
                        signal=signal,
                        quantity=reduced_contracts,
                        notional_value=adjusted_notional,
                        max_loss=adjusted_max_loss,
                        kelly_fraction=result.kelly_fraction,
                        reason=f"Position sized to meet CVaR limit: {cvar:.2f} <= {max_cvar_dollars:.2f}"
                    )
        
        # If CVaR is acceptable, return original result
        return result 