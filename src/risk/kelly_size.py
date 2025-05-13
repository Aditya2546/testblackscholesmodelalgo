"""
Kelly Criterion Position Sizing Module

This module calculates optimal position sizes based on edge probability and risk-reward ratio,
using fractional Kelly criterion (default 0.25x) to calculate trade size.

Includes risk limits for various metrics to maintain portfolio-level risk control.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    symbol: str
    contract_id: str
    quantity: int
    notional_value: float
    max_loss: float
    target_gain: float
    kelly_fraction: float
    edge_probability: float
    risk_reward_ratio: float
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "contract_id": self.contract_id,
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "max_loss": self.max_loss,
            "target_gain": self.target_gain,
            "kelly_fraction": self.kelly_fraction,
            "edge_probability": self.edge_probability,
            "risk_reward_ratio": self.risk_reward_ratio,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RiskLimits:
    """Portfolio risk limits for various risk metrics."""
    # Position size limits
    max_position_size_pct: float = 0.05  # Max single position size as % of portfolio
    max_position_size_usd: float = 50000.0  # Max single position size in dollars
    
    # Greeks exposure limits
    max_total_delta: float = 1000000.0   # Max delta in dollars
    max_total_gamma: float = 250000.0    # Max gamma in dollars
    max_total_vega: float = 50000.0      # Max vega in dollars
    max_total_theta: float = 25000.0     # Max theta per day in dollars
    
    # Concentration limits
    max_sector_exposure_pct: float = 0.2  # Max sector exposure as % of portfolio
    max_single_name_exposure_pct: float = 0.1  # Max single name exposure
    
    # Directional bias limits
    max_short_delta_pct: float = 0.5  # Max short delta as % of portfolio
    max_long_delta_pct: float = 0.5   # Max long delta as % of portfolio
    
    # Trading frequency limits
    max_trades_per_day: int = 50  # Max trades per day
    
    # Loss limits
    max_daily_loss_pct: float = 0.03  # Max daily loss as % of portfolio
    max_drawdown_pct: float = 0.08   # Max drawdown trigger for reduced sizing


class KellyPositionSizer:
    """
    Position sizer using Kelly criterion with risk limits.
    
    Calculates optimal position sizes based on:
    1. Edge probability (from ML model)
    2. Risk-reward ratio (from option price and targets)
    3. Portfolio-level risk constraints
    """
    
    def __init__(self, 
                portfolio_value: float,
                risk_limits: Optional[RiskLimits] = None,
                kelly_fraction: float = 0.25,
                min_probability_threshold: float = 0.6):
        """
        Initialize Kelly position sizer.
        
        Args:
            portfolio_value: Total portfolio value in dollars
            risk_limits: Risk limits to use (uses defaults if None)
            kelly_fraction: Fraction of full Kelly size to use (0.0-1.0)
            min_probability_threshold: Minimum probability for taking a trade
        """
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits or RiskLimits()
        self.kelly_fraction = kelly_fraction
        self.min_probability_threshold = min_probability_threshold
        
        # Current portfolio exposures
        self.current_delta = 0.0
        self.current_gamma = 0.0
        self.current_vega = 0.0
        self.current_theta = 0.0
        
        # Concentration tracking
        self.symbol_exposures: Dict[str, float] = {}
        self.sector_exposures: Dict[str, float] = {}
        
        # Daily activity tracking
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.max_portfolio_value = portfolio_value
        self.current_drawdown_pct = 0.0
        
        # Initialize last reset time
        self.last_reset_time = datetime.now()
        
        # Cache for symbol sector mapping
        self._sector_cache: Dict[str, str] = {}
    
    def calculate_position_size(self, 
                              symbol: str,
                              contract_id: str,
                              edge_probability: float,
                              entry_price: float,
                              stop_loss: float,
                              target_price: float,
                              contract_multiplier: float = 100.0,
                              min_quantity: int = 1,
                              greeks: Optional[Dict[str, float]] = None,
                              sector: Optional[str] = None) -> PositionSizeResult:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Trading symbol
            contract_id: Option contract identifier
            edge_probability: Probability of success (0.0-1.0)
            entry_price: Entry price per contract
            stop_loss: Stop loss price per contract
            target_price: Target price per contract
            contract_multiplier: Contract multiplier (100 for standard options)
            min_quantity: Minimum quantity to trade
            greeks: Option greeks (delta, gamma, vega, theta)
            sector: Stock sector
            
        Returns:
            Position size result
        """
        # Validate inputs
        if edge_probability < 0.0 or edge_probability > 1.0:
            return PositionSizeResult(
                symbol=symbol,
                contract_id=contract_id,
                quantity=0,
                notional_value=0.0,
                max_loss=0.0,
                target_gain=0.0,
                kelly_fraction=0.0,
                edge_probability=edge_probability,
                risk_reward_ratio=0.0,
                reason="Invalid edge probability"
            )
        
        # Skip if below minimum probability threshold
        if edge_probability < self.min_probability_threshold:
            return PositionSizeResult(
                symbol=symbol,
                contract_id=contract_id,
                quantity=0,
                notional_value=0.0,
                max_loss=0.0,
                target_gain=0.0,
                kelly_fraction=0.0,
                edge_probability=edge_probability,
                risk_reward_ratio=0.0,
                reason=f"Edge probability {edge_probability:.2f} below threshold {self.min_probability_threshold:.2f}"
            )
        
        # Calculate risk-reward ratio
        if entry_price <= 0.0:
            return PositionSizeResult(
                symbol=symbol,
                contract_id=contract_id,
                quantity=0,
                notional_value=0.0,
                max_loss=0.0,
                target_gain=0.0,
                kelly_fraction=0.0,
                edge_probability=edge_probability,
                risk_reward_ratio=0.0,
                reason="Invalid entry price"
            )
        
        # Calculate profit and loss per contract
        if target_price > entry_price:
            # Long position
            max_loss_per_contract = (entry_price - stop_loss) * contract_multiplier
            target_gain_per_contract = (target_price - entry_price) * contract_multiplier
        else:
            # Short position
            max_loss_per_contract = (stop_loss - entry_price) * contract_multiplier
            target_gain_per_contract = (entry_price - target_price) * contract_multiplier
        
        # Ensure max_loss is positive
        max_loss_per_contract = abs(max_loss_per_contract)
        
        # Calculate risk-reward ratio
        if max_loss_per_contract <= 0.0:
            return PositionSizeResult(
                symbol=symbol,
                contract_id=contract_id,
                quantity=0,
                notional_value=0.0,
                max_loss=0.0,
                target_gain=0.0,
                kelly_fraction=0.0,
                edge_probability=edge_probability,
                risk_reward_ratio=0.0,
                reason="Invalid risk-reward calculation"
            )
        
        risk_reward_ratio = target_gain_per_contract / max_loss_per_contract
        
        # Calculate Kelly fraction
        # Kelly formula: f* = (p(b+1) - 1) / b where p is win probability and b is win/loss ratio
        kelly_value = (edge_probability * (risk_reward_ratio + 1) - 1) / risk_reward_ratio
        
        # Apply fractional Kelly
        fractional_kelly = kelly_value * self.kelly_fraction
        
        # Ensure Kelly fraction is non-negative
        fractional_kelly = max(0.0, fractional_kelly)
        
        # Calculate Kelly position size as percentage of portfolio
        kelly_position_size_pct = fractional_kelly
        
        # Apply position size limit as percentage of portfolio
        position_size_pct = min(kelly_position_size_pct, self.risk_limits.max_position_size_pct)
        
        # Calculate dollar position size
        position_size_usd = self.portfolio_value * position_size_pct
        
        # Apply absolute dollar limit
        position_size_usd = min(position_size_usd, self.risk_limits.max_position_size_usd)
        
        # Apply drawdown adjustment
        if self.current_drawdown_pct > self.risk_limits.max_drawdown_pct:
            drawdown_factor = 0.5  # Reduce sizing by 50% if in drawdown
            position_size_usd *= drawdown_factor
            
            logger.warning(
                f"Reducing position size due to drawdown of {self.current_drawdown_pct:.1%}, "
                f"above limit of {self.risk_limits.max_drawdown_pct:.1%}"
            )
        
        # Calculate quantity
        notional_per_contract = entry_price * contract_multiplier
        quantity = int(position_size_usd / notional_per_contract)
        
        # Ensure minimum quantity
        quantity = max(quantity, 0)
        
        # If quantity is zero but Kelly is positive, use minimum quantity
        if quantity == 0 and fractional_kelly > 0:
            quantity = min_quantity
        
        # Calculate notional value
        notional_value = quantity * notional_per_contract
        
        # Calculate max loss
        max_loss = quantity * max_loss_per_contract
        
        # Calculate target gain
        target_gain = quantity * target_gain_per_contract
        
        # Check risk limits if there are Greeks
        if greeks:
            # Check if adding this position would exceed Greek limits
            if not self._check_greek_limits(greeks, quantity, contract_multiplier):
                return PositionSizeResult(
                    symbol=symbol,
                    contract_id=contract_id,
                    quantity=0,
                    notional_value=0.0,
                    max_loss=0.0,
                    target_gain=0.0,
                    kelly_fraction=fractional_kelly,
                    edge_probability=edge_probability,
                    risk_reward_ratio=risk_reward_ratio,
                    reason="Would exceed portfolio Greek limits"
                )
        
        # Check concentration limits
        if not self._check_concentration_limits(symbol, sector, notional_value):
            return PositionSizeResult(
                symbol=symbol,
                contract_id=contract_id,
                quantity=0,
                notional_value=0.0,
                max_loss=0.0,
                target_gain=0.0,
                kelly_fraction=fractional_kelly,
                edge_probability=edge_probability,
                risk_reward_ratio=risk_reward_ratio,
                reason="Would exceed concentration limits"
            )
        
        # Check trade frequency limit
        if self.daily_trade_count >= self.risk_limits.max_trades_per_day:
            return PositionSizeResult(
                symbol=symbol,
                contract_id=contract_id,
                quantity=0,
                notional_value=0.0,
                max_loss=0.0,
                target_gain=0.0,
                kelly_fraction=fractional_kelly,
                edge_probability=edge_probability,
                risk_reward_ratio=risk_reward_ratio,
                reason=f"Daily trade limit of {self.risk_limits.max_trades_per_day} reached"
            )
        
        # If we get here, all checks passed
        return PositionSizeResult(
            symbol=symbol,
            contract_id=contract_id,
            quantity=quantity,
            notional_value=notional_value,
            max_loss=max_loss,
            target_gain=target_gain,
            kelly_fraction=fractional_kelly,
            edge_probability=edge_probability,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _check_greek_limits(self, 
                          greeks: Dict[str, float], 
                          quantity: int,
                          contract_multiplier: float) -> bool:
        """
        Check if a position would violate Greek risk limits.
        
        Args:
            greeks: Option Greeks (delta, gamma, vega, theta)
            quantity: Position quantity
            contract_multiplier: Contract multiplier
            
        Returns:
            True if within limits, False otherwise
        """
        # Extract Greeks
        delta = greeks.get('delta', 0.0) * quantity * contract_multiplier
        gamma = greeks.get('gamma', 0.0) * quantity * contract_multiplier
        vega = greeks.get('vega', 0.0) * quantity * contract_multiplier
        theta = greeks.get('theta', 0.0) * quantity * contract_multiplier
        
        # Calculate new total exposures
        new_delta = self.current_delta + delta
        new_gamma = self.current_gamma + gamma
        new_vega = self.current_vega + vega
        new_theta = self.current_theta + theta
        
        # Check against limits
        if abs(new_delta) > self.risk_limits.max_total_delta:
            logger.warning(
                f"Delta limit exceeded: {abs(new_delta):.0f} > {self.risk_limits.max_total_delta:.0f}"
            )
            return False
            
        if abs(new_gamma) > self.risk_limits.max_total_gamma:
            logger.warning(
                f"Gamma limit exceeded: {abs(new_gamma):.0f} > {self.risk_limits.max_total_gamma:.0f}"
            )
            return False
            
        if abs(new_vega) > self.risk_limits.max_total_vega:
            logger.warning(
                f"Vega limit exceeded: {abs(new_vega):.0f} > {self.risk_limits.max_total_vega:.0f}"
            )
            return False
            
        if abs(new_theta) > self.risk_limits.max_total_theta:
            logger.warning(
                f"Theta limit exceeded: {abs(new_theta):.0f} > {self.risk_limits.max_total_theta:.0f}"
            )
            return False
        
        # Check directional bias
        if new_delta > 0 and new_delta > self.portfolio_value * self.risk_limits.max_long_delta_pct:
            logger.warning(
                f"Long delta limit exceeded: {new_delta:.0f} > "
                f"{self.portfolio_value * self.risk_limits.max_long_delta_pct:.0f}"
            )
            return False
            
        if new_delta < 0 and abs(new_delta) > self.portfolio_value * self.risk_limits.max_short_delta_pct:
            logger.warning(
                f"Short delta limit exceeded: {abs(new_delta):.0f} > "
                f"{self.portfolio_value * self.risk_limits.max_short_delta_pct:.0f}"
            )
            return False
        
        return True
    
    def _check_concentration_limits(self, 
                                  symbol: str,
                                  sector: Optional[str],
                                  notional_value: float) -> bool:
        """
        Check if a position would violate concentration limits.
        
        Args:
            symbol: Trading symbol
            sector: Stock sector
            notional_value: Position notional value
            
        Returns:
            True if within limits, False otherwise
        """
        # Calculate new symbol exposure
        current_symbol_exposure = self.symbol_exposures.get(symbol, 0.0)
        new_symbol_exposure = current_symbol_exposure + notional_value
        
        # Check symbol limit
        if new_symbol_exposure > self.portfolio_value * self.risk_limits.max_single_name_exposure_pct:
            logger.warning(
                f"Symbol exposure limit exceeded for {symbol}: {new_symbol_exposure:.0f} > "
                f"{self.portfolio_value * self.risk_limits.max_single_name_exposure_pct:.0f}"
            )
            return False
        
        # Check sector limit if sector provided
        if sector:
            current_sector_exposure = self.sector_exposures.get(sector, 0.0)
            new_sector_exposure = current_sector_exposure + notional_value
            
            if new_sector_exposure > self.portfolio_value * self.risk_limits.max_sector_exposure_pct:
                logger.warning(
                    f"Sector exposure limit exceeded for {sector}: {new_sector_exposure:.0f} > "
                    f"{self.portfolio_value * self.risk_limits.max_sector_exposure_pct:.0f}"
                )
                return False
        
        return True
    
    def update_portfolio_risk(self, 
                            symbol: str,
                            sector: Optional[str],
                            quantity: int,
                            notional_value: float,
                            greeks: Optional[Dict[str, float]] = None,
                            contract_multiplier: float = 100.0) -> None:
        """
        Update portfolio risk metrics after executing a trade.
        
        Args:
            symbol: Trading symbol
            sector: Stock sector
            quantity: Position quantity
            notional_value: Position notional value
            greeks: Option Greeks (delta, gamma, vega, theta)
            contract_multiplier: Contract multiplier
        """
        # Update Greek exposures
        if greeks:
            self.current_delta += greeks.get('delta', 0.0) * quantity * contract_multiplier
            self.current_gamma += greeks.get('gamma', 0.0) * quantity * contract_multiplier
            self.current_vega += greeks.get('vega', 0.0) * quantity * contract_multiplier
            self.current_theta += greeks.get('theta', 0.0) * quantity * contract_multiplier
        
        # Update concentration metrics
        self.symbol_exposures[symbol] = self.symbol_exposures.get(symbol, 0.0) + notional_value
        
        if sector:
            self.sector_exposures[sector] = self.sector_exposures.get(sector, 0.0) + notional_value
            
            # Cache sector for this symbol
            self._sector_cache[symbol] = sector
        
        # Update daily trade count
        self.daily_trade_count += 1
        
        # Check if we need to reset daily metrics
        self._check_reset_daily_metrics()
    
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update portfolio value and related metrics.
        
        Args:
            new_value: New portfolio value
        """
        # Calculate P&L
        pnl = new_value - self.portfolio_value
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update max portfolio value
        if new_value > self.max_portfolio_value:
            self.max_portfolio_value = new_value
        
        # Calculate drawdown
        if self.max_portfolio_value > 0:
            self.current_drawdown_pct = 1.0 - (new_value / self.max_portfolio_value)
        
        # Update portfolio value
        self.portfolio_value = new_value
        
        # Log significant changes
        if abs(pnl) > self.portfolio_value * 0.01:  # 1% change
            logger.info(f"Portfolio value updated to {new_value:.2f} (change: {pnl:.2f})")
        
        # Check for daily loss limit
        if self.daily_pnl < -self.portfolio_value * self.risk_limits.max_daily_loss_pct:
            logger.warning(
                f"Daily loss limit exceeded: {self.daily_pnl:.2f} < "
                f"{-self.portfolio_value * self.risk_limits.max_daily_loss_pct:.2f}"
            )
        
        # Check if we need to reset daily metrics
        self._check_reset_daily_metrics()
    
    def close_position(self, 
                     symbol: str,
                     quantity: int,
                     notional_value: float,
                     greeks: Optional[Dict[str, float]] = None,
                     contract_multiplier: float = 100.0) -> None:
        """
        Update portfolio risk metrics after closing a position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            notional_value: Position notional value
            greeks: Option Greeks (delta, gamma, vega, theta)
            contract_multiplier: Contract multiplier
        """
        # Update Greek exposures (opposite sign)
        if greeks:
            self.current_delta -= greeks.get('delta', 0.0) * quantity * contract_multiplier
            self.current_gamma -= greeks.get('gamma', 0.0) * quantity * contract_multiplier
            self.current_vega -= greeks.get('vega', 0.0) * quantity * contract_multiplier
            self.current_theta -= greeks.get('theta', 0.0) * quantity * contract_multiplier
        
        # Update concentration metrics
        if symbol in self.symbol_exposures:
            self.symbol_exposures[symbol] = max(0.0, self.symbol_exposures[symbol] - notional_value)
            
            # Remove symbol if exposure is zero
            if self.symbol_exposures[symbol] == 0.0:
                del self.symbol_exposures[symbol]
        
        # Update sector exposure
        sector = self._sector_cache.get(symbol)
        if sector and sector in self.sector_exposures:
            self.sector_exposures[sector] = max(0.0, self.sector_exposures[sector] - notional_value)
            
            # Remove sector if exposure is zero
            if self.sector_exposures[sector] == 0.0:
                del self.sector_exposures[sector]
    
    def _check_reset_daily_metrics(self) -> None:
        """Check if daily metrics should be reset based on time."""
        now = datetime.now()
        
        # Reset if day has changed
        if now.date() != self.last_reset_time.date():
            # Reset daily metrics
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            
            # Update last reset time
            self.last_reset_time = now
            
            logger.info("Daily risk metrics reset")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            "portfolio_value": self.portfolio_value,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_portfolio_value": self.max_portfolio_value,
            "daily_pnl": self.daily_pnl,
            "daily_trade_count": self.daily_trade_count,
            "greek_exposures": {
                "delta": self.current_delta,
                "gamma": self.current_gamma,
                "vega": self.current_vega,
                "theta": self.current_theta
            },
            "concentration": {
                "symbols": self.symbol_exposures,
                "sectors": self.sector_exposures
            },
            "limits": {
                "max_position_size_pct": self.risk_limits.max_position_size_pct,
                "max_position_size_usd": self.risk_limits.max_position_size_usd,
                "max_total_delta": self.risk_limits.max_total_delta,
                "max_total_gamma": self.risk_limits.max_total_gamma,
                "max_total_vega": self.risk_limits.max_total_vega,
                "max_total_theta": self.risk_limits.max_total_theta,
                "max_sector_exposure_pct": self.risk_limits.max_sector_exposure_pct,
                "max_single_name_exposure_pct": self.risk_limits.max_single_name_exposure_pct,
                "max_short_delta_pct": self.risk_limits.max_short_delta_pct,
                "max_long_delta_pct": self.risk_limits.max_long_delta_pct,
                "max_trades_per_day": self.risk_limits.max_trades_per_day,
                "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                "max_drawdown_pct": self.risk_limits.max_drawdown_pct
            }
        }


# Global instance for convenience
_position_sizer = None

def get_position_sizer(portfolio_value: float = 100000.0,
                      risk_limits: Optional[RiskLimits] = None,
                      kelly_fraction: float = 0.25) -> KellyPositionSizer:
    """
    Get the global position sizer instance.
    
    Args:
        portfolio_value: Portfolio value (for initialization)
        risk_limits: Risk limits (for initialization)
        kelly_fraction: Kelly fraction (for initialization)
        
    Returns:
        Global KellyPositionSizer instance
    """
    global _position_sizer
    
    if _position_sizer is None:
        _position_sizer = KellyPositionSizer(
            portfolio_value=portfolio_value,
            risk_limits=risk_limits,
            kelly_fraction=kelly_fraction
        )
    
    return _position_sizer 