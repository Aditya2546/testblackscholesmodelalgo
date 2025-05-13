"""
Multi-leg spread strategy detection module.

Identifies opportunities for option spread strategies such as
vertical spreads, iron condors, strangles, and straddles.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Set, Any

import numpy as np
import pandas as pd

from ..data.market_data import OptionChain, OptionQuote, OptionRight, Option
from .signal import Signal, SignalType, SignalGenerator


class SpreadType(Enum):
    """Types of option spread strategies."""
    VERTICAL = "vertical"           # Bull/bear spread
    IRON_CONDOR = "iron_condor"     # Iron condor
    IRON_BUTTERFLY = "iron_butterfly"  # Iron butterfly
    STRANGLE = "strangle"           # Strangle
    STRADDLE = "straddle"           # Straddle
    CALENDAR = "calendar"           # Calendar spread


@dataclass
class SpreadLeg:
    """Individual leg of a spread strategy."""
    expiration: datetime
    strike: float
    right: OptionRight
    quantity: int  # Positive for long, negative for short
    price: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpreadStrategy:
    """Complete spread strategy with multiple legs."""
    spread_type: SpreadType
    symbol: str
    legs: List[SpreadLeg]
    net_debit: float = 0.0  # Positive for debit spreads, negative for credit spreads
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_points: List[float] = field(default_factory=list)
    probability_of_profit: float = 0.0
    expected_value: float = 0.0
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        if self.max_loss == 0:
            return float('inf')
        return abs(self.max_profit / self.max_loss)
    
    def to_signal(self) -> Signal:
        """Convert to Signal object."""
        # Determine overall direction (long/short)
        direction = -1 if self.net_debit < 0 else 1
        
        # Find first leg's expiration for the signal
        expiration = min(leg.expiration for leg in self.legs)
        
        # Create signal
        signal = Signal(
            timestamp=datetime.now(),
            signal_type=SignalType.SPREAD,
            symbol=self.symbol,
            direction=direction,
            expiration=expiration,
            expected_edge=self.probability_of_profit * 100,  # Convert to percentage
            expected_pnl=self.expected_value * 100,  # Multiply by contract size
            confidence=self.probability_of_profit
        )
        
        # Add legs information
        for leg in self.legs:
            leg_info = {
                "expiration": leg.expiration,
                "strike": leg.strike,
                "right": leg.right,
                "quantity": leg.quantity,
                "price": leg.price
            }
            signal.legs.append(leg_info)
        
        return signal


class SpreadSignalGenerator(SignalGenerator):
    """
    Signal generator for multi-leg spread strategies.
    
    Analyzes option chains to identify profitable spread opportunities
    based on volatility and price discrepancies.
    """
    
    def __init__(self, min_profit_prob: float = 0.6, min_risk_reward: float = 1.5):
        """
        Initialize SpreadSignalGenerator.
        
        Args:
            min_profit_prob: Minimum probability of profit to generate a signal
            min_risk_reward: Minimum risk-reward ratio to generate a signal
        """
        self.min_profit_prob = min_profit_prob
        self.min_risk_reward = min_risk_reward
    
    def _find_vertical_spreads(self, chain: OptionChain, underlying_price: float, 
                              expiration: datetime) -> List[SpreadStrategy]:
        """
        Find vertical spread opportunities (bull/bear call/put spreads).
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            expiration: Expiration date to analyze
            
        Returns:
            List of vertical spread strategies
        """
        spreads = []
        
        # Get options for this expiration
        options = chain.get_options_by_expiration(expiration)
        
        # Group by option right (call/put)
        call_options = {}
        put_options = {}
        
        for (strike, right), quote in options.items():
            if right == OptionRight.CALL:
                call_options[strike] = quote
            else:
                put_options[strike] = quote
        
        # Find bull call spreads (buy lower strike call, sell higher strike call)
        strike_list = sorted(call_options.keys())
        for i in range(len(strike_list) - 1):
            for j in range(i + 1, min(i + 5, len(strike_list))):  # Limit to 5 strike widths max
                lower_strike = strike_list[i]
                higher_strike = strike_list[j]
                
                # Skip if strikes are too far apart
                if higher_strike - lower_strike > 20:  # Limit to reasonable strike widths
                    continue
                
                # Get quotes
                long_call = call_options[lower_strike]
                short_call = call_options[higher_strike]
                
                # Calculate net debit
                net_debit = long_call.ask_price - short_call.bid_price
                
                # Calculate max profit and loss
                max_profit = (higher_strike - lower_strike) * 100 - net_debit * 100
                max_loss = net_debit * 100
                
                # Calculate breakeven
                breakeven = lower_strike + net_debit
                
                # Calculate probability of profit using delta approximation
                # Bull call spread profits if price > breakeven at expiration
                delta_long = long_call.delta
                probability_profit = delta_long  # Simple approximation
                
                # Calculate expected value
                expected_value = (probability_profit * max_profit - (1 - probability_profit) * max_loss) / 100
                
                # Check if spread meets criteria
                if probability_profit >= self.min_profit_prob and (max_profit / max_loss) >= self.min_risk_reward:
                    # Create spread strategy
                    spread = SpreadStrategy(
                        spread_type=SpreadType.VERTICAL,
                        symbol=chain.underlying_symbol,
                        legs=[
                            SpreadLeg(
                                expiration=expiration,
                                strike=lower_strike,
                                right=OptionRight.CALL,
                                quantity=1,
                                price=long_call.ask_price,
                                greeks={"delta": long_call.delta, "gamma": long_call.gamma, 
                                      "theta": long_call.theta, "vega": long_call.vega}
                            ),
                            SpreadLeg(
                                expiration=expiration,
                                strike=higher_strike,
                                right=OptionRight.CALL,
                                quantity=-1,
                                price=short_call.bid_price,
                                greeks={"delta": short_call.delta, "gamma": short_call.gamma, 
                                      "theta": short_call.theta, "vega": short_call.vega}
                            )
                        ],
                        net_debit=net_debit,
                        max_profit=max_profit / 100,  # Convert to per contract
                        max_loss=max_loss / 100,      # Convert to per contract
                        breakeven_points=[breakeven],
                        probability_of_profit=probability_profit,
                        expected_value=expected_value
                    )
                    spreads.append(spread)
        
        # Find bear put spreads (buy higher strike put, sell lower strike put)
        strike_list = sorted(put_options.keys())
        for i in range(len(strike_list) - 1):
            for j in range(i + 1, min(i + 5, len(strike_list))):  # Limit to 5 strike widths max
                lower_strike = strike_list[i]
                higher_strike = strike_list[j]
                
                # Skip if strikes are too far apart
                if higher_strike - lower_strike > 20:  # Limit to reasonable strike widths
                    continue
                
                # Get quotes
                long_put = put_options[higher_strike]
                short_put = put_options[lower_strike]
                
                # Calculate net debit
                net_debit = long_put.ask_price - short_put.bid_price
                
                # Calculate max profit and loss
                max_profit = (higher_strike - lower_strike) * 100 - net_debit * 100
                max_loss = net_debit * 100
                
                # Calculate breakeven
                breakeven = higher_strike - net_debit
                
                # Calculate probability of profit using delta approximation
                # Bear put spread profits if price < breakeven at expiration
                delta_long = long_put.delta
                probability_profit = -delta_long  # Simple approximation
                
                # Calculate expected value
                expected_value = (probability_profit * max_profit - (1 - probability_profit) * max_loss) / 100
                
                # Check if spread meets criteria
                if probability_profit >= self.min_profit_prob and (max_profit / max_loss) >= self.min_risk_reward:
                    # Create spread strategy
                    spread = SpreadStrategy(
                        spread_type=SpreadType.VERTICAL,
                        symbol=chain.underlying_symbol,
                        legs=[
                            SpreadLeg(
                                expiration=expiration,
                                strike=higher_strike,
                                right=OptionRight.PUT,
                                quantity=1,
                                price=long_put.ask_price,
                                greeks={"delta": long_put.delta, "gamma": long_put.gamma, 
                                      "theta": long_put.theta, "vega": long_put.vega}
                            ),
                            SpreadLeg(
                                expiration=expiration,
                                strike=lower_strike,
                                right=OptionRight.PUT,
                                quantity=-1,
                                price=short_put.bid_price,
                                greeks={"delta": short_put.delta, "gamma": short_put.gamma, 
                                      "theta": short_put.theta, "vega": short_put.vega}
                            )
                        ],
                        net_debit=net_debit,
                        max_profit=max_profit / 100,  # Convert to per contract
                        max_loss=max_loss / 100,      # Convert to per contract
                        breakeven_points=[breakeven],
                        probability_of_profit=probability_profit,
                        expected_value=expected_value
                    )
                    spreads.append(spread)
        
        return spreads
    
    def _find_iron_condors(self, chain: OptionChain, underlying_price: float, 
                          expiration: datetime) -> List[SpreadStrategy]:
        """
        Find iron condor opportunities.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            expiration: Expiration date to analyze
            
        Returns:
            List of iron condor spread strategies
        """
        spreads = []
        
        # Get options for this expiration
        options = chain.get_options_by_expiration(expiration)
        
        # Group by option right
        call_options = {}
        put_options = {}
        
        for (strike, right), quote in options.items():
            if right == OptionRight.CALL:
                call_options[strike] = quote
            else:
                put_options[strike] = quote
        
        # Get sorted strikes
        call_strikes = sorted(call_options.keys())
        put_strikes = sorted(put_options.keys())
        
        # Find strikes around the current price
        atm_call_idx = 0
        for i, strike in enumerate(call_strikes):
            if strike >= underlying_price:
                atm_call_idx = i
                break
                
        atm_put_idx = 0
        for i, strike in enumerate(put_strikes):
            if strike >= underlying_price:
                atm_put_idx = i
                break
        
        # Get potential strikes for iron condor
        # For calls: sell ATM+1, buy ATM+2 or ATM+3
        # For puts: sell ATM-1, buy ATM-2 or ATM-3
        
        # Check if we have enough strikes for an iron condor
        if (atm_call_idx + 3 >= len(call_strikes)) or (atm_put_idx < 3):
            return []
        
        # Try different iron condor widths
        for call_spread_width in [1, 2]:
            for put_spread_width in [1, 2]:
                # Sell call strike
                sell_call_idx = atm_call_idx + 1
                # Buy call strike
                buy_call_idx = sell_call_idx + call_spread_width
                
                # Sell put strike
                sell_put_idx = atm_put_idx - 1
                # Buy put strike
                buy_put_idx = sell_put_idx - put_spread_width
                
                # Check if indices are valid
                if (buy_call_idx >= len(call_strikes)) or (buy_put_idx < 0):
                    continue
                
                # Get strikes
                sell_call_strike = call_strikes[sell_call_idx]
                buy_call_strike = call_strikes[buy_call_idx]
                sell_put_strike = put_strikes[sell_put_idx]
                buy_put_strike = put_strikes[buy_put_idx]
                
                # Get quotes
                sell_call = call_options[sell_call_strike]
                buy_call = call_options[buy_call_strike]
                sell_put = put_options[sell_put_strike]
                buy_put = put_options[buy_put_strike]
                
                # Calculate credit received
                credit = (sell_call.bid_price - buy_call.ask_price + 
                         sell_put.bid_price - buy_put.ask_price)
                
                # Set net_debit for a credit spread (negative value)
                net_debit = -credit
                
                # Calculate max profit and loss
                call_spread_width_dollars = (buy_call_strike - sell_call_strike) * 100
                put_spread_width_dollars = (sell_put_strike - buy_put_strike) * 100
                
                max_profit = credit * 100
                max_loss = max(call_spread_width_dollars, put_spread_width_dollars) - max_profit
                
                # Calculate breakevens
                upper_breakeven = sell_call_strike + credit
                lower_breakeven = sell_put_strike - credit
                
                # Calculate probability of profit
                # Iron condor profits if price stays between breakevens
                delta_sell_call = sell_call.delta
                delta_sell_put = sell_put.delta
                
                # Approximate probability using deltas
                probability_above_upper = delta_sell_call
                probability_below_lower = -delta_sell_put
                probability_profit = 1 - probability_above_upper - probability_below_lower
                
                # Calculate expected value
                expected_value = (probability_profit * max_profit - (1 - probability_profit) * max_loss) / 100
                
                # Check if spread meets criteria
                if probability_profit >= self.min_profit_prob and (max_profit / max_loss) >= self.min_risk_reward:
                    # Create spread strategy
                    spread = SpreadStrategy(
                        spread_type=SpreadType.IRON_CONDOR,
                        symbol=chain.underlying_symbol,
                        legs=[
                            SpreadLeg(
                                expiration=expiration,
                                strike=sell_call_strike,
                                right=OptionRight.CALL,
                                quantity=-1,
                                price=sell_call.bid_price,
                                greeks={"delta": sell_call.delta, "gamma": sell_call.gamma, 
                                      "theta": sell_call.theta, "vega": sell_call.vega}
                            ),
                            SpreadLeg(
                                expiration=expiration,
                                strike=buy_call_strike,
                                right=OptionRight.CALL,
                                quantity=1,
                                price=buy_call.ask_price,
                                greeks={"delta": buy_call.delta, "gamma": buy_call.gamma, 
                                      "theta": buy_call.theta, "vega": buy_call.vega}
                            ),
                            SpreadLeg(
                                expiration=expiration,
                                strike=sell_put_strike,
                                right=OptionRight.PUT,
                                quantity=-1,
                                price=sell_put.bid_price,
                                greeks={"delta": sell_put.delta, "gamma": sell_put.gamma, 
                                      "theta": sell_put.theta, "vega": sell_put.vega}
                            ),
                            SpreadLeg(
                                expiration=expiration,
                                strike=buy_put_strike,
                                right=OptionRight.PUT,
                                quantity=1,
                                price=buy_put.ask_price,
                                greeks={"delta": buy_put.delta, "gamma": buy_put.gamma, 
                                      "theta": buy_put.theta, "vega": buy_put.vega}
                            )
                        ],
                        net_debit=net_debit,
                        max_profit=max_profit / 100,  # Convert to per contract
                        max_loss=max_loss / 100,      # Convert to per contract
                        breakeven_points=[lower_breakeven, upper_breakeven],
                        probability_of_profit=probability_profit,
                        expected_value=expected_value
                    )
                    spreads.append(spread)
        
        return spreads
    
    def _find_straddles(self, chain: OptionChain, underlying_price: float, 
                       expiration: datetime) -> List[SpreadStrategy]:
        """
        Find straddle opportunities (buy call and put at same strike).
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            expiration: Expiration date to analyze
            
        Returns:
            List of straddle spread strategies
        """
        spreads = []
        
        # Get options for this expiration
        options = chain.get_options_by_expiration(expiration)
        
        # Find ATM options (closest to underlying price)
        closest_strike = None
        min_distance = float('inf')
        
        for (strike, _), _ in options.items():
            distance = abs(strike - underlying_price)
            if distance < min_distance:
                min_distance = distance
                closest_strike = strike
        
        if not closest_strike:
            return []
        
        # Get call and put for the ATM strike
        call_quote = None
        put_quote = None
        
        for (strike, right), quote in options.items():
            if strike == closest_strike:
                if right == OptionRight.CALL:
                    call_quote = quote
                else:
                    put_quote = quote
        
        if not call_quote or not put_quote:
            return []
        
        # Calculate straddle price
        net_debit = call_quote.ask_price + put_quote.ask_price
        
        # Calculate breakevens
        upper_breakeven = closest_strike + net_debit
        lower_breakeven = closest_strike - net_debit
        
        # Simple expected value calculation
        # Straddle profits if price moves significantly in either direction
        # Use IV to estimate probability
        iv = (call_quote.implied_volatility + put_quote.implied_volatility) / 2
        
        # Calculate expected move based on IV
        time_to_expiry = (expiration - datetime.now()).days / 365
        expected_move = underlying_price * iv * np.sqrt(time_to_expiry)
        
        # Probability calculation is simplified
        # A straddle needs to move more than the net debit to be profitable
        probability_profit = 0.5  # Simplified - would use normal distribution in reality
        
        # Calculate simplified max profit/loss
        # Max loss is limited to the premium paid
        max_loss = net_debit * 100
        
        # Max profit is unlimited in theory, but estimate using expected move
        estimated_profit = max(0, expected_move - net_debit) * 100
        
        # Expected value
        expected_value = (probability_profit * estimated_profit - (1 - probability_profit) * max_loss) / 100
        
        # Create spread strategy
        spread = SpreadStrategy(
            spread_type=SpreadType.STRADDLE,
            symbol=chain.underlying_symbol,
            legs=[
                SpreadLeg(
                    expiration=expiration,
                    strike=closest_strike,
                    right=OptionRight.CALL,
                    quantity=1,
                    price=call_quote.ask_price,
                    greeks={"delta": call_quote.delta, "gamma": call_quote.gamma, 
                          "theta": call_quote.theta, "vega": call_quote.vega}
                ),
                SpreadLeg(
                    expiration=expiration,
                    strike=closest_strike,
                    right=OptionRight.PUT,
                    quantity=1,
                    price=put_quote.ask_price,
                    greeks={"delta": put_quote.delta, "gamma": put_quote.gamma, 
                          "theta": put_quote.theta, "vega": put_quote.vega}
                )
            ],
            net_debit=net_debit,
            max_profit=float('inf'),  # Unlimited upside
            max_loss=max_loss / 100,  # Convert to per contract
            breakeven_points=[lower_breakeven, upper_breakeven],
            probability_of_profit=probability_profit,
            expected_value=expected_value
        )
        
        spreads.append(spread)
        
        return spreads
    
    def generate_signals(self, chain: OptionChain, underlying_price: float) -> List[Signal]:
        """
        Generate spread strategy signals from an option chain.
        
        Args:
            chain: Option chain to analyze
            underlying_price: Current price of the underlying asset
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get unique expirations
        expirations = set()
        for (expiration, _, _) in chain.options:
            expirations.add(expiration)
        
        # Analyze each expiration
        for expiration in expirations:
            # Skip very short-dated options
            dte = (expiration - datetime.now()).days
            if dte < 7:
                continue
            
            # Find different spread strategies
            vertical_spreads = self._find_vertical_spreads(chain, underlying_price, expiration)
            iron_condors = self._find_iron_condors(chain, underlying_price, expiration)
            straddles = self._find_straddles(chain, underlying_price, expiration)
            
            # Convert to signals
            for spread in vertical_spreads + iron_condors + straddles:
                signals.append(spread.to_signal())
        
        # Sort by expected edge
        signals.sort(key=lambda s: s.expected_edge, reverse=True)
        
        return signals 