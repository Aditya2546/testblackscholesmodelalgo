#!/usr/bin/env python
"""
Realistic Options Trading Simulator

This module extends the basic options day trader simulator with more realistic trading factors:
1. Slippage and liquidity effects
2. Detailed commission structure
3. Fill probability modeling

These factors are critical for realistic backtesting and simulation.
"""

import numpy as np
import datetime
import time
from typing import Dict, List, Tuple, Optional

# Import base simulator
from options_day_trader_sim import OptionSignal, Config, MarketDataSimulator, OptionsDayTrader


class RealisticConfig(Config):
    """Extended configuration settings for realistic trading simulation."""
    
    # Slippage Configuration
    SLIPPAGE_BASE_PCT = 0.02  # Base slippage as percentage of mid price
    SLIPPAGE_VOLATILITY_MULTIPLIER = 0.5  # Higher IV = more slippage
    SLIPPAGE_VOLUME_FACTOR = 3.0  # Impact of low volume on slippage
    MIN_VOLUME_THRESHOLD = 50  # Volume below this is considered very low liquidity
    
    # Commission Structure
    BASE_COMMISSION_PER_CONTRACT = 0.65  # Base commission per contract
    EXCHANGE_FEE_PER_CONTRACT = 0.055  # Exchange fees
    REGULATORY_FEE_PER_CONTRACT = 0.022  # SEC/FINRA fees
    MIN_COMMISSION_PER_TRADE = 1.00  # Minimum commission per trade
    
    # Fill Probability
    MARKET_ORDER_FILL_PROBABILITY = 0.99  # Nearly always filled but at worse prices
    LIMIT_ORDER_AT_ASK_FILL_PROBABILITY = 0.95  # High probability at the ask
    LIMIT_ORDER_AT_MID_FILL_PROBABILITY = 0.65  # Medium probability at midpoint
    LIMIT_ORDER_AT_BID_FILL_PROBABILITY = 0.40  # Lower probability at the bid
    VOLUME_FILL_IMPACT = 0.2  # How much volume affects fill probability
    
    # Market Impact
    MARKET_IMPACT_FACTOR = 0.1  # How much your order moves the market
    MARKET_IMPACT_THRESHOLD = 0.1  # % of option volume where market impact starts


class LiquidityModel:
    """Models option liquidity and its effects on execution."""
    
    @staticmethod
    def calculate_slippage(option_price: float, bid_ask_spread: float, 
                          volume: int, iv: float, position_size: int) -> float:
        """
        Calculate realistic slippage based on market conditions.
        
        Args:
            option_price: Current option mid price
            bid_ask_spread: Current spread in dollars
            volume: Option volume today
            iv: Implied volatility percentage
            position_size: Number of contracts to trade
            
        Returns:
            Slippage amount in dollars per contract
        """
        # Base slippage as a percentage of option price
        base_slippage = option_price * RealisticConfig.SLIPPAGE_BASE_PCT
        
        # Volume factor - more slippage for lower volume
        volume_factor = max(1.0, RealisticConfig.SLIPPAGE_VOLUME_FACTOR * 
                           (RealisticConfig.MIN_VOLUME_THRESHOLD / max(volume, 1)))
        
        # IV factor - more slippage for higher IV options
        iv_factor = 1.0 + (iv / 100.0 * RealisticConfig.SLIPPAGE_VOLATILITY_MULTIPLIER)
        
        # Position size factor - larger positions have more impact
        if volume > 0:
            size_factor = 1.0 + max(0, (position_size / volume - 0.05) * 10)
        else:
            size_factor = 2.0  # Very high if no volume
        
        # Combine factors
        slippage = base_slippage * volume_factor * iv_factor * size_factor
        
        # Cap slippage at 50% of the bid-ask spread for realistic results
        max_slippage = bid_ask_spread * 0.5
        
        return min(slippage, max_slippage)
    
    @staticmethod
    def calculate_fill_probability(order_type: str, price_point: str, 
                                 volume: int, position_size: int) -> float:
        """
        Calculate the probability of an order being filled.
        
        Args:
            order_type: 'market' or 'limit'
            price_point: For limit orders: 'bid', 'mid', or 'ask'
            volume: Option volume 
            position_size: Number of contracts to trade
            
        Returns:
            Probability of fill (0-1)
        """
        # Base fill probability based on order type and price point
        if order_type == 'market':
            base_probability = RealisticConfig.MARKET_ORDER_FILL_PROBABILITY
        else:  # limit order
            if price_point == 'ask':
                base_probability = RealisticConfig.LIMIT_ORDER_AT_ASK_FILL_PROBABILITY
            elif price_point == 'mid':
                base_probability = RealisticConfig.LIMIT_ORDER_AT_MID_FILL_PROBABILITY
            elif price_point == 'bid':
                base_probability = RealisticConfig.LIMIT_ORDER_AT_BID_FILL_PROBABILITY
            else:
                base_probability = 0.5  # Default
        
        # Volume impact - lower volume means lower fill probability
        volume_factor = min(1.0, (volume / max(RealisticConfig.MIN_VOLUME_THRESHOLD, 1)) ** RealisticConfig.VOLUME_FILL_IMPACT)
        
        # Position size impact - larger positions have lower fill probability
        position_factor = 1.0
        if volume > 0:
            position_factor = max(0.5, 1.0 - (position_size / volume) * 2)
        
        # Combine factors
        fill_probability = base_probability * volume_factor * position_factor
        
        return min(1.0, max(0.01, fill_probability))


class CommissionModel:
    """Models realistic brokerage commissions and fees."""
    
    @staticmethod
    def calculate_commission(position_size: int, is_opening: bool = True) -> float:
        """
        Calculate realistic commissions and fees.
        
        Args:
            position_size: Number of contracts
            is_opening: Whether this is an opening trade (some fees differ)
            
        Returns:
            Total commission in dollars
        """
        # Base commission
        base_commission = position_size * RealisticConfig.BASE_COMMISSION_PER_CONTRACT
        
        # Exchange fees
        exchange_fees = position_size * RealisticConfig.EXCHANGE_FEE_PER_CONTRACT
        
        # Regulatory fees (usually only applied on sells)
        regulatory_fees = 0
        if not is_opening:
            regulatory_fees = position_size * RealisticConfig.REGULATORY_FEE_PER_CONTRACT
        
        # Calculate total commission
        total_commission = base_commission + exchange_fees + regulatory_fees
        
        # Apply minimum if needed
        total_commission = max(total_commission, RealisticConfig.MIN_COMMISSION_PER_TRADE)
        
        return total_commission


class RealisticOptionsDayTrader(OptionsDayTrader):
    """
    Extends the basic OptionsDayTrader with realistic trading factors.
    """
    
    def __init__(self, account_value: float = 25000.0):
        """Initialize the realistic trader."""
        super().__init__(account_value)
        self.liquidity_model = LiquidityModel()
        self.commission_model = CommissionModel()
        self.fill_failures = 0
        self.partial_fills = 0
        self.slippage_costs = 0.0
    
    def simulate_trade(self, signal: OptionSignal, hold_minutes: int = None, 
                      order_type: str = 'limit', price_point: str = 'mid') -> Dict:
        """
        Simulate a trade with realistic factors.
        
        Args:
            signal: The option signal to trade
            hold_minutes: How long to hold the position in minutes
            order_type: 'market' or 'limit'
            price_point: For limit orders: 'bid', 'mid', or 'ask'
            
        Returns:
            Dictionary with trade results
        """
        # Use config values if not specified
        if hold_minutes is None:
            hold_minutes = Config.MAX_HOLD_MINUTES
            
        print(f"\n🕐 SIMULATING REALISTIC TRADE FOR {signal.symbol} {signal.option_type.upper()} ${signal.strike}...")
        
        # Determine requested entry price based on order type and price point
        bid = signal.entry_price_range[0]
        ask = signal.entry_price_range[1]
        mid = (bid + ask) / 2
        
        if order_type == 'market':
            requested_price = ask  # Market orders typically fill at the ask
        else:  # limit order
            if price_point == 'bid':
                requested_price = bid
            elif price_point == 'ask':
                requested_price = ask
            else:  # mid
                requested_price = mid
                
        # Determine the bid-ask spread
        bid_ask_spread = ask - bid
        
        # Calculate position size
        position_size = self.get_position_size(mid)  # Base size on mid price
        
        # Calculate fill probability
        fill_probability = self.liquidity_model.calculate_fill_probability(
            order_type, price_point, signal.volume, position_size)
        
        # Determine if the order is filled
        is_filled = np.random.random() < fill_probability
        
        if not is_filled:
            print(f"⚠️ ORDER NOT FILLED (Fill Probability: {fill_probability:.1%})")
            self.fill_failures += 1
            
            # Return early with failure data
            return {
                "symbol": signal.symbol,
                "option_type": signal.option_type,
                "strike": signal.strike,
                "entry_time": signal.timestamp,
                "filled": False,
                "fill_probability": fill_probability,
                "requested_price": requested_price,
                "position_size": position_size,
                "outcome_type": "UNFILLED ORDER"
            }
        
        # Determine if it's a partial fill
        is_partial = False
        original_position_size = position_size
        
        if position_size > 1 and signal.volume < position_size * 2:
            partial_fill_probability = 0.3 + (position_size / max(signal.volume, 1)) * 0.4
            is_partial = np.random.random() < partial_fill_probability
            
            if is_partial:
                # Fill between 30-90% of requested size
                fill_percentage = np.random.uniform(0.3, 0.9)
                position_size = max(1, int(position_size * fill_percentage))
                self.partial_fills += 1
                print(f"⚠️ PARTIAL FILL: {position_size}/{original_position_size} contracts ({fill_percentage:.1%})")
        
        # Calculate entry slippage
        entry_slippage = self.liquidity_model.calculate_slippage(
            mid, bid_ask_spread, signal.volume, signal.iv, position_size)
        
        # Apply slippage to entry price
        if order_type == 'market':
            # Market orders typically get worse prices
            entry_price = ask + entry_slippage
        else:  # limit order
            entry_price = requested_price
        
        # Calculate entry commission
        entry_commission = self.commission_model.calculate_commission(position_size, is_opening=True)
        
        # Randomly determine outcome with a bias toward profitable trades (55/45)
        is_profitable = np.random.random() < 0.55
        
        # Time to outcome is random but shorter (ultra short-term trading)
        outcome_minutes = np.random.randint(5, min(hold_minutes, Config.MAX_HOLD_MINUTES))
        outcome_time = datetime.datetime.now() + datetime.timedelta(minutes=outcome_minutes)
        
        # Determine outcome price - more realistic for short-term option trades
        if is_profitable:
            # Hit target or somewhere between entry and target
            outcome_pct = np.random.uniform(0.03, 1.0) * Config.PROFIT_TARGET_PCT
            outcome_price_before_slippage = entry_price * (1 + outcome_pct)
            outcome_type = "TARGET HIT" if outcome_pct >= Config.PROFIT_TARGET_PCT * 0.95 else "PARTIAL PROFIT"
        else:
            # Hit stop loss or somewhere between entry and stop
            outcome_pct = np.random.uniform(0.3, 1.0) * -Config.STOP_LOSS_PCT
            outcome_price_before_slippage = entry_price * (1 + outcome_pct)
            outcome_type = "STOP LOSS HIT" if outcome_pct <= -Config.STOP_LOSS_PCT * 0.95 else "PARTIAL LOSS"
        
        # Calculate exit slippage - typically higher when exiting
        exit_slippage = self.liquidity_model.calculate_slippage(
            outcome_price_before_slippage, bid_ask_spread * 1.1, 
            signal.volume, signal.iv, position_size)
        
        # Apply exit slippage (always negative when selling)
        outcome_price = outcome_price_before_slippage - exit_slippage
        
        # Calculate exit commission
        exit_commission = self.commission_model.calculate_commission(position_size, is_opening=False)
        
        # Calculate profit/loss
        price_change = outcome_price - entry_price
        price_change_pct = price_change / entry_price * 100
        dollar_profit = price_change * position_size * 100  # 100 shares per contract
        
        # Calculate total commissions and fees
        total_commission = entry_commission + exit_commission
        total_slippage_cost = (entry_slippage + exit_slippage) * position_size * 100
        
        # Track slippage costs for reporting
        self.slippage_costs += total_slippage_cost
        
        # Calculate net profit after all costs
        net_profit = dollar_profit - total_commission
        
        # Update account value
        previous_account_value = self.account_value
        self.account_value += net_profit
        
        # Print the outcome with realistic details
        print("\n" + "=" * 80)
        print(f"REALISTIC TRADE OUTCOME: {signal.symbol} {signal.option_type.upper()} ${signal.strike}")
        print("-" * 80)
        print(f"ORDER TYPE:   {order_type.upper()} order at {price_point.upper()} price")
        print(f"FILL PROBABILITY: {fill_probability:.1%}")
        if is_partial:
            print(f"PARTIAL FILL: {position_size}/{original_position_size} contracts")
        print(f"ENTRY TIME:   {signal.timestamp.strftime('%H:%M:%S')}")
        print(f"ENTRY PRICE:  ${entry_price:.2f} x {position_size} contracts")
        print(f"ENTRY SLIPPAGE: ${entry_slippage:.2f} per contract (${entry_slippage * position_size * 100:.2f} total)")
        print(f"POSITION SIZE: ${(entry_price * position_size * 100):.2f} ({(position_size * entry_price * 100 / previous_account_value * 100):.1f}% of account)")
        print(f"EXIT TIME:    {outcome_time.strftime('%H:%M:%S')} ({outcome_minutes} minutes later)")
        print(f"EXIT PRICE:   ${outcome_price:.2f} ({price_change_pct:.1f}%)")
        print(f"EXIT SLIPPAGE: ${exit_slippage:.2f} per contract (${exit_slippage * position_size * 100:.2f} total)")
        print(f"OUTCOME:      {outcome_type}")
        print(f"COMMISSIONS:  ${total_commission:.2f} (Entry: ${entry_commission:.2f}, Exit: ${exit_commission:.2f})")
        print(f"TOTAL COSTS:  ${(total_commission + total_slippage_cost):.2f} (${(total_commission + total_slippage_cost) / position_size:.2f} per contract)")
        print(f"GROSS P&L:    ${dollar_profit:.2f}")
        print(f"NET P&L:      ${net_profit:.2f}")
        
        if net_profit > 0:
            profit_per_minute = net_profit / outcome_minutes
            print(f"PROFIT RATE:  ${profit_per_minute:.2f} per minute")
        
        # Show account changes
        print("-" * 80)
        print(f"PREVIOUS BALANCE: ${previous_account_value:.2f}")
        print(f"CURRENT BALANCE:  ${self.account_value:.2f}")
        account_change_pct = (self.account_value / previous_account_value - 1) * 100
        print(f"ACCOUNT CHANGE:   {account_change_pct:+.2f}%")
        overall_change_pct = (self.account_value / self.initial_account_value - 1) * 100
        print(f"DAY'S RETURN:     {overall_change_pct:+.2f}%")
        print("=" * 80)
        
        result = {
            "symbol": signal.symbol,
            "option_type": signal.option_type,
            "strike": signal.strike,
            "order_type": order_type,
            "price_point": price_point,
            "fill_probability": fill_probability,
            "filled": True,
            "partial_fill": is_partial,
            "original_size": original_position_size,
            "filled_size": position_size,
            "entry_time": signal.timestamp,
            "entry_price": entry_price,
            "entry_slippage": entry_slippage * position_size * 100,
            "entry_commission": entry_commission,
            "position_size": position_size,
            "exit_time": outcome_time,
            "exit_price": outcome_price,
            "exit_slippage": exit_slippage * position_size * 100,
            "exit_commission": exit_commission,
            "hold_time_minutes": outcome_minutes,
            "price_change_pct": price_change_pct,
            "total_commission": total_commission,
            "total_slippage": total_slippage_cost,
            "gross_profit": dollar_profit,
            "net_profit": net_profit,
            "outcome_type": outcome_type,
            "previous_account": previous_account_value,
            "new_account": self.account_value
        }
        
        self.trade_history.append(result)
        return result


def simulate_realistic_trading(num_trades: int = 10, account_value: float = 25000.0):
    """
    Run a realistic trading simulation for a specified number of trades.
    
    Args:
        num_trades: Number of trades to simulate
        account_value: Starting account value
    """
    trader = RealisticOptionsDayTrader(account_value=account_value)
    
    print(f"\n=== REALISTIC TRADING SIMULATION ===")
    print(f"Starting account: ${trader.account_value:.2f}")
    print(f"Simulating {num_trades} trades with realistic factors:")
    print(f"  - Slippage and liquidity effects")
    print(f"  - Detailed commission structure")
    print(f"  - Fill probability modeling\n")
    
    completed_trades = 0
    attempt_count = 0
    order_types = ['limit', 'market']
    price_points = ['bid', 'mid', 'ask']
    
    # Keep attempting trades until we have the requested number of completed trades
    while completed_trades < num_trades and attempt_count < num_trades * 2:
        attempt_count += 1
        
        # Scan for signals
        signals = trader.scan_for_signals()
        
        if signals:
            signal = signals[0]  # Take the first signal
            
            # Randomly choose order type and price point for variety
            order_type = np.random.choice(order_types, p=[0.7, 0.3])  # 70% limit, 30% market
            
            if order_type == 'limit':
                # Price point probabilities (most traders use limit at mid)
                price_point = np.random.choice(price_points, p=[0.2, 0.6, 0.2])
            else:
                price_point = 'ask'  # Market orders execute at ask
            
            # Simulate the trade
            result = trader.simulate_trade(signal, order_type=order_type, price_point=price_point)
            
            # Count only filled trades
            if result.get("filled", False):
                completed_trades += 1
        
        # Small delay between attempts
        time.sleep(0.5)
    
    # Print summary statistics
    filled_trades = [t for t in trader.trade_history if t.get("filled", False)]
    unfilled_count = trader.fill_failures
    partial_count = trader.partial_fills
    
    if filled_trades:
        # Financial stats
        total_profit = sum(t["net_profit"] for t in filled_trades)
        gross_profit = sum(t["gross_profit"] for t in filled_trades)
        total_commission = sum(t["total_commission"] for t in filled_trades)
        total_slippage = sum(t.get("total_slippage", 0) for t in filled_trades)
        win_count = sum(1 for t in filled_trades if t["net_profit"] > 0)
        loss_count = len(filled_trades) - win_count
        win_rate = win_count / len(filled_trades) if filled_trades else 0
        
        # Calculate metrics
        average_profit_per_trade = total_profit / len(filled_trades)
        average_hold_time = sum(t["hold_time_minutes"] for t in filled_trades) / len(filled_trades)
        profit_per_minute = total_profit / sum(t["hold_time_minutes"] for t in filled_trades) if sum(t["hold_time_minutes"] for t in filled_trades) > 0 else 0
        
        # Print overall summary
        print("\n" + "=" * 80)
        print(f"REALISTIC TRADING SIMULATION SUMMARY:")
        print("-" * 80)
        print(f"Trade Attempts:     {attempt_count}")
        print(f"Filled Trades:      {len(filled_trades)}")
        print(f"Unfilled Orders:    {unfilled_count} ({unfilled_count/attempt_count:.1%} of attempts)")
        print(f"Partial Fills:      {partial_count} ({partial_count/len(filled_trades):.1%} of filled trades)")
        print(f"Winning Trades:     {win_count} ({win_rate:.1%})")
        print(f"Losing Trades:      {loss_count}")
        print(f"Gross P&L:          ${gross_profit:.2f}")
        print(f"Total Commissions:  ${total_commission:.2f} ({total_commission/gross_profit*100:.1f}% of gross)")
        print(f"Total Slippage:     ${total_slippage:.2f} ({total_slippage/gross_profit*100:.1f}% of gross)")
        print(f"Net P&L:            ${total_profit:.2f}")
        print(f"Avg Profit/Trade:   ${average_profit_per_trade:.2f}")
        print(f"Avg Hold Time:      {average_hold_time:.1f} minutes")
        print(f"Profit Per Minute:  ${profit_per_minute:.2f}")
        print("-" * 80)
        print(f"Starting Balance:   ${trader.initial_account_value:.2f}")
        print(f"Ending Balance:     ${trader.account_value:.2f}")
        account_change_pct = (trader.account_value / trader.initial_account_value - 1) * 100
        print(f"Account Change:     {account_change_pct:+.2f}%")
        
        # Impact of realistic factors
        theoretical_profit = gross_profit
        actual_profit = total_profit
        realistic_factors_impact = theoretical_profit - actual_profit
        impact_percentage = (realistic_factors_impact / theoretical_profit) * 100 if theoretical_profit != 0 else 0
        
        print("-" * 80)
        print(f"IMPACT OF REALISTIC FACTORS:")
        print(f"Theoretical P&L (no costs): ${theoretical_profit:.2f}")
        print(f"Actual P&L (with costs):    ${actual_profit:.2f}")
        print(f"Total Impact of Costs:      ${realistic_factors_impact:.2f} ({impact_percentage:.1f}% reduction)")
        print("=" * 80)
    else:
        print("No trades were completed successfully.")


if __name__ == "__main__":
    # Run a realistic simulation with 20 trades
    simulate_realistic_trading(num_trades=20, account_value=25000.0) 