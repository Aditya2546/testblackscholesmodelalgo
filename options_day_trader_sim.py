#!/usr/bin/env python
"""
Options Day Trading Alert System - Simulation Mode

This script generates options trading alerts in simulation mode, regardless of
market conditions, to demonstrate how the alerts look and function.
"""

import pandas as pd
import numpy as np
import datetime
import time
import argparse
from typing import Dict, List, Tuple

# In a real implementation, you would import your broker's API
# import your_broker_api as broker

# Configuration settings
class Config:
    # Ticker symbols to monitor
    WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    
    # Signal settings
    MOMENTUM_THRESHOLD = 0.5  # % change to trigger momentum signal
    VOLUME_THRESHOLD = 1.5    # Multiple of avg volume to confirm signal
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Option settings
    DAYS_TO_EXPIRATION = 0     # For ultra-short term, use same-day expiration
    MAX_DELTA = 0.35           # Maximum option delta (closer to ATM)
    MIN_DELTA = 0.20           # Minimum option delta (not too far OTM)
    MIN_VOLUME = 100           # Minimum option volume
    MAX_SPREAD_PCT = 0.15      # Maximum bid-ask spread as % of option price
    
    # Risk management
    POSITION_SIZE_PCT = 0.02   # Maximum % of account per position
    STOP_LOSS_PCT = 0.15       # Stop loss at 15% of option premium (more realistic)
    PROFIT_TARGET_PCT = 0.10   # Take profit at 10% of option premium (more realistic)
    
    # Trade timing
    MAX_HOLD_MINUTES = 45      # Maximum time to hold a trade
    TARGET_HOLD_MINUTES = 30   # Target time to hold a trade
    
    # Simulation settings
    WIN_RATE = 0.65            # Win rate for simulation (65%)


class OptionSignal:
    """An options trading signal with entry and exit parameters."""
    
    def __init__(self, 
                symbol: str,
                option_type: str,  # 'call' or 'put'
                strike: float,
                expiration: datetime.date,
                current_price: float,
                underlying_price: float,
                entry_price_range: Tuple[float, float],
                stop_loss: float,
                target_price: float,
                signal_strength: float = 0.0,
                volume: int = 0,
                open_interest: int = 0,
                iv: float = 0.0,
                delta: float = 0.0):
        
        self.symbol = symbol
        self.option_type = option_type
        self.strike = strike
        self.expiration = expiration
        self.current_price = current_price
        self.underlying_price = underlying_price
        self.entry_price_range = entry_price_range  # (min, max) to pay
        self.stop_loss = stop_loss
        self.target_price = target_price
        self.signal_strength = signal_strength  # 0-1 scale
        self.volume = volume
        self.open_interest = open_interest
        self.iv = iv
        self.delta = delta
        self.timestamp = datetime.datetime.now()
        
    def __str__(self) -> str:
        """String representation of the signal."""
        expiry_str = self.expiration.strftime("%Y-%m-%d")
        return (f"ALERT: {self.symbol} {self.option_type.upper()} {self.strike} {expiry_str} "
                f"- Buy between ${self.entry_price_range[0]:.2f}-${self.entry_price_range[1]:.2f} "
                f"| Stop Loss: ${self.stop_loss:.2f} | Target: ${self.target_price:.2f} "
                f"| Signal Strength: {self.signal_strength:.2f}/1.0")
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            "symbol": self.symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration.strftime("%Y-%m-%d"),
            "current_price": self.current_price,
            "underlying_price": self.underlying_price,
            "entry_min": self.entry_price_range[0],
            "entry_max": self.entry_price_range[1],
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "signal_strength": self.signal_strength,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "iv": self.iv,
            "delta": self.delta,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }


class MarketDataSimulator:
    """
    Simulates market data for testing.
    In a real implementation, this would be replaced with live market data.
    """
    
    def __init__(self):
        """Initialize with some sample data."""
        self.prices = {}
        self.options_chain = {}
        
        # Generate some sample price data for testing
        for symbol in Config.WATCHLIST:
            base_price = {
                "SPY": 450.0, "QQQ": 380.0, "AAPL": 175.0, "MSFT": 390.0,
                "TSLA": 180.0, "AMZN": 160.0, "NVDA": 920.0, "META": 470.0
            }.get(symbol, 100.0)
            
            # Add some random noise to the price
            self.prices[symbol] = base_price * (1 + np.random.normal(0, 0.005))
            
            # Generate options chain
            self._generate_options_chain(symbol, self.prices[symbol])
    
    def _generate_options_chain(self, symbol: str, price: float) -> None:
        """Generate a simulated options chain for the given symbol."""
        today = datetime.date.today()
        
        # Generate expirations (today, tomorrow, this Friday, next Friday)
        expirations = [
            today,
            today + datetime.timedelta(days=1),
            today + datetime.timedelta(days=(4 - today.weekday()) % 7),  # This Friday
            today + datetime.timedelta(days=(4 - today.weekday()) % 7 + 7)  # Next Friday
        ]
        
        # Generate strikes around the current price
        strike_pct_range = np.arange(-0.10, 0.11, 0.01)  # -10% to +10% in 1% steps
        strikes = [round(price * (1 + pct) / 5) * 5 for pct in strike_pct_range]  # Round to nearest $5
        
        chain = {}
        for exp in expirations:
            exp_str = exp.strftime("%Y-%m-%d")
            chain[exp_str] = {"calls": {}, "puts": {}}
            
            days_to_exp = (exp - today).days
            if days_to_exp < 0:
                continue  # Skip expired options
            
            for strike in strikes:
                # Calculate simulated option prices
                time_factor = max(0.01, days_to_exp / 365.0)
                itm_call = max(0, price - strike)
                itm_put = max(0, strike - price)
                
                # Very simplified option pricing
                iv = 0.3 + abs(strike - price) / price  # Higher IV for OTM options
                
                # Calculate approximate delta
                call_delta = 0.5 + 0.5 * (price - strike) / (price * iv * np.sqrt(time_factor))
                call_delta = max(0.01, min(0.99, call_delta))
                put_delta = call_delta - 1.0
                
                # Option prices
                call_price = max(0.05, itm_call + price * iv * np.sqrt(time_factor) * 0.4)
                put_price = max(0.05, itm_put + price * iv * np.sqrt(time_factor) * 0.4)
                
                # Add random volume and open interest
                call_volume = int(np.random.exponential(500) * (1.5 - abs(call_delta - 0.5)))
                put_volume = int(np.random.exponential(500) * (1.5 - abs(put_delta + 0.5)))
                
                call_oi = int(call_volume * np.random.uniform(2, 5))
                put_oi = int(put_volume * np.random.uniform(2, 5))
                
                # Create option data
                chain[exp_str]["calls"][strike] = {
                    "bid": round(call_price * 0.95, 2),
                    "ask": round(call_price * 1.05, 2),
                    "last": round(call_price, 2),
                    "volume": call_volume,
                    "open_interest": call_oi,
                    "iv": round(iv, 2),
                    "delta": round(call_delta, 2),
                    "gamma": round(0.05 * np.exp(-(strike - price)**2 / (price * 0.1)**2), 2),
                    "theta": round(-call_price * 0.1 / max(1, days_to_exp), 2),
                    "vega": round(price * np.sqrt(time_factor) * 0.1, 2)
                }
                
                chain[exp_str]["puts"][strike] = {
                    "bid": round(put_price * 0.95, 2),
                    "ask": round(put_price * 1.05, 2),
                    "last": round(put_price, 2),
                    "volume": put_volume,
                    "open_interest": put_oi,
                    "iv": round(iv, 2),
                    "delta": round(put_delta, 2),
                    "gamma": round(0.05 * np.exp(-(strike - price)**2 / (price * 0.1)**2), 2),
                    "theta": round(-put_price * 0.1 / max(1, days_to_exp), 2),
                    "vega": round(price * np.sqrt(time_factor) * 0.1, 2)
                }
        
        self.options_chain[symbol] = chain
    
    def get_price(self, symbol: str) -> float:
        """Get the current price for a symbol."""
        if symbol not in self.prices:
            return 100.0  # Default price
        
        # Add some random noise to simulate price movement
        self.prices[symbol] *= (1 + np.random.normal(0, 0.001))
        return self.prices[symbol]
    
    def get_options_chain(self, symbol: str) -> Dict:
        """Get the options chain for a symbol."""
        if symbol not in self.options_chain:
            self._generate_options_chain(symbol, 100.0)
        return self.options_chain[symbol]
    
    def get_technical_indicators(self, symbol: str) -> Dict:
        """Get technical indicators for a symbol."""
        # In simulation mode, create indicators that will trigger signals
        if np.random.random() < 0.5:  # 50% chance of call signal
            return {
                "rsi": 25.0,  # Oversold
                "macd": -0.5,
                "bollinger_pct": -0.8,
                "volume_ratio": 1.8,
                "price_change_1d": -0.01  # Small downtrend
            }
        else:  # 50% chance of put signal
            return {
                "rsi": 75.0,  # Overbought
                "macd": 0.5,
                "bollinger_pct": 0.8,
                "volume_ratio": 1.8,
                "price_change_1d": 0.01  # Small uptrend
            }


class OptionsDayTrader:
    """
    Options day trading system that generates entry and exit signals.
    SIMULATION MODE - Always finds opportunities
    """
    
    def __init__(self, account_value: float = 25000.0):
        """
        Initialize the day trading system.
        
        Args:
            account_value: Total account value for position sizing
        """
        self.initial_account_value = account_value
        self.account_value = account_value
        self.market_data = MarketDataSimulator()
        self.active_signals = []
        self.positions = []
        self.trade_history = []
        
    def scan_for_signals(self, force_symbol: str = None) -> List[OptionSignal]:
        """
        Scan the market for options day trading signals.
        In simulation mode, always returns signals.
        
        Args:
            force_symbol: Force a signal for this specific symbol
            
        Returns:
            List of option trading signals
        """
        signals = []
        
        # In simulation mode, we'll generate at least one signal per scan
        symbols_to_check = [force_symbol] if force_symbol else Config.WATCHLIST
        
        for symbol in symbols_to_check:
            # Get current price and technical indicators
            price = self.market_data.get_price(symbol)
            technicals = self.market_data.get_technical_indicators(symbol)
            
            # In simulation mode, we'll alternate between calls and puts
            # based on the technical indicators from the simulator
            if technicals["rsi"] < 30:
                direction = "call"
                signal_strength = np.random.uniform(0.6, 0.9)  # High confidence signal
            else:
                direction = "put"
                signal_strength = np.random.uniform(0.6, 0.9)  # High confidence signal
            
            # Find the best option
            signal = self._find_best_option(symbol, price, direction, signal_strength)
            if signal:
                signals.append(signal)
                
                # In simulation mode, we'll just return one signal to keep it simple
                if not force_symbol:
                    break
        
        return signals
    
    def _find_best_option(self, symbol: str, price: float, option_type: str, signal_strength: float) -> OptionSignal:
        """
        Find the best option for day trading based on criteria.
        
        Args:
            symbol: Stock symbol
            price: Current stock price
            option_type: 'call' or 'put'
            signal_strength: Signal strength (0-1)
            
        Returns:
            Option signal or None if no suitable option found
        """
        # Get options chain
        chain = self.market_data.get_options_chain(symbol)
        
        # Find nearest expiration date (0-1 DTE for day trading)
        today = datetime.date.today()
        target_expiry = today + datetime.timedelta(days=Config.DAYS_TO_EXPIRATION)
        
        best_option = None
        best_score = 0
        
        # Look for the nearest expiration date
        for exp_str, exp_data in chain.items():
            expiry = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
            
            # Skip if too far in the future (want 0-1 DTE for day trading)
            days_out = (expiry - today).days
            if days_out > Config.DAYS_TO_EXPIRATION + 1:
                continue
            
            # Get the appropriate options (calls or puts)
            options = exp_data["calls"] if option_type == "call" else exp_data["puts"]
            
            for strike, option in options.items():
                # In simulation mode, we'll relax our criteria to ensure we get a signal
                if option["volume"] < 100:  # Relaxed from 500
                    continue
                    
                spread_pct = (option["ask"] - option["bid"]) / option["ask"]
                if spread_pct > 0.20:  # Relaxed from 0.10
                    continue
                
                delta = abs(option["delta"])
                if delta < 0.10 or delta > 0.40:  # Relaxed criteria
                    continue
                
                # Calculate a score for this option
                score = (
                    option["volume"] / 1000 * 0.3 +  # Higher volume is better
                    (1 - spread_pct) * 0.3 +         # Tighter spread is better
                    (1 - abs(delta - 0.25)) * 0.4    # Target delta around 0.25
                )
                
                if score > best_score:
                    best_score = score
                    
                    # Calculate entry, stop loss and target prices
                    mid_price = (option["bid"] + option["ask"]) / 2
                    entry_low = option["bid"]
                    entry_high = min(mid_price * 1.02, option["ask"])  # Don't pay full ask
                    
                    stop_loss = mid_price * (1 - Config.STOP_LOSS_PCT)
                    target = mid_price * (1 + Config.PROFIT_TARGET_PCT)
                    
                    best_option = OptionSignal(
                        symbol=symbol,
                        option_type=option_type,
                        strike=strike,
                        expiration=expiry,
                        current_price=mid_price,
                        underlying_price=price,
                        entry_price_range=(entry_low, entry_high),
                        stop_loss=stop_loss,
                        target_price=target,
                        signal_strength=signal_strength,
                        volume=option["volume"],
                        open_interest=option["open_interest"],
                        iv=option["iv"],
                        delta=option["delta"]
                    )
        
        return best_option
    
    def get_position_size(self, option_price: float) -> int:
        """
        Calculate the number of contracts to trade.
        
        Args:
            option_price: Current option price
            
        Returns:
            Number of contracts to trade
        """
        max_position_value = self.account_value * Config.POSITION_SIZE_PCT
        contract_value = option_price * 100  # Each option contract is for 100 shares
        
        # Calculate number of contracts (round down to be conservative)
        num_contracts = int(max_position_value / contract_value)
        
        # Ensure at least 1 contract but not too many
        return max(1, min(10, num_contracts))
    
    def print_alert(self, signal: OptionSignal) -> None:
        """
        Print a formatted alert for the given signal.
        
        Args:
            signal: Option signal to alert
        """
        symbol_str = f"{signal.symbol} ${signal.underlying_price:.2f}"
        contract_str = f"{signal.option_type.upper()} ${signal.strike} {signal.expiration.strftime('%m/%d')}"
        entry_str = f"${signal.entry_price_range[0]:.2f}-${signal.entry_price_range[1]:.2f}"
        stop_str = f"${signal.stop_loss:.2f}"
        target_str = f"${signal.target_price:.2f}"
        
        # Calculate potential returns
        avg_entry = (signal.entry_price_range[0] + signal.entry_price_range[1]) / 2
        stop_pct = (signal.stop_loss - avg_entry) / avg_entry * 100
        target_pct = (signal.target_price - avg_entry) / avg_entry * 100
        
        risk_reward = abs(target_pct / stop_pct) if stop_pct != 0 else 0
        
        position_size = self.get_position_size(avg_entry)
        max_risk = position_size * 100 * (avg_entry - signal.stop_loss)
        max_profit = position_size * 100 * (signal.target_price - avg_entry)
        
        print("\n" + "=" * 80)
        print(f"🔔 OPTION ALERT: {symbol_str} - {contract_str}")
        print("-" * 80)
        print(f"ACTION:      BUY TO OPEN")
        print(f"CONTRACTS:   {position_size} (${position_size * avg_entry * 100:.2f})")
        print(f"ENTRY PRICE: {entry_str}")
        print(f"STOP LOSS:   {stop_str} ({stop_pct:.1f}%) - Max Risk: ${max_risk:.2f}")
        print(f"TARGET:      {target_str} ({target_pct:.1f}%) - Max Profit: ${max_profit:.2f}")
        print(f"RISK/REWARD: 1:{risk_reward:.2f}")
        print(f"EXPIRES:     {signal.expiration.strftime('%A, %B %d, %Y')} ({(signal.expiration - datetime.date.today()).days} days)")
        print(f"SIGNAL STRENGTH: {signal.signal_strength:.2f}/1.0")
        print("-" * 80)
        print(f"VOLUME: {signal.volume} | OPEN INT: {signal.open_interest} | IV: {signal.iv:.1f}% | DELTA: {signal.delta:.2f}")
        print("=" * 80)

    def simulate_trade(self, signal: OptionSignal, hold_minutes: int = None) -> Dict:
        """
        Simulate a trade based on the signal.
        
        Args:
            signal: The option signal to trade
            hold_minutes: How long to hold the position in minutes
            
        Returns:
            Dictionary with trade results
        """
        # Use config values if not specified
        if hold_minutes is None:
            hold_minutes = Config.MAX_HOLD_MINUTES
            
        print(f"\n🕐 SIMULATING TRADE FOR {signal.symbol} {signal.option_type.upper()} ${signal.strike}...")
        
        # Entry price is the midpoint of the entry range
        entry_price = (signal.entry_price_range[0] + signal.entry_price_range[1]) / 2
        position_size = self.get_position_size(entry_price)
        
        # Randomly determine outcome with a bias toward profitable trades (55/45)
        is_profitable = np.random.random() < 0.55
        
        # Time to outcome is random but shorter (ultra short-term trading)
        outcome_minutes = np.random.randint(5, min(hold_minutes, Config.MAX_HOLD_MINUTES))
        outcome_time = datetime.datetime.now() + datetime.timedelta(minutes=outcome_minutes)
        
        # Determine outcome price - more realistic for short-term option trades
        if is_profitable:
            # Hit target or somewhere between entry and target
            outcome_pct = np.random.uniform(0.03, 1.0) * Config.PROFIT_TARGET_PCT
            outcome_price = entry_price * (1 + outcome_pct)
            outcome_type = "TARGET HIT" if outcome_pct >= Config.PROFIT_TARGET_PCT * 0.95 else "PARTIAL PROFIT"
        else:
            # Hit stop loss or somewhere between entry and stop
            outcome_pct = np.random.uniform(0.3, 1.0) * -Config.STOP_LOSS_PCT
            outcome_price = entry_price * (1 + outcome_pct)
            outcome_type = "STOP LOSS HIT" if outcome_pct <= -Config.STOP_LOSS_PCT * 0.95 else "PARTIAL LOSS"
        
        # Calculate profit/loss
        price_change = outcome_price - entry_price
        price_change_pct = price_change / entry_price * 100
        dollar_profit = price_change * position_size * 100  # 100 shares per contract
        
        # Calculate commissions (simulate $1 per contract)
        commission = position_size * 1.0 * 2  # $1 per contract, entry and exit
        net_profit = dollar_profit - commission
        
        # Update account value
        previous_account_value = self.account_value
        self.account_value += net_profit
        
        # Print the outcome
        print("\n" + "=" * 80)
        print(f"TRADE OUTCOME: {signal.symbol} {signal.option_type.upper()} ${signal.strike}")
        print("-" * 80)
        print(f"ENTRY TIME:  {signal.timestamp.strftime('%H:%M:%S')}")
        print(f"ENTRY PRICE: ${entry_price:.2f} x {position_size} contracts")
        print(f"POSITION SIZE: ${(entry_price * position_size * 100):.2f} ({(position_size * entry_price * 100 / previous_account_value * 100):.1f}% of account)")
        print(f"EXIT TIME:   {outcome_time.strftime('%H:%M:%S')} ({outcome_minutes} minutes later)")
        print(f"EXIT PRICE:  ${outcome_price:.2f} ({price_change_pct:.1f}%)")
        print(f"OUTCOME:     {outcome_type}")
        print(f"COMMISSION:  ${commission:.2f}")
        print(f"NET P&L:     ${net_profit:.2f}")
        if net_profit > 0:
            profit_per_minute = net_profit / outcome_minutes
            print(f"PROFIT RATE: ${profit_per_minute:.2f} per minute")
        
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
            "entry_time": signal.timestamp,
            "entry_price": entry_price,
            "position_size": position_size,
            "exit_time": outcome_time,
            "exit_price": outcome_price,
            "hold_time_minutes": outcome_minutes,
            "price_change_pct": price_change_pct,
            "commission": commission,
            "gross_profit": dollar_profit,
            "net_profit": net_profit,
            "outcome_type": outcome_type,
            "previous_account": previous_account_value,
            "new_account": self.account_value
        }
        
        self.trade_history.append(result)
        return result


def simulate_specific_symbols(symbols: List[str] = None):
    """
    Simulate trades for specific symbols.
    
    Args:
        symbols: List of symbols to simulate trades for. If None, use a default set.
    """
    symbols = symbols or ["SPY", "AAPL", "NVDA"]
    trader = OptionsDayTrader(account_value=25000.0)
    
    # Simulate market open time
    market_open_time = datetime.datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    if market_open_time > datetime.datetime.now():
        market_open_time -= datetime.timedelta(days=1)  # Use yesterday if before market open
    
    print(f"\n === MARKET OPEN SIMULATION - {market_open_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    print(f"Simulating day trading opportunities for: {', '.join(symbols)}")
    print(f"Starting account: ${trader.account_value:.2f}")
    
    all_results = []
    time_elapsed = 0  # Track simulated minutes elapsed
    
    for symbol in symbols:
        print(f"\n--- Scanning {symbol} for trading opportunities ---")
        
        # Force the scanner to find opportunities for this symbol
        signals = trader.scan_for_signals(force_symbol=symbol)
        
        if signals:
            for signal in signals:
                # Update signal timestamp to reflect market time + elapsed time
                current_time = market_open_time + datetime.timedelta(minutes=time_elapsed)
                signal.timestamp = current_time
                
                trader.print_alert(signal)
                
                # Ask if user wants to simulate this trade
                choice = input("\nDo you want to simulate this trade? (y/n): ").lower()
                if choice.startswith('y'):
                    result = trader.simulate_trade(signal)
                    all_results.append(result)
                    time_elapsed += result["hold_time_minutes"] + 5  # Add time to find next trade
        else:
            print(f"No trading opportunities found for {symbol}")
    
    # Print summary
    if all_results:
        total_profit = sum(r["net_profit"] for r in all_results)
        win_count = sum(1 for r in all_results if r["net_profit"] > 0)
        loss_count = len(all_results) - win_count
        win_rate = win_count / len(all_results) * 100 if all_results else 0
        
        # Calculate metrics
        total_commissions = sum(r["commission"] for r in all_results)
        average_profit_per_trade = total_profit / len(all_results) if all_results else 0
        average_hold_time = sum(r["hold_time_minutes"] for r in all_results) / len(all_results) if all_results else 0
        profit_per_minute = total_profit / sum(r["hold_time_minutes"] for r in all_results) if sum(r["hold_time_minutes"] for r in all_results) > 0 else 0
        
        print("\n" + "=" * 80)
        print(f"TRADING SESSION SUMMARY:")
        print("-" * 80)
        print(f"Total Trades:       {len(all_results)}")
        print(f"Winning Trades:     {win_count} ({win_rate:.1f}%)")
        print(f"Losing Trades:      {loss_count}")
        print(f"Total Net P&L:      ${total_profit:.2f}")
        print(f"Total Commissions:  ${total_commissions:.2f}")
        print(f"Avg Profit/Trade:   ${average_profit_per_trade:.2f}")
        print(f"Avg Hold Time:      {average_hold_time:.1f} minutes")
        print(f"Profit Per Minute:  ${profit_per_minute:.2f}")
        print("-" * 80)
        print(f"Starting Balance:   ${trader.initial_account_value:.2f}")
        print(f"Ending Balance:     ${trader.account_value:.2f}")
        account_change_pct = (trader.account_value / trader.initial_account_value - 1) * 100
        print(f"Account Change:     {account_change_pct:+.2f}%")
        print(f"Simulated Time:     {time_elapsed} minutes ({time_elapsed/60:.1f} hours)")
        print("=" * 80)


def run_scanner(update_interval: int = 60, num_alerts: int = 5):
    """
    Run the scanner continuously, generating alerts at specified intervals.
    
    Args:
        update_interval: Seconds between updates
        num_alerts: Number of alerts to generate (for testing)
    """
    trader = OptionsDayTrader()
    
    print(f"Starting Options Day Trading Scanner (SIMULATION MODE)...")
    print(f"Scanning {len(Config.WATCHLIST)} symbols every {update_interval} seconds")
    print(f"Press Ctrl+C to exit")
    
    alerts_generated = 0
    try:
        while alerts_generated < num_alerts:
            print(f"\nScanning market at {datetime.datetime.now().strftime('%H:%M:%S')}...")
            
            # Scan for new signals
            signals = trader.scan_for_signals()
            
            if signals:
                alerts_generated += len(signals)
                for signal in signals:
                    trader.print_alert(signal)
                    
                    # In simulation mode, also show a simulated trade outcome
                    choice = input("\nDo you want to simulate this trade? (y/n): ").lower()
                    if choice.startswith('y'):
                        trader.simulate_trade(signal)
            else:
                print("No signals detected in this scan.")
            
            if alerts_generated < num_alerts:
                print(f"Waiting {update_interval} seconds for next scan...")
                time.sleep(update_interval)
        
        print("\nScan complete. Generated requested number of alerts.")
        
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options Day Trading Scanner (Simulation Mode)")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in seconds")
    parser.add_argument("--alerts", type=int, default=5, help="Number of alerts to generate")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to simulate")
    parser.add_argument("--mode", choices=["scan", "symbols"], default="scan", 
                        help="Mode: 'scan' for continuous scanning, 'symbols' for specific symbols")
    args = parser.parse_args()
    
    if args.mode == "symbols":
        symbols = args.symbols.split(",") if args.symbols else None
        simulate_specific_symbols(symbols)
    else:
        run_scanner(args.interval, args.alerts) 