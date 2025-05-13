#!/usr/bin/env python
"""
Options Day Trading Alert System

This script scans for potential day trading options opportunities
and provides clear entry, exit, and risk management signals.
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
    DAYS_TO_EXPIRATION = 1     # For day trading, use 0-1 DTE
    MAX_DELTA = 0.30           # Maximum option delta (closer to ATM)
    MIN_DELTA = 0.15           # Minimum option delta (not too far OTM)
    MIN_VOLUME = 500           # Minimum option volume
    MAX_SPREAD_PCT = 0.10      # Maximum bid-ask spread as % of option price
    
    # Risk management
    POSITION_SIZE_PCT = 0.02   # Maximum % of account per position
    STOP_LOSS_PCT = 0.30       # Stop loss at 30% of option premium
    PROFIT_TARGET_PCT = 0.50   # Take profit at 50% of option premium


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
        # In a real implementation, this would calculate actual indicators
        return {
            "rsi": np.random.uniform(20, 80),
            "macd": np.random.normal(0, 1),
            "bollinger_pct": np.random.uniform(-1, 1),
            "volume_ratio": np.random.uniform(0.5, 2.0),
            "price_change_1d": np.random.normal(0, 0.02)
        }


class OptionsDayTrader:
    """
    Options day trading system that generates entry and exit signals.
    """
    
    def __init__(self, account_value: float = 25000.0):
        """
        Initialize the day trading system.
        
        Args:
            account_value: Total account value for position sizing
        """
        self.account_value = account_value
        self.market_data = MarketDataSimulator()
        self.active_signals = []
        self.positions = []
        
    def scan_for_signals(self) -> List[OptionSignal]:
        """
        Scan the market for options day trading signals.
        
        Returns:
            List of option trading signals
        """
        signals = []
        
        # Check each symbol in the watchlist
        for symbol in Config.WATCHLIST:
            # Get current price and technical indicators
            price = self.market_data.get_price(symbol)
            technicals = self.market_data.get_technical_indicators(symbol)
            
            # Decide on direction (calls or puts)
            if technicals["rsi"] < Config.RSI_OVERSOLD and technicals["price_change_1d"] < -Config.MOMENTUM_THRESHOLD/100:
                # Oversold and downtrending - look for calls (bullish reversal)
                direction = "call"
                signal_strength = (Config.RSI_OVERSOLD - technicals["rsi"]) / 15 + abs(technicals["price_change_1d"]) * 10
                signal_strength = min(1.0, signal_strength)
            elif technicals["rsi"] > Config.RSI_OVERBOUGHT and technicals["price_change_1d"] > Config.MOMENTUM_THRESHOLD/100:
                # Overbought and uptrending - look for puts (bearish reversal)
                direction = "put"
                signal_strength = (technicals["rsi"] - Config.RSI_OVERBOUGHT) / 15 + abs(technicals["price_change_1d"]) * 10
                signal_strength = min(1.0, signal_strength)
            else:
                # No clear signal
                continue
            
            # If we have a directional bias, find the best option
            if signal_strength > 0.3:  # Minimum threshold for signal quality
                signal = self._find_best_option(symbol, price, direction, signal_strength)
                if signal:
                    signals.append(signal)
        
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
                # Check if option meets our criteria
                if option["volume"] < Config.MIN_VOLUME:
                    continue
                    
                spread_pct = (option["ask"] - option["bid"]) / option["ask"]
                if spread_pct > Config.MAX_SPREAD_PCT:
                    continue
                
                delta = abs(option["delta"])
                if delta < Config.MIN_DELTA or delta > Config.MAX_DELTA:
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


def run_scanner(update_interval: int = 60, num_alerts: int = 5):
    """
    Run the scanner continuously, generating alerts at specified intervals.
    
    Args:
        update_interval: Seconds between updates
        num_alerts: Number of alerts to generate (for testing)
    """
    trader = OptionsDayTrader()
    
    print(f"Starting Options Day Trading Scanner...")
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
            else:
                print("No signals detected in this scan.")
            
            if alerts_generated < num_alerts:
                print(f"Waiting {update_interval} seconds for next scan...")
                time.sleep(update_interval)
        
        print("\nScan complete. Generated requested number of alerts.")
        
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options Day Trading Scanner")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in seconds")
    parser.add_argument("--alerts", type=int, default=5, help="Number of alerts to generate (for testing)")
    args = parser.parse_args()
    
    run_scanner(args.interval, args.alerts) 