#!/usr/bin/env python
"""
Alpaca Markets Integration

This script integrates Alpaca Market Data with the options trading system,
allowing the system to use real market data instead of simulated data.
"""

import os
import datetime
import sys
from typing import Dict, List

# Import Alpaca market data provider
from alpaca_market_data import AlpacaMarketData

# Import your trading system components
from options_day_trader_sim import OptionSignal, Config
from src.data.market_data import OptionChain, OptionQuote, OptionRight

# Import the API credentials
from alpaca_config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, load_credentials

class AlpacaIntegration:
    """
    Integrates Alpaca Market Data with the options trading system.
    """
    
    def __init__(self, api_key=None, api_secret=None, base_url=None):
        """Initialize the integration with API credentials."""
        # Set default API keys if provided, otherwise use alpaca_config values
        if api_key is None:
            api_key = ALPACA_API_KEY
        if api_secret is None:
            api_secret = ALPACA_API_SECRET
        if base_url is None:
            base_url = ALPACA_API_BASE_URL
            
        # Set environment variables with these credentials
        os.environ["ALPACA_API_KEY"] = api_key
        os.environ["ALPACA_API_SECRET"] = api_secret
        os.environ["ALPACA_API_BASE_URL"] = base_url
            
        # Create the Alpaca market data provider
        self.alpaca = AlpacaMarketData()
        
        print("Alpaca integration initialized")
    
    def get_option_signals(self, symbols: List[str], max_signals: int = 5) -> List[OptionSignal]:
        """
        Generate option trading signals using real market data.
        
        Args:
            symbols: List of symbols to scan for options
            max_signals: Maximum number of signals to return
            
        Returns:
            List of option trading signals
        """
        signals = []
        
        for symbol in symbols:
            print(f"Scanning {symbol} for option opportunities...")
            
            # Get current price
            underlying_price = self.alpaca.get_price(symbol)
            if not underlying_price or underlying_price <= 0:
                print(f"Could not get price for {symbol}, skipping")
                continue
                
            # Get technical indicators
            indicators = self.alpaca.get_technical_indicators(symbol)
            
            # Get options chain
            chain = self.alpaca.get_options_chain(symbol)
            if not chain:
                print(f"No options data available for {symbol}, skipping")
                continue
            
            # Process the chain to find signals
            for expiration_date, exp_data in chain.items():
                # Check if expiration is within our target range (e.g., 5 days)
                exp_date = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
                days_to_exp = (exp_date - datetime.datetime.now().date()).days
                
                if days_to_exp > Config.DAYS_TO_EXPIRATION:
                    continue
                
                # Analyze calls and puts
                for option_type, options in [("call", exp_data["calls"]), ("put", exp_data["puts"])]:
                    for strike_str, option_data in options.items():
                        strike = float(strike_str)
                        
                        # Skip if no valid prices
                        if option_data["bid"] <= 0 or option_data["ask"] <= 0:
                            continue
                            
                        # Calculate price and ranges
                        price = (option_data["bid"] + option_data["ask"]) / 2
                        entry_range = (option_data["bid"], option_data["ask"])
                        
                        # Calculate a signal strength based on various factors
                        signal_strength = 0.5  # Base score
                        
                        # Adjust based on RSI (example: prefer oversold for calls, overbought for puts)
                        rsi = indicators["rsi"]
                        if option_type == "call" and rsi < 40:
                            signal_strength += 0.1
                        elif option_type == "put" and rsi > 60:
                            signal_strength += 0.1
                            
                        # Adjust based on recent price movement
                        price_change = indicators["price_change_1d"]
                        if (option_type == "call" and price_change < -0.01) or \
                           (option_type == "put" and price_change > 0.01):
                            signal_strength += 0.1
                            
                        # Adjust based on implied volatility
                        iv = option_data["iv"]
                        if 20 <= iv <= 50:  # Moderate IV
                            signal_strength += 0.1
                            
                        # Create the signal
                        signal = OptionSignal(
                            symbol=symbol,
                            option_type=option_type,
                            strike=strike,
                            expiration=expiration_date,
                            current_price=price,
                            underlying_price=underlying_price,
                            entry_price_range=entry_range,
                            stop_loss=price * (1 - Config.STOP_LOSS_PCT),
                            target_price=price * (1 + Config.PROFIT_TARGET_PCT),
                            signal_strength=signal_strength,
                            volume=option_data.get("volume", 0),
                            open_interest=option_data.get("open_interest", 0),
                            iv=option_data.get("iv", 30.0),
                            delta=option_data.get("delta", 0.5 if option_type == "call" else -0.5)
                        )
                        
                        signals.append(signal)
        
        # Sort by signal strength and return top signals
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        return signals[:max_signals]
    
    def convert_to_option_chain(self, symbol: str, alpaca_chain: Dict) -> OptionChain:
        """
        Convert Alpaca options chain to the internal OptionChain format.
        
        Args:
            symbol: The underlying symbol
            alpaca_chain: Alpaca options chain data
            
        Returns:
            OptionChain in the internal format
        """
        # Create a new option chain
        chain = OptionChain(
            underlying_symbol=symbol,
            timestamp=datetime.datetime.now()
        )
        
        # Process each expiration date
        for exp_date_str, exp_data in alpaca_chain.items():
            exp_date = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d")
            
            # Process calls
            for strike_str, call_data in exp_data["calls"].items():
                strike = float(strike_str)
                
                # Create OptionQuote
                call_quote = OptionQuote(
                    timestamp=datetime.datetime.now(),
                    bid_price=call_data["bid"],
                    ask_price=call_data["ask"],
                    bid_size=10,  # Default since Alpaca doesn't provide sizes
                    ask_size=10,  # Default since Alpaca doesn't provide sizes
                    exchange="ALPACA",
                    underlying_symbol=symbol,
                    expiration=exp_date,
                    strike=strike,
                    right=OptionRight.CALL,
                    last_price=call_data.get("last", (call_data["bid"] + call_data["ask"]) / 2),
                    volume=call_data.get("volume", 0),
                    open_interest=call_data.get("open_interest", 0),
                    implied_volatility=call_data.get("iv", 30.0) / 100,  # Convert % to decimal
                    delta=call_data.get("delta", 0.5),
                    gamma=call_data.get("gamma", 0.01),
                    theta=call_data.get("theta", -0.01),
                    vega=call_data.get("vega", 0.1),
                    rho=call_data.get("rho", 0.01)
                )
                
                # Add to chain
                chain.add_option(call_quote)
            
            # Process puts
            for strike_str, put_data in exp_data["puts"].items():
                strike = float(strike_str)
                
                # Create OptionQuote
                put_quote = OptionQuote(
                    timestamp=datetime.datetime.now(),
                    bid_price=put_data["bid"],
                    ask_price=put_data["ask"],
                    bid_size=10,  # Default since Alpaca doesn't provide sizes
                    ask_size=10,  # Default since Alpaca doesn't provide sizes
                    exchange="ALPACA",
                    underlying_symbol=symbol,
                    expiration=exp_date,
                    strike=strike,
                    right=OptionRight.PUT,
                    last_price=put_data.get("last", (put_data["bid"] + put_data["ask"]) / 2),
                    volume=put_data.get("volume", 0),
                    open_interest=put_data.get("open_interest", 0),
                    implied_volatility=put_data.get("iv", 30.0) / 100,  # Convert % to decimal
                    delta=put_data.get("delta", -0.5),
                    gamma=put_data.get("gamma", 0.01),
                    theta=put_data.get("theta", -0.01),
                    vega=put_data.get("vega", 0.1),
                    rho=put_data.get("rho", -0.01)
                )
                
                # Add to chain
                chain.add_option(put_quote)
        
        return chain

if __name__ == "__main__":
    # Load credentials from config
    load_credentials()
    
    # Create the integration
    integration = AlpacaIntegration()
    
    # Get some signals
    symbols = ["SPY", "AAPL", "MSFT", "QQQ", "AMZN"]
    signals = integration.get_option_signals(symbols, max_signals=10)
    
    # Display results
    print(f"\nFound {len(signals)} trading signals:")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  {signal.symbol} {signal.option_type.upper()} ${signal.strike} expiring {signal.expiration}")
        print(f"  Current price: ${signal.current_price:.2f}")
        print(f"  Entry range: ${signal.entry_price_range[0]:.2f} - ${signal.entry_price_range[1]:.2f}")
        print(f"  Stop loss: ${signal.stop_loss:.2f}")
        print(f"  Target price: ${signal.target_price:.2f}")
        print(f"  Signal strength: {signal.signal_strength:.2f}") 