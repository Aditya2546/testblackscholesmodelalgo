#!/usr/bin/env python
"""
Market Data API Integration for options trading system
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from market_data_api import MarketDataAPI
from market_data_config import load_market_data_credentials

@dataclass
class TradingSignal:
    """Data class for trading signals generated from option analysis"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: str
    current_price: float
    entry_price_range: Tuple[float, float]
    target_price: float
    stop_loss: float
    signal_strength: float  # 0.0 to 1.0
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    underlying_price: float
    time_to_expiry: float  # in days
    premium_to_strike_ratio: float
    volume: int = 0
    open_interest: int = 0

class MarketDataIntegration:
    """
    Integration class for Market Data API to obtain options data and generate trading signals
    """
    
    def __init__(self):
        """Initialize the Market Data API integration"""
        credentials = load_market_data_credentials()
        self.api = MarketDataAPI(api_token=credentials["token"])
        
    def get_option_data(self, option_symbol: str) -> Dict:
        """
        Get option data for a specific option contract
        
        Args:
            option_symbol: The OCC option symbol (e.g., AAPL250117C00150000)
            
        Returns:
            Dictionary with option data
        """
        return self.api.get_option_quote(option_symbol)
    
    def get_option_history(self, option_symbol: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get historical option data
        
        Args:
            option_symbol: The OCC option symbol
            days_back: Number of days of historical data to retrieve
            
        Returns:
            DataFrame with historical option data
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        return self.api.get_option_quote_as_dataframe(
            option_symbol=option_symbol,
            from_date=start_date,
            to_date=end_date
        )
    
    def calculate_time_to_expiry(self, expiration_date: str) -> float:
        """
        Calculate time to expiry in days
        
        Args:
            expiration_date: Option expiration date (YYYY-MM-DD)
            
        Returns:
            Time to expiry in days
        """
        expiry = datetime.strptime(expiration_date, "%Y-%m-%d")
        today = datetime.now()
        return (expiry - today).days
    
    def get_expiration_dates(self, ticker: str) -> List[str]:
        """
        Get available expiration dates for options on a given ticker
        
        Note: This is a placeholder. The Market Data API doesn't have a direct endpoint 
        for expiration dates, so this would need to be implemented with a different API
        or hard-coded for demo purposes.
        
        Args:
            ticker: The stock ticker symbol
            
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        # For demo purposes, return some common expiration dates
        # In a real implementation, this would fetch actual available expirations
        today = datetime.now()
        
        # Generate some future expiration dates (3rd Friday of next months)
        expirations = []
        for i in range(1, 13):  # Next 12 months
            year = today.year + ((today.month + i - 1) // 12)
            month = ((today.month + i - 1) % 12) + 1
            
            # Find the third Friday of the month
            day = 1
            first_day = datetime(year, month, day)
            # Find the first Friday
            while first_day.weekday() != 4:  # 4 is Friday
                day += 1
                first_day = datetime(year, month, day)
            
            # Add two weeks to get the third Friday
            third_friday = first_day + timedelta(days=14)
            expirations.append(third_friday.strftime("%Y-%m-%d"))
        
        return expirations
    
    def extract_option_data(self, option_data: Dict) -> Dict:
        """
        Extract option data from the API response, which may be in array format
        
        Args:
            option_data: Dictionary with option data from API
            
        Returns:
            Dictionary with extracted option data
        """
        result = {}
        
        # Check if data is in array format
        if "bid" in option_data and isinstance(option_data["bid"], list):
            # Extract the first item from each array
            for key, value in option_data.items():
                if isinstance(value, list) and len(value) > 0:
                    result[key] = value[0]
            
            # Map specific field names from API to our expected format
            field_mapping = {
                "underlyingPrice": "underlying_price",
                "openInterest": "open_interest",
            }
            
            for api_field, our_field in field_mapping.items():
                if api_field in result:
                    result[our_field] = result[api_field]
        else:
            # Data is already in the expected format
            result = option_data
        
        return result
    
    def analyze_option(self, option_data: Dict) -> Dict:
        """
        Analyze an option contract to determine if it's a good trading opportunity
        
        Args:
            option_data: Dictionary with option data
            
        Returns:
            Dictionary with analysis results
        """
        # Extract data, handling both array and direct formats
        data = self.extract_option_data(option_data)
        
        # Extract relevant data
        iv = data.get("iv", 0)
        delta = data.get("delta", 0)
        gamma = data.get("gamma", 0)
        theta = data.get("theta", 0)
        vega = data.get("vega", 0)
        bid = data.get("bid", 0)
        ask = data.get("ask", 0)
        
        # Basic analysis
        spread = ask - bid
        spread_percentage = spread / ask if ask > 0 else 0
        liquidity_score = 1.0 - spread_percentage  # Higher is better
        
        # Options with extreme IV might be overpriced
        iv_score = 0.5 - abs(iv - 0.3) if iv > 0 else 0
        
        # Options with high delta are more responsive to price changes
        # But we'll favor options around delta 0.5 for balanced risk/reward
        delta_score = 1.0 - abs(abs(delta) - 0.5) if delta != 0 else 0
        
        # Theta decay is negative, so less negative is better
        theta_score = min(1.0, max(0, (theta + 0.01) / 0.01)) if theta < 0 else 0
        
        # Combine scores with weights
        score = (
            liquidity_score * 0.3 +
            iv_score * 0.2 +
            delta_score * 0.3 +
            theta_score * 0.2
        )
        
        return {
            "score": score,
            "liquidity_score": liquidity_score,
            "iv_score": iv_score,
            "delta_score": delta_score,
            "theta_score": theta_score
        }
    
    def get_option_signals(self, symbols: List[str], max_signals: int = 5) -> List[TradingSignal]:
        """
        Generate trading signals for options on the given symbols
        
        Args:
            symbols: List of stock ticker symbols
            max_signals: Maximum number of signals to return
            
        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        # Current market prices estimation (as of May 2025)
        current_prices = {
            "SPY": 550.0,
            "AAPL": 210.0,
            "MSFT": 420.0,
            "QQQ": 450.0,
            "AMZN": 185.0,
            "GOOG": 180.0,
            "TSLA": 190.0,
            "META": 480.0
        }
        
        for symbol in symbols:
            # Get available expiration dates
            expirations = self.get_expiration_dates(symbol)
            
            # Filter to get only expirations within 30-60 days
            target_expirations = []
            for exp in expirations:
                days_to_expiry = self.calculate_time_to_expiry(exp)
                if 30 <= days_to_expiry <= 60:
                    target_expirations.append(exp)
            
            if not target_expirations:
                continue
                
            # For now, just use the first valid expiration
            expiration = target_expirations[0]
            
            # Get the estimated current price for this symbol
            current_price = current_prices.get(symbol, 100.0)  # Default to 100 if not found
            print(f"Processing {symbol} with estimated price: ${current_price}")
            
            # Generate standard strike prices based on the current price
            strikes = self.generate_standard_strikes(symbol, current_price)
            print(f"Generated strikes for {symbol}: {strikes}")
            
            # Analyze options at these strikes
            success_count = 0
            for strike in strikes:
                # Skip after a few successful fetches to prevent API overuse
                if success_count >= 3:
                    break
                    
                for option_type in ["C", "P"]:  # Call and Put
                    try:
                        # Format option symbol
                        option_symbol = self.api.format_option_symbol(
                            ticker=symbol,
                            expiration_date=expiration,
                            strike_price=strike,
                            option_type=option_type
                        )
                        
                        print(f"Checking {option_symbol}...")
                        
                        # Get option data
                        option_data = self.api.get_option_quote(option_symbol)
                        
                        if option_data and option_data.get("s") in ["ok", "error"] and \
                           (("bid" in option_data and isinstance(option_data["bid"], list)) or \
                            ("bid" in option_data and not isinstance(option_data["bid"], list))):
                            
                            # Extract data from potentially array-based response
                            extracted_data = self.extract_option_data(option_data)
                            
                            # Extract data points
                            bid = extracted_data.get("bid", 0)
                            ask = extracted_data.get("ask", 0)
                            
                            # Skip if no valid bid/ask
                            if bid <= 0 or ask <= 0:
                                print(f"Skipping {symbol} {strike} {option_type} - No valid bid/ask")
                                continue
                                
                            mid = extracted_data.get("mid", (bid + ask) / 2 if bid and ask else 0)
                            iv = extracted_data.get("iv", 0)
                            delta = extracted_data.get("delta", 0)
                            gamma = extracted_data.get("gamma", 0)
                            theta = extracted_data.get("theta", 0)
                            vega = extracted_data.get("vega", 0)
                            underlying_price = extracted_data.get("underlying_price", current_price)
                            
                            # Update our price estimation with actual underlying price if available
                            if underlying_price > 0:
                                current_prices[symbol] = underlying_price
                                print(f"Updated {symbol} price to ${underlying_price} based on market data")
                            
                            # Mark as success and update our strike information for next time
                            success_count += 1
                            
                            # Analyze this option
                            analysis = self.analyze_option(option_data)
                            score = analysis["score"]
                            
                            # If score is good enough, create a signal
                            if score > 0.6:  # Threshold for good signals
                                time_to_expiry = self.calculate_time_to_expiry(expiration)
                                premium_to_strike_ratio = mid / strike if strike > 0 else 0
                                
                                # Create trading signal
                                signal = TradingSignal(
                                    symbol=symbol,
                                    option_type="call" if option_type == "C" else "put",
                                    strike=strike,
                                    expiration=expiration,
                                    current_price=mid,
                                    entry_price_range=(bid * 1.02, ask * 0.98),  # Slightly better than market
                                    target_price=mid * 1.5,  # Target 50% gain
                                    stop_loss=mid * 0.7,  # Stop at 30% loss
                                    signal_strength=score,
                                    implied_volatility=iv,
                                    delta=delta,
                                    gamma=gamma,
                                    theta=theta,
                                    vega=vega,
                                    underlying_price=underlying_price,
                                    time_to_expiry=time_to_expiry,
                                    premium_to_strike_ratio=premium_to_strike_ratio
                                )
                                
                                signals.append(signal)
                                print(f"Added signal for {symbol} {option_type} ${strike}")
                    except Exception as e:
                        print(f"Error processing {symbol} {strike} {option_type}: {e}")
                        continue
        
        # Sort signals by strength and return top ones
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        return signals[:max_signals]
        
    def generate_standard_strikes(self, symbol: str, current_price: float) -> List[float]:
        """
        Generate standard strike prices for options based on the current price
        
        Args:
            symbol: The ticker symbol
            current_price: The current price of the underlying asset
            
        Returns:
            List of standard strike prices
        """
        # Define strike intervals based on stock price
        if symbol == "SPY" or symbol == "QQQ":
            # SPY and QQQ options typically have $5 strike intervals
            interval = 5.0
            # Round to nearest $5
            base = round(current_price / interval) * interval
            # Generate strikes around the base
            strikes = [
                base - 2 * interval,  # 2 strikes below
                base - interval,      # 1 strike below 
                base,                 # At-the-money
                base + interval,      # 1 strike above
                base + 2 * interval   # 2 strikes above
            ]
        elif current_price >= 200:
            # Higher priced stocks often have $5 or $10 strike intervals
            interval = 5.0
            # Round to nearest interval
            base = round(current_price / interval) * interval
            # Generate strikes around the base
            strikes = [
                base - 2 * interval,
                base - interval,
                base,
                base + interval,
                base + 2 * interval
            ]
        elif current_price >= 100:
            # Medium priced stocks typically have $5 strike intervals
            interval = 5.0
            # Round to nearest $5
            base = round(current_price / interval) * interval
            # Generate strikes around the base
            strikes = [
                base - 2 * interval,
                base - interval,
                base,
                base + interval,
                base + 2 * interval
            ]
        elif current_price >= 50:
            # Lower-medium priced stocks might have $2.5 strike intervals
            interval = 2.5
            # Round to nearest $2.5
            base = round(current_price / interval) * interval
            # Generate strikes around the base
            strikes = [
                base - 2 * interval,
                base - interval,
                base,
                base + interval,
                base + 2 * interval
            ]
        else:
            # Lower priced stocks often have $1 or $2.5 strike intervals
            interval = 1.0
            # Round to nearest $1
            base = round(current_price / interval) * interval
            # Generate strikes around the base
            strikes = [
                base - 2 * interval,
                base - interval,
                base,
                base + interval,
                base + 2 * interval
            ]
            
        return strikes 