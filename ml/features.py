#!/usr/bin/env python
"""
Options Trading ML Feature Generator

This module generates features for machine learning models to predict
edge in options pricing, based on market data and options characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import datetime

class FeatureConfig:
    """Configuration for feature generation."""
    
    # Returns-based features
    RETURN_WINDOWS = [1, 5, 10, 15, 30, 60, 120]  # Tick windows for return calculation
    
    # Volatility features
    VOLATILITY_WINDOWS = [5, 15, 30, 60]  # Windows for volatility calculation
    
    # Volume features
    VOLUME_WINDOWS = [5, 15, 30, 60]  # Windows for volume accumulation
    
    # Order flow features
    ORDER_IMBALANCE_WINDOWS = [10, 30, 60]  # Windows for order imbalance
    
    # Technical indicator settings
    RSI_WINDOW = 14  # Window for RSI calculation
    MACD_WINDOWS = (12, 26, 9)  # Fast, slow, signal windows
    
    # Option-specific features
    IV_PERCENTILE_LOOKBACK = 30  # Days for IV percentile calculation
    GREEKS_CHANGE_WINDOW = 5  # Ticks for Greek changes
    
    # Options surface features
    MONEYNESS_LEVELS = [-0.05, -0.025, 0, 0.025, 0.05]  # Moneyness levels for surface
    EXPIRY_DAYS = [1, 7, 30, 90]  # Days to expiration for surface


class FeatureGenerator:
    """
    Generate features for options trading ML models.
    
    This class transforms raw market data and options information into
    features suitable for machine learning models.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize feature generator.
        
        Args:
            config: Feature generation configuration
        """
        self.config = config or FeatureConfig()
    
    def generate_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from price data.
        
        Args:
            price_data: DataFrame with price history (columns: time, open, high, low, close, volume)
            
        Returns:
            DataFrame with price-based features
        """
        features = pd.DataFrame(index=price_data.index)
        
        # Returns over different windows
        for window in self.config.RETURN_WINDOWS:
            features[f'return_{window}'] = price_data['close'].pct_change(window)
        
        # Log returns (better for ML models)
        for window in self.config.RETURN_WINDOWS:
            features[f'log_return_{window}'] = np.log(price_data['close'] / price_data['close'].shift(window))
        
        # Volatility measures
        for window in self.config.VOLATILITY_WINDOWS:
            features[f'volatility_{window}'] = price_data['close'].pct_change().rolling(window).std()
            
        # Price change velocity (acceleration/deceleration)
        for window in self.config.RETURN_WINDOWS:
            if window > 1:
                features[f'return_velocity_{window}'] = features[f'return_{window}'] / features[f'return_1']
        
        # High-Low range relative to close
        features['hl_range_pct'] = (price_data['high'] - price_data['low']) / price_data['close']
        
        # Distance from daily high/low
        features['dist_from_high'] = (price_data['high'].rolling(390).max() - price_data['close']) / price_data['close']
        features['dist_from_low'] = (price_data['close'] - price_data['low'].rolling(390).min()) / price_data['close']
        
        return features
    
    def generate_volume_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volume-based features.
        
        Args:
            price_data: DataFrame with price history
            
        Returns:
            DataFrame with volume-based features
        """
        features = pd.DataFrame(index=price_data.index)
        
        # Volume profile
        for window in self.config.VOLUME_WINDOWS:
            features[f'volume_ma_{window}'] = price_data['volume'].rolling(window).mean()
            
        # Volume ratio against moving average
        for window in self.config.VOLUME_WINDOWS:
            features[f'volume_ratio_{window}'] = price_data['volume'] / features[f'volume_ma_{window}']
        
        # Up/down volume
        price_data['up_volume'] = np.where(price_data['close'] > price_data['close'].shift(1), 
                                          price_data['volume'], 0)
        price_data['down_volume'] = np.where(price_data['close'] < price_data['close'].shift(1), 
                                            price_data['volume'], 0)
        
        # Up/down volume ratios
        for window in self.config.VOLUME_WINDOWS:
            features[f'up_volume_ratio_{window}'] = price_data['up_volume'].rolling(window).sum() / price_data['volume'].rolling(window).sum()
        
        return features
    
    def generate_order_flow_features(self, order_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from order flow data.
        
        Args:
            order_data: DataFrame with order flow data (time, price, size, side, etc.)
            
        Returns:
            DataFrame with order flow features
        """
        features = pd.DataFrame(index=order_data.index)
        
        # Label buy/sell orders
        buy_mask = order_data['side'] == 'buy'
        order_data['buy_volume'] = np.where(buy_mask, order_data['size'], 0)
        order_data['sell_volume'] = np.where(~buy_mask, order_data['size'], 0)
        
        # Order imbalance
        for window in self.config.ORDER_IMBALANCE_WINDOWS:
            buy_vol = order_data['buy_volume'].rolling(window).sum()
            sell_vol = order_data['sell_volume'].rolling(window).sum()
            total_vol = buy_vol + sell_vol
            features[f'order_imbalance_{window}'] = (buy_vol - sell_vol) / total_vol
        
        # Large order detection
        order_data['large_buy'] = np.where(
            (order_data['side'] == 'buy') & (order_data['size'] > order_data['size'].mean() * 2),
            order_data['size'], 0)
        order_data['large_sell'] = np.where(
            (order_data['side'] == 'sell') & (order_data['size'] > order_data['size'].mean() * 2),
            order_data['size'], 0)
        
        # Large order imbalance
        for window in self.config.ORDER_IMBALANCE_WINDOWS:
            large_buy = order_data['large_buy'].rolling(window).sum()
            large_sell = order_data['large_sell'].rolling(window).sum()
            total_large = large_buy + large_sell
            features[f'large_order_imbalance_{window}'] = (large_buy - large_sell) / total_large.replace(0, 1)
        
        return features
    
    def generate_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicators as features.
        
        Args:
            price_data: DataFrame with price history
            
        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=price_data.index)
        
        # RSI
        delta = price_data['close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Make positive
        
        avg_gain = gain.rolling(self.config.RSI_WINDOW).mean()
        avg_loss = loss.rolling(self.config.RSI_WINDOW).mean()
        
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        fast, slow, signal = self.config.MACD_WINDOWS
        features['macd'] = price_data['close'].ewm(span=fast).mean() - price_data['close'].ewm(span=slow).mean()
        features['macd_signal'] = features['macd'].ewm(span=signal).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        window = 20
        features['bollinger_mid'] = price_data['close'].rolling(window).mean()
        features['bollinger_std'] = price_data['close'].rolling(window).std()
        features['bollinger_upper'] = features['bollinger_mid'] + 2 * features['bollinger_std']
        features['bollinger_lower'] = features['bollinger_mid'] - 2 * features['bollinger_std']
        features['bollinger_pct'] = (price_data['close'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])
        
        return features
    
    def generate_option_specific_features(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features specific to options.
        
        Args:
            option_data: DataFrame with option data
            
        Returns:
            DataFrame with option-specific features
        """
        features = pd.DataFrame(index=option_data.index)
        
        # Implied volatility features
        if 'implied_volatility' in option_data.columns:
            # IV percentile
            iv_history = option_data['implied_volatility'].rolling(self.config.IV_PERCENTILE_LOOKBACK).apply(
                lambda x: np.percentile(x, 50) if len(x) > 0 else np.nan)
            features['iv_percentile'] = option_data['implied_volatility'].rank(pct=True)
            
            # IV changes
            for window in self.config.RETURN_WINDOWS:
                features[f'iv_change_{window}'] = option_data['implied_volatility'].pct_change(window)
        
        # Greek changes
        greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
        for greek in greek_cols:
            if greek in option_data.columns:
                features[f'{greek}_change'] = option_data[greek].pct_change(self.config.GREEKS_CHANGE_WINDOW)
        
        # Option premium features
        if 'bid' in option_data.columns and 'ask' in option_data.columns:
            features['bid_ask_spread'] = (option_data['ask'] - option_data['bid']) / option_data['ask']
            features['mid_price'] = (option_data['bid'] + option_data['ask']) / 2
        
        # Time decay acceleration (not linear as expiration approaches)
        if 'days_to_expiry' in option_data.columns and 'theta' in option_data.columns:
            features['theta_per_day'] = option_data['theta'] / option_data['days_to_expiry'].replace(0, 0.01)
        
        # Moneyness features
        if 'strike' in option_data.columns and 'underlying_price' in option_data.columns:
            features['moneyness'] = option_data['underlying_price'] / option_data['strike'] - 1
            features['moneyness_squared'] = features['moneyness'] ** 2  # Non-linear effects
        
        return features
    
    def generate_all_features(self, price_data: pd.DataFrame, 
                             option_data: pd.DataFrame = None, 
                             order_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate all features from available data.
        
        Args:
            price_data: DataFrame with price history
            option_data: DataFrame with option data (optional)
            order_data: DataFrame with order flow data (optional)
            
        Returns:
            DataFrame with all features
        """
        # Generate features from each data source
        price_features = self.generate_price_features(price_data)
        vol_features = self.generate_volume_features(price_data)
        tech_features = self.generate_technical_indicators(price_data)
        
        # Combine all base features
        features = pd.concat([price_features, vol_features, tech_features], axis=1)
        
        # Add optional features if data is available
        if option_data is not None:
            option_features = self.generate_option_specific_features(option_data)
            features = pd.concat([features, option_features], axis=1)
        
        if order_data is not None:
            order_features = self.generate_order_flow_features(order_data)
            features = pd.concat([features, order_features], axis=1)
        
        # Drop NA values that might have been introduced by rolling windows
        features = features.dropna()
        
        return features


class StreamFeatureGenerator:
    """
    Generate features from streaming market data.
    
    This class is optimized for generating features from real-time data,
    without requiring full historical data.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize stream feature generator.
        
        Args:
            config: Feature generation configuration
        """
        self.config = config or FeatureConfig()
        
        # Initialize storage for historical data
        max_history = max(
            max(self.config.RETURN_WINDOWS),
            max(self.config.VOLATILITY_WINDOWS),
            max(self.config.VOLUME_WINDOWS),
            max(self.config.ORDER_IMBALANCE_WINDOWS),
            self.config.RSI_WINDOW,
            max(self.config.MACD_WINDOWS)
        )
        
        self.price_history = []
        self.volume_history = []
        self.max_history = max_history
    
    def update(self, price: float, volume: int = None, timestamp = None) -> None:
        """
        Update internal state with new data point.
        
        Args:
            price: Current price
            volume: Current volume (optional)
            timestamp: Timestamp (optional)
        """
        timestamp = timestamp or datetime.datetime.now()
        self.price_history.append((timestamp, price))
        
        if volume is not None:
            self.volume_history.append((timestamp, volume))
        
        # Trim histories to required length
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]
        
        if len(self.volume_history) > self.max_history:
            self.volume_history = self.volume_history[-self.max_history:]
    
    def extract_features(self) -> Dict[str, float]:
        """
        Extract features from current state.
        
        Returns:
            Dictionary of feature name to value
        """
        features = {}
        
        if len(self.price_history) < 2:
            return features  # Not enough data yet
        
        # Get current and past prices
        prices = [p for _, p in self.price_history]
        current_price = prices[-1]
        
        # Calculate return features
        for window in self.config.RETURN_WINDOWS:
            if len(prices) > window:
                features[f'return_{window}'] = (current_price / prices[-window-1]) - 1
        
        # Calculate volatility features
        for window in self.config.VOLATILITY_WINDOWS:
            if len(prices) > window + 1:
                window_prices = prices[-(window+1):]
                returns = [(window_prices[i+1] / window_prices[i]) - 1 for i in range(len(window_prices)-1)]
                features[f'volatility_{window}'] = np.std(returns)
        
        # Calculate volume features if available
        if self.volume_history:
            volumes = [v for _, v in self.volume_history]
            current_volume = volumes[-1]
            
            for window in self.config.VOLUME_WINDOWS:
                if len(volumes) > window:
                    avg_volume = np.mean(volumes[-window:])
                    features[f'volume_ratio_{window}'] = current_volume / avg_volume
        
        return features 