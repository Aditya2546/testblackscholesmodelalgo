"""
ML Feature Generation Module

This module generates features for the machine learning models used in edge detection.
Features include price returns over various time intervals, implied volatility changes,
order flow imbalance, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    # Return windows in minutes
    return_windows: List[int] = None
    # Implied volatility features
    include_iv_features: bool = True
    # Order flow features
    include_orderflow: bool = True
    orderflow_window: int = 60  # seconds
    # Level II (order book) features
    include_level2: bool = True
    # Market context features
    include_vix: bool = True
    include_etf_flow: bool = True
    # Other parameters
    normalize_features: bool = True
    
    def __post_init__(self):
        """Initialize default values if None."""
        if self.return_windows is None:
            self.return_windows = [1, 5, 15, 30, 60]


class FeatureGenerator:
    """
    Feature generator for ML-based edge detection.
    
    This class creates a set of features from market data that are used
    to detect trading opportunities in options markets.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature generator.
        
        Args:
            config: Configuration for feature generation
        """
        self.config = config if config is not None else FeatureConfig()
    
    def generate_price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price-based features including returns over multiple timeframes.
        
        Args:
            prices: DataFrame with timestamp index and 'price' column
            
        Returns:
            DataFrame with price features
        """
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        
        if 'price' not in prices.columns:
            raise ValueError("prices DataFrame must have a 'price' column")
        
        features = pd.DataFrame(index=prices.index)
        
        # Calculate returns over different windows
        for window in self.config.return_windows:
            # Convert window from minutes to periods (assuming data frequency matches)
            features[f'return_{window}m'] = prices['price'].pct_change(window)
            
            # Add realized volatility features
            features[f'realized_vol_{window}m'] = (
                prices['price']
                .pct_change()
                .rolling(window)
                .std() * np.sqrt(252 * 390 / window)  # Annualized vol
            )
        
        return features
    
    def generate_iv_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate implied volatility features.
        
        Args:
            options_data: DataFrame with options data including implied volatility
            
        Returns:
            DataFrame with IV features
        """
        if not self.config.include_iv_features:
            return pd.DataFrame(index=options_data.index)
        
        features = pd.DataFrame(index=options_data.index)
        
        # Calculate IV change
        if 'implied_volatility' in options_data.columns:
            features['iv_change'] = options_data['implied_volatility'].diff()
            features['iv_change_pct'] = options_data['implied_volatility'].pct_change()
        
        # Calculate IV skew if multiple strikes available
        if 'strike' in options_data.columns and 'option_type' in options_data.columns:
            # Group by timestamp and calculate IV skew
            skew_data = options_data.groupby(['timestamp', 'option_type']).apply(self._calculate_skew)
            if not skew_data.empty:
                features = features.join(skew_data, how='left')
        
        return features
    
    def _calculate_skew(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate implied volatility skew metrics from a group of options.
        
        Args:
            group: DataFrame group with options at different strikes
            
        Returns:
            Series with skew metrics
        """
        if len(group) < 2:
            return pd.Series({
                'iv_skew': np.nan,
                'iv_skew_slope': np.nan,
                'iv_skew_curvature': np.nan
            })
        
        # Sort by strike
        sorted_group = group.sort_values('strike')
        
        # Calculate simple ATM to OTM skew
        atm_iv = sorted_group.iloc[len(sorted_group) // 2]['implied_volatility']
        furthest_otm_iv = sorted_group.iloc[-1]['implied_volatility']
        
        iv_skew = furthest_otm_iv - atm_iv
        
        # Calculate slope and curvature if enough strikes
        if len(sorted_group) >= 3:
            strikes = sorted_group['strike'].values
            ivs = sorted_group['implied_volatility'].values
            
            # Normalize strikes around ATM
            atm_strike = strikes[len(strikes) // 2]
            norm_strikes = (strikes - atm_strike) / atm_strike
            
            # Calculate slope using linear regression
            if len(norm_strikes) > 1:  # At least 2 points for regression
                try:
                    slope = np.polyfit(norm_strikes, ivs, 1)[0]
                except:
                    slope = np.nan
            else:
                slope = np.nan
            
            # Calculate curvature using quadratic fit
            if len(norm_strikes) > 2:  # At least 3 points for quadratic fit
                try:
                    curvature = np.polyfit(norm_strikes, ivs, 2)[0]
                except:
                    curvature = np.nan
            else:
                curvature = np.nan
                
            return pd.Series({
                'iv_skew': iv_skew,
                'iv_skew_slope': slope,
                'iv_skew_curvature': curvature
            })
        else:
            return pd.Series({
                'iv_skew': iv_skew,
                'iv_skew_slope': np.nan,
                'iv_skew_curvature': np.nan
            })
    
    def generate_orderflow_features(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Generate order flow features including buy/sell imbalance.
        
        Args:
            trades: DataFrame with trade data including side, price, and size
            
        Returns:
            DataFrame with order flow features
        """
        if not self.config.include_orderflow or trades.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=trades.index.unique())
        
        # Calculate buy/sell imbalance over the specified window
        window_seconds = self.config.orderflow_window
        
        # Ensure trades have 'side' and 'size' columns
        if 'side' not in trades.columns or 'size' not in trades.columns:
            return features
        
        # Label trades as buy or sell
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']
        
        # Calculate rolling imbalance
        for idx in features.index:
            window_start = idx - pd.Timedelta(seconds=window_seconds)
            window_trades = trades[(trades.index >= window_start) & (trades.index <= idx)]
            
            window_buy_trades = window_trades[window_trades['side'] == 'buy']
            window_sell_trades = window_trades[window_trades['side'] == 'sell']
            
            buy_volume = window_buy_trades['size'].sum() if not window_buy_trades.empty else 0
            sell_volume = window_sell_trades['size'].sum() if not window_sell_trades.empty else 0
            
            # Calculate imbalance
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
            else:
                imbalance = 0
                
            features.at[idx, 'orderflow_imbalance'] = imbalance
            features.at[idx, 'order_volume'] = total_volume
        
        return features
    
    def generate_level2_features(self, orderbook: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from Level II (order book) data.
        
        Args:
            orderbook: DataFrame with order book data
            
        Returns:
            DataFrame with order book features
        """
        if not self.config.include_level2 or orderbook.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=orderbook.index.unique())
        
        # Calculate order book depth metrics
        if {'bid_size', 'ask_size', 'bid_price', 'ask_price'}.issubset(orderbook.columns):
            for idx in features.index:
                # Get current order book snapshot
                book = orderbook.loc[idx]
                
                # Calculate bid/ask ratio
                bid_size = book['bid_size']
                ask_size = book['ask_size']
                
                if ask_size > 0:
                    depth_ratio = bid_size / ask_size
                else:
                    depth_ratio = np.nan
                
                # Calculate spread
                spread = book['ask_price'] - book['bid_price']
                relative_spread = spread / book['bid_price'] if book['bid_price'] > 0 else np.nan
                
                features.at[idx, 'depth_ratio'] = depth_ratio
                features.at[idx, 'relative_spread'] = relative_spread
        
        return features
    
    def generate_market_context_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate market context features including VIX changes and ETF flows.
        
        Args:
            market_data: Dictionary of market data including 'vix' and 'etf_flows'
            
        Returns:
            DataFrame with market context features
        """
        # Start with empty features DataFrame
        index = None
        for df in market_data.values():
            if not df.empty:
                index = df.index
                break
                
        if index is None:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=index)
        
        # Add VIX features
        if self.config.include_vix and 'vix' in market_data and not market_data['vix'].empty:
            vix_data = market_data['vix']
            if 'price' in vix_data.columns:
                features['vix_level'] = vix_data['price']
                features['vix_change'] = vix_data['price'].diff()
                features['vix_change_pct'] = vix_data['price'].pct_change()
        
        # Add ETF flow features
        if (self.config.include_etf_flow and 'etf_flows' in market_data 
                and not market_data['etf_flows'].empty):
            flow_data = market_data['etf_flows']
            for etf in flow_data['symbol'].unique():
                etf_flow = flow_data[flow_data['symbol'] == etf]
                if 'flow' in etf_flow.columns:
                    features[f'{etf}_flow'] = etf_flow['flow']
        
        return features
    
    def generate_all_features(self, 
                             prices: pd.DataFrame,
                             options_data: Optional[pd.DataFrame] = None,
                             trades: Optional[pd.DataFrame] = None,
                             orderbook: Optional[pd.DataFrame] = None,
                             market_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Generate all features for edge detection.
        
        Args:
            prices: DataFrame with price data
            options_data: DataFrame with options data
            trades: DataFrame with trade data
            orderbook: DataFrame with order book data
            market_data: Dictionary of market data
            
        Returns:
            DataFrame with all features
        """
        # Generate individual feature sets
        price_features = self.generate_price_features(prices)
        
        iv_features = (self.generate_iv_features(options_data) 
                      if options_data is not None else pd.DataFrame(index=price_features.index))
        
        orderflow_features = (self.generate_orderflow_features(trades)
                             if trades is not None else pd.DataFrame(index=price_features.index))
        
        level2_features = (self.generate_level2_features(orderbook)
                          if orderbook is not None else pd.DataFrame(index=price_features.index))
        
        market_context_features = (self.generate_market_context_features(market_data)
                                  if market_data is not None else pd.DataFrame(index=price_features.index))
        
        # Combine all features
        all_features = price_features.join(
            [iv_features, orderflow_features, level2_features, market_context_features],
            how='left'
        )
        
        # Handle missing values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # Normalize features if configured
        if self.config.normalize_features:
            all_features = self._normalize_features(all_features)
        
        return all_features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to have zero mean and unit variance.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Normalized features
        """
        # Skip timestamp index
        numeric_features = features.select_dtypes(include=['number'])
        
        # Calculate means and standard deviations
        means = numeric_features.mean()
        stds = numeric_features.std()
        
        # Replace zero standard deviations with 1 to avoid division by zero
        stds = stds.replace(0, 1)
        
        # Normalize
        normalized = (numeric_features - means) / stds
        
        # Combine with non-numeric columns
        non_numeric = features.select_dtypes(exclude=['number'])
        return pd.concat([normalized, non_numeric], axis=1)

# Stream feature generator for real-time feature calculation
class StreamFeatureGenerator(FeatureGenerator):
    """
    Stream-based feature generator for real-time feature calculation.
    
    This class extends the base FeatureGenerator to work with streaming data,
    maintaining internal state for calculating time-based features.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the stream feature generator."""
        super().__init__(config)
        
        # Initialize data buffers
        self.price_buffer = pd.DataFrame(columns=['price'])
        self.options_buffer = pd.DataFrame()
        self.trades_buffer = pd.DataFrame()
        self.orderbook_buffer = pd.DataFrame()
        self.market_data_buffers = {}
        
        # Initialize feature stats for normalization
        self.feature_means = {}
        self.feature_stds = {}
        
        # Maximum buffer sizes (minutes)
        self.max_buffer_size = max(self.config.return_windows) * 2 if self.config.return_windows else 120
    
    def update_buffers(self, 
                      timestamp: pd.Timestamp,
                      price: Optional[float] = None,
                      options_data: Optional[Dict] = None,
                      trade: Optional[Dict] = None,
                      orderbook: Optional[Dict] = None,
                      market_updates: Optional[Dict[str, Dict]] = None) -> None:
        """
        Update internal data buffers with new data points.
        
        Args:
            timestamp: Current timestamp
            price: Current price (optional)
            options_data: Options data update (optional)
            trade: Trade data update (optional)
            orderbook: Order book update (optional)
            market_updates: Market data updates (optional)
        """
        # Update price buffer
        if price is not None:
            self.price_buffer.loc[timestamp, 'price'] = price
        
        # Update options buffer
        if options_data is not None:
            options_row = pd.DataFrame([options_data], index=[timestamp])
            self.options_buffer = pd.concat([self.options_buffer, options_row])
        
        # Update trades buffer
        if trade is not None:
            trade_row = pd.DataFrame([trade], index=[timestamp])
            self.trades_buffer = pd.concat([self.trades_buffer, trade_row])
        
        # Update orderbook buffer
        if orderbook is not None:
            orderbook_row = pd.DataFrame([orderbook], index=[timestamp])
            self.orderbook_buffer = pd.concat([self.orderbook_buffer, orderbook_row])
        
        # Update market data buffers
        if market_updates is not None:
            for market, data in market_updates.items():
                if market not in self.market_data_buffers:
                    self.market_data_buffers[market] = pd.DataFrame()
                
                market_row = pd.DataFrame([data], index=[timestamp])
                self.market_data_buffers[market] = pd.concat(
                    [self.market_data_buffers[market], market_row]
                )
        
        # Trim buffers to maintain performance
        self._trim_buffers(timestamp)
    
    def _trim_buffers(self, current_time: pd.Timestamp) -> None:
        """
        Trim buffers to contain only recent data.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - pd.Timedelta(minutes=self.max_buffer_size)
        
        # Trim each buffer
        self.price_buffer = self.price_buffer[self.price_buffer.index > cutoff_time]
        self.options_buffer = self.options_buffer[self.options_buffer.index > cutoff_time]
        self.trades_buffer = self.trades_buffer[self.trades_buffer.index > cutoff_time]
        self.orderbook_buffer = self.orderbook_buffer[self.orderbook_buffer.index > cutoff_time]
        
        # Trim market data buffers
        for market in self.market_data_buffers:
            self.market_data_buffers[market] = self.market_data_buffers[market][
                self.market_data_buffers[market].index > cutoff_time
            ]
    
    def calculate_latest_features(self) -> Dict[str, float]:
        """
        Calculate features based on current buffer state.
        
        Returns:
            Dictionary of feature values
        """
        if self.price_buffer.empty:
            return {}
        
        # Get current timestamp
        current_time = self.price_buffer.index[-1]
        
        # Generate features from buffers
        price_features = self.generate_price_features(self.price_buffer)
        
        iv_features = (self.generate_iv_features(self.options_buffer)
                      if not self.options_buffer.empty else pd.DataFrame(index=[current_time]))
        
        orderflow_features = (self.generate_orderflow_features(self.trades_buffer)
                             if not self.trades_buffer.empty else pd.DataFrame(index=[current_time]))
        
        level2_features = (self.generate_level2_features(self.orderbook_buffer)
                          if not self.orderbook_buffer.empty else pd.DataFrame(index=[current_time]))
        
        # Prepare market data for context features
        market_data = {k: v for k, v in self.market_data_buffers.items() if not v.empty}
        market_context_features = (self.generate_market_context_features(market_data)
                                  if market_data else pd.DataFrame(index=[current_time]))
        
        # Combine all features
        all_features = pd.concat(
            [price_features, iv_features, orderflow_features, level2_features, market_context_features],
            axis=1
        )
        
        # Get latest features
        if current_time in all_features.index:
            latest_features = all_features.loc[current_time].to_dict()
            
            # Update feature stats for normalization
            self._update_feature_stats(latest_features)
            
            # Normalize if configured
            if self.config.normalize_features:
                latest_features = self._normalize_latest_features(latest_features)
            
            return latest_features
        else:
            return {}
    
    def _update_feature_stats(self, latest_features: Dict[str, float]) -> None:
        """
        Update feature statistics for normalization.
        
        Args:
            latest_features: Latest feature values
        """
        for feature, value in latest_features.items():
            if np.isnan(value) or not np.isfinite(value):
                continue
                
            if feature not in self.feature_means:
                self.feature_means[feature] = value
                self.feature_stds[feature] = 1.0
            else:
                # Exponential moving average for mean and std
                alpha = 0.01  # Smoothing factor
                old_mean = self.feature_means[feature]
                self.feature_means[feature] = (1 - alpha) * old_mean + alpha * value
                
                # Update std using Welford's online algorithm
                old_std = self.feature_stds[feature]
                variance = (old_std ** 2) + alpha * ((value - old_mean) * (value - self.feature_means[feature]))
                self.feature_stds[feature] = np.sqrt(max(variance, 1e-8))
    
    def _normalize_latest_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize latest features using updated statistics.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Normalized features
        """
        normalized = {}
        for feature, value in features.items():
            if feature in self.feature_means and feature in self.feature_stds:
                if self.feature_stds[feature] > 0:
                    normalized[feature] = (value - self.feature_means[feature]) / self.feature_stds[feature]
                else:
                    normalized[feature] = 0.0
            else:
                normalized[feature] = value
        
        return normalized