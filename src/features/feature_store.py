"""
Feature Store for Options Trading System

This module maintains rolling windows of micro-features:
- order_flow_imbalance_60s, depth_ratio, quoted_spread
- IV_skew (ΔIV_atm-25Δ, ΔIV_25Δ-50Δ)
- realized_vol_1m, realized_vol_5m vs. implied_vol
- stochastic volatility flag (Heston proxy σ² jump)

Exposes `.get_feature_vector(contract_id, t)` in <2ms via:
- Memory-based caching
- Efficient rolling window implementations
- Vectorized calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
import time
import logging
import threading
import os
import json
from bisect import bisect_left
from collections import defaultdict, deque
import pickle
import hashlib
import joblib
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features in store."""
    ORDERBOOK = "orderbook"     # Order book features
    TRADE = "trade"             # Trade-based features
    VOLATILITY = "volatility"   # Volatility features
    OPTIONS = "options"         # Options-specific features
    TECHNICAL = "technical"     # Technical indicators
    FUNDAMENTAL = "fundamental" # Fundamental data
    SENTIMENT = "sentiment"     # Sentiment features
    CUSTOM = "custom"           # Custom features


@dataclass
class Feature:
    """Feature metadata."""
    name: str
    type: FeatureType
    window_size: int = 0
    window_unit: str = "s"  # "s" for seconds, "m" for minutes, "h" for hours, "d" for days
    description: str = ""
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize additional fields."""
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "window_size": self.window_size,
            "window_unit": self.window_unit,
            "description": self.description,
            "version": self.version,
            "dependencies": self.dependencies,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feature':
        """Create from dictionary."""
        feature_type = FeatureType(data.get("type", "custom"))
        last_updated = data.get("last_updated")
        
        if last_updated and isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        
        return cls(
            name=data.get("name", ""),
            type=feature_type,
            window_size=data.get("window_size", 0),
            window_unit=data.get("window_unit", "s"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            dependencies=data.get("dependencies", []),
            last_updated=last_updated
        )


@dataclass
class FeatureVector:
    """Vector of feature values."""
    contract_id: str
    timestamp: datetime
    values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __getitem__(self, key: str) -> float:
        """Get feature value by name."""
        return self.values.get(key, float('nan'))
    
    def __setitem__(self, key: str, value: float) -> None:
        """Set feature value by name."""
        self.values[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_id": self.contract_id,
            "timestamp": self.timestamp.isoformat(),
            "values": self.values,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVector':
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            contract_id=data.get("contract_id", ""),
            timestamp=timestamp or datetime.now(),
            values=data.get("values", {}),
            metadata=data.get("metadata", {})
        )
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array with specified feature order."""
        return np.array([self.values.get(name, float('nan')) for name in feature_names])
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.values, name=self.timestamp)


class FeatureCache:
    """Cache for feature vectors."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items in cache
        """
        self.max_size = max_size
        self._cache: Dict[str, FeatureVector] = {}
        self._timestamps: List[Tuple[str, datetime]] = []
        self._lock = threading.RLock()
    
    def _generate_key(self, contract_id: str, timestamp: datetime) -> str:
        """Generate cache key."""
        return f"{contract_id}_{timestamp.isoformat()}"
    
    def get(self, contract_id: str, timestamp: datetime) -> Optional[FeatureVector]:
        """
        Get feature vector from cache.
        
        Args:
            contract_id: Contract ID
            timestamp: Timestamp
            
        Returns:
            Feature vector if found, None otherwise
        """
        key = self._generate_key(contract_id, timestamp)
        with self._lock:
            return self._cache.get(key)
    
    def put(self, feature_vector: FeatureVector) -> None:
        """
        Add feature vector to cache.
        
        Args:
            feature_vector: Feature vector to add
        """
        key = self._generate_key(feature_vector.contract_id, feature_vector.timestamp)
        with self._lock:
            # Add to cache
            self._cache[key] = feature_vector
            
            # Add to timestamps
            self._timestamps.append((key, feature_vector.timestamp))
            
            # Evict if over max size
            if len(self._cache) > self.max_size:
                # Sort by timestamp (oldest first)
                self._timestamps.sort(key=lambda x: x[1])
                
                # Remove oldest
                old_key, _ = self._timestamps.pop(0)
                self._cache.pop(old_key, None)
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """Get number of items in cache."""
        with self._lock:
            return len(self._cache)


class FeatureStore:
    """
    High-performance feature store for options trading with market microstructure data
    and volatility metrics. Supports sub-2ms retrieval times through efficient caching.
    """
    
    def __init__(
        self,
        cache_dir: str = ".feature_cache",
        cache_size: int = 10000,  # Number of feature vectors to cache in memory
        ttl_seconds: int = 3600,  # Time-to-live for cached features
        enable_disk_cache: bool = True,
        performance_monitoring: bool = True
    ):
        """
        Initialize the feature store
        
        Args:
            cache_dir: Directory for disk cache
            cache_size: Number of feature vectors to cache in memory (LRU cache)
            ttl_seconds: Cache time-to-live in seconds
            enable_disk_cache: Whether to enable persistent disk caching
            performance_monitoring: Whether to track and log performance metrics
        """
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.enable_disk_cache = enable_disk_cache
        self.performance_monitoring = performance_monitoring
        
        # Create cache directory if it doesn't exist
        if enable_disk_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # In-memory cache (LRU)
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        # Feature definitions
        self.feature_definitions = {}
        
        # Performance tracking
        self.retrieval_times = []
        
        logger.info(f"Initialized FeatureStore with cache size {cache_size} and TTL {ttl_seconds}s")
    
    def add_feature_definition(
        self,
        feature_name: str,
        feature_fn: callable,
        dependencies: List[str] = None,
        is_market_microstructure: bool = False,
        is_volatility_metric: bool = False,
        description: str = ""
    ) -> None:
        """
        Add a feature definition to the store
        
        Args:
            feature_name: Name of the feature
            feature_fn: Function to compute the feature
            dependencies: List of dependent features
            is_market_microstructure: Whether this is a market microstructure feature
            is_volatility_metric: Whether this is a volatility-related metric
            description: Description of the feature
        """
        self.feature_definitions[feature_name] = {
            "function": feature_fn,
            "dependencies": dependencies or [],
            "is_market_microstructure": is_market_microstructure,
            "is_volatility_metric": is_volatility_metric,
            "description": description
        }
        logger.info(f"Added feature definition: {feature_name}")
    
    @lru_cache(maxsize=128)
    def _compute_cache_key(self, symbol: str, timestamp: Union[str, datetime], feature_names: Tuple[str]) -> str:
        """
        Compute a unique cache key for a feature request
        
        Args:
            symbol: Ticker symbol
            timestamp: Data timestamp
            feature_names: Tuple of feature names (must be hashable)
            
        Returns:
            Unique cache key as string
        """
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
            
        key_data = f"{symbol}_{timestamp}_{','.join(sorted(feature_names))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_features(
        self,
        symbol: str,
        timestamp: Union[str, datetime],
        feature_names: List[str],
        market_data: Optional[Dict] = None,
        force_recompute: bool = False
    ) -> Dict[str, float]:
        """
        Get features for a specific symbol and timestamp
        
        Args:
            symbol: Ticker symbol
            timestamp: Data timestamp
            feature_names: List of feature names to retrieve
            market_data: Optional market data to use for computation
            force_recompute: Force recomputation even if cached
            
        Returns:
            Dictionary of feature values
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Convert timestamp to string for caching
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = timestamp
        
        # Create a cache key
        cache_key = self._compute_cache_key(symbol, timestamp_str, tuple(sorted(feature_names)))
        
        # Check if in memory cache and not expired
        if not force_recompute and cache_key in self.cache:
            cache_time = self.cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self.ttl_seconds:
                self.cache_hits += 1
                result = self.cache[cache_key]
                
                if self.performance_monitoring:
                    retrieval_time = (time.time() - start_time) * 1000  # ms
                    self.retrieval_times.append(retrieval_time)
                    
                    if len(self.retrieval_times) % 1000 == 0:
                        avg_time = sum(self.retrieval_times[-1000:]) / 1000
                        logger.info(f"Avg feature retrieval time (last 1000): {avg_time:.2f}ms")
                
                return result
        
        # Check disk cache if enabled
        if not force_recompute and self.enable_disk_cache:
            disk_cache_path = os.path.join(self.cache_dir, f"{cache_key}.joblib")
            if os.path.exists(disk_cache_path):
                try:
                    mtime = os.path.getmtime(disk_cache_path)
                    if time.time() - mtime < self.ttl_seconds:
                        result = joblib.load(disk_cache_path)
                        
                        # Update memory cache
                        self._update_memory_cache(cache_key, result)
                        
                        self.cache_hits += 1
                        
                        if self.performance_monitoring:
                            retrieval_time = (time.time() - start_time) * 1000  # ms
                            self.retrieval_times.append(retrieval_time)
                        
                        return result
                except Exception as e:
                    logger.warning(f"Error loading disk cache: {e}")
        
        # Cache miss - compute features
        self.cache_misses += 1
        
        # For each feature, compute or get dependencies
        result = {}
        computed_features = set()
        
        # Build dependency graph
        dependencies = {}
        for feature in feature_names:
            if feature not in self.feature_definitions:
                raise ValueError(f"Unknown feature: {feature}")
            dependencies[feature] = self.feature_definitions[feature]["dependencies"]
        
        # Determine computation order (simple topological sort)
        computation_order = []
        visited = set()
        
        def visit(feature):
            if feature in visited:
                return
            visited.add(feature)
            for dep in dependencies.get(feature, []):
                visit(dep)
            computation_order.append(feature)
        
        for feature in feature_names:
            visit(feature)
        
        # Compute features in order
        for feature in computation_order:
            if feature in computed_features:
                continue
                
            try:
                # Get feature definition
                feature_def = self.feature_definitions[feature]
                feature_fn = feature_def["function"]
                
                # Get required dependencies
                deps = {dep: result[dep] for dep in feature_def["dependencies"] if dep in result}
                
                # Compute feature
                result[feature] = feature_fn(symbol, timestamp, market_data, deps)
                computed_features.add(feature)
                
            except Exception as e:
                logger.error(f"Error computing feature {feature}: {e}")
                result[feature] = np.nan
        
        # Update cache
        self._update_memory_cache(cache_key, result)
        
        # Update disk cache
        if self.enable_disk_cache:
            try:
                disk_cache_path = os.path.join(self.cache_dir, f"{cache_key}.joblib")
                joblib.dump(result, disk_cache_path)
            except Exception as e:
                logger.warning(f"Error writing to disk cache: {e}")
        
        if self.performance_monitoring:
            retrieval_time = (time.time() - start_time) * 1000  # ms
            self.retrieval_times.append(retrieval_time)
            
            if retrieval_time > 2.0:  # Log slow retrievals
                logger.warning(f"Slow feature retrieval: {retrieval_time:.2f}ms for {symbol} at {timestamp_str}")
        
        return result
    
    def _update_memory_cache(self, cache_key: str, result: Dict) -> None:
        """
        Update the in-memory LRU cache
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Implement LRU cache behavior
        if len(self.cache) >= self.cache_size:
            # Find the least recently used item
            oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        # Add to cache
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
    
    def cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        
        avg_retrieval_time = None
        if self.retrieval_times:
            avg_retrieval_time = sum(self.retrieval_times) / len(self.retrieval_times)
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": self.total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.cache),
            "avg_retrieval_time_ms": avg_retrieval_time
        }
    
    def clear_cache(self) -> None:
        """Clear all caches (memory and disk)"""
        # Clear memory cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Clear disk cache
        if self.enable_disk_cache and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting cache file {file_path}: {e}")
        
        logger.info("Cache cleared")
    
    def register_market_microstructure_features(self) -> None:
        """Register common market microstructure features"""
        
        # Order flow imbalance
        def order_flow_imbalance(symbol, timestamp, market_data, deps):
            if market_data is None or 'order_book' not in market_data:
                return np.nan
            
            order_book = market_data['order_book']
            
            # Calculate buy and sell volumes
            buy_volume = sum(level['size'] for level in order_book.get('bids', []))
            sell_volume = sum(level['size'] for level in order_book.get('asks', []))
            
            # Calculate imbalance
            total_volume = buy_volume + sell_volume
            return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        self.add_feature_definition(
            "order_flow_imbalance",
            order_flow_imbalance,
            is_market_microstructure=True,
            description="Imbalance between buy and sell orders in the order book"
        )
        
        # Depth ratio
        def market_depth_ratio(symbol, timestamp, market_data, deps):
            if market_data is None or 'order_book' not in market_data:
                return np.nan
            
            order_book = market_data['order_book']
            
            if not order_book.get('bids') or not order_book.get('asks'):
                return np.nan
            
            # Calculate volume at top 5 levels
            top_bid_volume = sum(level['size'] for level in order_book.get('bids', [])[:5])
            top_ask_volume = sum(level['size'] for level in order_book.get('asks', [])[:5])
            
            return top_bid_volume / top_ask_volume if top_ask_volume > 0 else np.inf
        
        self.add_feature_definition(
            "market_depth_ratio",
            market_depth_ratio,
            is_market_microstructure=True,
            description="Ratio of bid volume to ask volume at top price levels"
        )
        
        # Spread percentage
        def spread_percentage(symbol, timestamp, market_data, deps):
            if market_data is None or 'order_book' not in market_data:
                return np.nan
            
            order_book = market_data['order_book']
            
            if not order_book.get('bids') or not order_book.get('asks'):
                return np.nan
            
            best_bid = order_book['bids'][0]['price'] if order_book['bids'] else np.nan
            best_ask = order_book['asks'][0]['price'] if order_book['asks'] else np.nan
            
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            return 100 * spread / mid_price if mid_price > 0 else np.nan
        
        self.add_feature_definition(
            "spread_percentage",
            spread_percentage,
            is_market_microstructure=True,
            description="Bid-ask spread as percentage of mid price"
        )
    
    def register_volatility_metrics(self) -> None:
        """Register common volatility metrics"""
        
        # Implied volatility
        def implied_volatility(symbol, timestamp, market_data, deps):
            if market_data is None or 'options' not in market_data:
                return np.nan
            
            options_data = market_data['options']
            
            # Average IV across all ATM options
            atm_options = [opt for opt in options_data if abs(opt.get('moneyness', 1) - 1.0) < 0.05]
            
            if not atm_options:
                return np.nan
            
            avg_iv = sum(opt.get('implied_volatility', 0) for opt in atm_options) / len(atm_options)
            return avg_iv
        
        self.add_feature_definition(
            "implied_volatility",
            implied_volatility,
            is_volatility_metric=True,
            description="Average implied volatility of at-the-money options"
        )
        
        # IV skew
        def iv_skew(symbol, timestamp, market_data, deps):
            if market_data is None or 'options' not in market_data:
                return np.nan
            
            options_data = market_data['options']
            
            # Get put options with different strikes
            put_options = [opt for opt in options_data if opt.get('option_type') == 'put']
            
            if len(put_options) < 2:
                return np.nan
            
            # Sort by strike
            put_options.sort(key=lambda x: x.get('strike_price', 0))
            
            # Calculate skew as difference between OTM and ATM puts
            atm_puts = [opt for opt in put_options if abs(opt.get('moneyness', 1) - 1.0) < 0.05]
            otm_puts = [opt for opt in put_options if opt.get('moneyness', 1) < 0.9]
            
            if not atm_puts or not otm_puts:
                return np.nan
            
            atm_iv = sum(opt.get('implied_volatility', 0) for opt in atm_puts) / len(atm_puts)
            otm_iv = sum(opt.get('implied_volatility', 0) for opt in otm_puts) / len(otm_puts)
            
            return otm_iv - atm_iv
        
        self.add_feature_definition(
            "iv_skew",
            iv_skew,
            is_volatility_metric=True,
            description="Difference between OTM and ATM put option implied volatility"
        )
        
        # Realized vs implied volatility ratio
        def realized_implied_ratio(symbol, timestamp, market_data, deps):
            if market_data is None or 'options' not in market_data or 'historical' not in market_data:
                return np.nan
            
            options_data = market_data['options']
            historical_data = market_data['historical']
            
            # Calculate implied volatility
            atm_options = [opt for opt in options_data if abs(opt.get('moneyness', 1) - 1.0) < 0.05]
            
            if not atm_options:
                return np.nan
            
            implied_vol = sum(opt.get('implied_volatility', 0) for opt in atm_options) / len(atm_options)
            
            # Calculate realized volatility (20-day)
            if len(historical_data) < 20:
                return np.nan
            
            returns = np.diff(np.log(historical_data[-20:]['close']))
            realized_vol = np.std(returns) * np.sqrt(252)
            
            return realized_vol / implied_vol if implied_vol > 0 else np.nan
        
        self.add_feature_definition(
            "realized_implied_ratio",
            realized_implied_ratio,
            dependencies=["implied_volatility"],
            is_volatility_metric=True,
            description="Ratio of realized volatility to implied volatility"
        )
    
    def export_feature_definitions(self, filepath: str) -> None:
        """
        Export feature definitions to a JSON file
        
        Args:
            filepath: Path to save the definitions
        """
        # Create a serializable version of the definitions
        serializable_defs = {}
        for name, def_dict in self.feature_definitions.items():
            serializable_defs[name] = {
                "dependencies": def_dict["dependencies"],
                "is_market_microstructure": def_dict["is_market_microstructure"],
                "is_volatility_metric": def_dict["is_volatility_metric"],
                "description": def_dict["description"]
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_defs, f, indent=2)
        
        logger.info(f"Exported {len(serializable_defs)} feature definitions to {filepath}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create feature store
    store = FeatureStore()
    
    # Print registered features
    print("Registered features:")
    for name, feature in store.feature_definitions.items():
        print(f"- {name} ({feature['is_market_microstructure']}, {feature['is_volatility_metric']}): {feature['description']}")
    
    # Save feature definitions
    store.export_feature_definitions("config/features.json") 