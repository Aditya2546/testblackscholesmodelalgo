"""
Metrics and monitoring module for options trading system.

Provides capabilities for recording and analyzing system performance
metrics, latency tracking, and trade statistics.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import threading
import json
import os

# Configure logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric with value and metadata."""
    name: str
    value: float
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsRecorder:
    """
    Records and tracks system metrics.
    
    Provides functions to record counters, gauges, histograms,
    and timers with flexible tagging support.
    """
    
    def __init__(self, service_name: str = "options_trading",
                storage_path: Optional[str] = None):
        """
        Initialize metrics recorder.
        
        Args:
            service_name: Name of the service generating metrics
            storage_path: Path to store metrics (None for in-memory only)
        """
        self.service_name = service_name
        self.storage_path = storage_path
        
        # Storage for metrics
        self.metrics: Dict[str, List[Metric]] = {}
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        
        # Create storage directory if needed
        if storage_path and not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        # Thread lock for thread safety
        self._lock = threading.Lock()
    
    def _add_metric(self, metric: Metric) -> None:
        """
        Add a metric to storage.
        
        Args:
            metric: Metric to add
        """
        with self._lock:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []
            
            self.metrics[metric.name].append(metric)
            
            # Update counters or gauges
            if metric.type == MetricType.COUNTER:
                self.counters[metric.name] = metric.value
            elif metric.type == MetricType.GAUGE:
                self.gauges[metric.name] = metric.value
            
            # Write to storage if configured
            if self.storage_path:
                self._write_metric(metric)
    
    def _write_metric(self, metric: Metric) -> None:
        """
        Write a metric to storage.
        
        Args:
            metric: Metric to write
        """
        try:
            file_path = os.path.join(self.storage_path, f"{metric.name}.json")
            
            # Convert to dictionary
            metric_dict = {
                "name": metric.name,
                "value": metric.value,
                "type": metric.type.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags
            }
            
            # Append to file
            with open(file_path, "a") as f:
                f.write(json.dumps(metric_dict) + "\n")
        except Exception as e:
            logger.error(f"Error writing metric to storage: {e}")
    
    def increment_counter(self, name: str, value: float = 1.0,
                         tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            tags: Optional tags to associate with the metric
        """
        with self._lock:
            current = self.counters.get(name, 0.0)
            new_value = current + value
            self.counters[name] = new_value
        
        metric = Metric(
            name=name,
            value=new_value,
            type=MetricType.COUNTER,
            tags=tags or {}
        )
        
        self._add_metric(metric)
    
    def set_gauge(self, name: str, value: float,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Value to set
            tags: Optional tags to associate with the metric
        """
        metric = Metric(
            name=name,
            value=value,
            type=MetricType.GAUGE,
            tags=tags or {}
        )
        
        self._add_metric(metric)
    
    def record_histogram(self, name: str, value: float,
                        tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a histogram value.
        
        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags to associate with the metric
        """
        metric = Metric(
            name=name,
            value=value,
            type=MetricType.HISTOGRAM,
            tags=tags or {}
        )
        
        self._add_metric(metric)
    
    def time_this(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            name: Metric name
            tags: Optional tags to associate with the metric
            
        Returns:
            Context manager for timing
        """
        start_time = time.time()
        
        class TimingContextManager:
            def __init__(self, recorder, name, tags):
                self.recorder = recorder
                self.name = name
                self.tags = tags
                self.start_time = start_time
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                duration_ms = (end_time - self.start_time) * 1000.0
                
                self.recorder.record_histogram(
                    name=self.name,
                    value=duration_ms,
                    tags=self.tags
                )
        
        return TimingContextManager(self, name, tags)
    
    def get_counter(self, name: str) -> float:
        """
        Get current value of a counter.
        
        Args:
            name: Counter name
            
        Returns:
            Current counter value
        """
        return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """
        Get current value of a gauge.
        
        Args:
            name: Gauge name
            
        Returns:
            Current gauge value
        """
        return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a histogram.
        
        Args:
            name: Histogram name
            
        Returns:
            Dictionary of statistics (count, min, max, avg)
        """
        metrics = self.metrics.get(name, [])
        values = [m.value for m in metrics if m.type == MetricType.HISTOGRAM]
        
        if not values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0
            }
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values)
        }
    
    def clear_metrics(self) -> None:
        """Clear all metrics from memory."""
        with self._lock:
            self.metrics = {}
            self.counters = {}
            self.gauges = {}


# Global metrics recorder instance
metrics = MetricsRecorder()


def configure_metrics(service_name: str, storage_path: Optional[str] = None) -> None:
    """
    Configure the global metrics recorder.
    
    Args:
        service_name: Name of the service
        storage_path: Path to store metrics
    """
    global metrics
    metrics = MetricsRecorder(service_name, storage_path)


def increment_counter(name: str, value: float = 1.0,
                    tags: Optional[Dict[str, str]] = None) -> None:
    """
    Increment a counter metric.
    
    Args:
        name: Metric name
        value: Value to increment by
        tags: Optional tags to associate with the metric
    """
    metrics.increment_counter(name, value, tags)


def set_gauge(name: str, value: float,
             tags: Optional[Dict[str, str]] = None) -> None:
    """
    Set a gauge metric.
    
    Args:
        name: Metric name
        value: Value to set
        tags: Optional tags to associate with the metric
    """
    metrics.set_gauge(name, value, tags)


def record_histogram(name: str, value: float,
                    tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a histogram value.
    
    Args:
        name: Metric name
        value: Value to record
        tags: Optional tags to associate with the metric
    """
    metrics.record_histogram(name, value, tags)


def time_this(name: str, tags: Optional[Dict[str, str]] = None):
    """
    Context manager for timing operations.
    
    Args:
        name: Metric name
        tags: Optional tags to associate with the metric
        
    Returns:
        Context manager for timing
    """
    return metrics.time_this(name, tags)


class PerformanceTracker:
    """
    Tracks performance of the trading system.
    
    Records metrics related to signal generation, order execution,
    and overall system performance.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        # Register useful metrics
        set_gauge("system.start_time", time.time())
        increment_counter("system.starts")
    
    def track_signal_generation(self, count: int, duration_ms: float,
                              signal_type: str) -> None:
        """
        Track signal generation performance.
        
        Args:
            count: Number of signals generated
            duration_ms: Time taken in milliseconds
            signal_type: Type of signals
        """
        # Record counts
        increment_counter("signals.generated", count)
        increment_counter(f"signals.generated.{signal_type}", count)
        
        # Record timing
        record_histogram("signals.generation_time_ms", duration_ms)
        record_histogram(f"signals.generation_time_ms.{signal_type}", duration_ms)
        
        # Calculate rate
        if duration_ms > 0:
            rate = count / (duration_ms / 1000.0)
            set_gauge("signals.generation_rate", rate)
    
    def track_order_execution(self, order_id: str, status: str,
                            duration_ms: float) -> None:
        """
        Track order execution performance.
        
        Args:
            order_id: Order ID
            status: Order status
            duration_ms: Time taken in milliseconds
        """
        # Record counts
        increment_counter("orders.total")
        increment_counter(f"orders.status.{status}")
        
        # Record timing
        record_histogram("orders.execution_time_ms", duration_ms)
        
        # Record specific tags
        tags = {"order_id": order_id, "status": status}
        record_histogram("orders.execution_time_ms.tagged", duration_ms, tags)
    
    def track_portfolio_metrics(self, total_value: float, cash: float,
                              positions_value: float) -> None:
        """
        Track portfolio metrics.
        
        Args:
            total_value: Total portfolio value
            cash: Cash balance
            positions_value: Value of positions
        """
        set_gauge("portfolio.total_value", total_value)
        set_gauge("portfolio.cash", cash)
        set_gauge("portfolio.positions_value", positions_value)
    
    def track_trade(self, symbol: str, quantity: int, price: float,
                  side: str, pnl: float = 0.0) -> None:
        """
        Track a trade.
        
        Args:
            symbol: Symbol traded
            quantity: Quantity traded
            price: Execution price
            side: Trade side (buy/sell)
            pnl: Realized P&L
        """
        increment_counter("trades.total")
        increment_counter(f"trades.side.{side}")
        
        # Record trade value
        trade_value = abs(quantity * price)
        record_histogram("trades.value", trade_value)
        
        # Record P&L
        if pnl != 0.0:
            record_histogram("trades.pnl", pnl)
            if pnl > 0:
                increment_counter("trades.profitable")
            else:
                increment_counter("trades.unprofitable")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {}
        
        # Signal statistics
        signals_generated = metrics.get_counter("signals.generated")
        signal_time = metrics.get_histogram_stats("signals.generation_time_ms")
        
        stats["signals"] = {
            "count": signals_generated,
            "avg_time_ms": signal_time.get("avg", 0.0)
        }
        
        # Order statistics
        orders_total = metrics.get_counter("orders.total")
        orders_filled = metrics.get_counter("orders.status.filled")
        orders_time = metrics.get_histogram_stats("orders.execution_time_ms")
        
        stats["orders"] = {
            "total": orders_total,
            "filled": orders_filled,
            "fill_rate": orders_filled / orders_total if orders_total > 0 else 0.0,
            "avg_time_ms": orders_time.get("avg", 0.0)
        }
        
        # Trade statistics
        trades_total = metrics.get_counter("trades.total")
        trades_profitable = metrics.get_counter("trades.profitable")
        trades_pnl = metrics.get_histogram_stats("trades.pnl")
        
        stats["trades"] = {
            "total": trades_total,
            "profitable": trades_profitable,
            "win_rate": trades_profitable / trades_total if trades_total > 0 else 0.0,
            "avg_pnl": trades_pnl.get("avg", 0.0)
        }
        
        # Portfolio statistics
        portfolio_value = metrics.get_gauge("portfolio.total_value")
        
        stats["portfolio"] = {
            "value": portfolio_value
        }
        
        return stats


# Create a global performance tracker
performance = PerformanceTracker() 