"""
Backtesting module for options trading strategies.

Provides an event-driven backtesting engine with realistic market simulation
and comprehensive performance analysis.
"""

from .backtester import (
    Backtester, BacktestPortfolio, BacktestExecution, BacktestResults,
    Event, EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent
)

__all__ = [
    'Backtester',
    'BacktestPortfolio',
    'BacktestExecution',
    'BacktestResults',
    'Event',
    'EventType',
    'MarketDataEvent',
    'SignalEvent',
    'OrderEvent',
    'FillEvent'
] 