#!/usr/bin/env python
"""
Configuration Settings for Options Trading System

This file contains configuration settings and constants used throughout the application.
"""

class Config:
    """Configuration settings for the options trading system"""
    
    # General settings
    DEBUG_MODE = False
    
    # API settings
    API_RETRY_ATTEMPTS = 3
    API_RETRY_DELAY = 2  # seconds
    
    # Trading settings
    MAX_POSITION_SIZE_PCT = 0.05  # Maximum position size as percentage of account
    PROFIT_TARGET_PCT = 0.30      # Target profit percentage
    STOP_LOSS_PCT = 0.15          # Stop loss percentage
    MAX_TRADE_DURATION = 120      # Maximum trade duration in minutes
    COMMISSION_PER_CONTRACT = 0.65  # Commission per contract
    CONTRACT_MULTIPLIER = 100     # Standard options contract multiplier
    
    # Risk management
    DEFAULT_RISK_PER_TRADE = 0.02  # Default risk per trade (2% of account)
    MAX_DAILY_LOSS_PCT = 0.05      # Maximum daily loss percentage
    MAX_OPEN_POSITIONS = 5         # Maximum number of open positions
    
    # Scanner settings
    DEFAULT_SCAN_DAYS_TO_EXPIRY = 30  # Default days to expiration for scanner
    DEFAULT_MAX_SIGNALS = 10           # Default maximum number of signals
    
    # ML model settings
    MIN_ML_PROBABILITY = 0.4       # Minimum ML probability threshold
    DEFAULT_MODEL_PATH = "models"  # Default path for ML models
    
    # Data collection settings
    DATA_DIR = "training_data"
    HISTORICAL_DATA_DIR = f"{DATA_DIR}/historical"
    REALTIME_DATA_DIR = f"{DATA_DIR}/realtime"
    PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
    
    # UI settings
    DEFAULT_ACCOUNT_VALUE = 1000.0  # Default account value

    # Risk management settings
    POSITION_SIZE_PCT = 0.10  # 10% of account per trade
    STOP_LOSS_PCT = 0.10      # 10% stop loss
    PROFIT_TARGET_PCT = 0.20  # 20% profit target
    
    # Trading parameters
    DAYS_TO_EXPIRATION = 5    # Look for options expiring in 5 days
    MAX_HOLD_MINUTES = 75     # Maximum hold time in minutes
    
    # Default watchlist
    WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
    
    # Backtesting parameters
    BACKTEST_PERIODS = 3      # Number of backtest periods to run
    TRADES_PER_PERIOD = 20    # Number of trades per backtest period
    
    # Data collection settings
    HISTORICAL_DATA_DAYS = 30  # Days of historical data to collect
    
    # API limits and timeouts
    API_RATE_LIMIT = 10        # Requests per second
    API_TIMEOUT = 10           # Seconds to timeout API requests 