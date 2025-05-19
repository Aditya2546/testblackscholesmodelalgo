#!/usr/bin/env python
"""
Alpaca API Configuration

This module contains Alpaca API credentials and settings.
DO NOT commit this file with your actual credentials to version control.
Use environment variables in production.
"""

import os

# Alpaca API credentials
# For security, prefer setting these as environment variables:
# export ALPACA_API_KEY="your-api-key"
# export ALPACA_API_SECRET="your-api-secret"
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "PKH3TQ6KZ0MY8CB7LY4O")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "k6OTSqi7aHT39sNnziAIZ9ztK2RhdLIumSLnOy67")

# Trading Environment Settings
USE_PAPER_TRADING = True  # Set to False for live trading

# API Rate Limiting Configuration
API_CALL_LIMIT_PER_MINUTE = 10000  # High-tier Alpaca subscription (10,000 calls/min)
API_CALL_LIMIT_PER_SECOND = API_CALL_LIMIT_PER_MINUTE / 60  # Approximately 166 calls/second

# Performance Optimization Settings
PARALLEL_API_CALLS = True  # Enable parallel API calls for faster data fetching
MAX_PARALLEL_REQUESTS = 16  # Maximum number of concurrent API requests
CACHE_TTL_MINUTES = 60  # How long to cache API responses in minutes

# Training Cycle Configuration 
DEFAULT_TRAINING_INTERVAL = 60  # Run a training cycle every 60 seconds
DEFAULT_BATCH_SIZE = 1024  # Larger batch size for faster training

# Symbols to analyze
DEFAULT_SYMBOLS = ["SPY", "AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "GOOGL", "META", "NFLX", "AMD"]

# Technical Indicator Parameters - Optimized for high-frequency training
INDICATOR_SETTINGS = {
    "rsi_period": 14,
    "fast_ema": 12,
    "slow_ema": 26,
    "signal_period": 9,
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "atr_period": 14,
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou_b": 52
}

# Additional API endpoints - for advanced users
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"

# API URLs as per documentation
ALPACA_API_BASE_URL = ALPACA_PAPER_BASE_URL if USE_PAPER_TRADING else ALPACA_LIVE_BASE_URL
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_OPTIONS_URL = "https://api.alpaca.markets/v2/options/contracts"

# Helper function to load credentials into environment
def load_credentials():
    """Load Alpaca credentials into environment variables."""
    # Only set environment variables if they don't already exist
    if not os.environ.get("ALPACA_API_KEY"):
        os.environ["ALPACA_API_KEY"] = ALPACA_API_KEY
    if not os.environ.get("ALPACA_API_SECRET"):
        os.environ["ALPACA_API_SECRET"] = ALPACA_API_SECRET
    
    os.environ["ALPACA_API_BASE_URL"] = ALPACA_API_BASE_URL
    os.environ["ALPACA_DATA_URL"] = ALPACA_DATA_URL
    os.environ["ALPACA_OPTIONS_URL"] = ALPACA_OPTIONS_URL
    
    # Print masked credentials for debugging
    masked_key = f"{ALPACA_API_KEY[:4]}...{ALPACA_API_KEY[-4:]}" if len(ALPACA_API_KEY) > 8 else "Not set"
    print(f"Loaded Alpaca credentials for API key: {masked_key}")
    print(f"Using {'PAPER' if USE_PAPER_TRADING else 'LIVE'} trading environment")

if __name__ == "__main__":
    load_credentials()
    print("Credentials loaded successfully. They can now be accessed through os.environ.")
    print("\nIMPORTANT: Set your actual API keys in this file or use environment variables:")
    print("  export ALPACA_API_KEY='your-api-key'")
    print("  export ALPACA_API_SECRET='your-api-secret'") 