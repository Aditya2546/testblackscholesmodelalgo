#!/usr/bin/env python
"""
Alpaca API Configuration

This module contains Alpaca API credentials and settings.
Do not commit this file to version control.
"""

import os

# Alpaca API credentials
ALPACA_API_KEY = "AKTJN56J3HTI2KWEEZ8A"
ALPACA_API_SECRET = "9uhVQTGFRzSRBkk1RWI0ovhBGWg3UDH2u7woInaX"

# API URLs as per documentation
ALPACA_API_BASE_URL = "https://api.alpaca.markets/v2"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_OPTIONS_URL = "https://api.alpaca.markets/v2/options/contracts"

# Helper function to load credentials into environment
def load_credentials():
    """Load Alpaca credentials into environment variables."""
    os.environ["ALPACA_API_KEY"] = ALPACA_API_KEY
    os.environ["ALPACA_API_SECRET"] = ALPACA_API_SECRET
    os.environ["ALPACA_API_BASE_URL"] = ALPACA_API_BASE_URL
    os.environ["ALPACA_DATA_URL"] = ALPACA_DATA_URL
    os.environ["ALPACA_OPTIONS_URL"] = ALPACA_OPTIONS_URL
    
    print(f"Loaded Alpaca credentials for API key: {ALPACA_API_KEY[:4]}...{ALPACA_API_KEY[-4:]}")

if __name__ == "__main__":
    load_credentials()
    print("Credentials loaded successfully. They can now be accessed through os.environ.") 