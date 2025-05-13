# Market Data API Integration

This documentation explains how to use the Market Data API integration for the Black-Scholes Options Trading System.

## Overview

The Market Data API integration allows the trading system to fetch real-time and historical options data from the Market Data API. This is an alternative to using Alpaca for options data, especially while waiting for Alpaca account approval.

## Configuration

The integration uses the Market Data API token, which can be configured in multiple ways:

1. Environment Variable: Set the `MARKET_DATA_API_TOKEN` environment variable.
2. .env File: Create a `.env` file with `MARKET_DATA_API_TOKEN=your_token_here`.
3. Default Token: The system includes a default token for testing.

## Files

- `market_data_api.py`: Core API client for interacting with the Market Data API
- `market_data_config.py`: Configuration management for API tokens
- `integrate_market_data.py`: Integration class that generates trading signals based on options data
- `main_market_data.py`: Main entry point for running the system with Market Data API
- `test_market_data_api.py`: Test script demonstrating API features

## Usage

### Running the System

To run the system with Market Data API:

```bash
python main_market_data.py
```

This will:
1. Load credentials
2. Initialize the Market Data API integration
3. Scan a predefined set of symbols for options trading signals
4. Display the generated trading signals

### Testing the API

To test the API functionality:

```bash
python test_market_data_api.py
```

This will:
1. Format an example option symbol
2. Fetch real-time option quotes
3. Fetch historical option data
4. Generate a price chart

## Features

The Market Data API integration provides:

- Real-time option quotes including bid, ask, last price, etc.
- Greek values (delta, gamma, theta, vega, rho)
- Implied volatility
- Historical options data
- Option symbol formatting in OCC format
- Trading signal generation based on option analysis

## API Documentation

For more information about the Market Data API, visit:
- API Documentation: https://docs.marketdata.app/api/options/quotes
- Market Data Website: https://marketdata.app

## Sample Option Symbol Format

The system uses the OCC option symbol format:

```
{Symbol}{ExpYYMMDD}{C/P}{Strike price}
```

Example: `AAPL250117C00150000` represents:
- AAPL: Underlying stock symbol
- 250117: Expiration date (Jan 17, 2025)
- C: Call option (P for put)
- 00150000: Strike price of $150.00 (multiplied by 1000) 