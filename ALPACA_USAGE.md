# Using Alpaca Markets API with the Black-Scholes Options System

This document explains how to set up and use the Alpaca Markets API integration with our options trading system.

## Setup

1. Create an account on [Alpaca Markets](https://alpaca.markets/)
2. Get your API key and secret from the Alpaca dashboard
3. Update the `alpaca_config.py` file with your credentials

## Usage

Run the options trading system with Alpaca integration:

```bash
python3 main.py
```

This will:
1. Connect to Alpaca API
2. Fetch real-time market data
3. Generate trading signals based on current market conditions
4. Display recommended options trades

## Troubleshooting

### "No trading signals found from Alpaca API" Error

If you encounter this error, it may be due to one of the following reasons:

1. **API Access Level**: The free tier of Alpaca API doesn't include options market data. You need a paid subscription to access options data.

2. **Connection Issues**: There might be temporary connection issues with the Alpaca API.

3. **Market Hours**: Options data may not be available outside of market hours.

### Solutions

1. **Upgrade to Paid Subscription**: If you're serious about options trading, consider upgrading to Alpaca's paid tier that includes options market data.

2. **Use Simulated Data**: Our system includes a fallback mechanism that generates simulated options data when the API doesn't provide real data. This allows you to test the system's functionality without a paid subscription.

3. **Check API Status**: Visit [Alpaca Status](https://status.alpaca.markets/) to check if there are any known issues with the API.

## Simulated Data Mode

The system automatically falls back to using simulated options data when it can't retrieve real data from the API. This is indicated by messages like:

```
Error fetching options expirations: 404 - {"message":"Not Found"}
Falling back to simulated options data...
Generated simulated options chain for SPY with 2 expirations and 5 strikes
```

### Limitations of Simulated Data

While simulated data allows you to test the system, be aware of these limitations:

1. **Accuracy**: The simulated prices and Greeks are approximations and do not reflect actual market conditions.

2. **Expiration Dates**: We generate fixed expiration dates (today + 2 days and today + 5 days).

3. **Strike Prices**: We generate strikes around the current price, but they may not match actual available strikes.

4. **Volume and Open Interest**: These are generated values and do not reflect actual market liquidity.

## Contact

If you continue to experience issues, please contact our support team at support@blackscholesoptions.com. 