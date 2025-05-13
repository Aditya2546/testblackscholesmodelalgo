# Options Day Trading Alert System

A Python-based system for day trading options that generates alerts with specific entry, exit, and risk management parameters.

## Features

- Scans popular stocks for day trading options opportunities
- Provides clear entry price ranges to buy at
- Sets specific stop loss and target profit levels
- Includes expiration date information
- Calculates risk/reward ratios
- Recommends position sizing
- Shows key option metrics (volume, open interest, IV, delta)

## Usage

### Basic Usage

```bash
python3 options_day_trader.py
```

This will scan for options trading opportunities every 60 seconds and generate up to 5 alerts.

### Custom Scanning

```bash
python3 options_day_trader.py --interval 30 --alerts 10
```

- `--interval`: Seconds between market scans (default: 60)
- `--alerts`: Number of alerts to generate before exiting (default: 5)

## How to Day Trade with the Alerts

1. Run the scanner during market hours
2. When you get an alert, check if the current option price is within the entry range
3. If so, place a buy order with the specified number of contracts
4. Set your stop loss and take profit orders according to the alert
5. Monitor the position and close it if it hits your targets

## Sample Alert

```
================================================================================
🔔 OPTION ALERT: SPY $450.75 - CALL $455 05/12
--------------------------------------------------------------------------------
ACTION:      BUY TO OPEN
CONTRACTS:   2 ($836.00)
ENTRY PRICE: $4.02-$4.36
STOP LOSS:   $2.97 (-30.0%) - Max Risk: $258.00
TARGET:      $6.30 (50.0%) - Max Profit: $418.00
RISK/REWARD: 1:1.62
EXPIRES:     Monday, May 12, 2025 (0 days)
SIGNAL STRENGTH: 0.62/1.0
--------------------------------------------------------------------------------
VOLUME: 1523 | OPEN INT: 5412 | IV: 0.3% | DELTA: 0.25
================================================================================
```

## Customization

You can modify the trading parameters by editing the `Config` class in the script:

- Watchlist of symbols to monitor
- Signal thresholds (RSI, momentum)
- Option selection criteria (days to expiration, delta, volume)
- Risk management settings (position size, stop loss, profit target)

## Requirements

- Python 3.6+
- NumPy
- Pandas

## Installation

```bash
# Install required packages
pip install numpy pandas
```

## Disclaimer

This is a simulated trading system for educational purposes only. The alerts are based on simulated market data and technical indicators. In a real implementation, you would need to:

1. Connect to a broker API to get real market data
2. Implement actual technical indicators
3. Test thoroughly before using with real money

Always use proper risk management when trading options. Options trading involves significant risk and is not suitable for all investors. 