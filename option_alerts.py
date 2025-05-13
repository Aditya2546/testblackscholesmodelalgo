#!/usr/bin/env python
"""
Option Alerts Generator - Uses Market Data API to monitor for option trading opportunities
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from market_data_config import load_market_data_credentials
from integrate_market_data import MarketDataIntegration, TradingSignal

# Directory for storing alerts
ALERTS_DIR = "alerts"

@dataclass
class Alert:
    """Data class for option alerts"""
    id: str
    timestamp: str
    symbol: str
    option_type: str
    strike: float
    expiration: str
    current_price: float
    signal_strength: float
    underlying_price: float
    alert_type: str  # "entry", "exit", "target_hit", "stop_loss_hit"
    message: str
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    is_active: bool = True

def setup_alerts_directory():
    """Create the alerts directory if it doesn't exist"""
    if not os.path.exists(ALERTS_DIR):
        os.makedirs(ALERTS_DIR)
        print(f"Created alerts directory: {ALERTS_DIR}")

def generate_alert_id(symbol: str, option_type: str, strike: float, expiration: str) -> str:
    """Generate a unique alert ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{symbol}_{option_type}_{strike}_{expiration}_{timestamp}"

def save_alert(alert: Alert):
    """Save alert to a JSON file"""
    alert_dict = asdict(alert)
    filename = f"{ALERTS_DIR}/{alert.id}.json"
    
    with open(filename, 'w') as f:
        json.dump(alert_dict, f, indent=2)
    
    print(f"Alert saved to {filename}")

def signal_to_alert(signal: TradingSignal, alert_type: str, message: str) -> Alert:
    """Convert a trading signal to an alert"""
    alert_id = generate_alert_id(
        signal.symbol, 
        signal.option_type, 
        signal.strike, 
        signal.expiration
    )
    
    return Alert(
        id=alert_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbol=signal.symbol,
        option_type=signal.option_type,
        strike=signal.strike,
        expiration=signal.expiration,
        current_price=signal.current_price,
        signal_strength=signal.signal_strength,
        underlying_price=signal.underlying_price,
        alert_type=alert_type,
        message=message,
        implied_volatility=signal.implied_volatility,
        delta=signal.delta,
        gamma=signal.gamma,
        theta=signal.theta,
        vega=signal.vega
    )

def scan_for_alerts(integration: MarketDataIntegration, symbols: List[str], min_signal_strength: float = 0.6):
    """Scan for new alerts based on option signals"""
    signals = integration.get_option_signals(symbols, max_signals=30)
    alerts = []
    
    for signal in signals:
        if signal.signal_strength >= min_signal_strength:
            # Create entry alert
            message = (
                f"New {signal.option_type.upper()} option alert for {signal.symbol} at strike ${signal.strike}. "
                f"Current price: ${signal.current_price:.2f}, "
                f"Entry range: ${signal.entry_price_range[0]:.2f} - ${signal.entry_price_range[1]:.2f}, "
                f"Target: ${signal.target_price:.2f}, Stop loss: ${signal.stop_loss:.2f}"
            )
            
            alert = signal_to_alert(signal, "entry", message)
            alerts.append(alert)
            save_alert(alert)
    
    return alerts

def monitor_active_alerts(integration: MarketDataIntegration):
    """Monitor active alerts for target or stop loss hits"""
    # Get all active alerts
    active_alerts = []
    for filename in os.listdir(ALERTS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(ALERTS_DIR, filename), 'r') as f:
                alert_data = json.load(f)
                if alert_data.get('is_active', False):
                    active_alerts.append(alert_data)
    
    updated_alerts = []
    for alert_data in active_alerts:
        try:
            # Get current option data
            option_symbol = integration.api.format_option_symbol(
                ticker=alert_data['symbol'],
                expiration_date=alert_data['expiration'],
                strike_price=alert_data['strike'],
                option_type=alert_data['option_type'][0].upper()  # Convert 'call'/'put' to 'C'/'P'
            )
            
            option_data = integration.get_option_data(option_symbol)
            if option_data and option_data.get("s") in ["ok", "error"]:
                # Extract data
                extracted_data = integration.extract_option_data(option_data)
                
                bid = extracted_data.get("bid", 0)
                ask = extracted_data.get("ask", 0)
                mid = extracted_data.get("mid", (bid + ask) / 2 if bid and ask else 0)
                
                # Original alert details
                original_price = alert_data['current_price']
                target_price = original_price * 1.5  # 50% gain
                stop_loss = original_price * 0.7    # 30% loss
                
                # Check for target hit
                if mid >= target_price:
                    alert_data['is_active'] = False
                    alert_data['alert_type'] = 'target_hit'
                    alert_data['message'] = f"Target price of ${target_price:.2f} hit! Current price: ${mid:.2f}"
                    updated_alerts.append(alert_data)
                
                # Check for stop loss hit
                elif mid <= stop_loss:
                    alert_data['is_active'] = False
                    alert_data['alert_type'] = 'stop_loss_hit'
                    alert_data['message'] = f"Stop loss of ${stop_loss:.2f} hit! Current price: ${mid:.2f}"
                    updated_alerts.append(alert_data)
                
        except Exception as e:
            print(f"Error monitoring alert {alert_data['id']}: {e}")
    
    # Save updated alerts
    for alert_data in updated_alerts:
        filename = f"{ALERTS_DIR}/{alert_data['id']}.json"
        with open(filename, 'w') as f:
            json.dump(alert_data, f, indent=2)
        print(f"Updated alert: {alert_data['message']}")
    
    return updated_alerts

def main():
    """Main entry point for the option alerts system"""
    print("Options Trading Alert System")
    print("============================")
    
    # Setup alerts directory
    setup_alerts_directory()
    
    # Load Market Data API credentials
    try:
        credentials = load_market_data_credentials()
        print("Market Data API token loaded successfully")
    except Exception as e:
        print(f"Error loading Market Data API credentials: {e}")
        sys.exit(1)
    
    # Initialize MarketDataIntegration
    try:
        integration = MarketDataIntegration()
    except Exception as e:
        print(f"Error initializing Market Data integration: {e}")
        sys.exit(1)
    
    # Define symbols to scan
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "META"]
    
    # Set scan frequency parameters
    new_alerts_interval = 60 * 30  # 30 minutes for new alerts
    monitor_interval = 60 * 5      # 5 minutes for monitoring existing alerts
    
    try:
        last_new_scan = 0
        last_monitor_scan = 0
        
        print("Starting alert monitoring. Press Ctrl+C to exit.")
        while True:
            current_time = time.time()
            
            # Scan for new alerts
            if current_time - last_new_scan >= new_alerts_interval:
                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Scanning for new trading signals...")
                new_alerts = scan_for_alerts(integration, symbols)
                if new_alerts:
                    print(f"Found {len(new_alerts)} new alerts!")
                else:
                    print("No new alerts found.")
                last_new_scan = current_time
            
            # Monitor active alerts
            if current_time - last_monitor_scan >= monitor_interval:
                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Monitoring active alerts...")
                updated_alerts = monitor_active_alerts(integration)
                if updated_alerts:
                    print(f"Updated {len(updated_alerts)} alerts.")
                else:
                    print("No alerts updated.")
                last_monitor_scan = current_time
            
            # Sleep to avoid excessive API calls
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\nAlert monitoring stopped by user.")
    
    print("\nDone!")

def run_once():
    """Run a single scan for alerts without continuous monitoring"""
    print("Options Trading Alert System - Single Scan")
    print("=========================================")
    
    # Setup alerts directory
    setup_alerts_directory()
    
    # Load Market Data API credentials
    try:
        credentials = load_market_data_credentials()
        print("Market Data API token loaded successfully")
    except Exception as e:
        print(f"Error loading Market Data API credentials: {e}")
        sys.exit(1)
    
    # Initialize MarketDataIntegration
    try:
        integration = MarketDataIntegration()
    except Exception as e:
        print(f"Error initializing Market Data integration: {e}")
        sys.exit(1)
    
    # Define symbols to scan - smaller set for faster testing
    symbols = ["SPY", "QQQ", "AAPL"]
    
    # Scan for new alerts
    print("Scanning for new trading signals...")
    new_alerts = scan_for_alerts(integration, symbols)
    if new_alerts:
        print(f"Found {len(new_alerts)} new alerts!")
        for alert in new_alerts:
            print(f"- {alert.symbol} {alert.option_type.upper()} ${alert.strike} - {alert.message}")
    else:
        print("No new alerts found.")
    
    # Monitor existing alerts
    print("\nMonitoring existing alerts...")
    updated_alerts = monitor_active_alerts(integration)
    if updated_alerts:
        print(f"Updated {len(updated_alerts)} alerts.")
        for alert in updated_alerts:
            print(f"- {alert['symbol']} {alert['option_type'].upper()} ${alert['strike']} - {alert['message']}")
    else:
        print("No alerts updated.")
    
    print("\nDone!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once()
    else:
        main() 