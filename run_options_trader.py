#!/usr/bin/env python
"""
Options Trader Runner

This script runs the full options trading system, demonstrating all components
working together with either real market data or simulation.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/options_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import system components
from options_trader_system import OptionsTraderSystem

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Options Trader System Runner")
    
    # Mode selection
    parser.add_argument("--mode", choices=["live", "paper", "simulation"], default="simulation",
                       help="Trading mode (live, paper, or simulation)")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    # Simulation parameters
    parser.add_argument("--days", type=int, default=5,
                       help="Number of days to simulate (simulation mode only)")
    parser.add_argument("--account", type=float, default=100000,
                       help="Initial account balance")
    
    # Analysis only mode
    parser.add_argument("--analysis-only", action="store_true",
                       help="Run analysis only without executing trades")
    
    # API keys for command-line override
    parser.add_argument("--api-key", type=str, default=None,
                       help="Alpaca API key (overrides config/env)")
    parser.add_argument("--api-secret", type=str, default=None,
                       help="Alpaca API secret (overrides config/env)")
    
    # Schedule
    parser.add_argument("--run-at", type=str, default=None,
                       help="Run daily at specified time (format: HH:MM)")
    
    # Watchlist
    parser.add_argument("--watchlist", type=str, default=None,
                       help="Comma-separated list of symbols (overrides config)")
    
    return parser.parse_args()

def create_default_config(args):
    """
    Create default configuration if no config file is provided.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    config = {
        "market_data": {
            "primary_source": "alpaca",
            "backup_sources": ["yahoo"],
            "retry_count": 3,
            "retry_delay": 2
        },
        "analyzer": {
            "moneyness_range": 0.10,
            "min_volume": 50,
            "min_open_interest": 100,
            "max_spread_pct": 0.10,
            "days_to_expiry_min": 5,
            "days_to_expiry_max": 45,
            "target_profit_pct": 0.20,
            "stop_loss_pct": 0.10
        },
        "ml_model": {
            "model_type": "hybrid",
            "auto_update": True,
            "update_frequency": 24  # hours
        },
        "risk_management": {
            "max_position_size_pct": 0.05,  # % of account
            "max_total_exposure_pct": 0.25,  # % of account
            "max_per_symbol_exposure_pct": 0.10,  # % of account
            "kelly_fraction": 0.25,  # 1/4 Kelly criterion
            "max_delta_exposure": 50000,  # Total delta exposure
            "max_gamma_exposure": 5000,  # Total gamma exposure
            "max_vega_exposure": 10000,  # Total vega exposure
            "max_theta_exposure": -2000  # Max negative theta
        },
        "account": {
            "initial_balance": args.account,
            "commission_per_contract": 0.65,
            "slippage_model": "percent",
            "slippage_value": 0.01  # 1% slippage
        },
        "watchlist": [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"
        ],
        "signals": {
            "min_score": 0.35,
            "max_signals_per_symbol": 2,
            "max_total_signals": 20
        },
        "execution": {
            "trading_hours_start": "09:30",
            "trading_hours_end": "16:00",
            "timezone": "America/New_York"
        }
    }
    
    # Apply command line overrides
    if args.watchlist:
        config["watchlist"] = [s.strip() for s in args.watchlist.split(",")]
    
    # If API keys provided, store them (these will be used instead of creating config file)
    if args.api_key and args.api_secret:
        os.environ["APCA_API_KEY_ID"] = args.api_key
        os.environ["APCA_API_SECRET_KEY"] = args.api_secret
    
    return config

def save_config(config, filename="config.json"):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filename: Output filename
    """
    try:
        os.makedirs("config", exist_ok=True)
        path = os.path.join("config", filename)
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def run_analysis_only(trader):
    """
    Run analysis only mode.
    
    Args:
        trader: OptionsTraderSystem instance
    """
    logger.info("=== Analysis Only Mode ===")
    
    # Get trading signals
    signals_df = trader.get_trading_signals()
    
    if signals_df.empty:
        logger.warning("No valid trading signals found")
        return
    
    # Display top signals
    top_n = min(10, len(signals_df))
    logger.info(f"Top {top_n} trading signals:")
    
    for i, row in signals_df.head(top_n).iterrows():
        logger.info(f"{row['symbol']} {row['option_type'].upper()} ${row['strike']} "
                  f"Exp: {row['expiration']} "
                  f"Score: {row['combined_score']:.2f} "
                  f"Contracts: {row['max_contracts']} "
                  f"Size: ${row['actual_position_size']:.2f}")
    
    # Save analysis to file
    try:
        os.makedirs("output", exist_ok=True)
        output_file = f"output/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        signals_df.to_csv(output_file, index=False)
        logger.info(f"Analysis saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")

def run_simulation(trader, days):
    """
    Run simulation mode.
    
    Args:
        trader: OptionsTraderSystem instance
        days: Number of days to simulate
    """
    logger.info(f"=== Simulation Mode ({days} days) ===")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run simulation
    trader.simulate_trading_session(days=days)
    
    # Save trading history
    try:
        output_file = f"output/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            # Serialize datetime objects
            history = []
            for trade in trader.trading_history:
                trade_copy = trade.copy()
                trade_copy["timestamp"] = trade_copy["timestamp"].isoformat()
                if "exit_date" in trade_copy and trade_copy["exit_date"]:
                    trade_copy["exit_date"] = trade_copy["exit_date"].isoformat()
                if "expiration" in trade_copy and trade_copy["expiration"]:
                    trade_copy["expiration"] = trade_copy["expiration"].isoformat()
                history.append(trade_copy)
            
            json.dump(history, f, indent=2)
            
        logger.info(f"Simulation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving simulation results: {e}")

def run_paper_trading(trader):
    """
    Run paper trading mode.
    
    Args:
        trader: OptionsTraderSystem instance
    """
    logger.info("=== Paper Trading Mode ===")
    
    # Run trading cycle once
    trader.run_trading_cycle()
    
    # Display recent trades
    recent_trades = trader.trading_history[-10:] if len(trader.trading_history) > 10 else trader.trading_history
    
    if recent_trades:
        logger.info("Recent trades:")
        for trade in recent_trades:
            logger.info(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
                      f"{trade['symbol']} {trade['option_type'].upper()} ${trade['strike']} "
                      f"x{trade['quantity']} @ ${trade['entry_price']:.2f} "
                      f"Status: {trade['status']}")
    else:
        logger.info("No recent trades")

def run_live_trading(trader):
    """
    Run live trading mode.
    
    Args:
        trader: OptionsTraderSystem instance
    """
    logger.info("=== Live Trading Mode ===")
    logger.warning("Live trading is in development. Running in analysis-only mode for now.")
    
    # Run analysis only for now
    run_analysis_only(trader)

def schedule_daily_run(args):
    """
    Schedule the script to run daily at a specific time.
    
    Args:
        args: Command line arguments
    """
    if not args.run_at:
        return
    
    try:
        run_hour, run_minute = map(int, args.run_at.split(":"))
        
        while True:
            now = datetime.now()
            
            # Calculate time until next run
            target_time = now.replace(hour=run_hour, minute=run_minute, second=0, microsecond=0)
            if target_time < now:
                # If target time is in the past, schedule for tomorrow
                import datetime as dt
                target_time = target_time + dt.timedelta(days=1)
            
            # Calculate seconds until next run
            seconds_until_run = (target_time - now).total_seconds()
            
            logger.info(f"Scheduled to run at {target_time.strftime('%Y-%m-%d %H:%M:%S')} "
                       f"({seconds_until_run:.0f} seconds from now)")
            
            # Sleep until target time
            time.sleep(seconds_until_run)
            
            # Run the system
            logger.info(f"Running scheduled execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create a new trader system
            config_path = args.config
            trader = OptionsTraderSystem(config_path=config_path)
            
            # Run in specified mode
            if args.mode == "live":
                run_live_trading(trader)
            elif args.mode == "paper":
                run_paper_trading(trader)
            elif args.mode == "simulation":
                run_simulation(trader, args.days)
            
            logger.info("Scheduled run completed")
            
            # Wait a minute before checking again
            time.sleep(60)
    
    except ValueError:
        logger.error(f"Invalid time format for --run-at: {args.run_at}. Use HH:MM format.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
        sys.exit(0)

def main():
    """Main function"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    # Check if we should run in scheduled mode
    if args.run_at:
        schedule_daily_run(args)
        return
    
    # Create configuration
    if args.config and os.path.exists(args.config):
        config_path = args.config
        logger.info(f"Using configuration from {config_path}")
    else:
        # Create default config
        config = create_default_config(args)
        
        # Save config
        save_config(config)
        
        # Use saved config
        config_path = "config/config.json"
    
    # Create trader system
    trader = OptionsTraderSystem(config_path=config_path)
    
    # Run in specified mode
    if args.analysis_only:
        run_analysis_only(trader)
    elif args.mode == "live":
        run_live_trading(trader)
    elif args.mode == "paper":
        run_paper_trading(trader)
    elif args.mode == "simulation":
        run_simulation(trader, args.days)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1) 