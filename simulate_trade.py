#!/usr/bin/env python
"""
Options Trading System - Trade Simulation

This script simulates a specific options trade with detailed execution 
and performance tracking.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.market_data import OptionChain, Quote, OptionQuote, OptionRight
from src.strategy.signal_generator import SignalManager
from src.strategy.signal import SignalType, Signal
from src.pricing.model_interface import ModelFactory
from src.risk.position_sizing import KellyPositionSizer
from src.backtester.backtester import Backtester, BacktestResults
from src.observability.metrics import configure_metrics, performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spy_iron_condor(underlying_price: float = 450.0) -> Signal:
    """
    Create an iron condor signal for SPY.
    
    An iron condor is a neutral options strategy combining a bull put spread
    and a bear call spread.
    
    Args:
        underlying_price: Current price of the underlying
        
    Returns:
        Signal for an iron condor trade
    """
    expiration = datetime.now() + timedelta(days=30)
    
    # Calculate strikes (roughly 5% OTM for the short strikes)
    put_short_strike = round(underlying_price * 0.95, 0)
    put_long_strike = put_short_strike - 10
    call_short_strike = round(underlying_price * 1.05, 0)
    call_long_strike = call_short_strike + 10
    
    # Create option legs
    legs = [
        {
            # Sell put (short put)
            "expiration": expiration,
            "strike": put_short_strike,
            "right": OptionRight.PUT,
            "quantity": -1,  # Sell
            "price": 3.50,
            "greeks": {"delta": -0.30, "gamma": 0.01, "theta": 0.05, "vega": 0.10}
        },
        {
            # Buy put (long put)
            "expiration": expiration,
            "strike": put_long_strike,
            "right": OptionRight.PUT,
            "quantity": 1,  # Buy
            "price": 2.20,
            "greeks": {"delta": -0.20, "gamma": 0.01, "theta": 0.03, "vega": 0.08}
        },
        {
            # Sell call (short call)
            "expiration": expiration,
            "strike": call_short_strike,
            "right": OptionRight.CALL,
            "quantity": -1,  # Sell
            "price": 3.25,
            "greeks": {"delta": 0.30, "gamma": 0.01, "theta": 0.05, "vega": 0.10}
        },
        {
            # Buy call (long call)
            "expiration": expiration,
            "strike": call_long_strike,
            "right": OptionRight.CALL,
            "quantity": 1,  # Buy
            "price": 2.00,
            "greeks": {"delta": 0.20, "gamma": 0.01, "theta": 0.03, "vega": 0.08}
        }
    ]
    
    # Calculate net credit and max profit/loss
    net_credit = legs[0]["price"] - legs[1]["price"] + legs[2]["price"] - legs[3]["price"]
    max_profit = net_credit * 100  # Per contract
    max_loss = ((put_short_strike - put_long_strike) - net_credit) * 100  # Per contract
    
    signal = Signal(
        timestamp=datetime.now(),
        signal_type=SignalType.SPREAD,
        symbol="SPY",
        direction=1,  # Direction doesn't matter much for iron condor (neutral)
        expiration=expiration,
        market_price=net_credit,
        expected_edge=50.0,
        expected_pnl=max_profit * 0.7,  # Assuming we take profit at 70% of max
        confidence=0.7,
        legs=legs
    )
    
    logger.info(f"Created Iron Condor on SPY:")
    logger.info(f"  Short Put: Strike {put_short_strike} @ ${legs[0]['price']:.2f}")
    logger.info(f"  Long Put: Strike {put_long_strike} @ ${legs[1]['price']:.2f}")
    logger.info(f"  Short Call: Strike {call_short_strike} @ ${legs[2]['price']:.2f}")
    logger.info(f"  Long Call: Strike {call_long_strike} @ ${legs[3]['price']:.2f}")
    logger.info(f"  Net Credit: ${net_credit:.2f} per contract (${net_credit * 100:.2f} total)")
    logger.info(f"  Max Profit: ${max_profit:.2f}")
    logger.info(f"  Max Loss: ${max_loss:.2f}")
    
    return signal


def simulate_market_scenario(days: int = 30, price_drift: float = 0.0, vol_change: float = 0.0) -> pd.DataFrame:
    """
    Create a simulated market scenario.
    
    Args:
        days: Number of days to simulate
        price_drift: Percentage drift in price over the period (e.g., 0.05 = 5% up)
        vol_change: Percentage change in volatility (e.g., -0.1 = 10% decrease)
        
    Returns:
        DataFrame with price data
    """
    # Create dates (market days only)
    dates = []
    current_date = datetime.now()
    for _ in range(days):
        # Skip weekends
        if current_date.weekday() < 5:  # Monday to Friday
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Starting price
    start_price = 450.0
    end_price = start_price * (1 + price_drift)
    
    # Create price series
    # Linear trend + random noise
    price_trend = np.linspace(start_price, end_price, len(dates))
    price_noise = np.random.normal(0, 2, len(dates))
    prices = price_trend + price_noise
    
    # Vol variation
    start_vol = 0.20
    end_vol = start_vol * (1 + vol_change)
    vol_trend = np.linspace(start_vol, end_vol, len(dates))
    
    # Create DataFrame
    data = {
        'time': dates,
        'open': prices + np.random.normal(0, 0.5, len(dates)),
        'high': prices + np.random.normal(1, 0.5, len(dates)),
        'low': prices - np.random.normal(1, 0.5, len(dates)),
        'close': prices,
        'bid': prices - 0.05,
        'ask': prices + 0.05,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'implied_vol': vol_trend
    }
    
    return pd.DataFrame(data)


def run_iron_condor_simulation() -> BacktestResults:
    """
    Run a backtest simulation of an iron condor trade on SPY.
    
    Returns:
        Backtest results
    """
    logger.info("Running Iron Condor backtest simulation")
    
    # Create options trade
    current_price = 450.0
    signal = create_spy_iron_condor(current_price)
    
    # Create simulated market data (30 days with 0% drift = flat market)
    market_data = {'SPY': simulate_market_scenario(days=30, price_drift=0.0, vol_change=-0.2)}
    
    # Create custom signal generator that just returns our iron condor signal
    class CustomIronCondorGenerator(SignalManager):
        def generate_signals(self, chain, underlying_price):
            # Return our predefined signal
            return [signal]
    
    signal_gen = CustomIronCondorGenerator()
    
    # Create a backtest
    backtest = Backtester(
        strategy_generator=signal_gen,
        initial_capital=100000.0
    )
    
    # Create option chains for each day
    option_chains = {}
    for date in market_data['SPY']['time'].unique():
        # Create option chain for this date
        chain = OptionChain(
            underlying_symbol="SPY",
            timestamp=date
        )
        
        # Get price data for this date
        row = market_data['SPY'][market_data['SPY']['time'] == date].iloc[0]
        current_price = row['close']
        implied_vol = row['implied_vol']
        
        # Create options for the iron condor legs
        for leg in signal.legs:
            strike = leg["strike"]
            expiration = leg["expiration"]
            right = leg["right"]
            
            # Calculate appropriate price based on the market scenario
            # For demonstration purposes, we're using a simple pricing model
            # In a real system, you'd use a more sophisticated model
            days_to_expiry = (expiration - date).days
            if days_to_expiry < 1:
                days_to_expiry = 1
                
            time_decay_factor = days_to_expiry / 30  # Linear time decay (simplified)
            vol_adjust = implied_vol / 0.20  # Adjust for vol changes
            
            if right == OptionRight.PUT:
                # PUT option price adjustment
                moneyness = strike / current_price
                if moneyness < 0.95:  # Out of the money
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:  # At or in the money
                    price = max(0, strike - current_price) + leg["price"] * time_decay_factor * vol_adjust
            else:
                # CALL option price adjustment
                moneyness = current_price / strike
                if moneyness < 0.95:  # Out of the money
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:  # At or in the money
                    price = max(0, current_price - strike) + leg["price"] * time_decay_factor * vol_adjust
            
            # Create synthetic option quote
            option_quote = OptionQuote(
                timestamp=date,
                bid_price=price * 0.95,
                ask_price=price * 1.05,
                bid_size=10,
                ask_size=10,
                underlying_symbol="SPY",
                expiration=expiration,
                strike=strike,
                right=right,
                exchange="BACKTEST",
                last_price=price,
                volume=100,
                open_interest=1000,
                implied_volatility=implied_vol,
                delta=leg["greeks"]["delta"] * time_decay_factor,
                gamma=leg["greeks"]["gamma"] * time_decay_factor,
                theta=leg["greeks"]["theta"] / time_decay_factor,
                vega=leg["greeks"]["vega"] * time_decay_factor,
                rho=0.01
            )
            
            # Add option to chain
            chain.add_option(option_quote)
        
        # Add chain to dictionary
        if date not in option_chains:
            option_chains[date] = {}
        option_chains[date]['SPY'] = chain
    
    # Load data into backtest
    backtest.load_market_data(market_data, option_chains)
    
    # Run backtest
    results = backtest.run()
    
    return results


def analyze_results(results: BacktestResults) -> None:
    """
    Analyze and display backtest results.
    
    Args:
        results: Backtest results
    """
    logger.info("=== Iron Condor Backtest Results ===")
    logger.info(f"Initial capital: ${results.initial_capital:,.2f}")
    logger.info(f"Final capital: ${results.final_capital:,.2f}")
    logger.info(f"Total return: {results.total_return:.2f}%")
    logger.info(f"Annualized return: {results.annualized_return:.2f}%")
    logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max drawdown: {results.max_drawdown:.2f}%")
    logger.info(f"Win rate: {results.performance_metrics.get('win_rate', 0):.2f}%")
    logger.info(f"Total trades: {results.performance_metrics.get('total_trades', 0)}")
    
    # Show trades if any
    if not results.trades.empty:
        logger.info("\nTrade Details:")
        for i, trade in results.trades.iterrows():
            logger.info(f"Trade {i+1}:")
            logger.info(f"  Symbol: {trade.get('symbol')}")
            logger.info(f"  Entry: {trade.get('entry_time')} @ ${trade.get('entry_price'):.2f}")
            logger.info(f"  Exit: {trade.get('exit_time')} @ ${trade.get('exit_price'):.2f}")
            logger.info(f"  P&L: ${trade.get('pnl'):.2f} ({trade.get('return')*100:.2f}%)")
    
    # Show final positions if any
    if not results.positions.empty:
        logger.info("\nOpen Positions:")
        for i, pos in results.positions.iterrows():
            logger.info(f"Position {i+1}:")
            logger.info(f"  Symbol: {pos.get('symbol')}")
            logger.info(f"  Quantity: {pos.get('quantity')}")
            logger.info(f"  Entry Price: ${pos.get('entry_price'):.2f}")
            logger.info(f"  Current Price: ${pos.get('current_price'):.2f}")
            logger.info(f"  Current Value: ${pos.get('current_value'):.2f}")
            logger.info(f"  Unrealized P&L: ${pos.get('unrealized_pnl'):.2f}")


if __name__ == "__main__":
    # Configure metrics
    configure_metrics("iron_condor_simulation")
    
    # Run multiple market scenarios
    logger.info("=== Testing Iron Condor in Different Market Scenarios ===")
    
    # Scenario 1: Flat market with volatility decrease (ideal for iron condor)
    logger.info("\n=== SCENARIO 1: FLAT MARKET WITH VOL DECREASE ===")
    market_data = {'SPY': simulate_market_scenario(days=30, price_drift=0.0, vol_change=-0.2)}
    signal = create_spy_iron_condor(450.0)
    class Scenario1Generator(SignalManager):
        def generate_signals(self, chain, underlying_price):
            return [signal]
    
    backtest = Backtester(strategy_generator=Scenario1Generator(), initial_capital=100000.0)
    option_chains = {}
    for date in market_data['SPY']['time'].unique():
        chain = OptionChain(underlying_symbol="SPY", timestamp=date)
        row = market_data['SPY'][market_data['SPY']['time'] == date].iloc[0]
        current_price = row['close']
        implied_vol = row['implied_vol']
        
        for leg in signal.legs:
            strike = leg["strike"]
            expiration = leg["expiration"]
            right = leg["right"]
            
            days_to_expiry = max(1, (expiration - date).days)
            time_decay_factor = days_to_expiry / 30
            vol_adjust = implied_vol / 0.20
            
            if right == OptionRight.PUT:
                moneyness = strike / current_price
                if moneyness < 0.95:
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:
                    price = max(0, strike - current_price) + leg["price"] * time_decay_factor * vol_adjust
            else:
                moneyness = current_price / strike
                if moneyness < 0.95:
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:
                    price = max(0, current_price - strike) + leg["price"] * time_decay_factor * vol_adjust
            
            option_quote = OptionQuote(
                timestamp=date, bid_price=price * 0.95, ask_price=price * 1.05,
                bid_size=10, ask_size=10, underlying_symbol="SPY",
                expiration=expiration, strike=strike, right=right,
                exchange="BACKTEST", last_price=price, volume=100,
                open_interest=1000, implied_volatility=implied_vol,
                delta=leg["greeks"]["delta"] * time_decay_factor,
                gamma=leg["greeks"]["gamma"] * time_decay_factor,
                theta=leg["greeks"]["theta"] / time_decay_factor,
                vega=leg["greeks"]["vega"] * time_decay_factor, rho=0.01
            )
            chain.add_option(option_quote)
        
        if date not in option_chains:
            option_chains[date] = {}
        option_chains[date]['SPY'] = chain
    
    backtest.load_market_data(market_data, option_chains)
    results_scenario1 = backtest.run()
    analyze_results(results_scenario1)
    
    # Scenario 2: Bullish market that breaches the call spread
    logger.info("\n=== SCENARIO 2: BULLISH MARKET (BREACHES CALL SPREAD) ===")
    market_data = {'SPY': simulate_market_scenario(days=30, price_drift=0.08, vol_change=0.1)}
    signal = create_spy_iron_condor(450.0)
    class Scenario2Generator(SignalManager):
        def generate_signals(self, chain, underlying_price):
            return [signal]
    
    backtest = Backtester(strategy_generator=Scenario2Generator(), initial_capital=100000.0)
    option_chains = {}
    for date in market_data['SPY']['time'].unique():
        chain = OptionChain(underlying_symbol="SPY", timestamp=date)
        row = market_data['SPY'][market_data['SPY']['time'] == date].iloc[0]
        current_price = row['close']
        implied_vol = row['implied_vol']
        
        for leg in signal.legs:
            strike = leg["strike"]
            expiration = leg["expiration"]
            right = leg["right"]
            
            days_to_expiry = max(1, (expiration - date).days)
            time_decay_factor = days_to_expiry / 30
            vol_adjust = implied_vol / 0.20
            
            if right == OptionRight.PUT:
                moneyness = strike / current_price
                if moneyness < 0.95:
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:
                    price = max(0, strike - current_price) + leg["price"] * time_decay_factor * vol_adjust
            else:
                moneyness = current_price / strike
                if moneyness < 0.95:
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:
                    price = max(0, current_price - strike) + leg["price"] * time_decay_factor * vol_adjust
            
            option_quote = OptionQuote(
                timestamp=date, bid_price=price * 0.95, ask_price=price * 1.05,
                bid_size=10, ask_size=10, underlying_symbol="SPY",
                expiration=expiration, strike=strike, right=right,
                exchange="BACKTEST", last_price=price, volume=100,
                open_interest=1000, implied_volatility=implied_vol,
                delta=leg["greeks"]["delta"] * time_decay_factor,
                gamma=leg["greeks"]["gamma"] * time_decay_factor,
                theta=leg["greeks"]["theta"] / time_decay_factor,
                vega=leg["greeks"]["vega"] * time_decay_factor, rho=0.01
            )
            chain.add_option(option_quote)
        
        if date not in option_chains:
            option_chains[date] = {}
        option_chains[date]['SPY'] = chain
    
    backtest.load_market_data(market_data, option_chains)
    results_scenario2 = backtest.run()
    analyze_results(results_scenario2)
    
    # Scenario 3: Bearish market that breaches the put spread
    logger.info("\n=== SCENARIO 3: BEARISH MARKET (BREACHES PUT SPREAD) ===")
    market_data = {'SPY': simulate_market_scenario(days=30, price_drift=-0.08, vol_change=0.15)}
    signal = create_spy_iron_condor(450.0)
    class Scenario3Generator(SignalManager):
        def generate_signals(self, chain, underlying_price):
            return [signal]
    
    backtest = Backtester(strategy_generator=Scenario3Generator(), initial_capital=100000.0)
    option_chains = {}
    for date in market_data['SPY']['time'].unique():
        chain = OptionChain(underlying_symbol="SPY", timestamp=date)
        row = market_data['SPY'][market_data['SPY']['time'] == date].iloc[0]
        current_price = row['close']
        implied_vol = row['implied_vol']
        
        for leg in signal.legs:
            strike = leg["strike"]
            expiration = leg["expiration"]
            right = leg["right"]
            
            days_to_expiry = max(1, (expiration - date).days)
            time_decay_factor = days_to_expiry / 30
            vol_adjust = implied_vol / 0.20
            
            if right == OptionRight.PUT:
                moneyness = strike / current_price
                if moneyness < 0.95:
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:
                    price = max(0, strike - current_price) + leg["price"] * time_decay_factor * vol_adjust
            else:
                moneyness = current_price / strike
                if moneyness < 0.95:
                    price = leg["price"] * time_decay_factor * vol_adjust
                else:
                    price = max(0, current_price - strike) + leg["price"] * time_decay_factor * vol_adjust
            
            option_quote = OptionQuote(
                timestamp=date, bid_price=price * 0.95, ask_price=price * 1.05,
                bid_size=10, ask_size=10, underlying_symbol="SPY",
                expiration=expiration, strike=strike, right=right,
                exchange="BACKTEST", last_price=price, volume=100,
                open_interest=1000, implied_volatility=implied_vol,
                delta=leg["greeks"]["delta"] * time_decay_factor,
                gamma=leg["greeks"]["gamma"] * time_decay_factor,
                theta=leg["greeks"]["theta"] / time_decay_factor,
                vega=leg["greeks"]["vega"] * time_decay_factor, rho=0.01
            )
            chain.add_option(option_quote)
        
        if date not in option_chains:
            option_chains[date] = {}
        option_chains[date]['SPY'] = chain
    
    backtest.load_market_data(market_data, option_chains)
    results_scenario3 = backtest.run()
    analyze_results(results_scenario3) 