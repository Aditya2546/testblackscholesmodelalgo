#!/usr/bin/env python
"""
Compare Basic vs Realistic Options Trading Simulations

This script runs simulations using both the basic options trading simulator
and the enhanced realistic simulator with advanced factors, then compares
the results to show the impact of realistic trading conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from typing import Dict, List, Tuple

# Import both simulation modules
from options_day_trader_sim import OptionsDayTrader, OptionSignal, Config
from realistic_options_sim import RealisticOptionsDayTrader, RealisticConfig


def run_comparison(num_trades: int = 20, account_value: float = 25000.0, 
                  randomize_seed: bool = False):
    """
    Run a side-by-side comparison of basic vs realistic simulations.
    
    Args:
        num_trades: Number of trades to simulate for each model
        account_value: Starting account value
        randomize_seed: Whether to use different random seeds (True) or same seed for comparison (False)
    """
    # Set random seed for reproducibility if not randomizing
    if not randomize_seed:
        np.random.seed(42)
    
    print("\n" + "=" * 100)
    print("OPTIONS TRADING SIMULATION COMPARISON: BASIC vs REALISTIC")
    print("=" * 100)
    print(f"Starting Account: ${account_value:.2f}")
    print(f"Target Trades: {num_trades} per simulation")
    print("-" * 100)
    
    # Initialize both trader types
    basic_trader = OptionsDayTrader(account_value=account_value)
    realistic_trader = RealisticOptionsDayTrader(account_value=account_value)
    
    # Track metrics for comparison
    basic_trades = []
    realistic_trades = []
    basic_attempts = 0
    realistic_attempts = 0
    basic_failed_fills = 0
    realistic_failed_fills = 0
    
    # Lists to track equity curves
    basic_equity = [account_value]
    realistic_equity = [account_value]
    
    # Run each simulation
    for sim_type in ["basic", "realistic"]:
        if sim_type == "basic":
            trader = basic_trader
            trades_list = basic_trades
            print("\n🔹 RUNNING BASIC SIMULATION - No realistic factors")
        else:
            if not randomize_seed:
                np.random.seed(42)  # Reset seed for fair comparison
            trader = realistic_trader
            trades_list = realistic_trades
            print("\n🔸 RUNNING REALISTIC SIMULATION - Including slippage, commission structure, and fill probability")
            
        completed_trades = 0
        attempt_count = 0
            
        # Keep attempting trades until we have the requested number of completed trades
        while completed_trades < num_trades and attempt_count < num_trades * 4:
            attempt_count += 1
            
            # Scan for signals
            signals = trader.scan_for_signals()
            
            if signals:
                signal = signals[0]  # Take the first signal
                
                if sim_type == "basic":
                    # Basic trader just executes trades
                    result = trader.simulate_trade(signal)
                    trades_list.append(result)
                    completed_trades += 1
                    # Track equity for each trade
                    basic_equity.append(trader.account_value)
                else:
                    # Realistic trader has more sophisticated execution
                    order_types = ['limit', 'market']
                    price_points = ['bid', 'mid', 'ask']
                    
                    # Randomly choose order type (70% limit, 30% market)
                    order_type = np.random.choice(order_types, p=[0.7, 0.3])
                    
                    if order_type == 'limit':
                        # Price point distribution (20% bid, 60% mid, 20% ask)
                        price_point = np.random.choice(price_points, p=[0.2, 0.6, 0.2])
                    else:
                        price_point = 'ask'  # Market orders execute at ask
                    
                    # Execute the trade with realistic factors
                    result = trader.simulate_trade(signal, order_type=order_type, price_point=price_point)
                    
                    # Only count filled trades toward completed count
                    if result.get("filled", True):  # Basic trades are always "filled"
                        completed_trades += 1
                        trades_list.append(result)
                        realistic_equity.append(trader.account_value)
                    else:
                        realistic_failed_fills += 1
            
            # Small delay between attempts
            time.sleep(0.1)
            
        # Store metrics
        if sim_type == "basic":
            basic_attempts = attempt_count
        else:
            realistic_attempts = attempt_count
    
    # Print comparison results
    print("\n" + "=" * 100)
    print("SIMULATION COMPARISON RESULTS")
    print("=" * 100)
    
    # Filter out unfilled trades from realistic simulation
    filled_realistic_trades = [t for t in realistic_trades if t.get("filled", True)]
    
    # Calculate metrics for each simulation
    metrics = {
        "Basic": {
            "completed_trades": len(basic_trades),
            "attempt_count": basic_attempts,
            "fill_rate": 1.0,  # Always 100% in basic
            "win_count": sum(1 for t in basic_trades if t["net_profit"] > 0),
            "total_profit": sum(t["net_profit"] for t in basic_trades),
            "gross_profit": sum(t["gross_profit"] for t in basic_trades),
            "commission": sum(t["commission"] for t in basic_trades),
            "slippage": 0.0,  # No slippage in basic
            "avg_profit_per_trade": sum(t["net_profit"] for t in basic_trades) / len(basic_trades) if basic_trades else 0,
            "final_account": basic_equity[-1],
            "return_pct": (basic_equity[-1] / account_value - 1) * 100
        },
        "Realistic": {
            "completed_trades": len(filled_realistic_trades),
            "attempt_count": realistic_attempts,
            "fill_rate": len(filled_realistic_trades) / realistic_attempts if realistic_attempts > 0 else 0,
            "win_count": sum(1 for t in filled_realistic_trades if t["net_profit"] > 0),
            "total_profit": sum(t["net_profit"] for t in filled_realistic_trades),
            "gross_profit": sum(t["gross_profit"] for t in filled_realistic_trades),
            "commission": sum(t["total_commission"] for t in filled_realistic_trades),
            "slippage": sum(t.get("total_slippage", 0) for t in filled_realistic_trades),
            "avg_profit_per_trade": sum(t["net_profit"] for t in filled_realistic_trades) / len(filled_realistic_trades) if filled_realistic_trades else 0,
            "final_account": realistic_equity[-1],
            "return_pct": (realistic_equity[-1] / account_value - 1) * 100
        }
    }
    
    # Calculate win rates
    metrics["Basic"]["win_rate"] = metrics["Basic"]["win_count"] / metrics["Basic"]["completed_trades"] if metrics["Basic"]["completed_trades"] > 0 else 0
    metrics["Realistic"]["win_rate"] = metrics["Realistic"]["win_count"] / metrics["Realistic"]["completed_trades"] if metrics["Realistic"]["completed_trades"] > 0 else 0
    
    # Print comparison table
    print("Performance Metrics:")
    print("-" * 100)
    print(f"{'Metric':<30} {'Basic Simulation':<25} {'Realistic Simulation':<25} {'Difference':<20}")
    print("-" * 100)
    print(f"{'Trade Attempts':<30} {metrics['Basic']['attempt_count']:<25} {metrics['Realistic']['attempt_count']:<25} {metrics['Realistic']['attempt_count'] - metrics['Basic']['attempt_count']:<20}")
    print(f"{'Completed Trades':<30} {metrics['Basic']['completed_trades']:<25} {metrics['Realistic']['completed_trades']:<25} {metrics['Realistic']['completed_trades'] - metrics['Basic']['completed_trades']:<20}")
    
    # Fix the format specifier issue
    basic_fill = metrics['Basic']['fill_rate'] * 100
    realistic_fill = metrics['Realistic']['fill_rate'] * 100
    diff_fill = (metrics['Realistic']['fill_rate'] - metrics['Basic']['fill_rate']) * 100
    print(f"{'Fill Rate':<30} {basic_fill:.2f}%{'':<20} {realistic_fill:.2f}%{'':<20} {diff_fill:.2f}%{'':<15}")
    
    # Fix the win rate format too
    basic_win = metrics['Basic']['win_rate'] * 100
    realistic_win = metrics['Realistic']['win_rate'] * 100
    diff_win = (metrics['Realistic']['win_rate'] - metrics['Basic']['win_rate']) * 100
    print(f"{'Win Rate':<30} {basic_win:.2f}%{'':<20} {realistic_win:.2f}%{'':<20} {diff_win:.2f}%{'':<15}")
    
    print(f"{'Total Profit':<30} ${metrics['Basic']['total_profit']:.2f}{'':>18} ${metrics['Realistic']['total_profit']:.2f}{'':>18} ${metrics['Realistic']['total_profit'] - metrics['Basic']['total_profit']:.2f}{'':>13}")
    print(f"{'Gross Profit':<30} ${metrics['Basic']['gross_profit']:.2f}{'':>18} ${metrics['Realistic']['gross_profit']:.2f}{'':>18} ${metrics['Realistic']['gross_profit'] - metrics['Basic']['gross_profit']:.2f}{'':>13}")
    print(f"{'Commission Costs':<30} ${metrics['Basic']['commission']:.2f}{'':>18} ${metrics['Realistic']['commission']:.2f}{'':>18} ${metrics['Realistic']['commission'] - metrics['Basic']['commission']:.2f}{'':>13}")
    print(f"{'Slippage Costs':<30} ${metrics['Basic']['slippage']:.2f}{'':>18} ${metrics['Realistic']['slippage']:.2f}{'':>18} ${metrics['Realistic']['slippage']}{'':>13}")
    print(f"{'Avg Profit/Trade':<30} ${metrics['Basic']['avg_profit_per_trade']:.2f}{'':>18} ${metrics['Realistic']['avg_profit_per_trade']:.2f}{'':>18} ${metrics['Realistic']['avg_profit_per_trade'] - metrics['Basic']['avg_profit_per_trade']:.2f}{'':>13}")
    print(f"{'Final Account':<30} ${metrics['Basic']['final_account']:.2f}{'':>18} ${metrics['Realistic']['final_account']:.2f}{'':>18} ${metrics['Realistic']['final_account'] - metrics['Basic']['final_account']:.2f}{'':>13}")
    print(f"{'Return %':<30} {metrics['Basic']['return_pct']:+.2f}%{'':>18} {metrics['Realistic']['return_pct']:+.2f}%{'':>18} {metrics['Realistic']['return_pct'] - metrics['Basic']['return_pct']:+.2f}%{'':>13}")
    
    # Calculate impact of realistic factors
    impact = {
        "commission_impact": metrics["Realistic"]["commission"] - metrics["Basic"]["commission"],
        "slippage_impact": metrics["Realistic"]["slippage"],
        "fill_rate_impact": metrics["Basic"]["completed_trades"] - metrics["Realistic"]["completed_trades"],
    }
    
    # Total impact as percentage of basic simulation profit
    if metrics["Basic"]["total_profit"] != 0:
        total_profit_impact_pct = (metrics["Basic"]["total_profit"] - metrics["Realistic"]["total_profit"]) / metrics["Basic"]["total_profit"] * 100
    else:
        total_profit_impact_pct = 0
    
    print("\n" + "-" * 100)
    print("IMPACT OF REALISTIC FACTORS:")
    print("-" * 100)
    print(f"Commission Structure Impact: ${impact['commission_impact']:.2f}")
    print(f"Slippage Impact:             ${impact['slippage_impact']:.2f}")
    print(f"Fill Probability Impact:     {impact['fill_rate_impact']} missed trades")
    print(f"Total Profit Impact:         ${metrics['Basic']['total_profit'] - metrics['Realistic']['total_profit']:.2f} ({total_profit_impact_pct:.1f}% reduction)")
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(basic_equity, label='Basic Simulation', linewidth=2)
    plt.plot(realistic_equity, label='Realistic Simulation', linewidth=2)
    plt.axhline(y=account_value, color='r', linestyle='--', label='Starting Capital')
    plt.legend()
    plt.title('Account Equity Comparison: Basic vs Realistic Simulation')
    plt.xlabel('Trade Number')
    plt.ylabel('Account Value ($)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('simulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEquity curve comparison saved to 'simulation_comparison.png'")
    print("=" * 100)
    

if __name__ == "__main__":
    # Run with 20 trades at $25,000 starting account
    run_comparison(num_trades=20, account_value=25000.0, randomize_seed=False) 