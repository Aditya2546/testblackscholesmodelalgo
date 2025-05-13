#!/usr/bin/env python
"""
Realistic Smart Options Trader

This module extends the basic SmartTrader with realistic trading factors:
1. Slippage and liquidity effects
2. Detailed commission structure 
3. Fill probability modeling

These factors enhance the learning system by providing more accurate simulation.
"""

import numpy as np
import datetime
import time
import os
import json
from typing import Dict, List, Tuple, Optional

# Import base smart trader and realistic simulation models
from smart_trader import SmartTrader, TradingPattern
from realistic_options_sim import RealisticConfig, LiquidityModel, CommissionModel
from options_day_trader_sim import OptionSignal


class RealisticSmartTrader(SmartTrader):
    """
    Extends the smart trader with realistic trading factors.
    """
    
    def __init__(self, account_value: float = 25000.0):
        """Initialize the realistic smart trader."""
        # Initialize parent class
        super().__init__(account_value)
        
        # Get current data source
        self.data_source = os.environ.get("TRADING_DATA_SOURCE", "Simulated")
        self.source_tag = "real" if self.data_source == "Alpaca Real-Time" else "sim"
        
        # Override the patterns object with our own source-specific one
        self.patterns = TradingPattern()
        self.patterns.data_source = self.data_source
        self.patterns.source_tag = self.source_tag
        
        # Load patterns - ensure we're loading from the correct source
        pattern_file = f"realistic_smart_trader_patterns_{self.source_tag}.json"
        if not os.path.exists(pattern_file):
            # Try legacy file
            legacy_file = "realistic_smart_trader_patterns.json"
            if os.path.exists(legacy_file):
                pattern_file = legacy_file
        
        if not self.patterns.load(pattern_file):
            print(f"Starting with a new realistic pattern database for {self.data_source} - will learn from scratch.")
        else:
            print(f"Loaded realistic trading patterns for {self.data_source}.")
            self.patterns.print_stats()
        
        # Initialize realistic models
        self.liquidity_model = LiquidityModel()
        self.commission_model = CommissionModel()
        self.fill_failures = 0
        self.partial_fills = 0
        self.slippage_costs = 0.0
    
    # Override the save method to use the correct filename
    def save_patterns(self):
        """Save patterns to the correct file for this trader type."""
        pattern_file = f"realistic_smart_trader_patterns_{self.source_tag}.json"
        self.patterns.save(pattern_file)
        
    # Override the execute_trade method to save patterns with the correct filename
    def execute_trade(self, trade_info: Dict, order_type: str = 'limit', 
                     price_point: str = 'mid') -> Dict:
        """Execute trade with realistic factors and save to the correct pattern file."""
        result = super().execute_trade(trade_info, order_type, price_point)
        
        # Save patterns to the correct file after learning
        self.save_patterns()
        
        return result
    
    def print_realistic_trade_result(self, result: Dict):
        """Print detailed trade result with realistic factors."""
        print("\n" + "=" * 80)
        print(f"REALISTIC SMART TRADE: {result['symbol']} {result['option_type'].upper()} ${result['strike']}")
        print("-" * 80)
        print(f"PATTERN SCORE:  {result['score']:.2f} (Win probability: {result['win_probability']:.1%})")
        print(f"ORDER TYPE:    {result['order_type'].upper()} order at {result['price_point'].upper()} price")
        print(f"FILL PROBABILITY: {result['fill_probability']:.1%}")
        
        if result.get("partial_fill", False):
            print(f"PARTIAL FILL: {result['filled_size']}/{result['original_size']} contracts")
            
        print(f"ENTRY TIME:    {result['entry_time'].strftime('%H:%M:%S')}")
        print(f"ENTRY PRICE:   ${result['entry_price']:.2f} x {result['position_size']} contracts")
        print(f"ENTRY SLIPPAGE: ${result['entry_slippage']:.2f}")
        print(f"POSITION SIZE: ${(result['entry_price'] * result['position_size'] * 100):.2f} " + 
              f"({(result['position_size'] * result['entry_price'] * 100 / result['previous_account'] * 100):.1f}% of account)")
        print(f"EXIT TIME:     {result['exit_time'].strftime('%H:%M:%S')} ({result['hold_time_minutes']} minutes later)")
        print(f"EXIT PRICE:    ${result['exit_price']:.2f} ({result['price_change_pct']:.1f}%)")
        print(f"EXIT SLIPPAGE: ${result['exit_slippage']:.2f}")
        print(f"OUTCOME:       {result['outcome_type']}")
        print(f"COMMISSIONS:   ${result['total_commission']:.2f}")
        print(f"TOTAL COSTS:   ${(result['total_commission'] + result['total_slippage']):.2f}")
        print(f"GROSS P&L:     ${result['gross_profit']:.2f}")
        print(f"NET P&L:       ${result['net_profit']:.2f}")
        
        if result['net_profit'] > 0:
            profit_per_minute = result['net_profit'] / result['hold_time_minutes']
            print(f"PROFIT RATE:   ${profit_per_minute:.2f} per minute")
            
        # Show account changes
        print("-" * 80)
        print(f"PREVIOUS BALANCE: ${result['previous_account']:.2f}")
        print(f"CURRENT BALANCE:  ${result['new_account']:.2f}")
        account_change_pct = (result['new_account'] / result['previous_account'] - 1) * 100
        print(f"ACCOUNT CHANGE:   {account_change_pct:+.2f}%")
        print("=" * 80)
    
    def print_session_summary(self):
        """Print summary of the current trading session with realistic metrics."""
        super().print_session_summary()  # Call the base class method
        
        # Add additional realistic metrics
        if self.current_session:
            filled_trades = [t for t in self.current_session if t.get("filled", True)]
            unfilled_orders = [t for t in self.current_session if not t.get("filled", True)]
            partial_fills = sum(1 for t in filled_trades if t.get("partial_fill", False))
            
            total_slippage = sum(t.get("total_slippage", 0) for t in filled_trades)
            total_commission = sum(t.get("total_commission", 0) for t in filled_trades)
            
            # Calculate impact percentages
            if filled_trades:
                gross_profit = sum(t["gross_profit"] for t in filled_trades)
                net_profit = sum(t["net_profit"] for t in filled_trades)
                if gross_profit != 0:
                    slippage_impact_pct = total_slippage / abs(gross_profit) * 100
                    commission_impact_pct = total_commission / abs(gross_profit) * 100
                    total_cost_impact_pct = (total_slippage + total_commission) / abs(gross_profit) * 100
                else:
                    slippage_impact_pct = commission_impact_pct = total_cost_impact_pct = 0
                
                print("\nREALISTIC FACTORS IMPACT:")
                print(f"Fill Rate:         {len(filled_trades)/(len(filled_trades) + len(unfilled_orders)):.1%}")
                print(f"Partial Fills:     {partial_fills} ({partial_fills/len(filled_trades) if filled_trades else 0:.1%})")
                print(f"Total Slippage:    ${total_slippage:.2f} ({slippage_impact_pct:.1f}% of gross P&L)")
                print(f"Total Commission:  ${total_commission:.2f} ({commission_impact_pct:.1f}% of gross P&L)")
                print(f"Total Cost Impact: ${total_slippage + total_commission:.2f} ({total_cost_impact_pct:.1f}% of gross P&L)")
                
                if net_profit != 0 and gross_profit != 0:
                    profit_reduction = (gross_profit - net_profit) / gross_profit * 100
                    print(f"Profit Reduction:  {profit_reduction:.1f}% due to costs")
    
    def run_trading_session(self, symbols: List[str] = None, max_trades: int = 10, 
                           use_realistic: bool = True, order_type: str = "limit",
                           price_point: str = "mid"):
        """
        Run a simulated trading session with realistic factors.
        
        Args:
            symbols: List of symbols to trade
            max_trades: Maximum number of trades
            use_realistic: Whether to use realistic factors
            order_type: 'market' or 'limit'
            price_point: For limit orders: 'bid', 'mid', or 'ask'
        """
        symbols = symbols or RealisticConfig.WATCHLIST
        self.current_session = []
        self.session_count += 1
        
        print(f"\n=== STARTING TRADING SESSION #{self.session_count} ===")
        print(f"Account value: ${self.account_value:.2f}")
        print(f"Watchlist: {', '.join(symbols)}")
        print(f"Max trades: {max_trades}")
        
        if use_realistic:
            print(f"Using realistic trading factors - Order type: {order_type}, Price point: {price_point}")
        
        trades_executed = 0
        attempts = 0
        max_attempts = max_trades * 3  # Allow some extra attempts for unfilled orders
        
        while trades_executed < max_trades and attempts < max_attempts:
            attempts += 1
            
            # Scan for opportunities
            opportunities = self.scan_for_trades(symbols)
            
            if opportunities:
                # Take the top opportunity
                trade_info = opportunities[0]
                
                if use_realistic:
                    result = self.execute_trade(trade_info, order_type=order_type, price_point=price_point)
                    if result.get("filled", False):
                        trades_executed += 1
                else:
                    # Use the standard execution method from the parent class
                    result = super().execute_trade(trade_info)
                    trades_executed += 1
            
            # Add a small delay
            time.sleep(0.1)
        
        # Print session summary
        self.print_session_summary()
        
        # Save updated patterns
        self.patterns.save()


if __name__ == "__main__":
    # Create a realistic smart trader and run some simulations
    trader = RealisticSmartTrader(account_value=25000.0)
    
    # Run a session with realistic factors
    trader.run_trading_session(max_trades=10, use_realistic=True, order_type="limit", price_point="mid")
    
    # Display pattern statistics
    trader.patterns.print_stats() 