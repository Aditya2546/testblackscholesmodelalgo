#!/usr/bin/env python
"""
Simple Options Trading Learning Simulation

Demonstrates how a trading system can learn from past trades to improve its win rate over time.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Trading patterns we'll learn
PATTERNS = {
    "SPY_morning_oversold": {"base_win_rate": 0.65},
    "SPY_morning_overbought": {"base_win_rate": 0.40},
    "SPY_afternoon_oversold": {"base_win_rate": 0.55},
    "SPY_afternoon_overbought": {"base_win_rate": 0.45},
    "QQQ_morning_oversold": {"base_win_rate": 0.60},
    "QQQ_morning_overbought": {"base_win_rate": 0.45},
    "AAPL_morning_oversold": {"base_win_rate": 0.58},
    "AAPL_afternoon_overbought": {"base_win_rate": 0.52},
}

class LearningTrader:
    """Simple trader that learns from experience."""
    
    def __init__(self):
        """Initialize with empty knowledge."""
        self.pattern_results = defaultdict(lambda: {"wins": 0, "losses": 0})
        self.performance_history = []
        self.account = 25000
        self.starting_account = 25000
        self.trade_history = []
    
    def get_win_probability(self, pattern):
        """Get probability of winning based on past experience."""
        results = self.pattern_results[pattern]
        total = results["wins"] + results["losses"]
        
        if total < 5:
            # Not enough data, use base rate
            return PATTERNS[pattern]["base_win_rate"]
        else:
            # Use observed win rate
            return results["wins"] / total
    
    def select_trade(self, available_patterns):
        """Select the best pattern to trade based on learned probabilities."""
        best_pattern = None
        best_prob = 0
        
        for pattern in available_patterns:
            win_prob = self.get_win_probability(pattern)
            if win_prob > best_prob:
                best_prob = win_prob
                best_pattern = pattern
        
        return best_pattern, best_prob
    
    def execute_trade(self, pattern, position_size=100):
        """Simulate trade execution and record results."""
        # Determine actual outcome (hidden from trader)
        actual_win_rate = PATTERNS[pattern]["base_win_rate"]
        is_win = random.random() < actual_win_rate
        
        # Record result in our knowledge
        if is_win:
            self.pattern_results[pattern]["wins"] += 1
            profit = position_size * 0.10  # 10% profit
        else:
            self.pattern_results[pattern]["losses"] += 1
            profit = -position_size * 0.15  # 15% loss
        
        # Update account
        self.account += profit
        
        # Record trade
        self.trade_history.append({
            "pattern": pattern,
            "profit": profit,
            "is_win": is_win,
            "account": self.account,
            "predicted_prob": self.get_win_probability(pattern)
        })
        
        return is_win, profit
    
    def run_trading_session(self, num_trades=10):
        """Simulate a full trading session."""
        session_starting_account = self.account
        session_trades = []
        
        for i in range(num_trades):
            # Randomly select available patterns (simulates market conditions)
            available_patterns = random.sample(list(PATTERNS.keys()), min(3, len(PATTERNS)))
            
            # Select best pattern based on our learned probabilities
            selected_pattern, predicted_win_prob = self.select_trade(available_patterns)
            
            # Execute trade
            is_win, profit = self.execute_trade(selected_pattern)
            
            # Record results
            session_trades.append({
                "trade_num": i+1,
                "pattern": selected_pattern,
                "predicted_prob": predicted_win_prob,
                "result": "WIN" if is_win else "LOSS",
                "profit": profit,
                "account": self.account
            })
            
            # Print results
            print(f"Trade {i+1}: {selected_pattern} - Predicted win prob: {predicted_win_prob:.1%}")
            print(f"Result: {'WIN' if is_win else 'LOSS'} (${profit:.2f})")
            print(f"Account: ${self.account:.2f}\n")
        
        # Calculate session performance
        win_count = sum(1 for t in session_trades if t["result"] == "WIN")
        session_win_rate = win_count / len(session_trades)
        session_profit = self.account - session_starting_account
        
        self.performance_history.append({
            "trades": num_trades,
            "win_rate": session_win_rate,
            "profit": session_profit,
            "account": self.account
        })
        
        return session_trades
    
    def print_knowledge(self):
        """Print what we've learned about trading patterns."""
        print("\nLEARNED PATTERN INFORMATION:")
        print("-" * 40)
        print(f"{'Pattern':<25} {'Trades':<8} {'Win Rate':<10} {'Actual':<10}")
        print("-" * 40)
        
        for pattern, results in self.pattern_results.items():
            total = results["wins"] + results["losses"]
            if total > 0:
                win_rate = results["wins"] / total
                print(f"{pattern:<25} {total:<8} {win_rate:.1%}      {PATTERNS[pattern]['base_win_rate']:.1%}")
    
    def plot_performance(self, show=True, save=False):
        """Plot performance over time."""
        if not self.trade_history:
            print("No trades to plot.")
            return
        
        # Extract data
        trades = range(1, len(self.trade_history) + 1)
        account_values = [t["account"] for t in self.trade_history]
        win_rates = []
        
        # Calculate running win rate
        wins = 0
        for i, trade in enumerate(self.trade_history, 1):
            if trade["is_win"]:
                wins += 1
            win_rates.append(wins / i)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Account value plot
        ax1.plot(trades, account_values, 'b-')
        ax1.set_title('Account Value Over Time')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Account Value ($)')
        ax1.grid(True)
        
        # Win rate plot
        ax2.plot(trades, win_rates, 'g-')
        ax2.set_title('Win Rate Over Time')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim([0, 1])
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('learning_performance.png')
        
        if show:
            plt.show()


def run_simulation():
    """Run the learning trader simulation."""
    trader = LearningTrader()
    
    print("=== OPTIONS TRADING LEARNING SIMULATION ===\n")
    print("Starting with NO knowledge of trading patterns")
    print(f"Starting account: ${trader.account:.2f}\n")
    
    # Run multiple trading sessions
    num_sessions = 5
    trades_per_session = 10
    
    for i in range(num_sessions):
        print(f"=== TRADING SESSION {i+1} ===")
        trader.run_trading_session(trades_per_session)
        
        # Print what we've learned so far
        trader.print_knowledge()
        
        # Print session summary
        session = trader.performance_history[-1]
        print(f"\nSession {i+1} Summary:")
        print(f"Win Rate: {session['win_rate']:.1%}")
        print(f"Profit: ${session['profit']:.2f}")
        print(f"Account: ${session['account']:.2f}")
        print(f"Return: {(session['account']/trader.starting_account - 1)*100:.1f}%")
        
        print("\n" + "="*50 + "\n")
    
    # Plot the performance
    trader.plot_performance()
    
    print("SIMULATION COMPLETE!")
    print(f"Final Account Value: ${trader.account:.2f}")
    print(f"Total Return: {(trader.account/trader.starting_account - 1)*100:.1f}%")


if __name__ == "__main__":
    run_simulation() 