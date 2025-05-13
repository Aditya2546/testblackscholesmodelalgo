#!/usr/bin/env python
"""
Options Day Trading System with Learning Capabilities

This implements a simple machine learning system that can learn from past trades
and improve its decision making over time.
"""

import numpy as np
import pandas as pd
import datetime
import pickle
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Import the base simulator
from options_day_trader_sim import OptionSignal, Config, MarketDataSimulator

class TradeFeatures:
    """Extracts features from a trading opportunity for ML input."""
    
    @staticmethod
    def extract_features(symbol: str, price: float, option_type: str, 
                         strike: float, days_to_exp: int, time_of_day: float,
                         iv: float, delta: float, technicals: Dict) -> np.ndarray:
        """
        Extract a feature vector from trade information.
        
        Args:
            symbol: Ticker symbol
            price: Current price of underlying
            option_type: 'call' or 'put'
            strike: Strike price
            days_to_exp: Days to expiration
            time_of_day: Time of day as decimal (9.5 = 9:30 AM)
            iv: Implied volatility
            delta: Option delta
            technicals: Dictionary of technical indicators
            
        Returns:
            Feature vector as numpy array
        """
        # Calculate derived features
        moneyness = price / strike - 1.0 if option_type == 'call' else strike / price - 1.0
        time_to_close = min(6.5, max(0, 16.0 - time_of_day))  # Hours until market close
        
        # Build feature vector
        features = np.array([
            moneyness,
            days_to_exp,
            time_of_day,
            time_to_close,
            iv,
            abs(delta),
            technicals.get('rsi', 50) / 100.0,  # Normalize to 0-1
            technicals.get('price_change_1d', 0) * 100,  # Convert to percentage
            1.0 if option_type == 'call' else 0.0,  # Option type as binary
            technicals.get('volume_ratio', 1.0),
        ])
        
        return features


class SimpleModel:
    """
    A simple machine learning model for trading decisions.
    Uses a basic linear model with weights updated through reinforcement learning.
    """
    
    def __init__(self, feature_dim: int = 10):
        """Initialize the model with random weights."""
        self.weights = np.random.normal(0, 0.1, feature_dim)
        self.bias = 0.0
        self.learning_rate = 0.01
        self.history = []
        self.trade_memory = []
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict the quality score of a trade (0-1).
        
        Args:
            features: Feature vector
            
        Returns:
            Score between 0-1, higher is better
        """
        raw_score = np.dot(features, self.weights) + self.bias
        score = 1.0 / (1.0 + np.exp(-raw_score))  # Sigmoid to get 0-1
        return score
    
    def decide_trade(self, features: np.ndarray, threshold: float = 0.4) -> bool:
        """
        Decide whether to take a trade based on features.
        
        Args:
            features: Feature vector
            threshold: Minimum score to take trade
            
        Returns:
            True if should take trade, False otherwise
        """
        score = self.predict(features)
        return score > threshold, score
    
    def record_trade(self, features: np.ndarray, score: float, profit_pct: float):
        """
        Record a completed trade for learning.
        
        Args:
            features: Feature vector used for decision
            score: Model's score for this trade
            profit_pct: Actual profit/loss percentage
        """
        self.trade_memory.append((features, score, profit_pct))
        self.history.append(profit_pct)
        
        # Limit memory to last 1000 trades
        if len(self.trade_memory) > 1000:
            self.trade_memory.pop(0)
            self.history.pop(0)
    
    def learn_from_trades(self, batch_size: int = 32):
        """
        Update model weights based on past trades.
        
        Args:
            batch_size: Number of random trades to learn from
        """
        if len(self.trade_memory) < batch_size:
            return  # Not enough data to learn
        
        # Sample random trades from memory
        batch_indices = np.random.choice(len(self.trade_memory), batch_size, replace=False)
        batch = [self.trade_memory[i] for i in batch_indices]
        
        for features, score, profit_pct in batch:
            # Calculate target (actual outcome was good or bad)
            # Convert profit to a target between 0-1
            target = 1.0 / (1.0 + np.exp(-10 * profit_pct))  # Sharper sigmoid
            
            # Calculate error
            error = target - score
            
            # Update weights with simple gradient step
            self.weights += self.learning_rate * error * features
            self.bias += self.learning_rate * error
    
    def save_model(self, filename: str = "trade_model.pkl"):
        """Save model to file."""
        with open(filename, "wb") as f:
            pickle.dump((self.weights, self.bias, self.history), f)
    
    def load_model(self, filename: str = "trade_model.pkl") -> bool:
        """Load model from file, return success."""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, "rb") as f:
                self.weights, self.bias, self.history = pickle.load(f)
            return True
        except:
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get model performance statistics."""
        if not self.history:
            return {"win_rate": 0, "avg_return": 0, "trades": 0}
        
        win_rate = sum(1 for p in self.history if p > 0) / len(self.history)
        avg_return = sum(self.history) / len(self.history)
        
        return {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "trades": len(self.history),
            "recent_win_rate": sum(1 for p in self.history[-50:] if p > 0) / min(50, len(self.history))
        }


class LearningTrader:
    """
    Options trading system that learns from past trades to improve decisions.
    """
    
    def __init__(self, account_value: float = 25000.0):
        """Initialize the learning trader."""
        self.account_value = account_value
        self.initial_account_value = account_value
        self.market_data = MarketDataSimulator()
        self.model = SimpleModel()
        self.trade_history = []
        self.current_day_trades = []
        self.session_count = 0
        
        # Try to load saved model
        if not self.model.load_model():
            print("Starting with a new model - will learn from scratch.")
        else:
            stats = self.model.get_performance_stats()
            print(f"Loaded model with {stats['trades']} trades and {stats['win_rate']:.1%} win rate.")
    
    def evaluate_option(self, symbol: str, price: float, option_type: str,
                        strike: float, expiry: datetime.date, option_data: Dict,
                        technicals: Dict) -> Tuple[bool, float, np.ndarray]:
        """
        Evaluate if an option is a good trade opportunity.
        
        Args:
            symbol: Ticker symbol
            price: Current underlying price
            option_type: 'call' or 'put'
            strike: Strike price
            expiry: Expiration date
            option_data: Option chain data
            technicals: Technical indicators
            
        Returns:
            (should_trade, score, features)
        """
        # Get current time
        now = datetime.datetime.now()
        time_of_day = now.hour + now.minute / 60.0
        
        # Calculate days to expiration
        days_to_exp = (expiry - datetime.date.today()).days
        
        # Extract features
        features = TradeFeatures.extract_features(
            symbol=symbol,
            price=price,
            option_type=option_type,
            strike=strike,
            days_to_exp=days_to_exp,
            time_of_day=time_of_day,
            iv=option_data["iv"],
            delta=option_data["delta"],
            technicals=technicals
        )
        
        # Get model's decision
        should_trade, score = self.model.decide_trade(features)
        
        return should_trade, score, features
    
    def scan_for_trades(self, symbols: List[str] = None) -> List[Dict]:
        """
        Scan for trading opportunities and use the model to select the best ones.
        
        Args:
            symbols: List of symbols to check, or None for default watchlist
            
        Returns:
            List of trade opportunities with scores
        """
        symbols = symbols or Config.WATCHLIST
        opportunities = []
        
        for symbol in symbols:
            price = self.market_data.get_price(symbol)
            technicals = self.market_data.get_technical_indicators(symbol)
            chain = self.market_data.get_options_chain(symbol)
            
            for exp_str, exp_data in chain.items():
                expiry = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
                
                # Skip if too far in the future
                days_out = (expiry - datetime.date.today()).days
                if days_out > Config.DAYS_TO_EXPIRATION + 1:
                    continue
                
                # Check both calls and puts
                for option_type in ["calls", "puts"]:
                    for strike, option in exp_data[option_type].items():
                        # Basic filtering before spending compute on ML evaluation
                        delta = abs(option["delta"])
                        if (option["volume"] < Config.MIN_VOLUME or
                            delta < Config.MIN_DELTA or delta > Config.MAX_DELTA or
                            (option["ask"] - option["bid"]) / option["ask"] > Config.MAX_SPREAD_PCT):
                            continue
                        
                        # Evaluate with the model
                        should_trade, score, features = self.evaluate_option(
                            symbol=symbol,
                            price=price,
                            option_type=option_type.rstrip("s"),  # Remove plural
                            strike=float(strike),
                            expiry=expiry,
                            option_data=option,
                            technicals=technicals
                        )
                        
                        if should_trade:
                            opportunities.append({
                                "symbol": symbol,
                                "price": price,
                                "option_type": option_type.rstrip("s"),
                                "strike": float(strike),
                                "expiry": expiry,
                                "bid": option["bid"],
                                "ask": option["ask"],
                                "delta": option["delta"],
                                "volume": option["volume"],
                                "open_interest": option["open_interest"],
                                "iv": option["iv"],
                                "score": score,
                                "features": features
                            })
        
        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities
    
    def execute_trade(self, trade_info: Dict) -> Dict:
        """
        Simulate executing a trade and its outcome.
        
        Args:
            trade_info: Trade opportunity information
            
        Returns:
            Trade result with P&L
        """
        # Create option signal
        mid_price = (trade_info["bid"] + trade_info["ask"]) / 2
        entry_low = trade_info["bid"]
        entry_high = min(mid_price * 1.02, trade_info["ask"])
        
        signal = OptionSignal(
            symbol=trade_info["symbol"],
            option_type=trade_info["option_type"],
            strike=trade_info["strike"],
            expiration=trade_info["expiry"],
            current_price=mid_price,
            underlying_price=trade_info["price"],
            entry_price_range=(entry_low, entry_high),
            stop_loss=mid_price * (1 - Config.STOP_LOSS_PCT),
            target_price=mid_price * (1 + Config.PROFIT_TARGET_PCT),
            signal_strength=trade_info["score"],
            volume=trade_info["volume"],
            open_interest=trade_info["open_interest"],
            iv=trade_info["iv"],
            delta=trade_info["delta"]
        )
        
        # Entry price is the midpoint of the entry range
        entry_price = (signal.entry_price_range[0] + signal.entry_price_range[1]) / 2
        
        # Position sizing
        max_position_value = self.account_value * Config.POSITION_SIZE_PCT
        contract_value = entry_price * 100
        position_size = max(1, min(10, int(max_position_value / contract_value)))
        
        # Simulate outcome based on model score and some randomness
        # Better model predictions get better probability of success
        base_win_prob = 0.5  # Base 50% chance
        score_boost = (trade_info["score"] - 0.5) * 0.5  # Model's confidence adjustment
        win_probability = min(0.8, max(0.2, base_win_prob + score_boost))  # Limit to 20-80%
        
        is_profitable = np.random.random() < win_probability
        
        # Determine outcome details
        outcome_minutes = np.random.randint(5, Config.MAX_HOLD_MINUTES)
        outcome_time = datetime.datetime.now() + datetime.timedelta(minutes=outcome_minutes)
        
        if is_profitable:
            # Profitable outcome
            outcome_pct = np.random.uniform(0.03, 1.0) * Config.PROFIT_TARGET_PCT
            outcome_price = entry_price * (1 + outcome_pct)
            outcome_type = "TARGET HIT" if outcome_pct >= Config.PROFIT_TARGET_PCT * 0.95 else "PARTIAL PROFIT"
        else:
            # Loss outcome
            outcome_pct = np.random.uniform(0.3, 1.0) * -Config.STOP_LOSS_PCT
            outcome_price = entry_price * (1 + outcome_pct)
            outcome_type = "STOP LOSS HIT" if outcome_pct <= -Config.STOP_LOSS_PCT * 0.95 else "PARTIAL LOSS"
        
        # Calculate P&L
        price_change = outcome_price - entry_price
        price_change_pct = price_change / entry_price * 100
        dollar_profit = price_change * position_size * 100
        commission = position_size * 1.0 * 2  # $1 per contract, entry and exit
        net_profit = dollar_profit - commission
        
        # Update account value
        previous_account_value = self.account_value
        self.account_value += net_profit
        
        # Record the result for model learning
        self.model.record_trade(
            features=trade_info["features"],
            score=trade_info["score"],
            profit_pct=price_change_pct / 100  # Convert to decimal
        )
        
        result = {
            "symbol": signal.symbol,
            "option_type": signal.option_type,
            "strike": signal.strike,
            "entry_time": datetime.datetime.now(),
            "entry_price": entry_price,
            "position_size": position_size,
            "exit_time": outcome_time,
            "exit_price": outcome_price,
            "hold_time_minutes": outcome_minutes,
            "price_change_pct": price_change_pct,
            "commission": commission,
            "gross_profit": dollar_profit,
            "net_profit": net_profit,
            "outcome_type": outcome_type,
            "previous_account": previous_account_value,
            "new_account": self.account_value,
            "model_score": trade_info["score"],
            "win_probability": win_probability
        }
        
        self.trade_history.append(result)
        self.current_day_trades.append(result)
        
        return result
    
    def print_trade_result(self, result: Dict):
        """Print a formatted trade result."""
        symbol_str = f"{result['symbol']} - {result['option_type'].upper()} ${result['strike']}"
        
        print("\n" + "=" * 80)
        print(f"TRADE OUTCOME: {symbol_str}")
        print("-" * 80)
        print(f"MODEL SCORE:   {result['model_score']:.2f} (predicted win prob: {result['win_probability']:.1%})")
        print(f"ENTRY TIME:    {result['entry_time'].strftime('%H:%M:%S')}")
        print(f"ENTRY PRICE:   ${result['entry_price']:.2f} x {result['position_size']} contracts")
        print(f"POSITION SIZE: ${(result['entry_price'] * result['position_size'] * 100):.2f}")
        print(f"EXIT TIME:     {result['exit_time'].strftime('%H:%M:%S')} ({result['hold_time_minutes']} min)")
        print(f"EXIT PRICE:    ${result['exit_price']:.2f} ({result['price_change_pct']:.1f}%)")
        print(f"OUTCOME:       {result['outcome_type']}")
        print(f"COMMISSION:    ${result['commission']:.2f}")
        print(f"NET P&L:       ${result['net_profit']:.2f}")
        
        # Show account changes
        print("-" * 80)
        print(f"PREVIOUS BALANCE: ${result['previous_account']:.2f}")
        print(f"CURRENT BALANCE:  ${result['new_account']:.2f}")
        account_change_pct = (result['new_account'] / result['previous_account'] - 1) * 100
        print(f"ACCOUNT CHANGE:   {account_change_pct:+.2f}%")
        overall_change_pct = (self.account_value / self.initial_account_value - 1) * 100
        print(f"DAY'S RETURN:     {overall_change_pct:+.2f}%")
        print("=" * 80)
    
    def print_session_summary(self):
        """Print summary of the current trading session."""
        if not self.current_day_trades:
            print("\nNo trades executed in this session.")
            return
        
        total_profit = sum(r["net_profit"] for r in self.current_day_trades)
        win_count = sum(1 for r in self.current_day_trades if r["net_profit"] > 0)
        loss_count = len(self.current_day_trades) - win_count
        win_rate = win_count / len(self.current_day_trades) if self.current_day_trades else 0
        
        # Calculate metrics
        total_commissions = sum(r["commission"] for r in self.current_day_trades)
        avg_profit_per_trade = total_profit / len(self.current_day_trades) if self.current_day_trades else 0
        avg_hold_time = sum(r["hold_time_minutes"] for r in self.current_day_trades) / len(self.current_day_trades) if self.current_day_trades else 0
        profit_per_minute = total_profit / sum(r["hold_time_minutes"] for r in self.current_day_trades) if sum(r["hold_time_minutes"] for r in self.current_day_trades) > 0 else 0
        
        # Model statistics
        model_stats = self.model.get_performance_stats()
        
        print("\n" + "=" * 80)
        print(f"TRADING SESSION #{self.session_count} SUMMARY:")
        print("-" * 80)
        print(f"Total Trades:       {len(self.current_day_trades)}")
        print(f"Winning Trades:     {win_count} ({win_rate:.1%})")
        print(f"Losing Trades:      {loss_count}")
        print(f"Total Net P&L:      ${total_profit:.2f}")
        print(f"Total Commissions:  ${total_commissions:.2f}")
        print(f"Avg Profit/Trade:   ${avg_profit_per_trade:.2f}")
        print(f"Avg Hold Time:      {avg_hold_time:.1f} minutes")
        print(f"Profit Per Minute:  ${profit_per_minute:.2f}")
        print("-" * 80)
        print(f"Starting Balance:   ${self.initial_account_value:.2f}")
        print(f"Ending Balance:     ${self.account_value:.2f}")
        account_change_pct = (self.account_value / self.initial_account_value - 1) * 100
        print(f"Account Change:     {account_change_pct:+.2f}%")
        print("-" * 80)
        print(f"MODEL STATISTICS:")
        print(f"All-time Win Rate:  {model_stats['win_rate']:.1%} ({model_stats['trades']} trades)")
        print(f"Recent Win Rate:    {model_stats['recent_win_rate']:.1%} (last 50 trades)")
        print(f"Average Return:     {model_stats['avg_return']*100:+.2f}% per trade")
        print("=" * 80)
    
    def learn(self, iterations: int = 5):
        """Train the model on past trades."""
        for _ in range(iterations):
            self.model.learn_from_trades()
        
        # Save model after learning
        self.model.save_model()
        
        stats = self.model.get_performance_stats()
        print(f"Model trained. Current statistics:")
        print(f"Win Rate: {stats['win_rate']:.1%} | Avg Return: {stats['avg_return']*100:+.2f}% | Trades: {stats['trades']}")
    
    def run_training_session(self, symbols: List[str] = None, max_trades: int = 20):
        """
        Run a trading session to generate training data.
        
        Args:
            symbols: List of symbols to trade, or None for default watchlist
            max_trades: Maximum number of trades to execute
        """
        self.session_count += 1
        self.current_day_trades = []
        
        # Reset account for the day
        self.account_value = self.initial_account_value
        
        symbols = symbols or Config.WATCHLIST
        print(f"\n=== TRAINING SESSION #{self.session_count} - STARTING ===\n")
        print(f"Training on: {', '.join(symbols)}")
        print(f"Starting account: ${self.account_value:.2f}")
        print(f"Model stats: {self.model.get_performance_stats()}\n")
        
        trades_executed = 0
        
        while trades_executed < max_trades:
            # Scan for opportunities
            opportunities = self.scan_for_trades(symbols)
            
            if not opportunities:
                print("No viable trading opportunities found in this scan.")
                break
            
            # Take the best opportunity
            best_trade = opportunities[0]
            
            print(f"\nFound trading opportunity: {best_trade['symbol']} {best_trade['option_type'].upper()} ${best_trade['strike']}")
            print(f"Model score: {best_trade['score']:.2f}")
            
            # Execute the trade
            result = self.execute_trade(best_trade)
            self.print_trade_result(result)
            
            trades_executed += 1
            
            # Basic learning after every few trades
            if trades_executed % 5 == 0:
                self.learn(1)
        
        # Training session complete
        self.print_session_summary()
        
        # More thorough learning at the end of session
        self.learn(10)
    
    def run_multiple_sessions(self, num_sessions: int = 5, trades_per_session: int = 20):
        """Run multiple training sessions to improve the model."""
        starting_value = self.initial_account_value
        
        for i in range(num_sessions):
            print(f"\n\n{'='*40} SESSION {i+1}/{num_sessions} {'='*40}\n")
            self.run_training_session(max_trades=trades_per_session)
            
            # Re-initialize account between sessions
            self.initial_account_value = self.account_value
        
        # Final stats
        final_value = self.account_value
        total_return = (final_value / starting_value - 1) * 100
        
        print(f"\n\n{'='*30} TRAINING COMPLETE {'='*30}")
        print(f"Initial account value: ${starting_value:.2f}")
        print(f"Final account value:   ${final_value:.2f}")
        print(f"Overall return:        {total_return:+.2f}%")
        print(f"Model statistics:")
        stats = self.model.get_performance_stats()
        print(f"Win Rate: {stats['win_rate']:.1%} | Avg Return: {stats['avg_return']*100:+.2f}% | Trades: {stats['trades']}")


def main():
    """Run the learning trader program."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Options Day Trading System with Learning")
    parser.add_argument("--sessions", type=int, default=5, help="Number of training sessions")
    parser.add_argument("--trades", type=int, default=20, help="Trades per session")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to trade")
    args = parser.parse_args()
    
    trader = LearningTrader()
    
    symbols = args.symbols.split(",") if args.symbols else None
    
    trader.run_multiple_sessions(
        num_sessions=args.sessions,
        trades_per_session=args.trades
    )


if __name__ == "__main__":
    main() 