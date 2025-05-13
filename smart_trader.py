#!/usr/bin/env python
"""
Smart Trader Module

Provides trading logic and trade simulation capabilities.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from config import Config
import json
import os

class SmartTrader:
    """
    Smart trader class that provides trading logic and trade simulation.
    """
    
    def __init__(self, account_value: float = 25000.0):
        """
        Initialize the smart trader.
        
        Args:
            account_value: Initial account value
        """
        self.trading_history = []
        self.open_positions = []
        self.account_value = account_value
        self.trade_history = []
        self.current_session = []
        self.session_count = 0
        self.patterns = TradingPattern()
        
        # Try to load existing patterns
        pattern_file = "smart_trader_patterns.json"
        if os.path.exists(pattern_file):
            self.patterns.load(pattern_file)
    
    def simulate_trade(self, trade_info: Dict, risk_pct: float = 0.02, 
                      account_value: float = 25000.0) -> Dict:
        """
        Simulate a trade outcome based on real market data and ML predictions.
        
        Args:
            trade_info: Dictionary with trade information
            risk_pct: Risk percentage per trade
            account_value: Account value
            
        Returns:
            Dictionary with trade outcome information
        """
        # Extract trade information
        symbol = trade_info.get('symbol', 'UNKNOWN')
        option_type = trade_info.get('option_type', 'call')
        strike = trade_info.get('strike', 0.0)
        entry_price = trade_info.get('mid', 0.0)
        contracts = trade_info.get('contracts', 1)
        ml_probability = trade_info.get('ml_probability', 0.5)
        
        # Calculate position value
        position_value = entry_price * contracts * 100
        
        # Generate a simulated outcome based on ML probability
        # Higher ML probability = better chance of profit
        # Adjust the probability slightly based on random factors
        adjusted_probability = ml_probability * 0.8 + random.random() * 0.2
        
        # Determine the outcome
        outcome_roll = random.random()
        
        # Define possible outcomes
        if outcome_roll < adjusted_probability:
            # Profitable trade
            if random.random() < 0.7:  # 70% chance of full profit
                exit_price = entry_price * (1 + Config.PROFIT_TARGET_PCT)
                outcome = "PROFIT"
            else:  # 30% chance of partial profit
                exit_price = entry_price * (1 + Config.PROFIT_TARGET_PCT * random.uniform(0.3, 0.8))
                outcome = "PARTIAL PROFIT"
        else:
            # Losing trade
            if random.random() < 0.6:  # 60% chance of stop loss
                exit_price = entry_price * (1 - Config.STOP_LOSS_PCT)
                outcome = "STOP LOSS HIT"
            else:  # 40% chance of partial loss
                exit_price = entry_price * (1 - Config.STOP_LOSS_PCT * random.uniform(0.3, 0.8))
                outcome = "PARTIAL LOSS"
        
        # Calculate P&L
        pnl = (exit_price - entry_price) * contracts * 100
        
        # Subtract commissions
        commission = contracts * Config.COMMISSION_PER_CONTRACT * 2  # Entry and exit
        net_pnl = pnl - commission
        
        # Simulate trade duration
        trade_duration = random.randint(15, 90)  # Between 15 and 90 minutes
        
        # Create trade outcome
        trade_outcome = {
            'symbol': f"{symbol} {option_type.upper()} ${strike}",
            'entry_price': entry_price,
            'exit_price': exit_price,
            'contracts': contracts,
            'position_value': position_value,
            'commission': commission,
            'pnl': net_pnl,
            'outcome': outcome,
            'duration': trade_duration,
            'ml_probability': ml_probability,
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(minutes=trade_duration)
        }
        
        # Add to trading history
        self.trading_history.append(trade_outcome)
        
        return trade_outcome
    
    def get_trading_statistics(self) -> Dict:
        """
        Calculate trading statistics based on trading history.
        
        Returns:
            Dictionary with trading statistics
        """
        if not self.trading_history:
            return {
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'total_profit': 0.0,
                'largest_profit': 0.0,
                'largest_loss': 0.0,
                'avg_hold_time': 0
            }
        
        # Calculate statistics
        total_trades = len(self.trading_history)
        winning_trades = [t for t in self.trading_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trading_history if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t['pnl'] for t in self.trading_history)
        total_profit_wins = sum(t['pnl'] for t in winning_trades)
        total_loss = sum(abs(t['pnl']) for t in losing_trades)
        
        profit_factor = total_profit_wins / total_loss if total_loss > 0 else float('inf')
        
        largest_profit = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        avg_hold_time = np.mean([t['duration'] for t in self.trading_history])
        
        # Return statistics
        return {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'total_profit': total_profit,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'avg_hold_time': avg_hold_time
        }
    
    def reset(self):
        """Reset the trading history and open positions."""
        self.trading_history = []
        self.open_positions = []
        
    def run_multiple_sessions(self, num_sessions: int = 3, trades_per_session: int = 10) -> None:
        """
        Run multiple trading sessions to simulate a series of trades.
        
        Args:
            num_sessions: Number of trading sessions to run
            trades_per_session: Number of trades per session
        """
        print(f"\n=== RUNNING {num_sessions} TRADING SESSIONS WITH {trades_per_session} TRADES EACH ===")
        
        self.session_count = 0
        starting_account = self.account_value
        
        for i in range(num_sessions):
            self.session_count += 1
            self.run_trading_session(max_trades=trades_per_session)
            
            # Calculate session statistics
            session_trades = self.current_session
            win_count = sum(1 for t in session_trades if t.get('net_profit', 0) > 0)
            total_profit = sum(t.get('net_profit', 0) for t in session_trades)
            win_rate = win_count / len(session_trades) if session_trades else 0
            
            print(f"Session {i+1} complete: {win_rate:.1%} win rate, ${total_profit:.2f} profit")
        
        # Calculate overall statistics
        total_return = (self.account_value / starting_account - 1) * 100
        print(f"\n=== MULTIPLE SESSIONS COMPLETE ===")
        print(f"Starting balance: ${starting_account:.2f}")
        print(f"Final balance: ${self.account_value:.2f}")
        print(f"Total return: {total_return:+.2f}%")
    
    def run_trading_session(self, symbols: List[str] = None, max_trades: int = 10) -> None:
        """
        Run a simulated trading session.
        
        Args:
            symbols: List of symbols to trade. If None, uses default watchlist
            max_trades: Maximum number of trades in the session
        """
        symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
        
        # Initialize session
        self.current_session = []
        self.session_count += 1
        session_starting_account = self.account_value
        
        print(f"\n=== STARTING TRADING SESSION #{self.session_count} ===")
        print(f"Account value: ${self.account_value:.2f}")
        print(f"Watchlist: {', '.join(symbols)}")
        print(f"Max trades: {max_trades}")
        
        # Run trades
        trades_executed = 0
        
        while trades_executed < max_trades:
            # Scan for opportunities
            opportunities = self.scan_for_trades(symbols)
            
            if opportunities:
                # Take the top opportunity
                trade_info = opportunities[0]
                
                # Execute the trade
                result = self.execute_trade(trade_info)
                
                # Add to session history
                self.current_session.append(result)
                self.trade_history.append(result)
                
                trades_executed += 1
        
        # Calculate session results
        session_return = (self.account_value / session_starting_account - 1) * 100
        session_trades = len(self.current_session)
        winning_trades = sum(1 for t in self.current_session if t['net_profit'] > 0)
        win_rate = winning_trades / session_trades if session_trades > 0 else 0
        
        print(f"\n=== TRADING SESSION #{self.session_count} SUMMARY ===")
        print(f"Session return: {session_return:+.2f}%")
        print(f"Trades executed: {session_trades}")
        print(f"Win rate: {win_rate:.1%}")
        
        # Learn from trades if applicable
        if hasattr(self, 'patterns'):
            for trade in self.current_session:
                self.patterns.learn_from_trade(trade)
            
            # Save patterns
            if hasattr(self, 'save_patterns'):
                self.save_patterns()
    
    def scan_for_trades(self, symbols: List[str]) -> List[Dict]:
        """
        Scan for trading opportunities in the given symbols.
        
        Args:
            symbols: List of symbols to scan
            
        Returns:
            List of trading opportunities as dictionaries
        """
        opportunities = []
        
        for symbol in symbols:
            # Simulate market data for the symbol
            price = random.uniform(100, 500)  # Random price between 100 and 500
            
            # Generate some strikes around the current price
            strikes = [
                round(price * 0.9, 1),
                round(price * 0.95, 1),
                round(price, 1),
                round(price * 1.05, 1),
                round(price * 1.1, 1)
            ]
            
            # Create options opportunities
            for strike in strikes:
                for option_type in ['call', 'put']:
                    # Skip some opportunities randomly
                    if random.random() < 0.7:  # 70% chance to skip
                    continue
                
                    # Calculate option price using a simplified model
                    if option_type == 'call':
                        intrinsic = max(0, price - strike)
                else:
                        intrinsic = max(0, strike - price)
                    
                    time_value = price * 0.05 * random.uniform(0.8, 1.2)  # Random time value
                    option_price = intrinsic + time_value
                    
                    # Calculate bid/ask for the option
                    bid = round(option_price * 0.95, 2)
                    ask = round(option_price * 1.05, 2)
                    mid = round((bid + ask) / 2, 2)
                    
                    # Generate greeks
                    delta = 0.5 if strike == price else (0.7 if price > strike else 0.3)
                    gamma = 0.05
                    theta = -0.03
                    vega = 0.1
                    
                    # Get the pattern score
                    trade_info = {
                        'symbol': symbol,
                        'option_type': option_type,
                        'strike': strike,
                        'price': price,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'delta': delta,
                        'gamma': gamma,
                        'theta': theta,
                        'vega': vega,
                        'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                        'volume': random.randint(100, 5000),
                        'open_interest': random.randint(500, 10000),
                        'rsi': random.randint(30, 70)
                    }
                    
                    # Get a score based on trading patterns
                    score = self.patterns.get_pattern_score(trade_info)
                    
                    # Add score to the opportunity
                    trade_info['score'] = score
                    trade_info['win_probability'] = self.patterns.get_win_probability(trade_info)
                    
                    opportunities.append(trade_info)
        
        # Sort by score (descending)
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
    
    def execute_trade(self, trade_info: Dict, order_type: str = 'limit', 
                     price_point: str = 'mid') -> Dict:
        """
        Execute a trade based on the trade information.
        
        Args:
            trade_info: Dictionary with trade information
            order_type: Type of order ('market' or 'limit')
            price_point: For limit orders: 'bid', 'mid', or 'ask'
            
        Returns:
            Dictionary with trade outcome
        """
        # Extract trade information
        symbol = trade_info.get('symbol', 'UNKNOWN')
        option_type = trade_info.get('option_type', 'call')
        strike = trade_info.get('strike', 0.0)
        price = trade_info.get('price', 0.0)  # Underlying price
        
        # Get entry price based on order type and price point
        if order_type == 'market':
            entry_price = trade_info.get('ask', 0.0)  # Market orders execute at ask
        else:  # limit order
            if price_point == 'bid':
                entry_price = trade_info.get('bid', 0.0)
            elif price_point == 'ask':
                entry_price = trade_info.get('ask', 0.0)
            else:  # mid
                entry_price = trade_info.get('mid', 0.0)
        
        # Calculate position size (10% of account, minimum 1 contract)
        max_position_value = self.account_value * Config.POSITION_SIZE_PCT
        contracts = max(1, int(max_position_value / (entry_price * 100)))
        
        # Record the previous account value
        previous_account = self.account_value
        
        # Update account for entry
        position_value = entry_price * contracts * 100
        self.account_value -= position_value
        
        # Calculate entry commission
        commission = contracts * Config.COMMISSION_PER_CONTRACT
        self.account_value -= commission
        
        # Simulate holding period
        hold_time_minutes = random.randint(15, Config.MAX_HOLD_MINUTES)
        
        # Get a win probability based on patterns and/or ML
        win_probability = trade_info.get('win_probability', 0.5)
        
        # Determine the outcome
        outcome_roll = random.random()
        
        if outcome_roll < win_probability:
            # Profitable trade
            if random.random() < 0.7:  # 70% chance of full profit
                exit_price = entry_price * (1 + Config.PROFIT_TARGET_PCT)
                outcome_type = "PROFIT_TARGET"
            else:  # 30% chance of partial profit
                exit_price = entry_price * (1 + Config.PROFIT_TARGET_PCT * random.uniform(0.3, 0.8))
                outcome_type = "PARTIAL_PROFIT"
        else:
            # Losing trade
            if random.random() < 0.6:  # 60% chance of stop loss
                exit_price = entry_price * (1 - Config.STOP_LOSS_PCT)
                outcome_type = "STOP_LOSS"
            else:  # 40% chance of partial loss
                exit_price = entry_price * (1 - Config.STOP_LOSS_PCT * random.uniform(0.3, 0.8))
                outcome_type = "PARTIAL_LOSS"
        
        # Calculate price change percentage
        price_change_pct = (exit_price / entry_price - 1) * 100
        
        # Calculate gross profit
        gross_profit = (exit_price - entry_price) * contracts * 100
        
        # Calculate exit commission
        exit_commission = contracts * Config.COMMISSION_PER_CONTRACT
        self.account_value -= exit_commission
        
        # Calculate net profit
        net_profit = gross_profit - commission - exit_commission
        
        # Update account value for exit
        self.account_value += (position_value + gross_profit)
        
        # Create trade result
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(minutes=hold_time_minutes)
        
        trade_result = {
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'underlying_price': price,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': contracts,
            'hold_time_minutes': hold_time_minutes,
            'commission': commission + exit_commission,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'price_change_pct': price_change_pct,
            'outcome_type': outcome_type,
            'pattern_score': trade_info.get('score', 0.5),
            'win_probability': win_probability,
            'previous_account': previous_account,
            'new_account': self.account_value,
            'total_commission': commission + exit_commission
        }
        
        # Add to trading history
        self.trade_history.append(trade_result)
        
        # Print trade result if verbose
        self._print_trade_result(trade_result)
        
        return trade_result
    
    def _print_trade_result(self, result: Dict):
        """
        Print the result of a trade.
        
        Args:
            result: Dictionary with trade results
        """
        print("\n" + "=" * 80)
        print(f"TRADE OUTCOME: {result['symbol']} - {result['option_type'].upper()} ${result['strike']}")
        print("-" * 80)
        print(f"PATTERN SCORE: {result['pattern_score']:.2f} (predicted win prob: {result['win_probability']*100:.1f}%)")
        print(f"ENTRY TIME:    {result['entry_time'].strftime('%H:%M:%S')}")
        print(f"ENTRY PRICE:   ${result['entry_price']:.2f} x {result['position_size']} contracts")
        print(f"POSITION SIZE: ${(result['entry_price'] * result['position_size'] * 100):.2f}")
        print(f"EXIT TIME:     {result['exit_time'].strftime('%H:%M:%S')} ({result['hold_time_minutes']} min)")
        print(f"EXIT PRICE:    ${result['exit_price']:.2f} ({result['price_change_pct']:.1f}%)")
        print(f"OUTCOME:       {result['outcome_type']}")
        print(f"COMMISSION:    ${result['commission']:.2f}")
        print(f"NET P&L:       ${result['net_profit']:.2f}")
        print("-" * 80)
        print(f"PREVIOUS BALANCE: ${result['previous_account']:.2f}")
        print(f"CURRENT BALANCE:  ${result['new_account']:.2f}")
        account_change_pct = (result['new_account'] / result['previous_account'] - 1) * 100
        print(f"ACCOUNT CHANGE:   {account_change_pct:+.2f}%")
        print("=" * 80)
    
    def save_patterns(self):
        """Save the trading patterns to a file."""
        pattern_file = "smart_trader_patterns.json"
        self.patterns.save(pattern_file)

class TradingPattern:
    """
    Class to track and learn trading patterns based on market conditions.
    """
    
    def __init__(self):
        """Initialize the trading pattern recognition system."""
        self.patterns = {}
        self.data_source = "Simulated"
        self.source_tag = "sim"
        self.loaded = False
    
    def learn_from_trade(self, trade_info: Dict):
        """
        Learn from a trade outcome to improve future trading decisions.
        
        Args:
            trade_info: Dictionary with trade outcome information
        """
        # Extract key information
        symbol = trade_info.get('symbol', '').split()[0] if isinstance(trade_info.get('symbol'), str) else 'any'
        option_type = trade_info.get('option_type', 'call').lower()
        
        # Determine time of day
        entry_time = trade_info.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.strptime(entry_time, '%H:%M:%S')
                except:
                    entry_time = datetime.now()
                    
            hour = entry_time.hour
            
            if hour < 10:
                time_bucket = "morning"
            elif hour < 14:
                time_bucket = "midday"
            else:
                time_bucket = "afternoon"
        else:
            time_bucket = "midday"  # Default
        
        # Determine market condition
        rsi = trade_info.get('rsi', 50)
        
        if rsi < 30:
            market_condition = "oversold"
        elif rsi > 70:
            market_condition = "overbought"
        else:
            market_condition = "neutral"
        
        # Create pattern keys
        # Specific pattern for this symbol and option type
        specific_key = f"{symbol}_{option_type}_{time_bucket}_{market_condition}"
        
        # More general pattern for any symbol with this option type
        general_key = f"any_{option_type}_{time_bucket}_{market_condition}"
        
        # Determine outcome
        profitable = trade_info.get('net_profit', 0) > 0
        
        # Update pattern statistics
        for key in [specific_key, general_key]:
            if key not in self.patterns:
                self.patterns[key] = {'wins': 0, 'losses': 0}
            
            if profitable:
                self.patterns[key]['wins'] += 1
            else:
                self.patterns[key]['losses'] += 1
    
    def get_pattern_score(self, trade_info: Dict) -> float:
        """
        Calculate a score for a potential trade based on past patterns.
        
        Args:
            trade_info: Dictionary with trade information
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not self.patterns:
            return 0.5  # Default score if no patterns
        
        # Extract information
        symbol = trade_info.get('symbol', '').split()[0] if isinstance(trade_info.get('symbol'), str) else 'any'
        option_type = trade_info.get('option_type', 'call').lower()
        
        # Determine time of day
        hour = datetime.now().hour
        
        if hour < 10:
            time_bucket = "morning"
        elif hour < 14:
            time_bucket = "midday"
        else:
            time_bucket = "afternoon"
        
        # Determine market condition
        rsi = trade_info.get('rsi', 50)
        
        if rsi < 30:
            market_condition = "oversold"
        elif rsi > 70:
            market_condition = "overbought"
        else:
            market_condition = "neutral"
        
        # Create pattern keys
        specific_key = f"{symbol}_{option_type}_{time_bucket}_{market_condition}"
        general_key = f"any_{option_type}_{time_bucket}_{market_condition}"
        
        # Calculate score based on past performance
        score = 0.5  # Default neutral score
        
        # Check specific pattern first
        if specific_key in self.patterns:
            pattern = self.patterns[specific_key]
            total = pattern['wins'] + pattern['losses']
            
            if total >= 5:  # Only consider pattern if we have enough data
                win_rate = pattern['wins'] / total if total > 0 else 0.5
                confidence = min(1.0, total / 20)  # Max confidence at 20 samples
                
                # Adjust score based on win rate and confidence
                score = win_rate * 0.6 + 0.4  # Weighted win rate (0.4 to 1.0)
                
                # If we have a strong signal from the specific pattern, return it
                if total >= 10 and (win_rate >= 0.6 or win_rate <= 0.4):
                    return max(0.2, min(0.9, score))  # Cap between 0.2 and 0.9
        
        # Check the general pattern if specific isn't strong
        if general_key in self.patterns:
            pattern = self.patterns[general_key]
            total = pattern['wins'] + pattern['losses']
            
            if total >= 10:  # Need more samples for general pattern
                win_rate = pattern['wins'] / total if total > 0 else 0.5
                
                # Mix with the current score
                score = (score + win_rate) / 2
        
        return max(0.2, min(0.9, score))  # Cap between 0.2 and 0.9
    
    def get_win_probability(self, trade_info: Dict) -> float:
        """
        Get win probability for a trade based on pattern matching.
        
        Args:
            trade_info: Dictionary with trade information
            
        Returns:
            Win probability between 0.0 and 1.0
        """
        score = self.get_pattern_score(trade_info)
        
        # Convert score to probability
        return score
    
    def save(self, filename: str = "trading_patterns.json") -> bool:
        """
        Save patterns to a JSON file.
        
        Args:
            filename: Name of the file to save patterns to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'patterns': self.patterns,
                    'data_source': self.data_source,
                    'source_tag': self.source_tag
                }, f)
            return True
        except Exception as e:
            print(f"Error saving patterns: {e}")
            return False
    
    def load(self, filename: str = "trading_patterns.json") -> bool:
        """
        Load patterns from a JSON file.
        
        Args:
            filename: Name of the file to load patterns from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filename):
                return False
                
            with open(filename, 'r') as f:
                data = json.load(f)
                
            if 'patterns' in data:
                self.patterns = data['patterns']
                
                # Update source info if available
                if 'data_source' in data:
                    self.data_source = data['data_source']
                if 'source_tag' in data:
                    self.source_tag = data['source_tag']
                    
                self.loaded = True
                return True
        except Exception as e:
            print(f"Error loading patterns: {e}")
            
        return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the current patterns.
        
        Returns:
            Dictionary with pattern statistics
        """
        if not self.patterns:
            return {
                'total_patterns': 0,
                'total_trades': 0,
                'overall_win_rate': 0.0,
                'best_patterns': []
            }
        
        # Calculate overall statistics
        total_patterns = len(self.patterns)
        total_wins = sum(p['wins'] for p in self.patterns.values())
        total_losses = sum(p['losses'] for p in self.patterns.values())
        total_trades = total_wins + total_losses
        
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0.0
        
        # Find best performing patterns
        pattern_stats = []
        
        for key, data in self.patterns.items():
            total = data['wins'] + data['losses']
            
            if total >= 5:  # Only consider patterns with enough data
                win_rate = data['wins'] / total
                
                pattern_stats.append({
                    'pattern': key,
                    'win_rate': win_rate,
                    'total': total
                })
        
        # Sort by win rate and total trades
        pattern_stats.sort(key=lambda x: (-x['win_rate'], -x['total']))
        
        # Return statistics
        return {
            'total_patterns': total_patterns,
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'best_patterns': pattern_stats[:10]  # Top 10 patterns
        }
    
    def print_stats(self):
        """Print statistics about the current patterns."""
        stats = self.get_stats()
        
        print("TRADING PATTERN STATISTICS:")
        print(f"Total Patterns: {stats['total_patterns']}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Overall Win Rate: {stats['overall_win_rate']:.1%}")
        
        if stats['best_patterns']:
            print("TOP PERFORMING PATTERNS:")
            for i, pattern in enumerate(stats['best_patterns'][:5], 1):
                pattern_parts = pattern['pattern'].split('_')
                pattern_desc = f"{pattern_parts[0]} {pattern_parts[1].upper()} in {pattern_parts[2]} when {pattern_parts[3]}"
                print(f"{i}. {pattern_desc}")
                print(f"   Win Rate: {pattern['win_rate']:.1%} ({pattern['total']} trades)") 