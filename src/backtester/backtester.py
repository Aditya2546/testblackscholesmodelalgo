"""
Event-driven backtester for options trading strategies.

Provides tick-by-tick simulation of market events with realistic
order execution and analysis of strategy performance.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from collections import deque

from ..data.market_data import Asset, Quote, Trade, OptionChain, MarketDataBar, BarPeriod
from ..strategy.signal import Signal, SignalGenerator
from ..pricing.model_interface import OptionModel, ModelFactory
from ..execution.order_manager import Order, OrderStatus, OrderLeg, OrderType, TimeInForce
from ..risk.position_sizing import PositionSizer, KellyPositionSizer

# Configure logging
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of backtest events."""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    TIMER = "timer"
    CUSTOM = "custom"


@dataclass
class Event:
    """Base class for backtester events."""
    type: EventType
    time: datetime
    data: Any = None


class MarketDataEvent(Event):
    """Market data update event."""
    
    def __init__(self, time: datetime, asset: Asset, data: Any):
        super().__init__(EventType.MARKET_DATA, time, data)
        self.asset = asset


class SignalEvent(Event):
    """Trading signal event."""
    
    def __init__(self, time: datetime, signal: Signal):
        super().__init__(EventType.SIGNAL, time, signal)
        self.signal = signal


class OrderEvent(Event):
    """Order event (creation, update, etc.)."""
    
    def __init__(self, time: datetime, order: Order):
        super().__init__(EventType.ORDER, time, order)
        self.order = order


class FillEvent(Event):
    """Order fill event."""
    
    def __init__(self, time: datetime, order: Order, fill_price: float, 
                fill_quantity: int, commission: float = 0.0):
        super().__init__(EventType.FILL, time, order)
        self.order = order
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity
        self.commission = commission


class TimerEvent(Event):
    """Timer event for periodic actions."""
    
    def __init__(self, time: datetime, interval: timedelta):
        super().__init__(EventType.TIMER, time, interval)
        self.interval = interval


@dataclass
class BacktestPosition:
    """Position held in the backtest portfolio."""
    asset: Asset
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class BacktestResults:
    """Results of a backtest run."""
    start_time: datetime
    end_time: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    signals: pd.DataFrame
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class BacktestExecution:
    """
    Simulated order execution for backtesting.
    
    Models realistic order execution with slippage, partial fills,
    and delays to simulate real-world market conditions.
    """
    
    def __init__(self, slippage_model: Optional[Callable] = None, 
                 fill_probability: float = 1.0,
                 fill_delay_seconds: float = 0.0):
        """
        Initialize backtest execution model.
        
        Args:
            slippage_model: Function to calculate price slippage
            fill_probability: Probability of order being filled (0.0-1.0)
            fill_delay_seconds: Simulated delay in seconds for fills
        """
        self.slippage_model = slippage_model or self._default_slippage
        self.fill_probability = fill_probability
        self.fill_delay_seconds = fill_delay_seconds
    
    def _default_slippage(self, order: Order, quote: Quote) -> float:
        """
        Default slippage model based on order size and bid-ask spread.
        
        Args:
            order: Order being executed
            quote: Current market quote
            
        Returns:
            Slippage amount in percentage
        """
        # Calculate bid-ask spread as percentage
        spread_pct = (quote.ask_price - quote.bid_price) / quote.midpoint
        
        # Base slippage is half the spread (aggressive)
        base_slippage = spread_pct / 2
        
        # Scale with square root of quantity for larger orders
        total_quantity = sum(abs(leg.quantity) for leg in order.legs)
        size_factor = np.sqrt(total_quantity / 10) if total_quantity > 10 else 1.0
        
        return base_slippage * size_factor
    
    def process_order(self, order: Order, quote: Quote, 
                     current_time: datetime) -> Optional[FillEvent]:
        """
        Process an order against current market conditions.
        
        Args:
            order: Order to process
            quote: Current market quote
            current_time: Current simulation time
            
        Returns:
            Fill event if order is filled, None otherwise
        """
        # Check if order is fillable
        if order.status not in [OrderStatus.CREATED, OrderStatus.VALIDATED, OrderStatus.SENT]:
            return None
        
        # Only process the first leg for now (simplified model)
        # In a full implementation, would handle multi-leg orders properly
        if not order.legs:
            return None
            
        leg = order.legs[0]
        
        # Determine fill price based on order type and direction
        fill_price = None
        
        if leg.order_type == OrderType.MARKET:
            # Market orders execute at bid/ask with slippage
            if leg.action == "BUY":
                # Buy at ask price plus slippage
                slippage = self.slippage_model(order, quote)
                fill_price = quote.ask_price * (1.0 + slippage)
            else:
                # Sell at bid price minus slippage
                slippage = self.slippage_model(order, quote)
                fill_price = quote.bid_price * (1.0 - slippage)
                
        elif leg.order_type == OrderType.LIMIT:
            # Limit orders execute if price is favorable
            if leg.action == "BUY" and quote.ask_price <= leg.limit_price:
                # Can buy at or better than limit
                fill_price = min(quote.ask_price, leg.limit_price)
            elif leg.action == "SELL" and quote.bid_price >= leg.limit_price:
                # Can sell at or better than limit
                fill_price = max(quote.bid_price, leg.limit_price)
        
        # If we determined a fill price, create fill event
        if fill_price:
            # Apply fill probability
            if np.random.random() <= self.fill_probability:
                # Calculate fill quantity (could be partial)
                fill_quantity = leg.quantity
                
                # Simple commission model ($1 per contract)
                commission = fill_quantity * 1.0
                
                # Calculate fill time with delay
                fill_time = current_time
                if self.fill_delay_seconds > 0:
                    fill_time += timedelta(seconds=self.fill_delay_seconds)
                
                # Create fill event
                return FillEvent(
                    time=fill_time,
                    order=order,
                    fill_price=fill_price,
                    fill_quantity=fill_quantity,
                    commission=commission
                )
        
        return None


class BacktestPortfolio:
    """
    Portfolio management for backtesting.
    
    Tracks positions, calculates P&L, and handles order management
    during backtest simulation.
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                position_sizer: Optional[PositionSizer] = None):
        """
        Initialize backtest portfolio.
        
        Args:
            initial_capital: Initial capital in dollars
            position_sizer: Position sizer for determining trade size
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_sizer = position_sizer or KellyPositionSizer(initial_capital)
        
        # Current positions
        self.positions: Dict[str, BacktestPosition] = {}
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.pending_orders: List[Order] = []
        
        # Performance tracking
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self.metrics: Dict[str, float] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0
        }
    
    def _generate_position_key(self, asset: Asset) -> str:
        """
        Generate a unique key for a position.
        
        Args:
            asset: Asset for the position
            
        Returns:
            Unique position key
        """
        # For options, include expiration, strike, and right
        if hasattr(asset, "expiration") and hasattr(asset, "strike") and hasattr(asset, "right"):
            return f"{asset.symbol}_{asset.expiration.strftime('%Y%m%d')}_{asset.strike}_{asset.right.value}"
        else:
            return asset.symbol
    
    def _update_position(self, asset: Asset, quantity: int, price: float, 
                       time: datetime, commission: float = 0.0) -> None:
        """
        Update a position in the portfolio.
        
        Args:
            asset: Asset being traded
            quantity: Quantity being traded (positive for buy, negative for sell)
            price: Execution price
            time: Execution time
            commission: Commission paid
        """
        position_key = self._generate_position_key(asset)
        
        # If position exists, update it
        if position_key in self.positions:
            position = self.positions[position_key]
            
            # Calculate cost and P&L
            cost = quantity * price
            commissions = commission
            
            # Update position
            old_quantity = position.quantity
            old_value = old_quantity * position.current_price
            
            # If reducing or closing position, calculate realized P&L
            if (old_quantity > 0 and quantity < 0) or (old_quantity < 0 and quantity > 0):
                # Closing or reducing position
                closing_quantity = min(abs(old_quantity), abs(quantity))
                if old_quantity > 0:
                    # Long position being reduced
                    realized_pnl = (price - position.entry_price) * closing_quantity - commissions
                else:
                    # Short position being reduced
                    realized_pnl = (position.entry_price - price) * closing_quantity - commissions
                
                position.realized_pnl += realized_pnl
                self.metrics["total_pnl"] += realized_pnl
                
                # Record trade if fully closed
                if abs(closing_quantity) == abs(old_quantity) or (old_quantity + quantity) * old_quantity <= 0:
                    trade_result = {
                        "symbol": asset.symbol,
                        "entry_time": position.entry_time,
                        "entry_price": position.entry_price,
                        "exit_time": time,
                        "exit_price": price,
                        "quantity": old_quantity,
                        "pnl": position.realized_pnl,
                        "return": position.realized_pnl / (abs(old_quantity) * position.entry_price)
                    }
                    self.trades.append(trade_result)
                    self.metrics["total_trades"] += 1
                    
                    if position.realized_pnl > 0:
                        self.metrics["winning_trades"] += 1
                    else:
                        self.metrics["losing_trades"] += 1
            
            # Update position quantity and value
            position.quantity += quantity
            position.current_price = price
            position.current_value = position.quantity * price
            
            # If position is fully closed, remove it
            if position.quantity == 0:
                del self.positions[position_key]
            else:
                # Otherwise, recalculate entry price for additions
                if (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                    # Adding to position, update entry price as weighted average
                    position.entry_price = ((old_quantity * position.entry_price + quantity * price) / 
                                          (old_quantity + quantity))
                
                # Update unrealized P&L
                if position.quantity > 0:
                    position.unrealized_pnl = (price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - price) * abs(position.quantity)
                
                position.total_pnl = position.realized_pnl + position.unrealized_pnl
        
        else:
            # Create new position
            position = BacktestPosition(
                asset=asset,
                quantity=quantity,
                entry_price=price,
                entry_time=time,
                current_price=price,
                current_value=quantity * price,
                unrealized_pnl=0.0,
                realized_pnl=-commission,  # Initial commission impacts realized P&L
                total_pnl=-commission
            )
            
            self.positions[position_key] = position
        
        # Update capital
        self.current_capital -= (quantity * price + commission)
    
    def process_fill(self, fill: FillEvent) -> None:
        """
        Process a fill event in the portfolio.
        
        Args:
            fill: Fill event
        """
        order = fill.order
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_time = fill.time
        
        # Update positions for each leg
        for leg in order.legs:
            # Create simplified asset for now
            # In a full implementation, would create the proper asset type
            from ..data.market_data import Option, OptionRight
            
            asset = Option(
                symbol=leg.symbol,
                underlying_symbol=leg.symbol.split("_")[0] if "_" in leg.symbol else leg.symbol,
                expiration=leg.expiration,
                strike=leg.strike,
                right=leg.right,
                exchange=leg.exchange
            )
            
            # Determine quantity (positive for buy, negative for sell)
            quantity = leg.quantity if leg.action == "BUY" else -leg.quantity
            
            # Update position
            self._update_position(
                asset=asset,
                quantity=quantity,
                price=fill.fill_price,
                time=fill.time,
                commission=fill.commission
            )
        
        # Remove from pending orders
        if order in self.pending_orders:
            self.pending_orders.remove(order)
    
    def create_order(self, signal: Signal, current_time: datetime) -> Optional[Order]:
        """
        Create an order from a signal.
        
        Args:
            signal: Trading signal
            current_time: Current simulation time
            
        Returns:
            Created order or None if no order created
        """
        from ..execution.order_manager import Order, OrderLeg, OrderType, TimeInForce, OrderStatus
        import uuid
        
        # Use position sizer to determine quantity
        # This requires updating the portfolio value in the position sizer
        self.position_sizer.portfolio_value = self.current_capital
        position_size = self.position_sizer.calculate_position_size(signal)
        
        if position_size.quantity <= 0:
            # No position to take
            return None
        
        # Create order legs
        legs = []
        
        if len(signal.legs) > 0:
            # Multi-leg order
            for leg in signal.legs:
                action = "BUY" if leg.get("quantity", 0) > 0 else "SELL"
                
                order_leg = OrderLeg(
                    symbol=signal.symbol,
                    expiration=leg.get("expiration"),
                    strike=leg.get("strike"),
                    right=leg.get("right"),
                    action=action,
                    quantity=abs(leg.get("quantity", 0)) * position_size.quantity,
                    order_type=OrderType.LIMIT,
                    limit_price=leg.get("price", 0),
                    exchange="BACKTEST"
                )
                
                legs.append(order_leg)
        else:
            # Single-leg order
            action = "BUY" if signal.direction > 0 else "SELL"
            
            order_leg = OrderLeg(
                symbol=signal.symbol,
                expiration=signal.expiration,
                strike=signal.strike,
                right=signal.right,
                action=action,
                quantity=position_size.quantity,
                order_type=OrderType.LIMIT,
                limit_price=signal.market_price if signal.market_price else 0.0,
                exchange="BACKTEST"
            )
            
            legs.append(order_leg)
        
        # Create order
        order = Order(
            id=str(uuid.uuid4()),
            legs=legs,
            time_in_force=TimeInForce.DAY,
            status=OrderStatus.CREATED,
            signal_id=getattr(signal, 'id', None),
            created_time=current_time
        )
        
        # Add to orders and pending orders
        self.orders[order.id] = order
        self.pending_orders.append(order)
        
        return order
    
    def update_portfolio_value(self, quotes: Dict[str, Quote], current_time: datetime) -> None:
        """
        Update portfolio value based on current market quotes.
        
        Args:
            quotes: Current market quotes
            current_time: Current simulation time
        """
        total_portfolio_value = self.current_capital
        unrealized_pnl = 0.0
        
        # Update position values
        for key, position in self.positions.items():
            # Find quote for this asset
            asset_symbol = position.asset.symbol
            if asset_symbol in quotes:
                quote = quotes[asset_symbol]
                
                # Update position price and value
                old_price = position.current_price
                position.current_price = quote.midpoint
                position.current_value = position.quantity * position.current_price
                
                # Recalculate unrealized P&L
                if position.quantity > 0:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * abs(position.quantity)
                
                position.total_pnl = position.realized_pnl + position.unrealized_pnl
                
                # Add to portfolio value
                unrealized_pnl += position.unrealized_pnl
            
            # Add position value to total
            total_portfolio_value += position.current_value
        
        # Record equity curve point
        equity_point = {
            "time": current_time,
            "equity": total_portfolio_value,
            "cash": self.current_capital,
            "positions_value": total_portfolio_value - self.current_capital,
            "unrealized_pnl": unrealized_pnl
        }
        
        self.equity_curve.append(equity_point)
        
        # Update max drawdown
        if len(self.equity_curve) > 1:
            peak = max(point["equity"] for point in self.equity_curve[:len(self.equity_curve)-1])
            if total_portfolio_value < peak:
                drawdown = (peak - total_portfolio_value) / peak
                self.metrics["max_drawdown"] = max(self.metrics["max_drawdown"], drawdown)
    
    def get_results(self, start_time: datetime, end_time: datetime) -> BacktestResults:
        """
        Get backtest results.
        
        Args:
            start_time: Backtest start time
            end_time: Backtest end time
            
        Returns:
            Backtest results
        """
        # Convert lists to DataFrames
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate performance metrics
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1.0) * 100.0
        
        # Calculate annualized return
        days = (end_time - start_time).days
        if days > 0:
            years = days / 365.0
            annualized_return = ((1 + total_return / 100.0) ** (1 / years) - 1) * 100.0
        else:
            annualized_return = 0.0
        
        # Calculate Sharpe ratio if we have enough data
        if len(equity_df) > 1:
            equity_df["daily_return"] = equity_df["equity"].pct_change()
            daily_returns = equity_df["daily_return"].dropna()
            
            if len(daily_returns) > 0:
                avg_return = daily_returns.mean()
                std_return = daily_returns.std()
                sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Create positions DataFrame
        positions_data = []
        for key, pos in self.positions.items():
            positions_data.append({
                "symbol": pos.asset.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "current_value": pos.current_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "total_pnl": pos.total_pnl
            })
        
        positions_df = pd.DataFrame(positions_data)
        
        # Add additional metrics
        win_rate = (self.metrics["winning_trades"] / self.metrics["total_trades"] 
                   if self.metrics["total_trades"] > 0 else 0.0)
        
        metrics = {
            "total_trades": self.metrics["total_trades"],
            "winning_trades": self.metrics["winning_trades"],
            "losing_trades": self.metrics["losing_trades"],
            "win_rate": win_rate * 100.0,
            "total_pnl": self.metrics["total_pnl"],
            "max_drawdown": self.metrics["max_drawdown"] * 100.0
        }
        
        # Create dummy signals DataFrame if not tracking signals
        signals_df = pd.DataFrame()
        
        # Create results object
        results = BacktestResults(
            start_time=start_time,
            end_time=end_time,
            initial_capital=self.initial_capital,
            final_capital=final_equity,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.metrics["max_drawdown"] * 100.0,
            trades=trades_df,
            equity_curve=equity_df,
            positions=positions_df,
            signals=signals_df,
            performance_metrics=metrics
        )
        
        return results 


class Backtester:
    """
    Event-driven backtester for options trading strategies.
    
    Simulates trading with historical data and evaluates
    strategy performance with realistic execution.
    """
    
    def __init__(self, strategy_generator: SignalGenerator, 
                 initial_capital: float = 100000.0,
                 commission_model: Callable[[Order], float] = None):
        """
        Initialize backtester.
        
        Args:
            strategy_generator: Signal generator for strategy
            initial_capital: Initial capital
            commission_model: Model for calculating commissions
        """
        self.strategy_generator = strategy_generator
        self.initial_capital = initial_capital
        self.commission_model = commission_model or (lambda order: 1.0 * sum(abs(leg.quantity) for leg in order.legs))
        
        # Market data
        self.quotes: Dict[str, Quote] = {}
        self.option_chains: Dict[str, OptionChain] = {}
        
        # Portfolio and execution
        self.portfolio = BacktestPortfolio(initial_capital)
        self.execution = BacktestExecution()
        
        # Event queue
        self.events = deque()
        
        # Tracking
        self.current_time = datetime.now()
        self.start_time = None
        self.end_time = None
        self.processing = False
        self.signals: List[Signal] = []
    
    def load_market_data(self, data: Dict[str, pd.DataFrame],
                        option_chains: Dict[datetime, Dict[str, OptionChain]] = None) -> None:
        """
        Load historical market data for backtesting.
        
        Args:
            data: Dictionary mapping symbols to price dataframes
            option_chains: Option chains by date and underlying
        """
        self.market_data = data
        self.historical_option_chains = option_chains or {}
        
        # Find earliest and latest timestamps
        all_dates = []
        for df in data.values():
            if not df.empty and 'time' in df.columns:
                all_dates.extend(df['time'].tolist())
        
        if all_dates:
            self.start_time = min(all_dates)
            self.end_time = max(all_dates)
            self.current_time = self.start_time
    
    def _generate_market_events(self, current_time: datetime) -> List[Event]:
        """
        Generate market data events for the current timestamp.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of market data events
        """
        events = []
        
        # Process price data
        for symbol, df in self.market_data.items():
            # Filter data for this timestamp
            if 'time' in df.columns:
                data_at_time = df[df['time'] == current_time]
                
                if not data_at_time.empty:
                    row = data_at_time.iloc[0]
                    
                    # Create quote
                    if all(col in row.index for col in ['bid', 'ask']):
                        quote = Quote(
                            timestamp=current_time,
                            bid_price=row['bid'],
                            ask_price=row['ask'],
                            bid_size=row.get('bidSize', 100),
                            ask_size=row.get('askSize', 100)
                        )
                        
                        # Store quote
                        self.quotes[symbol] = quote
                        
                        # Create asset (simplified)
                        from ..data.market_data import Equity
                        asset = Equity(symbol=symbol)
                        
                        # Create event
                        event = MarketDataEvent(current_time, asset, quote)
                        events.append(event)
        
        # Process option chains
        if current_time in self.historical_option_chains:
            chains_at_time = self.historical_option_chains[current_time]
            
            for symbol, chain in chains_at_time.items():
                # Store chain
                self.option_chains[symbol] = chain
                
                # Create asset (simplified)
                from ..data.market_data import Equity
                asset = Equity(symbol=symbol)
                
                # Create event
                event = MarketDataEvent(current_time, asset, chain)
                events.append(event)
        
        return events
    
    def run(self) -> BacktestResults:
        """
        Run the backtest simulation.
        
        Returns:
            Backtest results
        """
        if not self.start_time or not self.end_time:
            raise ValueError("Market data must be loaded before running backtest")
        
        logger.info(f"Starting backtest from {self.start_time} to {self.end_time}")
        
        self.processing = True
        self.current_time = self.start_time
        
        # Main event loop
        while self.current_time <= self.end_time and self.processing:
            # Generate market events
            market_events = self._generate_market_events(self.current_time)
            for event in market_events:
                self.events.append(event)
            
            # Process all events for this timestamp
            while self.events and self.processing:
                event = self.events.popleft()
                self._process_event(event)
            
            # Move to next timestamp
            self._advance_time()
        
        # Calculate and return results
        return self.portfolio.get_results(self.start_time, self.end_time)
    
    def _process_event(self, event: Event) -> None:
        """
        Process a single event in the simulation.
        
        Args:
            event: Event to process
        """
        if event.type == EventType.MARKET_DATA:
            # Process market data update
            self._process_market_data(event)
            
        elif event.type == EventType.SIGNAL:
            # Process trading signal
            self._process_signal(event)
            
        elif event.type == EventType.ORDER:
            # Process order update
            self._process_order(event)
            
        elif event.type == EventType.FILL:
            # Process order fill
            self._process_fill(event)
            
        elif event.type == EventType.TIMER:
            # Process timer event
            self._process_timer(event)
    
    def _process_market_data(self, event: MarketDataEvent) -> None:
        """
        Process market data event.
        
        Args:
            event: Market data event
        """
        # Update portfolio value
        self.portfolio.update_portfolio_value(self.quotes, self.current_time)
        
        # Run signal generator if we have option chains
        if self.option_chains:
            for symbol, chain in self.option_chains.items():
                # Get underlying price
                if symbol in self.quotes:
                    underlying_price = self.quotes[symbol].midpoint
                    
                    # Generate signals
                    signals = self.strategy_generator.generate_signals(chain, underlying_price)
                    
                    # Create signal events
                    for signal in signals:
                        signal_event = SignalEvent(self.current_time, signal)
                        self.events.append(signal_event)
                        self.signals.append(signal)
    
    def _process_signal(self, event: SignalEvent) -> None:
        """
        Process signal event.
        
        Args:
            event: Signal event
        """
        # Create order from signal
        order = self.portfolio.create_order(event.signal, self.current_time)
        
        if order:
            # Create order event
            order_event = OrderEvent(self.current_time, order)
            self.events.append(order_event)
    
    def _process_order(self, event: OrderEvent) -> None:
        """
        Process order event.
        
        Args:
            event: Order event
        """
        order = event.order
        
        # Find quote for the order's symbol
        symbol = order.legs[0].symbol if order.legs else None
        
        if symbol and symbol in self.quotes:
            quote = self.quotes[symbol]
            
            # Try to execute order
            fill_event = self.execution.process_order(order, quote, self.current_time)
            
            if fill_event:
                self.events.append(fill_event)
    
    def _process_fill(self, event: FillEvent) -> None:
        """
        Process fill event.
        
        Args:
            event: Fill event
        """
        # Apply fill to portfolio
        self.portfolio.process_fill(event)
    
    def _process_timer(self, event: TimerEvent) -> None:
        """
        Process timer event.
        
        Args:
            event: Timer event
        """
        # Add any periodic logic here
        pass
    
    def _advance_time(self) -> None:
        """Advance simulation to next timestamp."""
        # Find next timestamp in data
        next_time = None
        
        for df in self.market_data.values():
            if 'time' in df.columns:
                # Find times after current time
                future_times = df[df['time'] > self.current_time]['time']
                
                if not future_times.empty:
                    # Get earliest future time
                    earliest = min(future_times)
                    
                    # Update next_time if this is earlier
                    if next_time is None or earliest < next_time:
                        next_time = earliest
        
        # If no more data, end simulation
        if next_time is None:
            self.processing = False
        else:
            self.current_time = next_time
    
    def stop(self) -> None:
        """Stop the backtest simulation."""
        self.processing = False 