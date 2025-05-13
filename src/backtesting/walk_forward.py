"""
Walk-Forward Backtesting for Options Trading System

This module implements rolling window backtesting with walk-forward analysis:
- 60d train / 20d test rolling windows
- Injects realistic latency & micro-slippage based on SpiderRock benchmarks
- Calculates key metrics: Sharpe, Sortino, max DD, win_rate, profit_factor

The walk-forward approach helps prevent overfitting by continuously retraining
the model on recent data and validating on out-of-sample periods.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# Import local modules
from src.ml.stacked_ensemble import StackedEnsemble
from src.risk.kelly_size import KellyPositionSizer
from src.risk.greeks_cap import GreeksCap

# Configure logging
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types for backtesting."""
    CSV = "csv"
    INFLUXDB = "influxdb"
    PARQUET = "parquet"
    SQLITE = "sqlite"


@dataclass
class SlippageModel:
    """Model for simulating execution slippage."""
    base_slippage_bps: float = 5.0  # Base slippage in basis points
    volatility_factor: float = 1.2  # Higher vol = higher slippage
    liquidity_factor: float = 0.8   # Higher liquidity = lower slippage
    market_impact_factor: float = 0.5  # Impact of order size
    
    def calculate_slippage(self, 
                         price: float,
                         size: int,
                         volatility: float,
                         avg_volume: int) -> float:
        """
        Calculate execution slippage.
        
        Args:
            price: Option price
            size: Order size in contracts
            volatility: Implied volatility
            avg_volume: Average daily volume
            
        Returns:
            Slippage amount in dollars per contract
        """
        # Base slippage as percentage of price
        base_slip = price * (self.base_slippage_bps / 10000)
        
        # Adjust for volatility (higher vol = more slippage)
        vol_adjustment = 1.0 + (volatility - 0.2) * self.volatility_factor
        vol_adjustment = max(0.5, min(vol_adjustment, 3.0))  # Limit range
        
        # Adjust for liquidity (higher liquidity = less slippage)
        liquidity_ratio = min(1.0, size / max(1, avg_volume * 0.01))
        liquidity_adjustment = 1.0 + (liquidity_ratio * self.liquidity_factor)
        
        # Market impact (higher size relative to avg volume = more slippage)
        market_impact = 1.0 + (size / max(1, avg_volume * 0.1)) * self.market_impact_factor
        
        # Combine factors
        total_slippage = base_slip * vol_adjustment * liquidity_adjustment * market_impact
        
        # Cap at reasonable percentage of price
        max_slippage = price * 0.05  # Max 5% slippage
        return min(total_slippage, max_slippage)


@dataclass
class LatencyModel:
    """Model for simulating execution latency."""
    base_latency_ms: float = 15.0  # Base latency in milliseconds
    volatility_factor: float = 5.0  # Higher vol = higher latency
    market_load_factor: float = 10.0  # Market-wide activity impact
    
    def calculate_latency(self, 
                        volatility: float,
                        market_load: float) -> float:
        """
        Calculate execution latency.
        
        Args:
            volatility: Implied volatility
            market_load: Market activity level (0.0-1.0)
            
        Returns:
            Latency in milliseconds
        """
        # Base latency
        latency = self.base_latency_ms
        
        # Adjust for volatility
        vol_adjustment = 1.0 + (volatility - 0.2) * self.volatility_factor
        vol_adjustment = max(0.5, min(vol_adjustment, 3.0))  # Limit range
        
        # Adjust for market load
        load_adjustment = 1.0 + (market_load * self.market_load_factor)
        
        # Combine factors
        total_latency = latency * vol_adjustment * load_adjustment
        
        # Add randomness (±20%)
        randomness = 0.8 + (0.4 * np.random.random())
        
        return total_latency * randomness


@dataclass
class BacktestMetrics:
    """Metrics for backtest evaluation."""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    
    # Trade metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Exposure metrics
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Other
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration_days": self.drawdown_duration_days,
            "volatility": self.volatility,
            "downside_deviation": self.downside_deviation,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_position_size": self.avg_position_size,
            "max_position_size": self.max_position_size,
            "avg_trade_duration": self.avg_trade_duration,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BacktestMetrics':
        """Create metrics from dictionary."""
        metrics = cls()
        metrics.total_return = data.get("total_return", 0.0)
        metrics.annualized_return = data.get("annualized_return", 0.0)
        metrics.sharpe_ratio = data.get("sharpe_ratio", 0.0)
        metrics.sortino_ratio = data.get("sortino_ratio", 0.0)
        metrics.max_drawdown = data.get("max_drawdown", 0.0)
        metrics.drawdown_duration_days = int(data.get("drawdown_duration_days", 0))
        metrics.volatility = data.get("volatility", 0.0)
        metrics.downside_deviation = data.get("downside_deviation", 0.0)
        metrics.win_rate = data.get("win_rate", 0.0)
        metrics.profit_factor = data.get("profit_factor", 0.0)
        metrics.avg_win = data.get("avg_win", 0.0)
        metrics.avg_loss = data.get("avg_loss", 0.0)
        metrics.largest_win = data.get("largest_win", 0.0)
        metrics.largest_loss = data.get("largest_loss", 0.0)
        metrics.avg_position_size = data.get("avg_position_size", 0.0)
        metrics.max_position_size = data.get("max_position_size", 0.0)
        metrics.avg_trade_duration = data.get("avg_trade_duration", 0.0)
        metrics.total_trades = int(data.get("total_trades", 0))
        metrics.winning_trades = int(data.get("winning_trades", 0))
        metrics.losing_trades = int(data.get("losing_trades", 0))
        return metrics


@dataclass
class BacktestTrade:
    """Individual trade in backtest."""
    symbol: str
    contract_id: str
    option_type: str  # "CALL" or "PUT"
    strike: float
    expiration: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    direction: str = "long"  # "long" or "short"
    edge_probability: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trade_duration: float = 0.0  # In minutes
    slippage: float = 0.0
    commission: float = 0.0
    iv_entry: Optional[float] = None
    iv_exit: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "symbol": self.symbol,
            "contract_id": self.contract_id,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "direction": self.direction,
            "edge_probability": self.edge_probability,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "trade_duration": self.trade_duration,
            "slippage": self.slippage,
            "commission": self.commission,
            "iv_entry": self.iv_entry,
            "iv_exit": self.iv_exit
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    # Data parameters
    start_date: datetime 
    end_date: datetime
    symbols: List[str]
    data_source: DataSource = DataSource.CSV
    data_path: str = "data"
    
    # Walk-forward parameters
    train_window_days: int = 60
    test_window_days: int = 20
    step_size_days: int = 20
    
    # Model parameters
    retrain_model: bool = True
    model_path: str = "models"
    
    # Trading parameters
    initial_capital: float = 100000.0
    max_position_pct: float = 0.05
    kelly_fraction: float = 0.25
    min_edge_probability: float = 0.67
    
    # Execution parameters
    commission_per_contract: float = 0.65
    slippage_model: Optional[SlippageModel] = None
    latency_model: Optional[LatencyModel] = None
    
    # Output parameters
    output_path: str = "backtest_results"
    save_trades: bool = True
    plot_results: bool = True
    
    def __post_init__(self):
        """Initialize defaults for models."""
        if self.slippage_model is None:
            self.slippage_model = SlippageModel()
        
        if self.latency_model is None:
            self.latency_model = LatencyModel()


class WalkForwardBacktest:
    """
    Walk-forward backtest for options trading strategies.
    
    Implements rolling window backtesting with walk-forward analysis
    to prevent overfitting by continuously retraining on recent data
    and validating on out-of-sample periods.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize walk-forward backtest.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        
        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.position_sizer = KellyPositionSizer(
            portfolio_value=self.config.initial_capital,
            kelly_fraction=self.config.kelly_fraction
        )
        self.greeks_cap = GreeksCap()
        
        # Backtest state
        self.current_capital = self.config.initial_capital
        self.trades: List[BacktestTrade] = []
        self.daily_returns: List[float] = []
        self.daily_equity: List[float] = []
        self.dates: List[datetime] = []
        
        # Performance metrics
        self.metrics = BacktestMetrics()
        self.window_metrics: List[BacktestMetrics] = []
        
        # Data cache
        self._data_cache = {}
        
        logger.info(f"Initialized walk-forward backtest from {config.start_date} to {config.end_date}")
    
    def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load data for symbol.
        
        Args:
            symbol: Symbol to load data for
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with data
        """
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        # Check cache
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Load data based on source
        if self.config.data_source == DataSource.CSV:
            # Load from CSV
            file_path = os.path.join(self.config.data_path, f"{symbol}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path, parse_dates=["date"])
            
            # Filter by date
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            
        elif self.config.data_source == DataSource.PARQUET:
            # Load from Parquet
            file_path = os.path.join(self.config.data_path, f"{symbol}.parquet")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_parquet(file_path)
            
            # Ensure date column exists and is datetime
            if "date" not in df.columns:
                raise ValueError("Data must have a 'date' column")
            
            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df["date"] = pd.to_datetime(df["date"])
            
            # Filter by date
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            
        elif self.config.data_source == DataSource.INFLUXDB:
            # Not implemented in this version
            raise NotImplementedError("InfluxDB data source not implemented")
            
        elif self.config.data_source == DataSource.SQLITE:
            # Not implemented in this version
            raise NotImplementedError("SQLite data source not implemented")
            
        else:
            raise ValueError(f"Unknown data source: {self.config.data_source}")
        
        # Verify data has required columns
        required_columns = ["date", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Data must have column: {col}")
        
        # Sort by date
        df = df.sort_values("date")
        
        # Add to cache
        self._data_cache[cache_key] = df
        
        return df
    
    def generate_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.
        
        Returns:
            List of tuples with (train_start, train_end, test_start, test_end)
        """
        windows = []
        
        current_date = self.config.start_date
        while current_date + timedelta(days=self.config.train_window_days + self.config.test_window_days) <= self.config.end_date:
            # Define window
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_window_days)
            
            # Add window
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            current_date += timedelta(days=self.config.step_size_days)
        
        return windows
    
    def train_model(self, train_data: Dict[str, pd.DataFrame]) -> StackedEnsemble:
        """
        Train model on training data.
        
        Args:
            train_data: Dictionary of DataFrames with training data by symbol
            
        Returns:
            Trained model
        """
        # Initialize model
        model = StackedEnsemble()
        
        # Collect training features and labels
        features = {}
        labels = []
        
        # Process each symbol
        for symbol, df in train_data.items():
            # Extract features
            symbol_features = self._extract_features(df, symbol)
            
            # Extract labels
            symbol_labels = self._extract_labels(df)
            
            # Add to training data
            features[symbol] = symbol_features
            labels.extend(symbol_labels)
        
        # Convert labels to numpy array
        labels = np.array(labels)
        
        # Train model
        model.train(features, labels)
        
        return model
    
    def _extract_features(self, df: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        """
        Extract features from data for model training.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol
            
        Returns:
            Dictionary with feature arrays
        """
        # This is a simplified feature extraction example
        # In a real implementation, you would extract micro-structure features,
        # option chain data, market indicators, etc.
        
        # Calculate basic price features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = df["range"] / df["close"]
        
        # Calculate technical indicators
        df["rsi_14"] = self._calculate_rsi(df["close"], window=14)
        
        # Create tabular features
        tabular_features = df[["returns", "log_returns", "range_pct", "rsi_14"]].dropna().values
        
        # Create sequence features (last 20 days)
        sequence_length = 20
        sequences = []
        for i in range(sequence_length, len(df)):
            seq = df.iloc[i-sequence_length:i][["returns", "log_returns", "range_pct"]].values
            sequences.append(seq)
        
        sequence_features = np.array(sequences) if sequences else np.empty((0, sequence_length, 3))
        
        return {
            "tabular": tabular_features,
            "sequence": sequence_features
        }
    
    def _extract_labels(self, df: pd.DataFrame) -> List[float]:
        """
        Extract labels from data for model training.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of labels (1 for positive edge, 0 for negative)
        """
        # This is a simplified label extraction example
        # In a real implementation, you would define edge based on
        # specific criteria like expected value, statistical edge, etc.
        
        # Calculate future returns (next 5 days)
        df["future_return_5d"] = df["close"].shift(-5) / df["close"] - 1
        
        # Define edge: positive if future return > 1%
        edge = (df["future_return_5d"] > 0.01).astype(int)
        
        # Remove NaNs
        edge = edge.dropna().tolist()
        
        return edge
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def run_backtest(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> BacktestMetrics:
        """
        Run walk-forward backtest.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Backtest metrics
        """
        # Generate windows
        windows = self.generate_windows()
        
        if not windows:
            raise ValueError("No valid windows generated. Check date range and window parameters.")
        
        # Report total number of windows
        logger.info(f"Running walk-forward backtest with {len(windows)} windows")
        
        # Run backtest for each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Update progress
            progress_pct = (i / len(windows)) * 100
            window_desc = f"Window {i+1}/{len(windows)}: {train_start.date()} to {test_end.date()}"
            
            if progress_callback:
                progress_callback(progress_pct, window_desc)
            
            logger.info(f"Processing {window_desc}")
            
            # Run window
            window_metrics = self._run_window(train_start, train_end, test_start, test_end)
            
            # Store window metrics
            self.window_metrics.append(window_metrics)
            
        # Calculate overall metrics
        self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        # Plot results if requested
        if self.config.plot_results:
            self._plot_results()
            
        return self.metrics
    
    def _run_window(self, train_start: datetime, train_end: datetime, 
                    test_start: datetime, test_end: datetime) -> BacktestMetrics:
        """
        Run a single window of the backtest.
        
        Args:
            train_start: Start of training period
            train_end: End of training period
            test_start: Start of testing period
            test_end: End of testing period
            
        Returns:
            Metrics for this window
        """
        # Load training data
        train_data = {}
        for symbol in self.config.symbols:
            df = self.load_data(symbol, train_start, train_end)
            train_data[symbol] = df
        
        # Train model if configured to do so
        if self.config.retrain_model:
            self.model = self.train_model(train_data)
            
            # Save model if path is specified
            if self.config.model_path:
                model_filename = f"model_{train_end.strftime('%Y%m%d')}.joblib"
                model_path = os.path.join(self.config.model_path, model_filename)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save(model_path)
                logger.info(f"Saved model to {model_path}")
        
        # Load test data
        test_data = {}
        for symbol in self.config.symbols:
            df = self.load_data(symbol, test_start, test_end)
            test_data[symbol] = df
        
        # Run test data through model and simulate trading
        window_trades = self._simulate_trading(test_data)
        
        # Calculate metrics for this window
        window_metrics = self._calculate_window_metrics(window_trades, test_start, test_end)
        
        return window_metrics
    
    def _find_opportunities(self, market_data: Dict[str, pd.DataFrame], 
                            trade_date: datetime) -> List[Tuple[str, str, float, float, str, float, str, float]]:
        """
        Find trading opportunities on a specific date.
        
        Args:
            market_data: Dictionary of DataFrames with market data by symbol
            trade_date: Date to find opportunities for
            
        Returns:
            List of tuples with (symbol, contract_id, edge_prob, price, option_type, strike, expiration, vol)
        """
        opportunities = []
        
        for symbol, df in market_data.items():
            # Filter to specific date
            day_data = df[df["date"].dt.date == trade_date.date()]
            
            if day_data.empty:
                continue
            
            # Extract features for prediction
            features = self._extract_features(day_data, symbol)
            
            # Get model prediction
            if self.model:
                edge_probabilities = self.model.predict_proba(features)
                
                # For demonstration - in reality, would process options chain data
                # and identify specific contracts with edge
                if len(edge_probabilities) > 0 and edge_probabilities[0] > self.config.min_edge_probability:
                    # Simplified - would use actual options data
                    current_price = day_data["close"].iloc[-1]
                    
                    # Create mock call option opportunity
                    call_contract_id = f"{symbol}_C_{trade_date.strftime('%Y%m%d')}"
                    call_strike = current_price * 1.05  # 5% OTM
                    call_price = current_price * 0.03  # Simplified pricing
                    call_expiration = (trade_date + timedelta(days=30)).strftime("%Y-%m-%d")
                    call_iv = 0.3  # Simplified IV
                    
                    # Create mock put option opportunity
                    put_contract_id = f"{symbol}_P_{trade_date.strftime('%Y%m%d')}"
                    put_strike = current_price * 0.95  # 5% OTM
                    put_price = current_price * 0.03  # Simplified pricing
                    put_expiration = (trade_date + timedelta(days=30)).strftime("%Y-%m-%d")
                    put_iv = 0.3  # Simplified IV
                    
                    # Add opportunities
                    opportunities.append((symbol, call_contract_id, edge_probabilities[0], 
                                        call_price, "CALL", call_strike, call_expiration, call_iv))
                    opportunities.append((symbol, put_contract_id, edge_probabilities[0], 
                                        put_price, "PUT", put_strike, put_expiration, put_iv))
        
        return opportunities
    
    def _simulate_trading(self, test_data: Dict[str, pd.DataFrame]) -> List[BacktestTrade]:
        """
        Simulate trading on test data.
        
        Args:
            test_data: Dictionary of DataFrames with test data by symbol
            
        Returns:
            List of trades
        """
        window_trades = []
        
        # Process each day in chronological order
        all_dates = set()
        for df in test_data.values():
            all_dates.update(df["date"].dt.date.unique())
        
        all_dates = sorted(all_dates)
        
        # Simulate each trading day
        for trade_date in all_dates:
            # Convert date to datetime for filtering
            trade_datetime = datetime.combine(trade_date, datetime.min.time())
            
            # Find trading opportunities for this day
            opportunities = self._find_opportunities(test_data, trade_datetime)
            
            # Sort opportunities by edge probability
            opportunities.sort(key=lambda x: x[2], reverse=True)
            
            # Execute trades
            for symbol, contract_id, edge_prob, entry_price, option_type, strike, expiration, vol in opportunities:
                # Skip if edge probability is below threshold
                if edge_prob < self.config.min_edge_probability:
                    continue
                
                # Calculate position size
                if edge_prob > 0:
                    # Calculate Kelly position size
                    risk_reward_ratio = 2.0  # Simplified - in reality would calculate based on options pricing
                    position_size = self.position_sizer.calculate_position_size(
                        edge_probability=edge_prob,
                        risk_reward_ratio=risk_reward_ratio,
                        current_price=entry_price
                    )
                    
                    # Convert position size to quantity
                    max_dollars = self.current_capital * self.config.max_position_pct
                    quantity = min(int(position_size / entry_price), int(max_dollars / entry_price))
                    
                    # Skip if quantity is too small
                    if quantity <= 0:
                        continue
                    
                    # Check Greeks risk limits
                    adjusted_quantity = self.greeks_cap.adjust_quantity(
                        symbol=symbol,
                        contract_id=contract_id,
                        option_type=option_type,
                        strike=strike,
                        expiration=expiration,
                        quantity=quantity,
                        price=entry_price,
                        implied_vol=vol
                    )
                    
                    # Skip if quantity was adjusted to zero
                    if adjusted_quantity <= 0:
                        logger.info(f"Trade skipped due to Greeks limits: {symbol} {option_type} {strike}")
                        continue
                    
                    # Calculate slippage
                    slippage = self.config.slippage_model.calculate_slippage(
                        price=entry_price,
                        size=adjusted_quantity,
                        volatility=vol,
                        avg_volume=1000  # Simplified - would use actual volume in real implementation
                    )
                    
                    # Adjust entry price for slippage (always against us)
                    adjusted_entry_price = entry_price * (1 + (0.01 * slippage))
                    
                    # Calculate commission
                    commission = adjusted_quantity * self.config.commission_per_contract
                    
                    # Create trade
                    trade = BacktestTrade(
                        symbol=symbol,
                        contract_id=contract_id,
                        option_type=option_type,
                        strike=strike,
                        expiration=expiration,
                        entry_time=trade_datetime,
                        entry_price=adjusted_entry_price,
                        quantity=adjusted_quantity,
                        direction="long",
                        edge_probability=edge_prob,
                        slippage=slippage,
                        commission=commission,
                        iv_entry=vol
                    )
                    
                    # Add trade to list
                    window_trades.append(trade)
                    
                    # Update portfolio (simplified)
                    self.current_capital -= (adjusted_entry_price * adjusted_quantity + commission)
            
            # Close trades as needed (simplified - would have more complex exit logic in reality)
            self._close_trades(window_trades, trade_datetime, test_data)
            
            # Update daily metrics
            self._update_daily_metrics(trade_datetime)
        
        return window_trades
    
    def _close_trades(self, trades: List[BacktestTrade], current_date: datetime, 
                      market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Close trades based on exit criteria.
        
        Args:
            trades: List of trades
            current_date: Current date
            market_data: Dictionary of DataFrames with market data by symbol
        """
        for trade in trades:
            # Skip if trade is already closed
            if trade.exit_time is not None:
                continue
            
            # Skip if trade was opened today
            if trade.entry_time.date() == current_date.date():
                continue
            
            # Get market data for symbol
            if trade.symbol not in market_data:
                continue
            
            df = market_data[trade.symbol]
            
            # Filter to specific date
            day_data = df[df["date"].dt.date == current_date.date()]
            
            if day_data.empty:
                continue
            
            # Simplified exit logic - in reality, would have more complex criteria
            days_held = (current_date - trade.entry_time).days
            
            # Close if held for 5 days or more
            if days_held >= 5:
                # Simplified exit price - in reality, would use options pricing model
                exit_price = trade.entry_price * (1 + (0.05 * np.random.randn()))  # Random P&L
                exit_price = max(0.01, exit_price)  # Ensure positive price
                
                # Adjust for slippage
                slippage = self.config.slippage_model.calculate_slippage(
                    price=exit_price,
                    size=trade.quantity,
                    volatility=trade.iv_entry,
                    avg_volume=1000  # Simplified
                )
                
                # Slippage always works against us
                adjusted_exit_price = exit_price * (1 - (0.01 * slippage))
                
                # Calculate P&L
                trade.exit_time = current_date
                trade.exit_price = adjusted_exit_price
                trade.pnl = (adjusted_exit_price - trade.entry_price) * trade.quantity - trade.commission
                trade.pnl_pct = (adjusted_exit_price / trade.entry_price - 1) * 100
                
                # Calculate trade duration in minutes (simplified)
                trade.trade_duration = days_held * 24 * 60
                
                # Update portfolio
                self.current_capital += (adjusted_exit_price * trade.quantity)
                
                logger.debug(f"Closed trade: {trade.symbol} {trade.option_type} {trade.strike}, P&L: {trade.pnl:.2f}")
    
    def _update_daily_metrics(self, current_date: datetime) -> None:
        """
        Update daily metrics for equity curve.
        
        Args:
            current_date: Current date
        """
        # Add date
        self.dates.append(current_date)
        
        # Add equity
        self.daily_equity.append(self.current_capital)
        
        # Calculate daily return
        if len(self.daily_equity) > 1:
            daily_return = self.daily_equity[-1] / self.daily_equity[-2] - 1
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)
    
    def _calculate_window_metrics(self, trades: List[BacktestTrade], 
                                  start_date: datetime, end_date: datetime) -> BacktestMetrics:
        """
        Calculate metrics for a window.
        
        Args:
            trades: List of trades
            start_date: Start date of window
            end_date: End date of window
            
        Returns:
            Metrics for window
        """
        metrics = BacktestMetrics()
        
        # Filter trades to those closed in this window
        window_trades = [t for t in trades if t.exit_time is not None and 
                        start_date <= t.exit_time <= end_date]
        
        if not window_trades:
            return metrics
        
        # Calculate trade metrics
        metrics.total_trades = len(window_trades)
        metrics.winning_trades = sum(1 for t in window_trades if t.pnl > 0)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # P&L metrics
        if metrics.winning_trades > 0:
            metrics.avg_win = sum(t.pnl for t in window_trades if t.pnl > 0) / metrics.winning_trades
            metrics.largest_win = max(t.pnl for t in window_trades if t.pnl > 0)
        
        if metrics.losing_trades > 0:
            metrics.avg_loss = sum(t.pnl for t in window_trades if t.pnl <= 0) / metrics.losing_trades
            metrics.largest_loss = min(t.pnl for t in window_trades if t.pnl <= 0)
        
        # Profit factor
        total_gain = sum(t.pnl for t in window_trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in window_trades if t.pnl <= 0))
        
        if total_loss > 0:
            metrics.profit_factor = total_gain / total_loss
        else:
            metrics.profit_factor = float('inf') if total_gain > 0 else 0
        
        # Position sizing
        metrics.avg_position_size = sum(t.entry_price * t.quantity for t in window_trades) / metrics.total_trades
        metrics.max_position_size = max(t.entry_price * t.quantity for t in window_trades)
        
        # Duration
        metrics.avg_trade_duration = sum(t.trade_duration for t in window_trades) / metrics.total_trades
        
        # Returns
        window_return = sum(t.pnl for t in window_trades) / self.config.initial_capital
        metrics.total_return = window_return
        
        days = (end_date - start_date).days
        if days > 0:
            metrics.annualized_return = (1 + window_return) ** (365 / days) - 1
        
        # Simplified drawdown calculation
        cumulative_pnl = 0
        peak = 0
        drawdown = 0
        
        for trade in sorted(window_trades, key=lambda t: t.exit_time):
            cumulative_pnl += trade.pnl
            peak = max(peak, cumulative_pnl)
            drawdown = min(drawdown, cumulative_pnl - peak)
        
        metrics.max_drawdown = abs(drawdown) / self.config.initial_capital if drawdown < 0 else 0
        
        return metrics
    
    def _calculate_metrics(self) -> None:
        """Calculate overall backtest metrics."""
        # Get all closed trades
        closed_trades = [t for t in self.trades if t.exit_time is not None]
        
        if not closed_trades:
            logger.warning("No closed trades to calculate metrics")
            return
        
        # Trade metrics
        self.metrics.total_trades = len(closed_trades)
        self.metrics.winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        self.metrics.losing_trades = self.metrics.total_trades - self.metrics.winning_trades
        
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # P&L metrics
        if self.metrics.winning_trades > 0:
            self.metrics.avg_win = sum(t.pnl for t in closed_trades if t.pnl > 0) / self.metrics.winning_trades
            self.metrics.largest_win = max(t.pnl for t in closed_trades if t.pnl > 0)
        
        if self.metrics.losing_trades > 0:
            self.metrics.avg_loss = sum(t.pnl for t in closed_trades if t.pnl <= 0) / self.metrics.losing_trades
            self.metrics.largest_loss = min(t.pnl for t in closed_trades if t.pnl <= 0)
        
        # Profit factor
        total_gain = sum(t.pnl for t in closed_trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in closed_trades if t.pnl <= 0))
        
        if total_loss > 0:
            self.metrics.profit_factor = total_gain / total_loss
        else:
            self.metrics.profit_factor = float('inf') if total_gain > 0 else 0
        
        # Position sizing
        self.metrics.avg_position_size = sum(t.entry_price * t.quantity for t in closed_trades) / self.metrics.total_trades
        self.metrics.max_position_size = max(t.entry_price * t.quantity for t in closed_trades)
        
        # Duration
        self.metrics.avg_trade_duration = sum(t.trade_duration for t in closed_trades) / self.metrics.total_trades
        
        # Equity and return metrics
        if self.daily_equity:
            start_equity = self.config.initial_capital
            end_equity = self.daily_equity[-1]
            
            self.metrics.total_return = (end_equity / start_equity) - 1
            
            days = (self.config.end_date - self.config.start_date).days
            if days > 0:
                self.metrics.annualized_return = (1 + self.metrics.total_return) ** (365 / days) - 1
            
            # Calculate volatility and Sharpe ratio
            if len(self.daily_returns) > 1:
                daily_returns_array = np.array(self.daily_returns)
                self.metrics.volatility = np.std(daily_returns_array) * np.sqrt(252)  # Annualized
                
                # Risk-free rate (simplified)
                risk_free_rate = 0.02  # 2% annual
                
                # Sharpe ratio
                excess_return = self.metrics.annualized_return - risk_free_rate
                self.metrics.sharpe_ratio = excess_return / self.metrics.volatility if self.metrics.volatility > 0 else 0
                
                # Sortino ratio (downside deviation)
                negative_returns = daily_returns_array[daily_returns_array < 0]
                if len(negative_returns) > 0:
                    self.metrics.downside_deviation = np.std(negative_returns) * np.sqrt(252)
                    self.metrics.sortino_ratio = excess_return / self.metrics.downside_deviation if self.metrics.downside_deviation > 0 else 0
            
            # Calculate drawdown
            max_dd, dd_duration = self._calculate_drawdown(self.daily_equity)
            self.metrics.max_drawdown = max_dd
            self.metrics.drawdown_duration_days = dd_duration
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Tuple of (max_drawdown_pct, max_duration_days)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0
        
        # Calculate drawdown series
        peak = equity_curve[0]
        drawdown_series = []
        
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = (equity - peak) / peak
            drawdown_series.append(drawdown)
        
        # Find maximum drawdown
        max_dd = min(drawdown_series)
        
        # Calculate drawdown duration
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        
        for dd in drawdown_series:
            if dd < 0:
                in_drawdown = True
                current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0
                    in_drawdown = False
        
        # Check if still in drawdown at end
        if in_drawdown:
            max_duration = max(max_duration, current_duration)
        
        return abs(max_dd), max_duration
    
    def _save_results(self) -> None:
        """Save backtest results to disk."""
        if not self.config.output_path:
            return
        
        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_path, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        # Save equity curve
        equity_file = os.path.join(self.config.output_path, "equity_curve.csv")
        equity_df = pd.DataFrame({
            "date": self.dates,
            "equity": self.daily_equity,
            "return": self.daily_returns
        })
        equity_df.to_csv(equity_file, index=False)
        
        # Save trades if requested
        if self.config.save_trades and self.trades:
            trades_file = os.path.join(self.config.output_path, "trades.csv")
            trades_data = [t.to_dict() for t in self.trades if t.exit_time is not None]
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"Saved backtest results to {self.config.output_path}")
    
    def _plot_results(self) -> None:
        """Plot backtest results."""
        if not (self.dates and self.daily_equity):
            logger.warning("No data to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.dates, self.daily_equity)
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        
        # Plot drawdown
        if len(self.daily_equity) > 1:
            plt.subplot(2, 1, 2)
            peak = pd.Series(self.daily_equity).cummax()
            drawdown = pd.Series(self.daily_equity) / peak - 1
            plt.fill_between(self.dates, 0, drawdown, color="red", alpha=0.3)
            plt.title("Drawdown")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_path, "equity_curve.png"))
        plt.close()
        
        # Plot trade distribution
        closed_trades = [t for t in self.trades if t.exit_time is not None]
        if closed_trades:
            plt.figure(figsize=(12, 8))
            
            # Plot P&L distribution
            pnl_values = [t.pnl for t in closed_trades]
            plt.subplot(2, 1, 1)
            sns.histplot(pnl_values, kde=True)
            plt.title("P&L Distribution")
            plt.xlabel("P&L ($)")
            plt.ylabel("Frequency")
            plt.grid(True)
            
            # Plot P&L by symbol
            plt.subplot(2, 1, 2)
            symbol_pnl = {}
            for t in closed_trades:
                if t.symbol not in symbol_pnl:
                    symbol_pnl[t.symbol] = []
                symbol_pnl[t.symbol].append(t.pnl)
            
            symbols = list(symbol_pnl.keys())
            avg_pnl = [np.mean(symbol_pnl[s]) for s in symbols]
            
            plt.bar(symbols, avg_pnl)
            plt.title("Average P&L by Symbol")
            plt.xlabel("Symbol")
            plt.ylabel("Average P&L ($)")
            plt.grid(True)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_path, "trade_analysis.png"))
            plt.close()
        
        logger.info("Generated backtest plots")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"],
        data_path="data/market_data",
        model_path="models",
        output_path="backtest_results/walkforward_test",
        initial_capital=100000.0,
        kelly_fraction=0.25,
        train_window_days=60,
        test_window_days=20,
        step_size_days=20
    )
    
    # Create and run backtest
    backtest = WalkForwardBacktest(config)
    metrics = backtest.run_backtest()
    
    # Print metrics
    print("=== BACKTEST RESULTS ===")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Total Trades: {metrics.total_trades}") 