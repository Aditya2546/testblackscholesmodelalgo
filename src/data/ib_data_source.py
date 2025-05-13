"""
Interactive Brokers market data source implementation.

Provides real-time and historical market data from Interactive Brokers
using the ib_insync library with optimized, low-latency data handling.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set

import pandas as pd
import numpy as np
from ib_insync import IB, Contract, Option as IBOption, Stock, BarData, util
from ib_insync import Ticker as IBTicker

from .market_data import (
    MarketDataSource, Asset, Equity, Option, Quote, Trade, MarketDataBar,
    OptionChain, OptionQuote, OptionRight, BarPeriod, AssetType,
    MarketDataException
)

# Configure logging
logger = logging.getLogger(__name__)


class IBDataSource(MarketDataSource):
    """
    Interactive Brokers market data source implementation.
    
    Provides real-time and historical market data from Interactive Brokers
    with optimized, low-latency data processing.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, 
                client_id: int = 1, account: Optional[str] = None,
                read_only: bool = True, timeout: int = 20,
                max_workers: int = 10):
        """
        Initialize the Interactive Brokers data source.
        
        Args:
            host: IB Gateway/TWS host address
            port: IB Gateway/TWS port
            client_id: Client ID for IB API connection
            account: IB account to use (None for default)
            read_only: Whether to use read-only mode
            timeout: Connection timeout in seconds
            max_workers: Maximum number of worker threads
        """
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account = account
        self._read_only = read_only
        self._timeout = timeout
        
        # Initialize IB connection
        self._ib = IB()
        
        # Quote subscriptions: asset symbol -> callback
        self._quote_callbacks: Dict[str, Set[Callable[[Quote], None]]] = {}
        
        # Trade subscriptions: asset symbol -> callback
        self._trade_callbacks: Dict[str, Set[Callable[[Trade], None]]] = {}
        
        # Bar subscriptions: (asset symbol, period) -> callback
        self._bar_callbacks: Dict[Tuple[str, BarPeriod], Set[Callable[[MarketDataBar], None]]] = {}
        
        # Option chain subscriptions: underlying symbol -> callback
        self._option_chain_callbacks: Dict[str, Set[Callable[[OptionChain], None]]] = {}
        
        # Keep track of subscribed contracts
        self._subscribed_contracts: Dict[str, Contract] = {}
        
        # Last quotes/trades for each asset
        self._last_quotes: Dict[str, Quote] = {}
        self._last_trades: Dict[str, Trade] = {}
        
        # Cache for option chains
        self._option_chains: Dict[str, OptionChain] = {}
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Event loop for async operations
        self._loop = None
        self._thread = None
        
    def connect(self) -> None:
        """
        Connect to Interactive Brokers.
        
        Raises:
            MarketDataException: If connection fails
        """
        try:
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                account=self._account,
                readonly=self._read_only,
                timeout=self._timeout
            )
            
            # Set up event handlers
            self._ib.pendingTickersEvent += self._on_tickers_update
            
            # Start a separate thread for the event loop
            self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self._thread.start()
            
            logger.info(f"Connected to Interactive Brokers at {self._host}:{self._port}")
        except Exception as e:
            logger.error(f"Failed to connect to Interactive Brokers: {e}")
            raise MarketDataException(f"Failed to connect to Interactive Brokers: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self._ib.isConnected():
            # Clean up subscriptions
            for contract in self._subscribed_contracts.values():
                self._ib.cancelMktData(contract)
            
            # Disconnect
            self._ib.disconnect()
            logger.info("Disconnected from Interactive Brokers")
    
    def _run_event_loop(self) -> None:
        """Run the event loop in a separate thread."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        self._loop = asyncio.get_event_loop()
        self._ib.run()
    
    def _on_tickers_update(self, tickers: List[IBTicker]) -> None:
        """
        Handle ticker updates from IB.
        
        Args:
            tickers: List of updated tickers
        """
        for ticker in tickers:
            contract = ticker.contract
            symbol = contract.symbol
            
            # Process equity/index tickers
            if isinstance(contract, Stock):
                # Create quote if bid/ask is available
                if hasattr(ticker, 'bid') and hasattr(ticker, 'ask') and ticker.bid and ticker.ask:
                    quote = Quote(
                        timestamp=datetime.now(),
                        bid_price=ticker.bid,
                        ask_price=ticker.ask,
                        bid_size=ticker.bidSize if hasattr(ticker, 'bidSize') else 0,
                        ask_size=ticker.askSize if hasattr(ticker, 'askSize') else 0,
                        exchange=contract.exchange
                    )
                    self._last_quotes[symbol] = quote
                    
                    # Notify quote subscribers
                    if symbol in self._quote_callbacks:
                        for callback in self._quote_callbacks[symbol]:
                            callback(quote)
                
                # Create trade if last price is available
                if hasattr(ticker, 'last') and ticker.last:
                    trade = Trade(
                        timestamp=datetime.now(),
                        price=ticker.last,
                        size=ticker.lastSize if hasattr(ticker, 'lastSize') else 0,
                        exchange=contract.exchange
                    )
                    self._last_trades[symbol] = trade
                    
                    # Notify trade subscribers
                    if symbol in self._trade_callbacks:
                        for callback in self._trade_callbacks[symbol]:
                            callback(trade)
            
            # Process option tickers
            elif isinstance(contract, IBOption):
                underlying = contract.symbol
                right = OptionRight.CALL if contract.right == 'C' else OptionRight.PUT
                expiration = datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d')
                
                # Only process if we have basic market data
                if hasattr(ticker, 'bid') and hasattr(ticker, 'ask') and ticker.bid and ticker.ask:
                    # Create option quote
                    option_quote = OptionQuote(
                        timestamp=datetime.now(),
                        bid_price=ticker.bid,
                        ask_price=ticker.ask,
                        bid_size=ticker.bidSize if hasattr(ticker, 'bidSize') else 0,
                        ask_size=ticker.askSize if hasattr(ticker, 'askSize') else 0,
                        exchange=contract.exchange,
                        underlying_symbol=underlying,
                        expiration=expiration,
                        strike=contract.strike,
                        right=right,
                        last_price=ticker.last if hasattr(ticker, 'last') and ticker.last else 0.0,
                        volume=ticker.volume if hasattr(ticker, 'volume') else 0,
                        open_interest=ticker.openInterest if hasattr(ticker, 'openInterest') else 0
                    )
                    
                    # Update option Greeks if available
                    if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks:
                        greeks = ticker.modelGreeks
                        option_quote.implied_volatility = greeks.impliedVol if hasattr(greeks, 'impliedVol') else 0.0
                        option_quote.delta = greeks.delta if hasattr(greeks, 'delta') else 0.0
                        option_quote.gamma = greeks.gamma if hasattr(greeks, 'gamma') else 0.0
                        option_quote.theta = greeks.theta if hasattr(greeks, 'theta') else 0.0
                        option_quote.vega = greeks.vega if hasattr(greeks, 'vega') else 0.0
                        option_quote.rho = greeks.rho if hasattr(greeks, 'rho') else 0.0
                    
                    # Update option chain
                    if underlying in self._option_chains:
                        chain = self._option_chains[underlying]
                        chain.add_option(option_quote)
                        chain.timestamp = datetime.now()
                        
                        # Notify option chain subscribers
                        if underlying in self._option_chain_callbacks:
                            for callback in self._option_chain_callbacks[underlying]:
                                callback(chain)
    
    def _create_contract(self, asset: Asset) -> Contract:
        """
        Create an IB contract from an asset.
        
        Args:
            asset: Asset to create contract for
            
        Returns:
            IB contract
        
        Raises:
            MarketDataException: If asset type is not supported
        """
        if asset.asset_type == AssetType.EQUITY:
            return Stock(
                symbol=asset.symbol,
                exchange=asset.exchange or "SMART",
                currency="USD"
            )
        elif asset.asset_type == AssetType.OPTION:
            option = asset  # type: Option
            return IBOption(
                symbol=option.underlying_symbol,
                lastTradeDateOrContractMonth=option.expiration.strftime('%Y%m%d'),
                strike=option.strike,
                right='C' if option.right == OptionRight.CALL else 'P',
                exchange=option.exchange or "SMART",
                currency="USD"
            )
        else:
            raise MarketDataException(f"Unsupported asset type: {asset.asset_type}")
    
    def _ensure_contract_subscribed(self, asset: Asset) -> Contract:
        """
        Ensure a contract is subscribed to market data.
        
        Args:
            asset: Asset to subscribe to
            
        Returns:
            Subscribed IB contract
        """
        symbol = asset.symbol
        
        if symbol not in self._subscribed_contracts:
            contract = self._create_contract(asset)
            
            # Request market data
            self._ib.reqMktData(contract, '', False, False)
            self._subscribed_contracts[symbol] = contract
            
            logger.debug(f"Subscribed to market data for {symbol}")
            
            # Add to option chain if it's an option
            if asset.asset_type == AssetType.OPTION:
                option = asset  # type: Option
                underlying = option.underlying_symbol
                
                # Create option chain if it doesn't exist
                if underlying not in self._option_chains:
                    self._option_chains[underlying] = OptionChain(
                        underlying_symbol=underlying,
                        timestamp=datetime.now()
                    )
        
        return self._subscribed_contracts[symbol]
    
    def subscribe_quote(self, asset: Asset, callback: Callable[[Quote], None]) -> None:
        """
        Subscribe to quotes for an asset.
        
        Args:
            asset: Asset to subscribe to
            callback: Callback function to receive quote updates
        """
        symbol = asset.symbol
        
        # Add callback to quote callbacks
        if symbol not in self._quote_callbacks:
            self._quote_callbacks[symbol] = set()
        self._quote_callbacks[symbol].add(callback)
        
        # Ensure contract is subscribed
        self._ensure_contract_subscribed(asset)
    
    def subscribe_trade(self, asset: Asset, callback: Callable[[Trade], None]) -> None:
        """
        Subscribe to trades for an asset.
        
        Args:
            asset: Asset to subscribe to
            callback: Callback function to receive trade updates
        """
        symbol = asset.symbol
        
        # Add callback to trade callbacks
        if symbol not in self._trade_callbacks:
            self._trade_callbacks[symbol] = set()
        self._trade_callbacks[symbol].add(callback)
        
        # Ensure contract is subscribed
        self._ensure_contract_subscribed(asset)
    
    def subscribe_bar(self, asset: Asset, period: BarPeriod, 
                     callback: Callable[[MarketDataBar], None]) -> None:
        """
        Subscribe to bars for an asset.
        
        Args:
            asset: Asset to subscribe to
            period: Bar period
            callback: Callback function to receive bar updates
        """
        symbol = asset.symbol
        key = (symbol, period)
        
        # Add callback to bar callbacks
        if key not in self._bar_callbacks:
            self._bar_callbacks[key] = set()
        self._bar_callbacks[key].add(callback)
        
        # Ensure contract is subscribed
        contract = self._ensure_contract_subscribed(asset)
        
        # Set up realtime bar subscription
        what_to_show = "TRADES"
        
        # Convert BarPeriod to IB bar size
        bar_size_map = {
            BarPeriod.SECOND: "1 secs",
            BarPeriod.MINUTE: "1 min",
            BarPeriod.FIVE_MINUTE: "5 mins",
            BarPeriod.FIFTEEN_MINUTE: "15 mins",
            BarPeriod.THIRTY_MINUTE: "30 mins",
            BarPeriod.HOUR: "1 hour",
            BarPeriod.DAY: "1 day",
            BarPeriod.WEEK: "1 week",
            BarPeriod.MONTH: "1 month"
        }
        
        if period in bar_size_map:
            # Start real-time bars
            self._ib.reqRealTimeBars(
                contract, 
                5,  # 5 second bars (minimum allowed by IB) 
                what_to_show, 
                False,  # Don't use regular trading hours only
                []  # No filter conditions
            )
            logger.debug(f"Subscribed to real-time bars for {symbol} with period {period}")
        else:
            logger.warning(f"Unsupported bar period for real-time bars: {period}")
    
    def subscribe_option_chain(self, symbol: str, 
                               callback: Callable[[OptionChain], None]) -> None:
        """
        Subscribe to an option chain for an underlying symbol.
        
        Args:
            symbol: Underlying symbol
            callback: Callback function to receive option chain updates
        """
        # Add callback to option chain callbacks
        if symbol not in self._option_chain_callbacks:
            self._option_chain_callbacks[symbol] = set()
        self._option_chain_callbacks[symbol].add(callback)
        
        # If we don't have an option chain for this symbol, fetch it
        if symbol not in self._option_chains:
            self._fetch_option_chain(symbol)
        else:
            # Otherwise notify with existing chain
            chain = self._option_chains[symbol]
            callback(chain)
    
    def _fetch_option_chain(self, symbol: str) -> None:
        """
        Fetch the complete option chain for a symbol.
        
        Args:
            symbol: Underlying symbol
        """
        underlying = Stock(symbol=symbol, exchange="SMART", currency="USD")
        
        # Request contract details to get available expirations and strikes
        try:
            # Request contract details for the underlying
            details = self._ib.reqContractDetails(underlying)
            
            if not details:
                logger.warning(f"No contract details found for {symbol}")
                return
            
            # Create option chain
            chain = OptionChain(
                underlying_symbol=symbol,
                timestamp=datetime.now()
            )
            self._option_chains[symbol] = chain
            
            # Get available expirations and strikes
            for detail in details:
                if hasattr(detail, 'summary') and detail.summary:
                    contract = detail.summary
                    
                    # Get available expirations and strikes
                    if hasattr(detail, 'marketName') and detail.marketName:
                        # Request the option chain
                        chains = self._ib.reqSecDefOptParams(
                            contract.symbol,
                            "",  # Exchange (empty for all)
                            contract.secType,
                            contract.conId
                        )
                        
                        for chain_data in chains:
                            for exp in chain_data.expirations:
                                for strike in chain_data.strikes:
                                    # Create call contract
                                    call_contract = IBOption(
                                        symbol=symbol,
                                        lastTradeDateOrContractMonth=exp,
                                        strike=strike,
                                        right='C',
                                        exchange=chain_data.exchange
                                    )
                                    
                                    # Create put contract
                                    put_contract = IBOption(
                                        symbol=symbol,
                                        lastTradeDateOrContractMonth=exp,
                                        strike=strike,
                                        right='P',
                                        exchange=chain_data.exchange
                                    )
                                    
                                    # Request market data for both contracts
                                    self._ib.reqMktData(call_contract, '', False, False)
                                    self._ib.reqMktData(put_contract, '', False, False)
                                    
                                    # Save contracts to subscribed contracts
                                    call_symbol = f"{symbol}_{exp}_C_{strike}"
                                    put_symbol = f"{symbol}_{exp}_P_{strike}"
                                    self._subscribed_contracts[call_symbol] = call_contract
                                    self._subscribed_contracts[put_symbol] = put_contract
            
            logger.info(f"Fetched option chain for {symbol}")
            
            # Notify option chain subscribers
            if symbol in self._option_chain_callbacks:
                for callback in self._option_chain_callbacks[symbol]:
                    callback(chain)
                    
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {e}")
    
    def unsubscribe(self, asset: Asset) -> None:
        """
        Unsubscribe from updates for an asset.
        
        Args:
            asset: Asset to unsubscribe from
        """
        symbol = asset.symbol
        
        # Remove callbacks
        if symbol in self._quote_callbacks:
            del self._quote_callbacks[symbol]
        
        if symbol in self._trade_callbacks:
            del self._trade_callbacks[symbol]
        
        # Remove bar callbacks for this symbol
        bar_keys_to_remove = []
        for key in self._bar_callbacks:
            if key[0] == symbol:
                bar_keys_to_remove.append(key)
        
        for key in bar_keys_to_remove:
            del self._bar_callbacks[key]
        
        # If it's an underlying, remove option chain callbacks
        if symbol in self._option_chain_callbacks:
            del self._option_chain_callbacks[symbol]
        
        # Cancel market data subscription
        if symbol in self._subscribed_contracts:
            contract = self._subscribed_contracts[symbol]
            self._ib.cancelMktData(contract)
            del self._subscribed_contracts[symbol]
        
        logger.debug(f"Unsubscribed from updates for {symbol}")
    
    def get_last_quote(self, asset: Asset) -> Quote:
        """
        Get the last quote for an asset.
        
        Args:
            asset: Asset to get the quote for
            
        Returns:
            Last quote for the asset
            
        Raises:
            MarketDataException: If no quote is available
        """
        symbol = asset.symbol
        
        # Ensure contract is subscribed
        self._ensure_contract_subscribed(asset)
        
        # If we have a last quote, return it
        if symbol in self._last_quotes:
            return self._last_quotes[symbol]
        
        # Otherwise, wait for a quote with timeout
        start_time = datetime.now()
        while datetime.now() - start_time < timedelta(seconds=3):
            # Check if we have a quote now
            if symbol in self._last_quotes:
                return self._last_quotes[symbol]
            
            # Sleep for a short time
            time.sleep(0.01)
        
        # If we still don't have a quote, try to get a ticker
        contract = self._subscribed_contracts[symbol]
        ticker = self._ib.reqTickers(contract)[0]
        
        if hasattr(ticker, 'bid') and hasattr(ticker, 'ask') and ticker.bid and ticker.ask:
            quote = Quote(
                timestamp=datetime.now(),
                bid_price=ticker.bid,
                ask_price=ticker.ask,
                bid_size=ticker.bidSize if hasattr(ticker, 'bidSize') else 0,
                ask_size=ticker.askSize if hasattr(ticker, 'askSize') else 0,
                exchange=contract.exchange
            )
            self._last_quotes[symbol] = quote
            return quote
        
        raise MarketDataException(f"No quote available for {symbol}")
    
    def get_option_chain(self, symbol: str) -> OptionChain:
        """
        Get the complete option chain for an underlying symbol.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            Option chain for the underlying symbol
        """
        # If we already have the chain, return it
        if symbol in self._option_chains:
            return self._option_chains[symbol]
        
        # Otherwise, fetch it
        self._fetch_option_chain(symbol)
        
        # If we still don't have it, create an empty one
        if symbol not in self._option_chains:
            chain = OptionChain(
                underlying_symbol=symbol,
                timestamp=datetime.now()
            )
            self._option_chains[symbol] = chain
        
        return self._option_chains[symbol]
    
    def get_historical_bars(self, asset: Asset, period: BarPeriod, 
                           start_time: datetime, end_time: datetime) -> List[MarketDataBar]:
        """
        Get historical bars for an asset.
        
        Args:
            asset: Asset to get historical data for
            period: Bar period
            start_time: Start time of the historical data
            end_time: End time of the historical data
            
        Returns:
            List of historical bars
        """
        contract = self._create_contract(asset)
        
        # Convert BarPeriod to IB bar size
        bar_size_map = {
            BarPeriod.MINUTE: "1 min",
            BarPeriod.FIVE_MINUTE: "5 mins",
            BarPeriod.FIFTEEN_MINUTE: "15 mins",
            BarPeriod.THIRTY_MINUTE: "30 mins",
            BarPeriod.HOUR: "1 hour",
            BarPeriod.DAY: "1 day",
            BarPeriod.WEEK: "1 week",
            BarPeriod.MONTH: "1 month"
        }
        
        # Handle case where we want tick data
        if period == BarPeriod.TICK:
            # For tick data, we need to use reqHistoricalTicks
            ticks = self._ib.reqHistoricalTicks(
                contract,
                start_time.strftime('%Y%m%d %H:%M:%S'),
                end_time.strftime('%Y%m%d %H:%M:%S'),
                1000,  # Number of ticks to return
                'TRADES',  # What to show
                1,  # Use regular trading hours
                False  # Don't ignore size
            )
            
            # Convert ticks to a list of MarketDataBar objects (each tick becomes a bar)
            bars = []
            for tick in ticks:
                bar = MarketDataBar(
                    timestamp=tick.time if hasattr(tick, 'time') else datetime.now(),
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    volume=tick.size if hasattr(tick, 'size') else 0
                )
                bars.append(bar)
            
            return bars
        
        # Map to IB bar size
        if period not in bar_size_map:
            raise MarketDataException(f"Unsupported bar period: {period}")
        
        bar_size = bar_size_map[period]
        
        # Request historical data
        bars = self._ib.reqHistoricalData(
            contract,
            end_time.strftime('%Y%m%d %H:%M:%S'),
            f"{(end_time - start_time).days} D",  # Duration
            bar_size,  # Bar size
            'TRADES',  # What to show
            1,  # Use regular trading hours
            1,  # Date format (1 = string)
            False,  # Don't keep up to date
            []  # No filter conditions
        )
        
        # Convert IB bars to MarketDataBar objects
        result = []
        for bar in bars:
            market_bar = MarketDataBar(
                timestamp=bar.date if hasattr(bar, 'date') else datetime.now(),
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume if hasattr(bar, 'volume') else 0,
                vwap=bar.average if hasattr(bar, 'average') else None,
                trades=bar.barCount if hasattr(bar, 'barCount') else 0
            )
            result.append(market_bar)
        
        return result
    
    def get_historical_quotes(self, asset: Asset, start_time: datetime, 
                             end_time: datetime) -> List[Quote]:
        """
        Get historical quotes for an asset.
        
        Args:
            asset: Asset to get historical data for
            start_time: Start time of the historical data
            end_time: End time of the historical data
            
        Returns:
            List of historical quotes
        """
        contract = self._create_contract(asset)
        
        # Request historical ticks for quotes
        ticks = self._ib.reqHistoricalTicks(
            contract,
            start_time.strftime('%Y%m%d %H:%M:%S'),
            end_time.strftime('%Y%m%d %H:%M:%S'),
            1000,  # Number of ticks to return
            'BID_ASK',  # What to show
            1,  # Use regular trading hours
            False  # Don't ignore size
        )
        
        # Convert ticks to a list of Quote objects
        quotes = []
        for tick in ticks:
            if hasattr(tick, 'priceBid') and hasattr(tick, 'priceAsk'):
                quote = Quote(
                    timestamp=tick.time if hasattr(tick, 'time') else datetime.now(),
                    bid_price=tick.priceBid,
                    ask_price=tick.priceAsk,
                    bid_size=tick.sizeBid if hasattr(tick, 'sizeBid') else 0,
                    ask_size=tick.sizeAsk if hasattr(tick, 'sizeAsk') else 0,
                    exchange=contract.exchange
                )
                quotes.append(quote)
        
        return quotes 