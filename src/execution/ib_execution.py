"""
Interactive Brokers execution interface.

Implements the OrderRouter interface for Interactive Brokers,
providing order submission, cancellation, and tracking capabilities.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set

from ib_insync import IB, Contract, Order as IBOrder, Option as IBOption
from ib_insync import Stock, OrderStatus as IBOrderStatus, Trade as IBTrade

from .order_manager import OrderRouter, Order, OrderStatus, OrderType, Execution, OrderLeg
from ..data.market_data import OptionRight

# Configure logging
logger = logging.getLogger(__name__)


class IBExecutionInterface(OrderRouter):
    """
    Interactive Brokers implementation of the OrderRouter interface.
    
    Handles order submission and tracking for Interactive Brokers.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, 
                client_id: int = 1, account: Optional[str] = None,
                timeout: int = 20):
        """
        Initialize IB execution interface.
        
        Args:
            host: IB Gateway/TWS host address
            port: IB Gateway/TWS port
            client_id: Client ID for IB API connection
            account: IB account to use (None for default)
            timeout: Connection timeout in seconds
        """
        self._host = host
        self._port = port
        self._client_id = client_id
        self._account = account
        self._timeout = timeout
        
        # Initialize IB connection
        self._ib = IB()
        
        # Map of our order IDs to IB trades
        self._order_map: Dict[str, IBTrade] = {}
        
        # Map of IB order IDs to our order IDs
        self._ib_order_map: Dict[int, str] = {}
        
        # Set of order IDs we've already processed executions for
        self._processed_executions: Dict[str, Set[str]] = {}
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                account=self._account,
                readonly=False,
                timeout=self._timeout
            )
            
            # Set up event handlers
            self._ib.orderStatusEvent += self._on_order_status
            self._ib.execDetailsEvent += self._on_execution
            
            logger.info(f"Connected to Interactive Brokers at {self._host}:{self._port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Interactive Brokers: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from Interactive Brokers")
    
    def _on_order_status(self, trade: IBTrade) -> None:
        """
        Handle order status updates from IB.
        
        Args:
            trade: Updated IB trade
        """
        # Map IB permId to our order ID if we haven't already
        perm_id = trade.order.permId
        if perm_id and perm_id not in self._ib_order_map:
            # Try to find our order ID from the trade
            for our_id, ib_trade in self._order_map.items():
                if ib_trade.order.permId == perm_id:
                    self._ib_order_map[perm_id] = our_id
                    break
        
        # Log status change
        logger.debug(f"IB order status: {trade.order.orderId} -> {trade.orderStatus.status}")
    
    def _on_execution(self, trade: IBTrade, fill) -> None:
        """
        Handle execution details from IB.
        
        Args:
            trade: IB trade that was executed
            fill: Fill details
        """
        # Find our order ID
        our_order_id = None
        for order_id, ib_trade in self._order_map.items():
            if ib_trade.order.permId == trade.order.permId:
                our_order_id = order_id
                break
        
        if our_order_id:
            # Log execution
            logger.info(f"Execution for order {our_order_id}: {fill.execution.shares} @ {fill.execution.price}")
            
            # Track that we've processed this execution
            exec_id = fill.execution.execId
            if our_order_id not in self._processed_executions:
                self._processed_executions[our_order_id] = set()
            
            self._processed_executions[our_order_id].add(exec_id)
    
    def _create_ib_contract(self, leg: OrderLeg) -> Contract:
        """
        Create an IB contract from an order leg.
        
        Args:
            leg: Order leg
            
        Returns:
            IB contract
        """
        if leg.expiration:
            # Option contract
            right = "C" if leg.right == OptionRight.CALL else "P"
            
            return IBOption(
                symbol=leg.symbol,
                lastTradeDateOrContractMonth=leg.expiration.strftime("%Y%m%d"),
                strike=leg.strike,
                right=right,
                exchange=leg.exchange,
                currency="USD"
            )
        else:
            # Stock contract
            return Stock(
                symbol=leg.symbol,
                exchange=leg.exchange,
                currency="USD"
            )
    
    def _create_ib_order(self, leg: OrderLeg) -> IBOrder:
        """
        Create an IB order from an order leg.
        
        Args:
            leg: Order leg
            
        Returns:
            IB order
        """
        # Map order type
        if leg.order_type == OrderType.MARKET:
            ib_order_type = "MKT"
        elif leg.order_type == OrderType.LIMIT:
            ib_order_type = "LMT"
        elif leg.order_type == OrderType.STOP:
            ib_order_type = "STP"
        elif leg.order_type == OrderType.STOP_LIMIT:
            ib_order_type = "STP LMT"
        else:
            ib_order_type = "MKT"  # Default to market
        
        # Create IB order
        order = IBOrder(
            action=leg.action,
            totalQuantity=leg.quantity,
            orderType=ib_order_type
        )
        
        # Set prices if needed
        if leg.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            order.lmtPrice = leg.limit_price
        
        if leg.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            order.auxPrice = leg.stop_price
        
        return order
    
    def send_order(self, order: Order) -> Tuple[bool, str]:
        """
        Send an order to Interactive Brokers.
        
        Args:
            order: Order to send
            
        Returns:
            Tuple of (success, message)
        """
        if not self._ib.isConnected():
            try:
                if not self.connect():
                    return False, "Failed to connect to Interactive Brokers"
            except Exception as e:
                return False, f"Failed to connect to Interactive Brokers: {e}"
        
        try:
            # Handle different order types
            if len(order.legs) == 1:
                # Single-leg order
                leg = order.legs[0]
                
                # Create contract and order
                contract = self._create_ib_contract(leg)
                ib_order = self._create_ib_order(leg)
                
                # Send order
                trade = self._ib.placeOrder(contract, ib_order)
                
                # Store mapping
                self._order_map[order.id] = trade
                
                # Map IB permId to our order ID if available
                if trade.order.permId:
                    self._ib_order_map[trade.order.permId] = order.id
                
                logger.info(f"Sent single-leg order {order.id} to IB: {leg.action} {leg.quantity} {leg.symbol}")
                
                return True, f"Order sent: {trade.order.orderId}"
                
            else:
                # Multi-leg order (use combo order)
                # Only support for simple spread types for now
                
                # For a basic two-leg vertical spread
                if len(order.legs) == 2:
                    # Extract legs
                    leg1, leg2 = order.legs
                    
                    # Create contracts for each leg
                    contract1 = self._create_ib_contract(leg1)
                    contract2 = self._create_ib_contract(leg2)
                    
                    # Create a combo/spread contract
                    from ib_insync import ComboLeg, Contract
                    
                    combo_legs = [
                        ComboLeg(
                            conId=self._ib.qualifyContracts(contract1)[0].conId,
                            ratio=1,
                            action=leg1.action,
                            exchange=leg1.exchange
                        ),
                        ComboLeg(
                            conId=self._ib.qualifyContracts(contract2)[0].conId,
                            ratio=1,
                            action=leg2.action,
                            exchange=leg2.exchange
                        )
                    ]
                    
                    combo = Contract(
                        symbol=leg1.symbol,
                        secType="BAG",
                        exchange=leg1.exchange,
                        currency="USD",
                        comboLegs=combo_legs
                    )
                    
                    # Create order
                    # Use a net debit/credit limit order
                    ib_order = IBOrder(
                        action="BUY",  # For spreads, we're always "buying" the combo
                        totalQuantity=1,  # One unit of the spread
                        orderType="NET"  # Net debit/credit order
                    )
                    
                    # Calculate net price
                    if leg1.limit_price and leg2.limit_price:
                        # Calculate net price based on leg actions
                        if leg1.action == "BUY" and leg2.action == "SELL":
                            net_price = leg1.limit_price - leg2.limit_price
                        elif leg1.action == "SELL" and leg2.action == "BUY":
                            net_price = leg2.limit_price - leg1.limit_price
                        else:
                            # If both legs are the same direction, sum prices
                            net_price = leg1.limit_price + leg2.limit_price
                        
                        ib_order.lmtPrice = net_price
                    
                    # Send order
                    trade = self._ib.placeOrder(combo, ib_order)
                    
                    # Store mapping
                    self._order_map[order.id] = trade
                    
                    # Map IB permId to our order ID if available
                    if trade.order.permId:
                        self._ib_order_map[trade.order.permId] = order.id
                    
                    logger.info(f"Sent combo order {order.id} to IB")
                    
                    return True, f"Order sent: {trade.order.orderId}"
                
                else:
                    # For complex multi-leg orders, place individual legs
                    trades = []
                    
                    for i, leg in enumerate(order.legs):
                        # Create contract and order
                        contract = self._create_ib_contract(leg)
                        ib_order = self._create_ib_order(leg)
                        
                        # For linked orders, add order group ID
                        ib_order.faGroup = f"Group_{order.id}"
                        ib_order.faMethod = "PctChange"  # All legs filled at same percentage
                        
                        # Send order
                        trade = self._ib.placeOrder(contract, ib_order)
                        trades.append(trade)
                        
                        # Add brief delay between sending legs
                        if i < len(order.legs) - 1:
                            time.sleep(0.1)
                    
                    # Store mapping
                    self._order_map[order.id] = trades[0]  # Store the first leg's trade
                    
                    # Map IB permId to our order ID if available
                    for trade in trades:
                        if trade.order.permId:
                            self._ib_order_map[trade.order.permId] = order.id
                    
                    logger.info(f"Sent {len(trades)}-leg order {order.id} to IB as individual legs")
                    
                    return True, f"Order sent: {trades[0].order.orderId}"
            
        except Exception as e:
            logger.exception(f"Error sending order {order.id} to IB")
            return False, f"Failed to send order: {e}"
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, message)
        """
        if not self._ib.isConnected():
            try:
                if not self.connect():
                    return False, "Failed to connect to Interactive Brokers"
            except Exception as e:
                return False, f"Failed to connect to Interactive Brokers: {e}"
        
        try:
            # Find IB trade for this order
            if order_id not in self._order_map:
                return False, f"Order {order_id} not found"
            
            trade = self._order_map[order_id]
            
            # If this is a list of trades (multi-leg), cancel all
            if isinstance(trade, list):
                for t in trade:
                    self._ib.cancelOrder(t.order)
                
                logger.info(f"Canceled multi-leg order {order_id}")
            else:
                # Cancel single trade
                self._ib.cancelOrder(trade.order)
                
                logger.info(f"Canceled order {order_id}")
            
            return True, "Order canceled"
            
        except Exception as e:
            logger.exception(f"Error canceling order {order_id}")
            return False, f"Failed to cancel order: {e}"
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Current order status, or None if not found
        """
        if not self._ib.isConnected():
            try:
                if not self.connect():
                    return None
            except Exception:
                return None
        
        try:
            # Find IB trade for this order
            if order_id not in self._order_map:
                return None
            
            trade = self._order_map[order_id]
            
            # If this is a list of trades (multi-leg), check all
            if isinstance(trade, list):
                # Get status of all legs
                statuses = [t.orderStatus.status for t in trade]
                
                # If all legs are filled, order is filled
                if all(s == "Filled" for s in statuses):
                    return OrderStatus.FILLED
                
                # If any leg is partially filled, order is partially filled
                elif any(s == "PartiallyFilled" for s in statuses):
                    return OrderStatus.PARTIALLY_FILLED
                
                # If any leg is cancelled, order is cancelled
                elif any(s == "Cancelled" for s in statuses):
                    return OrderStatus.CANCELED
                
                # If any leg is submitted/pending, order is sent
                elif any(s in ["PendingSubmit", "PendingCancel", "PreSubmitted", "Submitted"] for s in statuses):
                    return OrderStatus.SENT
                
                # If any leg is rejected, order is rejected
                elif any(s == "Rejected" for s in statuses):
                    return OrderStatus.REJECTED
                
                # Default to sent
                return OrderStatus.SENT
                
            else:
                # Map IB status to our status
                ib_status = trade.orderStatus.status
                
                if ib_status == "Filled":
                    return OrderStatus.FILLED
                elif ib_status == "PartiallyFilled":
                    return OrderStatus.PARTIALLY_FILLED
                elif ib_status == "Cancelled":
                    return OrderStatus.CANCELED
                elif ib_status in ["PendingSubmit", "PendingCancel", "PreSubmitted", "Submitted"]:
                    return OrderStatus.SENT
                elif ib_status == "Rejected":
                    return OrderStatus.REJECTED
                else:
                    return OrderStatus.SENT
            
        except Exception:
            logger.exception(f"Error getting status for order {order_id}")
            return None
    
    def get_executions(self, order_id: str) -> List[Execution]:
        """
        Get executions for an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            List of executions for the order
        """
        if not self._ib.isConnected():
            try:
                if not self.connect():
                    return []
            except Exception:
                return []
        
        try:
            # Find IB trade for this order
            if order_id not in self._order_map:
                return []
            
            trade = self._order_map[order_id]
            trades_to_check = [trade] if not isinstance(trade, list) else trade
            
            # Get executions for these trades
            executions = []
            
            for t in trades_to_check:
                # For each fill in the trade
                for fill in t.fills:
                    # Skip if we've already processed this execution
                    if (order_id in self._processed_executions and 
                        fill.execution.execId in self._processed_executions[order_id]):
                        continue
                    
                    # Create execution object
                    leg_index = 0  # Default to first leg
                    
                    # Try to match to specific leg for multi-leg orders
                    if isinstance(trade, list) and len(trade) > 1:
                        for i, leg_trade in enumerate(trade):
                            if leg_trade.contract.conId == fill.contract.conId:
                                leg_index = i
                                break
                    
                    execution = Execution(
                        order_id=order_id,
                        leg_index=leg_index,
                        execution_id=fill.execution.execId,
                        time=datetime.fromtimestamp(fill.execution.time),
                        quantity=fill.execution.shares,
                        price=fill.execution.price,
                        exchange=fill.execution.exchange,
                        commission=fill.commissionReport.commission if fill.commissionReport else 0.0
                    )
                    
                    executions.append(execution)
                    
                    # Mark as processed
                    if order_id not in self._processed_executions:
                        self._processed_executions[order_id] = set()
                    
                    self._processed_executions[order_id].add(fill.execution.execId)
            
            return executions
            
        except Exception:
            logger.exception(f"Error getting executions for order {order_id}")
            return [] 