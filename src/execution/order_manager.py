"""
Order management and execution module.

Handles order creation, validation, routing, and tracking through
to execution with smart order routing capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Set, Any, Callable
import logging
import uuid
import time

from ..data.market_data import Option, OptionRight
from ..strategy.signal import Signal

# Configure logging
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enum."""
    CREATED = "created"
    VALIDATED = "validated"
    SENT = "sent"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"
    ERROR = "error"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force enum."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderLeg:
    """Individual leg of a multi-leg order."""
    symbol: str
    expiration: Optional[datetime]
    strike: float
    right: OptionRight
    action: str  # "BUY" or "SELL"
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    exchange: str = "SMART"
    filled_quantity: int = 0
    filled_price: float = 0.0


@dataclass
class Order:
    """Order class for single or multi-leg orders."""
    id: str
    legs: List[OrderLeg]
    time_in_force: TimeInForce
    status: OrderStatus
    signal_id: Optional[str] = None
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    sent_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    execution_details: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class Execution:
    """Execution details for an order."""
    order_id: str
    leg_index: int
    execution_id: str
    time: datetime
    quantity: int
    price: float
    exchange: str
    commission: float = 0.0


class OrderValidator:
    """
    Validates orders before they are sent to the exchange.
    
    Performs a series of checks to ensure orders are properly formed
    and won't be rejected by the exchange.
    """
    
    @staticmethod
    def validate_order(order: Order) -> Tuple[bool, str]:
        """
        Validate an order.
        
        Args:
            order: Order to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check basic order properties
        if not order.id:
            return False, "Order ID is required"
            
        if not order.legs:
            return False, "Order must have at least one leg"
        
        # Validate each leg
        for i, leg in enumerate(order.legs):
            # Check required fields
            if not leg.symbol:
                return False, f"Leg {i}: Symbol is required"
                
            if not leg.expiration:
                return False, f"Leg {i}: Expiration is required"
                
            if not leg.action:
                return False, f"Leg {i}: Action is required"
                
            if leg.action not in ["BUY", "SELL"]:
                return False, f"Leg {i}: Action must be 'BUY' or 'SELL'"
                
            if leg.quantity <= 0:
                return False, f"Leg {i}: Quantity must be greater than zero"
                
            # Check price constraints based on order type
            if leg.order_type == OrderType.LIMIT and leg.limit_price is None:
                return False, f"Leg {i}: Limit price is required for limit orders"
                
            if leg.order_type == OrderType.STOP and leg.stop_price is None:
                return False, f"Leg {i}: Stop price is required for stop orders"
                
            if leg.order_type == OrderType.STOP_LIMIT:
                if leg.stop_price is None:
                    return False, f"Leg {i}: Stop price is required for stop-limit orders"
                if leg.limit_price is None:
                    return False, f"Leg {i}: Limit price is required for stop-limit orders"
        
        return True, ""


class OrderRouter(ABC):
    """Abstract base class for order routing interfaces."""
    
    @abstractmethod
    def send_order(self, order: Order) -> Tuple[bool, str]:
        """
        Send an order to the exchange.
        
        Args:
            order: Order to send
            
        Returns:
            Tuple of (success, message)
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, message)
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Current order status, or None if not found
        """
        pass
    
    @abstractmethod
    def get_executions(self, order_id: str) -> List[Execution]:
        """
        Get executions for an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            List of executions for the order
        """
        pass


class SmartOrderRouter:
    """
    Smart order router that selects the best execution venue.
    
    Manages multiple order router implementations and selects the
    best one based on performance and availability.
    """
    
    def __init__(self, primary_router: OrderRouter, fallback_router: Optional[OrderRouter] = None,
                 latency_threshold_ms: int = 500):
        """
        Initialize smart order router.
        
        Args:
            primary_router: Primary order router
            fallback_router: Backup order router to use if primary fails
            latency_threshold_ms: Latency threshold to switch to fallback router
        """
        self.primary_router = primary_router
        self.fallback_router = fallback_router
        self.latency_threshold_ms = latency_threshold_ms
        
        self.primary_router_latency_ms = 0
        self.fallback_router_latency_ms = 0
        self.primary_router_errors = 0
        self.fallback_router_errors = 0
        
    def send_order(self, order: Order) -> Tuple[bool, str]:
        """
        Send an order using the best available router.
        
        Args:
            order: Order to send
            
        Returns:
            Tuple of (success, message)
        """
        # Validate order first
        is_valid, error_message = OrderValidator.validate_order(order)
        if not is_valid:
            order.status = OrderStatus.REJECTED
            order.notes = error_message
            logger.error(f"Order validation failed: {error_message}")
            return False, error_message
        
        # Mark order as validated
        order.status = OrderStatus.VALIDATED
        
        # Decide which router to use
        use_fallback = False
        
        if self.primary_router_errors > 3:  # Switch to fallback after 3 consecutive errors
            use_fallback = True
        elif self.primary_router_latency_ms > self.latency_threshold_ms:
            use_fallback = True
        
        # If we should use fallback but don't have one, use primary anyway
        if use_fallback and not self.fallback_router:
            use_fallback = False
            logger.warning("Should use fallback router but none available")
        
        # Send order with selected router
        router = self.fallback_router if use_fallback else self.primary_router
        router_name = "fallback" if use_fallback else "primary"
        
        order.sent_time = datetime.now()
        
        try:
            start_time = time.time()
            success, message = router.send_order(order)
            end_time = time.time()
            
            # Calculate latency in ms
            latency_ms = (end_time - start_time) * 1000
            
            # Update latency metrics
            if use_fallback:
                self.fallback_router_latency_ms = latency_ms
            else:
                self.primary_router_latency_ms = latency_ms
            
            # Update order status
            if success:
                order.status = OrderStatus.SENT
                # Reset error count for the router that succeeded
                if use_fallback:
                    self.fallback_router_errors = 0
                else:
                    self.primary_router_errors = 0
                    
                logger.info(f"Order {order.id} sent via {router_name} router in {latency_ms:.2f}ms")
            else:
                order.status = OrderStatus.ERROR
                order.notes = message
                
                # Increment error count
                if use_fallback:
                    self.fallback_router_errors += 1
                else:
                    self.primary_router_errors += 1
                    
                logger.error(f"Order {order.id} failed via {router_name} router: {message}")
                
                # Try fallback if primary failed and fallback is available
                if not use_fallback and self.fallback_router and self.primary_router_errors > 0:
                    logger.info(f"Retrying order {order.id} with fallback router")
                    return self.send_order(order)
            
            return success, message
            
        except Exception as e:
            # Handle unexpected errors
            order.status = OrderStatus.ERROR
            order.notes = str(e)
            
            # Increment error count
            if use_fallback:
                self.fallback_router_errors += 1
            else:
                self.primary_router_errors += 1
                
            logger.exception(f"Unexpected error sending order {order.id} via {router_name} router")
            
            # Try fallback if primary failed and fallback is available
            if not use_fallback and self.fallback_router:
                logger.info(f"Retrying order {order.id} with fallback router after exception")
                return self.send_order(order)
                
            return False, str(e)
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an order using the router that sent it.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, message)
        """
        # Try both routers to cancel, starting with primary
        try:
            success, message = self.primary_router.cancel_order(order_id)
            if success:
                return True, message
        except Exception as e:
            logger.exception(f"Error canceling order {order_id} via primary router")
        
        # If primary failed and we have a fallback, try that
        if self.fallback_router:
            try:
                return self.fallback_router.cancel_order(order_id)
            except Exception as e:
                logger.exception(f"Error canceling order {order_id} via fallback router")
                return False, str(e)
        
        return False, "Failed to cancel order"
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current status of an order from either router.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Current order status, or None if not found
        """
        # Try primary router first
        try:
            status = self.primary_router.get_order_status(order_id)
            if status:
                return status
        except Exception:
            logger.exception(f"Error getting status for order {order_id} from primary router")
        
        # If not found or error, try fallback router
        if self.fallback_router:
            try:
                return self.fallback_router.get_order_status(order_id)
            except Exception:
                logger.exception(f"Error getting status for order {order_id} from fallback router")
        
        return None
    
    def get_executions(self, order_id: str) -> List[Execution]:
        """
        Get executions for an order from either router.
        
        Args:
            order_id: ID of the order
            
        Returns:
            List of executions for the order
        """
        executions = []
        
        # Try primary router first
        try:
            primary_executions = self.primary_router.get_executions(order_id)
            executions.extend(primary_executions)
        except Exception:
            logger.exception(f"Error getting executions for order {order_id} from primary router")
        
        # Also try fallback router
        if self.fallback_router:
            try:
                fallback_executions = self.fallback_router.get_executions(order_id)
                executions.extend(fallback_executions)
            except Exception:
                logger.exception(f"Error getting executions for order {order_id} from fallback router")
        
        return executions


class OrderManager:
    """
    Manages the lifecycle of orders in the system.
    
    Tracks orders from creation through execution or cancellation,
    provides updates and handles order events.
    """
    
    def __init__(self, router: Union[OrderRouter, SmartOrderRouter], 
                 poll_interval_ms: int = 200):
        """
        Initialize order manager.
        
        Args:
            router: Order router to use for sending orders
            poll_interval_ms: Interval to poll for order updates in milliseconds
        """
        self.router = router
        self.poll_interval_ms = poll_interval_ms
        
        # Store of all orders
        self.orders: Dict[str, Order] = {}
        
        # Callbacks for order status changes
        self.status_callbacks: Dict[str, List[Callable[[Order], None]]] = {}
        
        # Callbacks for executions
        self.execution_callbacks: Dict[str, List[Callable[[Order, Execution], None]]] = {}
    
    def create_order_from_signal(self, signal: Signal, quantity: int, 
                                order_type: OrderType = OrderType.LIMIT, 
                                time_in_force: TimeInForce = TimeInForce.DAY,
                                limit_price: Optional[float] = None,
                                client_order_id: Optional[str] = None) -> Order:
        """
        Create an order from a trading signal.
        
        Args:
            signal: Trading signal
            quantity: Number of contracts to trade
            order_type: Type of order
            time_in_force: Time in force for the order
            limit_price: Limit price for limit orders
            client_order_id: Optional client order ID
            
        Returns:
            Created order
        """
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order legs based on signal type
        legs = []
        
        if len(signal.legs) > 0:
            # Multi-leg order
            for leg in signal.legs:
                leg_expiration = leg.get("expiration")
                leg_strike = leg.get("strike")
                leg_right = leg.get("right")
                leg_quantity = leg.get("quantity")
                
                # Determine action based on direction
                action = "BUY" if leg_quantity > 0 else "SELL"
                
                # For limit orders, use the provided price or calculate one
                leg_limit_price = None
                if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                    if limit_price:
                        # If an overall limit price is provided, use it as a reference
                        leg_price = leg.get("price", 0)
                        leg_limit_price = leg_price
                        
                        # Adjust leg limit price to be slightly better for buys, worse for sells
                        if action == "BUY":
                            leg_limit_price *= 1.01  # 1% higher for buys (conservative)
                        else:
                            leg_limit_price *= 0.99  # 1% lower for sells (conservative)
                    else:
                        # If no limit price provided, use market prices with a buffer
                        leg_price = leg.get("price", 0)
                        leg_limit_price = leg_price
                        
                        # Adjust leg limit price to be slightly better for buys, worse for sells
                        if action == "BUY":
                            leg_limit_price *= 1.05  # 5% higher for buys (conservative)
                        else:
                            leg_limit_price *= 0.95  # 5% lower for sells (conservative)
                
                # Create order leg
                order_leg = OrderLeg(
                    symbol=signal.symbol,
                    expiration=leg_expiration,
                    strike=leg_strike,
                    right=leg_right,
                    action=action,
                    quantity=abs(leg_quantity) * quantity,  # Scale by overall quantity
                    order_type=order_type,
                    limit_price=leg_limit_price,
                    stop_price=None,  # Not using stop orders for multi-leg
                    exchange="SMART"
                )
                
                legs.append(order_leg)
        else:
            # Single-leg order
            # Determine action based on direction
            action = "BUY" if signal.direction > 0 else "SELL"
            
            # For limit orders, use the provided price or the signal price
            leg_limit_price = None
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if limit_price:
                    leg_limit_price = limit_price
                elif signal.market_price:
                    # Use market price with a buffer
                    leg_limit_price = signal.market_price
                    if action == "BUY":
                        leg_limit_price *= 1.05  # 5% higher for buys (conservative)
                    else:
                        leg_limit_price *= 0.95  # 5% lower for sells (conservative)
            
            # Create order leg
            order_leg = OrderLeg(
                symbol=signal.symbol,
                expiration=signal.expiration,
                strike=signal.strike,
                right=signal.right,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=leg_limit_price,
                stop_price=None,  # Not using stop orders for now
                exchange="SMART"
            )
            
            legs.append(order_leg)
        
        # Create order
        order = Order(
            id=order_id,
            legs=legs,
            time_in_force=time_in_force,
            status=OrderStatus.CREATED,
            signal_id=getattr(signal, 'id', None),
            client_order_id=client_order_id
        )
        
        # Add to order store
        self.orders[order_id] = order
        
        return order
    
    def send_order(self, order: Order) -> Tuple[bool, str]:
        """
        Send an order to the exchange.
        
        Args:
            order: Order to send
            
        Returns:
            Tuple of (success, message)
        """
        # Send order via router
        success, message = self.router.send_order(order)
        
        # Update order in store
        self.orders[order.id] = order
        
        # Notify callbacks if status changed
        self._notify_status_change(order)
        
        return success, message
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, message)
        """
        # Check if order exists
        if order_id not in self.orders:
            return False, f"Order {order_id} not found"
        
        order = self.orders[order_id]
        
        # Check if order can be canceled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            return False, f"Order {order_id} cannot be canceled (status: {order.status.value})"
        
        # Cancel order via router
        success, message = self.router.cancel_order(order_id)
        
        if success:
            # Update order status
            order.status = OrderStatus.CANCELED
            order.updated_time = datetime.now()
            
            # Notify callbacks
            self._notify_status_change(order)
        
        return success, message
    
    def register_status_callback(self, order_id: str, callback: Callable[[Order], None]) -> None:
        """
        Register a callback for order status changes.
        
        Args:
            order_id: ID of the order to watch
            callback: Callback function to call when status changes
        """
        if order_id not in self.status_callbacks:
            self.status_callbacks[order_id] = []
        
        self.status_callbacks[order_id].append(callback)
    
    def register_execution_callback(self, order_id: str, 
                                   callback: Callable[[Order, Execution], None]) -> None:
        """
        Register a callback for order executions.
        
        Args:
            order_id: ID of the order to watch
            callback: Callback function to call when an execution occurs
        """
        if order_id not in self.execution_callbacks:
            self.execution_callbacks[order_id] = []
        
        self.execution_callbacks[order_id].append(callback)
    
    def update_order_status(self, order_id: str) -> None:
        """
        Update the status of an order from the exchange.
        
        Args:
            order_id: ID of the order to update
        """
        # Check if order exists
        if order_id not in self.orders:
            logger.warning(f"Cannot update status for unknown order {order_id}")
            return
        
        order = self.orders[order_id]
        
        # Don't update if already in a terminal state
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            return
        
        # Get current status from router
        new_status = self.router.get_order_status(order_id)
        
        if new_status and new_status != order.status:
            # Update order status
            old_status = order.status
            order.status = new_status
            order.updated_time = datetime.now()
            
            # If order is filled, update filled time
            if new_status == OrderStatus.FILLED:
                order.filled_time = datetime.now()
            
            logger.info(f"Order {order_id} status changed: {old_status.value} -> {new_status.value}")
            
            # Notify callbacks
            self._notify_status_change(order)
    
    def update_executions(self, order_id: str) -> None:
        """
        Update executions for an order from the exchange.
        
        Args:
            order_id: ID of the order to update
        """
        # Check if order exists
        if order_id not in self.orders:
            logger.warning(f"Cannot update executions for unknown order {order_id}")
            return
        
        order = self.orders[order_id]
        
        # Get executions from router
        executions = self.router.get_executions(order_id)
        
        # Process new executions
        for execution in executions:
            # Check if this is a new execution
            if execution.execution_id not in order.execution_details:
                # Update order with execution
                order.execution_details[execution.execution_id] = {
                    "time": execution.time,
                    "quantity": execution.quantity,
                    "price": execution.price,
                    "exchange": execution.exchange,
                    "commission": execution.commission
                }
                
                # Update leg fill information
                if 0 <= execution.leg_index < len(order.legs):
                    leg = order.legs[execution.leg_index]
                    leg.filled_quantity += execution.quantity
                    
                    # Update filled price as weighted average
                    if leg.filled_quantity > 0:
                        leg.filled_price = (
                            (leg.filled_price * (leg.filled_quantity - execution.quantity) + 
                             execution.price * execution.quantity) / leg.filled_quantity
                        )
                
                # Check if order is now fully filled
                all_filled = True
                for leg in order.legs:
                    if leg.filled_quantity < leg.quantity:
                        all_filled = False
                        break
                
                if all_filled and order.status != OrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                    order.filled_time = datetime.now()
                    order.updated_time = datetime.now()
                    
                    # Notify status callbacks
                    self._notify_status_change(order)
                elif not all_filled and order.status != OrderStatus.PARTIALLY_FILLED:
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.updated_time = datetime.now()
                    
                    # Notify status callbacks
                    self._notify_status_change(order)
                
                # Notify execution callbacks
                self._notify_execution(order, execution)
    
    def _notify_status_change(self, order: Order) -> None:
        """
        Notify callbacks of an order status change.
        
        Args:
            order: Order with updated status
        """
        if order.id in self.status_callbacks:
            for callback in self.status_callbacks[order.id]:
                try:
                    callback(order)
                except Exception:
                    logger.exception(f"Error in status callback for order {order.id}")
    
    def _notify_execution(self, order: Order, execution: Execution) -> None:
        """
        Notify callbacks of an order execution.
        
        Args:
            order: Order that was executed
            execution: Execution details
        """
        if order.id in self.execution_callbacks:
            for callback in self.execution_callbacks[order.id]:
                try:
                    callback(order, execution)
                except Exception:
                    logger.exception(f"Error in execution callback for order {order.id}")
    
    def poll_for_updates(self) -> None:
        """Poll for updates for all active orders."""
        for order_id, order in self.orders.items():
            # Skip orders in terminal states
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                continue
            
            # Update order status and executions
            self.update_order_status(order_id)
            self.update_executions(order_id)
    
    def start_polling(self) -> None:
        """Start background polling for order updates."""
        import threading
        
        def polling_thread():
            while True:
                try:
                    self.poll_for_updates()
                except Exception:
                    logger.exception("Error polling for order updates")
                
                time.sleep(self.poll_interval_ms / 1000)
        
        thread = threading.Thread(target=polling_thread, daemon=True)
        thread.start()
        
        logger.info(f"Started order polling thread with interval {self.poll_interval_ms}ms")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: ID of the order to get
            
        Returns:
            Order if found, None otherwise
        """
        return self.orders.get(order_id)
    
    def get_all_orders(self) -> List[Order]:
        """
        Get all orders.
        
        Returns:
            List of all orders
        """
        return list(self.orders.values())
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders.
        
        Returns:
            List of active orders
        """
        return [order for order in self.orders.values() 
                if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]] 