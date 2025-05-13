"""
Alert & Execution Bus for Options Trading System

This module generates and distributes alerts when an edge is detected with high probability.
It supports various distribution methods including Slack, Discord, and Twilio SMS.

Flow:
1. If edge_prob >= 0.67 AND risk_checks pass -> build alert object
2. Send via configured channels
3. Log alert to InfluxDB measurement 'trade_signals'
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import threading
import requests
import queue
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert distribution channels."""
    SLACK = "slack"
    DISCORD = "discord"
    TWILIO = "twilio"
    EMAIL = "email"
    WEB = "web"
    INFLUXDB = "influxdb"


@dataclass
class TradeAlert:
    """Trading alert object."""
    contract_id: str
    symbol: str
    option_type: str  # "CALL" or "PUT"
    strike: float
    expiration: str
    target_price: float
    stop_loss: float
    entry_price_max: float
    timestamp: datetime = field(default_factory=datetime.now)
    edge_probability: float = 0.0
    delta: float = 0.0
    level: AlertLevel = AlertLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize ID if not provided."""
        if self.id is None:
            # Generate unique ID based on contract and timestamp
            self.id = f"{self.symbol}_{self.option_type}_{int(self.strike)}_{int(time.time())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "contract_id": self.contract_id,
            "symbol": self.symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "entry_price_max": self.entry_price_max,
            "timestamp": self.timestamp.isoformat(),
            "edge_probability": self.edge_probability,
            "delta": self.delta,
            "level": self.level.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeAlert':
        """Create alert from dictionary."""
        # Convert string level to enum
        level = data.get("level", "medium")
        if isinstance(level, str):
            level = AlertLevel(level)
        
        # Convert timestamp string to datetime
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            id=data.get("id"),
            contract_id=data.get("contract_id", ""),
            symbol=data.get("symbol", ""),
            option_type=data.get("option_type", ""),
            strike=data.get("strike", 0.0),
            expiration=data.get("expiration", ""),
            target_price=data.get("target_price", 0.0),
            stop_loss=data.get("stop_loss", 0.0),
            entry_price_max=data.get("entry_price_max", 0.0),
            timestamp=timestamp or datetime.now(),
            edge_probability=data.get("edge_probability", 0.0),
            delta=data.get("delta", 0.0),
            level=level,
            metadata=data.get("metadata", {})
        )
    
    def format_message(self) -> str:
        """Format alert as a human-readable message."""
        risk_reward = (self.target_price - self.entry_price_max) / (self.entry_price_max - self.stop_loss) if (self.entry_price_max - self.stop_loss) > 0 else 0
        
        # Calculate percentage changes
        target_pct = (self.target_price / self.entry_price_max - 1) * 100
        stop_pct = (self.stop_loss / self.entry_price_max - 1) * 100
        
        return (
            f"🚨 TRADE ALERT: {self.symbol} {self.option_type} ${self.strike} 🚨\n"
            f"Edge: {self.edge_probability:.2f} | Delta: {self.delta:.2f}\n"
            f"Entry: ≤ ${self.entry_price_max:.2f}\n"
            f"Target: ${self.target_price:.2f} (+{target_pct:.1f}%)\n"
            f"Stop Loss: ${self.stop_loss:.2f} ({stop_pct:.1f}%)\n"
            f"R/R: {risk_reward:.1f} | Exp: {self.expiration}"
        )
    
    def format_slack_message(self) -> Dict[str, Any]:
        """Format alert for Slack."""
        risk_reward = (self.target_price - self.entry_price_max) / (self.entry_price_max - self.stop_loss) if (self.entry_price_max - self.stop_loss) > 0 else 0
        
        # Pick color based on probability
        color = "#36a64f"  # green
        if self.edge_probability < 0.7:
            color = "#ffcc00"  # yellow
        
        # Calculate percentage changes
        target_pct = (self.target_price / self.entry_price_max - 1) * 100
        stop_pct = (self.stop_loss / self.entry_price_max - 1) * 100
        
        return {
            "attachments": [
                {
                    "color": color,
                    "title": f"Trade Alert: {self.symbol} {self.option_type} ${self.strike}",
                    "fields": [
                        {"title": "Edge", "value": f"{self.edge_probability:.2f}", "short": True},
                        {"title": "Delta", "value": f"{self.delta:.2f}", "short": True},
                        {"title": "Entry", "value": f"≤ ${self.entry_price_max:.2f}", "short": True},
                        {"title": "Target", "value": f"${self.target_price:.2f} (+{target_pct:.1f}%)", "short": True},
                        {"title": "Stop Loss", "value": f"${self.stop_loss:.2f} ({stop_pct:.1f}%)", "short": True},
                        {"title": "R/R", "value": f"{risk_reward:.1f}", "short": True},
                        {"title": "Expiration", "value": self.expiration, "short": True},
                        {"title": "ID", "value": self.id, "short": True}
                    ],
                    "footer": "ML Options Trading System",
                    "ts": int(self.timestamp.timestamp())
                }
            ]
        }


class AlertPublisher:
    """
    Alert distribution system supporting multiple channels.
    
    Handles alert formatting and distribution to various channels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert publisher.
        
        Args:
            config: Configuration dictionary with channel settings
        """
        self.config = config
        self.channels = {}
        
        # Initialize enabled channels
        self._initialize_channels()
        
        # Alert processing queue and thread
        self.queue = queue.Queue()
        self.running = False
        self.processing_thread = None
        
        logger.info(f"Alert publisher initialized with {len(self.channels)} channels")
    
    def _initialize_channels(self) -> None:
        """Initialize configured alert channels."""
        # Slack
        if self.config.get("slack", {}).get("enabled", False):
            webhook_url = self.config["slack"].get("webhook_url")
            if webhook_url:
                self.channels[AlertChannel.SLACK] = {
                    "webhook_url": webhook_url,
                    "channel": self.config["slack"].get("channel", "#alerts"),
                    "username": self.config["slack"].get("username", "Options Alert Bot")
                }
                logger.info("Slack alerts enabled")
        
        # Discord
        if self.config.get("discord", {}).get("enabled", False):
            webhook_url = self.config["discord"].get("webhook_url")
            if webhook_url:
                self.channels[AlertChannel.DISCORD] = {
                    "webhook_url": webhook_url,
                    "username": self.config["discord"].get("username", "Options Alert Bot")
                }
                logger.info("Discord alerts enabled")
        
        # Twilio SMS
        if self.config.get("twilio", {}).get("enabled", False):
            account_sid = self.config["twilio"].get("account_sid")
            auth_token = self.config["twilio"].get("auth_token")
            from_number = self.config["twilio"].get("from_number")
            to_numbers = self.config["twilio"].get("to_numbers", [])
            
            if account_sid and auth_token and from_number and to_numbers:
                self.channels[AlertChannel.TWILIO] = {
                    "account_sid": account_sid,
                    "auth_token": auth_token,
                    "from_number": from_number,
                    "to_numbers": to_numbers
                }
                logger.info("Twilio SMS alerts enabled")
        
        # InfluxDB
        if self.config.get("influxdb", {}).get("enabled", False):
            url = self.config["influxdb"].get("url")
            token = self.config["influxdb"].get("token")
            org = self.config["influxdb"].get("org")
            bucket = self.config["influxdb"].get("bucket")
            
            if url and token and org and bucket:
                self.channels[AlertChannel.INFLUXDB] = {
                    "url": url,
                    "token": token,
                    "org": org,
                    "bucket": bucket
                }
                logger.info("InfluxDB alerts enabled")
    
    def start(self) -> None:
        """Start the alert processing thread."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_alerts)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Alert processing thread started")
    
    def stop(self) -> None:
        """Stop the alert processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            self.processing_thread = None
        
        logger.info("Alert processing thread stopped")
    
    def publish(self, alert: TradeAlert) -> None:
        """
        Publish an alert to all configured channels.
        
        Args:
            alert: Alert to publish
        """
        self.queue.put(alert)
        logger.debug(f"Alert queued: {alert.id}")
    
    def _process_alerts(self) -> None:
        """Process alerts from the queue."""
        while self.running:
            try:
                # Get alert from queue with timeout
                try:
                    alert = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Publish to all enabled channels
                for channel, config in self.channels.items():
                    try:
                        if channel == AlertChannel.SLACK:
                            self._send_slack_alert(alert, config)
                        elif channel == AlertChannel.DISCORD:
                            self._send_discord_alert(alert, config)
                        elif channel == AlertChannel.TWILIO:
                            self._send_twilio_alert(alert, config)
                        elif channel == AlertChannel.INFLUXDB:
                            self._send_influxdb_alert(alert, config)
                    except Exception as e:
                        logger.error(f"Error sending alert to {channel.value}: {e}")
                
                # Mark as done
                self.queue.task_done()
                
                # Log success
                logger.info(f"Alert processed: {alert.id} - {alert.symbol} {alert.option_type} ${alert.strike}")
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
    
    def _send_slack_alert(self, alert: TradeAlert, config: Dict[str, Any]) -> None:
        """
        Send alert to Slack.
        
        Args:
            alert: Alert to send
            config: Slack configuration
        """
        webhook_url = config["webhook_url"]
        message = alert.format_slack_message()
        
        # Add channel and username
        message["channel"] = config.get("channel", "#alerts")
        message["username"] = config.get("username", "Options Alert Bot")
        
        response = requests.post(webhook_url, json=message)
        
        if response.status_code != 200:
            logger.warning(f"Slack API error: {response.status_code} - {response.text}")
    
    def _send_discord_alert(self, alert: TradeAlert, config: Dict[str, Any]) -> None:
        """
        Send alert to Discord.
        
        Args:
            alert: Alert to send
            config: Discord configuration
        """
        webhook_url = config["webhook_url"]
        
        # Discord uses a different format than Slack
        message = {
            "username": config.get("username", "Options Alert Bot"),
            "content": alert.format_message()
        }
        
        response = requests.post(webhook_url, json=message)
        
        if response.status_code != 204:
            logger.warning(f"Discord API error: {response.status_code} - {response.text}")
    
    def _send_twilio_alert(self, alert: TradeAlert, config: Dict[str, Any]) -> None:
        """
        Send alert via Twilio SMS.
        
        Args:
            alert: Alert to send
            config: Twilio configuration
        """
        # Import Twilio client here to avoid dependency requirement
        try:
            from twilio.rest import Client
        except ImportError:
            logger.error("Twilio package not installed. Install with 'pip install twilio'")
            return
        
        account_sid = config["account_sid"]
        auth_token = config["auth_token"]
        from_number = config["from_number"]
        to_numbers = config["to_numbers"]
        
        # Create Twilio client
        client = Client(account_sid, auth_token)
        
        # Format message
        message_body = alert.format_message()
        
        # Send to all numbers
        for to_number in to_numbers:
            try:
                message = client.messages.create(
                    body=message_body,
                    from_=from_number,
                    to=to_number
                )
                logger.debug(f"SMS sent to {to_number}: {message.sid}")
            except Exception as e:
                logger.error(f"Error sending SMS to {to_number}: {e}")
    
    def _send_influxdb_alert(self, alert: TradeAlert, config: Dict[str, Any]) -> None:
        """
        Send alert to InfluxDB.
        
        Args:
            alert: Alert to send
            config: InfluxDB configuration
        """
        # Import InfluxDB client here to avoid dependency requirement
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
        except ImportError:
            logger.error("InfluxDB client not installed. Install with 'pip install influxdb-client'")
            return
        
        url = config["url"]
        token = config["token"]
        org = config["org"]
        bucket = config["bucket"]
        
        # Create client
        client = InfluxDBClient(url=url, token=token, org=org)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Create point
        point = Point("trade_signals") \
            .tag("symbol", alert.symbol) \
            .tag("option_type", alert.option_type) \
            .tag("contract_id", alert.contract_id) \
            .field("strike", alert.strike) \
            .field("edge_probability", alert.edge_probability) \
            .field("delta", alert.delta) \
            .field("entry_price_max", alert.entry_price_max) \
            .field("target_price", alert.target_price) \
            .field("stop_loss", alert.stop_loss) \
            .time(alert.timestamp)
        
        # Write point
        write_api.write(bucket=bucket, record=point)
        
        # Close client
        client.close()


class AlertEngine:
    """
    Alert & Execution Bus for options trading system.
    
    Decides when to generate alerts based on edge probability and risk checks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize alert engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize alert publisher
        self.publisher = AlertPublisher(self.config.get("channels", {}))
        
        # Minimum edge probability
        self.min_edge_probability = self.config.get("min_edge_probability", 0.67)
        
        # Alert history
        self.alert_history: List[TradeAlert] = []
        self.alert_history_max_size = self.config.get("alert_history_max_size", 100)
        
        # Auto-execution mode (if enabled)
        self.auto_execute = self.config.get("auto_execute", False)
        self.execution_client = None
        
        # Start publisher
        self.publisher.start()
        
        logger.info(f"Alert engine initialized, min_edge_probability={self.min_edge_probability}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "min_edge_probability": 0.67,
            "alert_history_max_size": 100,
            "auto_execute": False,
            "channels": {
                "slack": {"enabled": False},
                "discord": {"enabled": False},
                "twilio": {"enabled": False},
                "influxdb": {"enabled": False}
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return default_config
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def check_edge(self, contract_id: str, symbol: str, option_type: str, strike: float,
                   expiration: str, edge_probability: float, delta: float, current_price: float,
                   target_price: float, stop_loss: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if an edge exists and generate alert if it does.
        
        Args:
            contract_id: Contract ID
            symbol: Symbol
            option_type: Option type ("CALL" or "PUT")
            strike: Strike price
            expiration: Expiration date
            edge_probability: Edge probability (0.0-1.0)
            delta: Option delta
            current_price: Current option price
            target_price: Target option price
            stop_loss: Stop loss price
            metadata: Additional metadata
            
        Returns:
            True if alert generated, False otherwise
        """
        # Check if edge probability meets threshold
        if edge_probability < self.min_edge_probability:
            logger.debug(f"Edge probability too low: {edge_probability} < {self.min_edge_probability}")
            return False
        
        # Create alert object
        alert = TradeAlert(
            contract_id=contract_id,
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            edge_probability=edge_probability,
            delta=delta,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_price_max=current_price,
            metadata=metadata or {}
        )
        
        # Check for duplicate alerts (same contract, similar price, recent)
        if self._is_duplicate_alert(alert):
            logger.debug(f"Duplicate alert suppressed: {alert.id}")
            return False
        
        # Add to history
        self._add_to_history(alert)
        
        # Publish alert
        self.publisher.publish(alert)
        
        # Auto-execute if enabled
        if self.auto_execute and self.execution_client:
            self._execute_trade(alert)
        
        return True
    
    def _is_duplicate_alert(self, alert: TradeAlert) -> bool:
        """
        Check if an alert is a duplicate of a recent alert.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if duplicate, False otherwise
        """
        # Check against recent alerts
        for recent in self.alert_history:
            # Only check alerts for the same contract
            if recent.contract_id != alert.contract_id:
                continue
            
            # Check if timestamp is within 1 hour
            time_diff = (alert.timestamp - recent.timestamp).total_seconds()
            if time_diff > 3600:
                continue
            
            # Check if price is within 5%
            price_diff_pct = abs(alert.entry_price_max - recent.entry_price_max) / recent.entry_price_max
            if price_diff_pct <= 0.05:
                return True
        
        return False
    
    def _add_to_history(self, alert: TradeAlert) -> None:
        """
        Add alert to history.
        
        Args:
            alert: Alert to add
        """
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.alert_history_max_size:
            self.alert_history = self.alert_history[-self.alert_history_max_size:]
    
    def _execute_trade(self, alert: TradeAlert) -> None:
        """
        Execute a trade based on an alert (if auto-execution is enabled).
        
        Args:
            alert: Alert with trade information
        """
        # This would interface with a broker API
        logger.info(f"Auto-execution not yet implemented for alert: {alert.id}")
    
    def get_recent_alerts(self, limit: int = 10) -> List[TradeAlert]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def shutdown(self) -> None:
        """Shut down alert engine."""
        if self.publisher:
            self.publisher.stop()
        
        logger.info("Alert engine shutdown")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create alert engine
    engine = AlertEngine()
    
    # Example alert
    engine.check_edge(
        contract_id="AAPL_230616C00150000",
        symbol="AAPL",
        option_type="CALL",
        strike=150.0,
        expiration="2023-06-16",
        edge_probability=0.75,
        delta=0.45,
        current_price=5.20,
        target_price=7.80,
        stop_loss=3.90,
        metadata={"source": "stacked_ensemble", "market_regime": "bullish"}
    )
    
    # Wait for alert to be processed
    time.sleep(2)
    
    # Shutdown
    engine.shutdown() 