"""
ADWIN-based Concept Drift Detection

This module monitors prediction streams for concept drift using the ADWIN algorithm,
which can detect changes in data distributions with statistical guarantees.

The module raises alerts when drift is detected, which can trigger:
1. Automatic position size reduction (50%)
2. Background model retraining
"""

import numpy as np
import pandas as pd
from river import drift
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
import threading
import time
import json
import os
import collections

# Configure logging
logger = logging.getLogger(__name__)


class DriftAlert(Exception):
    """Exception raised when concept drift is detected."""
    
    def __init__(self, message: str, detector_name: str, p_value: float, 
                window_size: int, recent_data: List[float]):
        """
        Initialize drift alert.
        
        Args:
            message: Alert message
            detector_name: Name of detector that triggered alert
            p_value: P-value of the change detection
            window_size: Size of window when drift was detected
            recent_data: Recent data points from detection window
        """
        self.detector_name = detector_name
        self.p_value = p_value
        self.window_size = window_size
        self.recent_data = recent_data
        self.timestamp = datetime.now()
        
        super().__init__(message)


@dataclass
class DriftMetadata:
    """Metadata for a drift detection event."""
    detector_name: str
    timestamp: datetime
    p_value: float
    window_size: int
    data_summary: Dict[str, float]
    alert_message: str
    actions_taken: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detector_name": self.detector_name,
            "timestamp": self.timestamp.isoformat(),
            "p_value": self.p_value,
            "window_size": self.window_size,
            "data_summary": self.data_summary,
            "alert_message": self.alert_message,
            "actions_taken": self.actions_taken
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DriftMetadata':
        """Create from dictionary."""
        metadata = cls(
            detector_name=data["detector_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            p_value=data["p_value"],
            window_size=data["window_size"],
            data_summary=data["data_summary"],
            alert_message=data["alert_message"]
        )
        
        if "actions_taken" in data:
            metadata.actions_taken = data["actions_taken"]
            
        return metadata


class ADWINGuard:
    """
    ADWIN-based concept drift detector for prediction streams.
    
    Continuously monitors data streams for distribution changes and
    triggers alerts with configurable sensitivity.
    """
    
    def __init__(self, 
                delta: float = 0.002,  # p-value threshold (default stricter than standard 0.05)
                max_window_size: int = 1000,
                min_window_size: int = 20,
                alert_callback: Optional[Callable[[DriftAlert], None]] = None,
                store_path: Optional[str] = None):
        """
        Initialize ADWIN guard.
        
        Args:
            delta: Detection threshold (lower = more sensitive)
            max_window_size: Maximum size of the sliding window
            min_window_size: Minimum window size for detection
            alert_callback: Function to call when drift is detected
            store_path: Path to store drift metadata (None = don't store)
        """
        self.delta = delta
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.alert_callback = alert_callback
        self.store_path = store_path
        
        # Initialize ADWIN detector
        self.detector = drift.ADWIN(delta=delta)
        
        # Data window for tracking recent values
        self.window = collections.deque(maxlen=max_window_size)
        
        # Metadata for detected drifts
        self.drift_history: List[DriftMetadata] = []
        
        # Threading lock for concurrent access
        self._lock = threading.Lock()
        
        # Performance monitoring
        self.call_count = 0
        self.alert_count = 0
        self.last_alert_time = None
        
        # Create store directory if needed
        if store_path and not os.path.exists(store_path):
            os.makedirs(store_path)
    
    def update(self, value: float) -> bool:
        """
        Update detector with new value.
        
        Args:
            value: New data point from the monitored stream
            
        Returns:
            True if drift detected, False otherwise
            
        Raises:
            DriftAlert: If drift is detected and no callback is set
        """
        with self._lock:
            # Increment call count
            self.call_count += 1
            
            # Add to window
            self.window.append(value)
            
            # Update detector
            self.detector.update(value)
            
            # Check if drift detected
            if self.detector.drift_detected:
                # Calculate window statistics
                recent_data = list(self.window)
                window_size = len(recent_data)
                
                # Only alert if we have enough data
                if window_size >= self.min_window_size:
                    # Compute data summary
                    data_summary = {
                        "mean": float(np.mean(recent_data)),
                        "std": float(np.std(recent_data)),
                        "min": float(np.min(recent_data)),
                        "max": float(np.max(recent_data)),
                        "p25": float(np.percentile(recent_data, 25)),
                        "p50": float(np.percentile(recent_data, 50)),
                        "p75": float(np.percentile(recent_data, 75))
                    }
                    
                    # Create alert message
                    message = (
                        f"Concept drift detected with p-value {self.delta:.4f} "
                        f"after {window_size} observations. "
                        f"Mean: {data_summary['mean']:.4f}, Std: {data_summary['std']:.4f}"
                    )
                    
                    # Create alert object
                    alert = DriftAlert(
                        message=message,
                        detector_name="ADWIN",
                        p_value=self.delta,
                        window_size=window_size,
                        recent_data=recent_data
                    )
                    
                    # Create metadata
                    metadata = DriftMetadata(
                        detector_name="ADWIN",
                        timestamp=datetime.now(),
                        p_value=self.delta,
                        window_size=window_size,
                        data_summary=data_summary,
                        alert_message=message
                    )
                    
                    # Store metadata
                    self.drift_history.append(metadata)
                    
                    # Save metadata if store path provided
                    if self.store_path:
                        self._save_metadata(metadata)
                    
                    # Update stats
                    self.alert_count += 1
                    self.last_alert_time = datetime.now()
                    
                    # Reset detector
                    self.detector = drift.ADWIN(delta=self.delta)
                    
                    # Call callback or raise alert
                    if self.alert_callback:
                        self.alert_callback(alert)
                        return True
                    else:
                        raise alert
            
            return False
    
    def _save_metadata(self, metadata: DriftMetadata) -> None:
        """
        Save drift metadata to disk.
        
        Args:
            metadata: Drift metadata to save
        """
        if not self.store_path:
            return
            
        try:
            # Create filename with timestamp
            timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"drift_{timestamp_str}.json"
            filepath = os.path.join(self.store_path, filename)
            
            # Save as JSON
            with open(filepath, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
                
            # Update history index
            self._update_history_index()
        except Exception as e:
            logger.error(f"Error saving drift metadata: {e}")
    
    def _update_history_index(self) -> None:
        """Update index of drift history."""
        if not self.store_path:
            return
            
        try:
            # Create index with summary of all drift events
            index_data = {
                "total_drifts": len(self.drift_history),
                "last_drift": self.drift_history[-1].timestamp.isoformat() if self.drift_history else None,
                "drifts": [m.to_dict() for m in self.drift_history]
            }
            
            # Save index
            index_path = os.path.join(self.store_path, "drift_index.json")
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating drift history index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            stats = {
                "call_count": self.call_count,
                "alert_count": self.alert_count,
                "alert_rate": self.alert_count / max(1, self.call_count),
                "last_alert_time": self.last_alert_time.isoformat() if self.last_alert_time else None,
                "delta": self.delta,
                "window_size": len(self.window),
                "max_window_size": self.max_window_size
            }
            
            # Add window statistics if there's data
            if self.window:
                window_data = list(self.window)
                stats.update({
                    "window_mean": float(np.mean(window_data)),
                    "window_std": float(np.std(window_data)),
                    "window_min": float(np.min(window_data)),
                    "window_max": float(np.max(window_data))
                })
            
            return stats
    
    def reset(self) -> None:
        """Reset detector state."""
        with self._lock:
            self.detector = drift.ADWIN(delta=self.delta)
            self.window.clear()


class PredictionDriftGuard:
    """
    Guard for monitoring prediction stream drift.
    
    Wraps model predictions and monitors for concept drift, triggering
    risk reduction and retraining when needed.
    """
    
    def __init__(self, 
                drift_delta: float = 0.002,
                position_size_reduction: float = 0.5,
                auto_retrain: bool = True,
                retrain_callback: Optional[Callable[[], None]] = None,
                store_path: Optional[str] = None):
        """
        Initialize prediction drift guard.
        
        Args:
            drift_delta: Sensitivity threshold for drift detection
            position_size_reduction: Factor to reduce position size by on drift
            auto_retrain: Whether to automatically trigger retraining
            retrain_callback: Function to call for retraining
            store_path: Path to store drift metadata
        """
        self.position_size_reduction = position_size_reduction
        self.auto_retrain = auto_retrain
        self.retrain_callback = retrain_callback
        
        # Create drift detectors for different streams
        self.prob_detector = ADWINGuard(
            delta=drift_delta,
            alert_callback=self._handle_prob_drift,
            store_path=store_path
        )
        
        self.win_rate_detector = ADWINGuard(
            delta=drift_delta,
            alert_callback=self._handle_win_rate_drift,
            store_path=store_path
        )
        
        # Position sizing adjustment
        self.current_position_multiplier = 1.0
        
        # Recovery settings
        self.recovery_counter = 0
        self.recovery_threshold = 50  # Number of good predictions before restoring size
        
        # Monitoring
        self.prediction_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.drift_alerts = []
        
        # Background retraining flag
        self.retraining_in_progress = False
        self.last_retrain_time = None
    
    def process_prediction(self, pred_prob: float) -> float:
        """
        Process a new prediction and check for drift.
        
        Args:
            pred_prob: Predicted probability from the model
            
        Returns:
            Original prediction (drift handling is done via callbacks)
        """
        # Update prediction detector
        self.prob_detector.update(pred_prob)
        
        # Increment prediction count
        self.prediction_count += 1
        
        # Increment recovery counter if we're in recovery mode
        if self.current_position_multiplier < 1.0:
            self.recovery_counter += 1
            
            # Check if we should restore position size
            if self.recovery_counter >= self.recovery_threshold:
                self.current_position_multiplier = 1.0
                self.recovery_counter = 0
                logger.info("Position size restored to 100% after recovery period")
        
        return pred_prob
    
    def process_outcome(self, outcome: bool) -> None:
        """
        Process a trade outcome and check for win rate drift.
        
        Args:
            outcome: Trade outcome (True = win, False = loss)
        """
        # Convert to float for detector
        outcome_value = 1.0 if outcome else 0.0
        
        # Update win rate detector
        self.win_rate_detector.update(outcome_value)
        
        # Update counters
        if outcome:
            self.win_count += 1
        else:
            self.loss_count += 1
    
    def get_position_multiplier(self) -> float:
        """
        Get current position size multiplier.
        
        Returns:
            Position size multiplier (0.0-1.0)
        """
        return self.current_position_multiplier
    
    def _handle_prob_drift(self, alert: DriftAlert) -> None:
        """
        Handle drift alert from probability detector.
        
        Args:
            alert: Drift alert from detector
        """
        # Log alert
        logger.warning(f"Probability drift detected: {alert.message}")
        
        # Reduce position size
        self._reduce_position_size("probability_drift")
        
        # Store alert
        self.drift_alerts.append({
            "type": "probability_drift",
            "timestamp": datetime.now().isoformat(),
            "message": alert.message,
            "p_value": alert.p_value,
            "window_size": alert.window_size
        })
        
        # Trigger retraining if enabled
        if self.auto_retrain and not self.retraining_in_progress:
            self._trigger_retraining()
    
    def _handle_win_rate_drift(self, alert: DriftAlert) -> None:
        """
        Handle drift alert from win rate detector.
        
        Args:
            alert: Drift alert from detector
        """
        # Log alert
        logger.warning(f"Win rate drift detected: {alert.message}")
        
        # Reduce position size
        self._reduce_position_size("win_rate_drift")
        
        # Store alert
        self.drift_alerts.append({
            "type": "win_rate_drift",
            "timestamp": datetime.now().isoformat(),
            "message": alert.message,
            "p_value": alert.p_value,
            "window_size": alert.window_size
        })
        
        # Trigger retraining if enabled
        if self.auto_retrain and not self.retraining_in_progress:
            self._trigger_retraining()
    
    def _reduce_position_size(self, reason: str) -> None:
        """
        Reduce position size due to drift.
        
        Args:
            reason: Reason for reduction
        """
        # Apply reduction
        self.current_position_multiplier *= self.position_size_reduction
        
        # Log action
        logger.warning(
            f"Reducing position size to {self.current_position_multiplier:.0%} "
            f"due to {reason}"
        )
        
        # Reset recovery counter
        self.recovery_counter = 0
    
    def _trigger_retraining(self) -> None:
        """Trigger model retraining in background."""
        if self.retrain_callback is None:
            logger.warning("Retraining triggered but no callback provided")
            return
        
        self.retraining_in_progress = True
        self.last_retrain_time = datetime.now()
        
        # Run in background thread
        thread = threading.Thread(
            target=self._run_retraining,
            daemon=True
        )
        thread.start()
        
        logger.info("Triggered background model retraining")
    
    def _run_retraining(self) -> None:
        """Run retraining callback and update status."""
        try:
            # Call retraining function
            self.retrain_callback()
            
            # Log completion
            retrain_duration = (datetime.now() - self.last_retrain_time).total_seconds()
            logger.info(f"Model retraining completed in {retrain_duration:.1f} seconds")
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
        finally:
            # Reset flag
            self.retraining_in_progress = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get guard statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "prediction_count": self.prediction_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_count / max(1, self.prediction_count),
            "current_position_multiplier": self.current_position_multiplier,
            "recovery_counter": self.recovery_counter,
            "recovery_threshold": self.recovery_threshold,
            "retraining_in_progress": self.retraining_in_progress,
            "drift_alerts_count": len(self.drift_alerts),
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else None
        }
        
        # Add detector stats
        stats["prob_detector"] = self.prob_detector.get_stats()
        stats["win_rate_detector"] = self.win_rate_detector.get_stats()
        
        return stats


# Create module-level instance for easy import
default_guard: Optional[PredictionDriftGuard] = None

def get_prediction_guard(drift_delta: float = 0.002,
                        position_size_reduction: float = 0.5,
                        auto_retrain: bool = True,
                        retrain_callback: Optional[Callable[[], None]] = None,
                        store_path: Optional[str] = None) -> PredictionDriftGuard:
    """
    Get the default prediction drift guard.
    
    Args:
        drift_delta: Sensitivity threshold for drift detection
        position_size_reduction: Factor to reduce position size by on drift
        auto_retrain: Whether to automatically trigger retraining
        retrain_callback: Function to call for retraining
        store_path: Path to store drift metadata
        
    Returns:
        Default PredictionDriftGuard instance
    """
    global default_guard
    
    if default_guard is None:
        default_guard = PredictionDriftGuard(
            drift_delta=drift_delta,
            position_size_reduction=position_size_reduction,
            auto_retrain=auto_retrain,
            retrain_callback=retrain_callback,
            store_path=store_path
        )
    
    return default_guard 