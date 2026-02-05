"""Alerting system for Adaptive MDAP."""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from threading import Lock
import logging

from adaptive_mdap.utils.logger import get_logger

logger = get_logger("monitoring.alerts")


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    name: str
    severity: AlertSeverity
    condition: str
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "condition": self.condition,
            "message": self.message,
            "status": self.status.value,
            "created_at": self.created_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    severity: AlertSeverity
    condition_fn: Callable[[Dict[str, Any]], bool]
    message_template: str
    evaluation_interval_seconds: int = 60
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes cooldown after resolution


class AlertingEngine:
    """Alerting engine for monitoring."""
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._rules: List[AlertRule] = []
        self._notification_callbacks: List[Callable[[Alert], None]] = []
        self._lock = Lock()
        self._last_evaluation: float = 0
        self._active_alerts: Dict[str, float] = {}  # alert_name -> last_triggered
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self._rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name."""
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.name == rule_name:
                    del self._rules[i]
                    return True
            return False
    
    def add_notification_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a notification callback."""
        self._notification_callbacks.append(callback)
    
    def evaluate_all(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against current metrics."""
        now = time.time()
        
        # Don't evaluate too frequently
        if now - self._last_evaluation < 1.0:
            return []
        
        self._last_evaluation = now
        triggered_alerts = []
        
        with self._lock:
            for rule in self._rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule.name in self._active_alerts:
                    if now - self._active_alerts[rule.name] < rule.cooldown_seconds:
                        continue
                
                try:
                    if rule.condition_fn(metrics):
                        # Create alert
                        alert_id = f"{rule.name}_{int(now)}"
                        message = rule.message_template.format(**metrics)
                        
                        alert = Alert(
                            alert_id=alert_id,
                            name=rule.name,
                            severity=rule.severity,
                            condition=rule.message_template,
                            message=message,
                            metadata={"metrics": metrics},
                        )
                        
                        self._alerts[alert_id] = alert
                        self._active_alerts[rule.name] = now
                        triggered_alerts.append(alert)
                        
                        # Notify callbacks
                        for callback in self._notification_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"Alert callback failed: {e}")
                
                except Exception as e:
                    logger.error(f"Alert rule evaluation failed: {rule.name}: {e}")
        
        return triggered_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                
                # Remove from active alerts
                if alert.name in self._active_alerts:
                    del self._active_alerts[alert.name]
                
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        with self._lock:
            return list(self._alerts.values())
    
    def get_alerts_by_status(self, status: AlertStatus) -> List[Alert]:
        """Get alerts by status."""
        with self._lock:
            return [a for a in self._alerts.values() if a.status == status]
    
    def clear_resolved(self) -> int:
        """Clear resolved alerts older than cooldown."""
        with self._lock:
            now = time.time()
            to_remove = []
            
            for alert_id, alert in self._alerts.items():
                if alert.status == AlertStatus.RESOLVED:
                    if now - alert.resolved_at > 3600:  # 1 hour after resolution
                        to_remove.append(alert_id)
            
            for alert_id in to_remove:
                del self._alerts[alert_id]
            
            return len(to_remove)


# Default alert rules
def create_default_rules() -> List[AlertRule]:
    """Create default alert rules."""
    
    def high_error_rate(metrics: Dict[str, Any]) -> bool:
        counters = metrics.get("counters", {})
        success = counters.get("classification_success", 0)
        failure = counters.get("classification_failure", 0)
        total = success + failure
        if total == 0:
            return False
        return (failure / total) > 0.1  # >10% error rate
    
    def high_latency(metrics: Dict[str, Any]) -> bool:
        timers = metrics.get("timers", {})
        latency = timers.get("classification_latency_ms", {})
        mean_latency = latency.get("mean_ms", 0)
        return mean_latency > 5000  # >5 seconds
    
    def low_cache_hit_rate(metrics: Dict[str, Any]) -> bool:
        cache = metrics.get("cache", {})
        hit_rate = cache.get("hit_rate", 1.0)
        return hit_rate < 0.3  # <30% hit rate
    
    return [
        AlertRule(
            name="high_error_rate",
            severity=AlertSeverity.WARNING,
            condition_fn=high_error_rate,
            message_template="Classification error rate is {failure}/{total} ({rate:.1%})",
            evaluation_interval_seconds=60,
        ),
        AlertRule(
            name="high_latency",
            severity=AlertSeverity.WARNING,
            condition_fn=high_latency,
            message_template="Classification latency is high: {mean_latency:.0f}ms",
            evaluation_interval_seconds=60,
        ),
        AlertRule(
            name="low_cache_hit_rate",
            severity=AlertSeverity.INFO,
            condition_fn=low_cache_hit_rate,
            message_template="Cache hit rate is low: {hit_rate:.1%}",
            evaluation_interval_seconds=300,
        ),
    ]


# Global alerting engine
_alerting_engine = AlertingEngine()

# Add default rules
for rule in create_default_rules():
    _alerting_engine.add_rule(rule)


def get_alerting_engine() -> AlertingEngine:
    """Get the global alerting engine."""
    return _alerting_engine


def check_and_alert(metrics: Dict[str, Any]) -> List[Alert]:
    """Evaluate all rules and return triggered alerts."""
    return get_alerting_engine().evaluate_all(metrics)


def get_active_alerts() -> List[Alert]:
    """Get all active alerts."""
    return get_alerting_engine().get_active_alerts()
