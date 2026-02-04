"""
Alerting System for Gauntlet Monitoring

Provides comprehensive alerting with configurable thresholds,
notification channels, and integration with external alerting systems.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC, timedelta
from threading import Lock
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class Alert:
    """
    Alert definition.

    Attributes:
        alert_id: Unique alert identifier
        name: Alert name
        severity: Alert severity
        status: Current status
        message: Alert message
        condition: Condition that triggered the alert
        value: Current value that triggered the alert
        threshold: Threshold that was exceeded
        metadata: Additional context
        created_at: Creation timestamp
        acknowledged_at: Acknowledgement timestamp
        resolved_at: Resolution timestamp
        labels: Labels for filtering/grouping
    """
    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    condition: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "condition": self.condition,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "labels": self.labels
        }

    def age_seconds(self) -> float:
        """Get alert age in seconds"""
        return time.time() - self.created_at


@dataclass
class AlertRule:
    """
    Alert rule definition.

    Attributes:
        name: Rule name
        severity: Alert severity
        condition_fn: Function that evaluates the rule
        message_template: Alert message template (can use {value}, {threshold})
        threshold: Threshold value
        comparison: Comparison operator ('gt', 'lt', 'eq', 'gte', 'lte')
        labels: Labels to attach to alerts
        enabled: Whether rule is enabled
        cooldown_seconds: Cooldown period between alerts
        evaluation_interval_seconds: How often to evaluate
    """
    name: str
    severity: AlertSeverity
    condition_fn: Callable[[Dict[str, Any]], bool]
    message_template: str
    threshold: Optional[float] = None
    comparison: str = "gt"
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    cooldown_seconds: int = 300
    evaluation_interval_seconds: int = 60


class NotificationChannel:
    """Base class for notification channels"""

    def send(self, alert: Alert) -> bool:
        """
        Send notification for alert.

        Args:
            alert: Alert to notify about

        Returns:
            True if notification sent successfully
        """
        raise NotImplementedError


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel (for testing/debugging)"""

    def __init__(self):
        self._notifications: List[Alert] = []

    def send(self, alert: Alert) -> bool:
        """Log the alert"""
        self._notifications.append(alert)

        log_msg = (
            f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}"
        )

        if alert.severity == AlertSeverity.CRITICAL:
            logger.error(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return True


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, url: str, timeout: int = 5):
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            import requests

            payload = {
                "alert": alert.to_dict(),
                "timestamp": time.time()
            }

            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class AlertingEngine:
    """
    Comprehensive alerting engine for gauntlet system.

    Features:
    - Configurable alert rules with thresholds
    - Multiple severity levels
    - Alert lifecycle management (active, acknowledged, resolved)
    - Multiple notification channels
    - Cooldown periods to prevent alert spam
    - Alert history and statistics

    Example:
        >>> engine = AlertingEngine()
        >>>
        >>> # Add a rule
        >>> engine.add_rule(AlertRule(
        ...     name="high_error_rate",
        ...     severity=AlertSeverity.WARNING,
        ...     condition_fn=lambda m: m.get("error_rate", 0) > 0.1,
        ...     message_template="Error rate is {value:.1%} (threshold: {threshold:.1%})",
        ...     threshold=0.1
        ... ))
        >>>
        >>> # Evaluate metrics
        >>> alerts = engine.evaluate({"error_rate": 0.15})
        >>>
        >>> # Get active alerts
        >>> active = engine.get_active_alerts()
    """

    def __init__(self):
        """Initialize alerting engine"""
        self._lock = Lock()

        # Alerts storage
        self._alerts: Dict[str, Alert] = {}
        self._rules: List[AlertRule] = []

        # Notification channels
        self._notification_channels: List[NotificationChannel] = []

        # Alert tracking
        self._last_triggered: Dict[str, float] = {}  # rule_name -> last_trigger_time
        self._alert_counter: int = 0

        # Add default log channel
        self.add_notification_channel(LogNotificationChannel())

        # Register default rules
        self._register_default_rules()

        logger.info("Gauntlet Alerting Engine initialized")

    def _register_default_rules(self):
        """Register default alert rules"""
        # High error rate rule
        self.add_rule(AlertRule(
            name="high_error_rate",
            severity=AlertSeverity.WARNING,
            condition_fn=self._check_high_error_rate,
            message_template="Gauntlet error rate is {value:.1%} (threshold: {threshold:.1%})",
            threshold=0.1,
            comparison="gt",
            labels={"component": "gauntlet"}
        ))

        # High latency rule
        self.add_rule(AlertRule(
            name="high_latency",
            severity=AlertSeverity.WARNING,
            condition_fn=self._check_high_latency,
            message_template="Gauntlet latency is {value:.0f}ms (threshold: {threshold:.0f}ms)",
            threshold=5000,
            comparison="gt",
            labels={"component": "gauntlet"}
        ))

        # Low pass rate rule
        self.add_rule(AlertRule(
            name="low_pass_rate",
            severity=AlertSeverity.CRITICAL,
            condition_fn=self._check_low_pass_rate,
            message_template="Gauntlet pass rate is {value:.1%} (threshold: {threshold:.1%})",
            threshold=0.5,
            comparison="lt",
            labels={"component": "gauntlet"}
        ))

        # Memory usage rule
        self.add_rule(AlertRule(
            name="high_memory_usage",
            severity=AlertSeverity.WARNING,
            condition_fn=self._check_memory_usage,
            message_template="Memory usage is {value:.1%} (threshold: {threshold:.1%})",
            threshold=0.85,
            comparison="gt",
            labels={"component": "system"}
        ))

        # CPU usage rule
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            severity=AlertSeverity.WARNING,
            condition_fn=self._check_cpu_usage,
            message_template="CPU usage is {value:.1%} (threshold: {threshold:.1%})",
            threshold=0.80,
            comparison="gt",
            labels={"component": "system"}
        ))

        # Low prediction accuracy rule
        self.add_rule(AlertRule(
            name="low_prediction_accuracy",
            severity=AlertSeverity.INFO,
            condition_fn=self._check_prediction_accuracy,
            message_template="Prediction accuracy is {value:.1%} (threshold: {threshold:.1%})",
            threshold=0.6,
            comparison="lt",
            labels={"component": "ml"}
        ))

    # ========== Default Rule Conditions ==========

    def _check_high_error_rate(self, metrics: Dict[str, Any]) -> bool:
        """Check for high error rate"""
        # Get from metrics collector if available
        try:
            from glue.adapters.gauntlet_adapter.monitoring.metrics import get_metrics_collector
            collector = get_metrics_collector()
            summary = collector.get_metric_summary()

            total = summary.get("total_executions", 0)
            failed = summary.get("total_failures", 0)

            if total > 0:
                error_rate = failed / total
                return error_rate > 0.1
        except Exception:
            pass

        return False

    def _check_high_latency(self, metrics: Dict[str, Any]) -> bool:
        """Check for high latency"""
        # Check gauges for latency
        latency = metrics.get("gauntlet_last_duration_ms", 0)
        return latency > 5000

    def _check_low_pass_rate(self, metrics: Dict[str, Any]) -> bool:
        """Check for low pass rate"""
        try:
            from glue.adapters.gauntlet_adapter.monitoring.metrics import get_metrics_collector
            collector = get_metrics_collector()
            summary = collector.get_metric_summary()

            pass_rate = summary.get("global_pass_rate", 1.0)
            return pass_rate < 0.5
        except Exception:
            pass

        return False

    def _check_memory_usage(self, metrics: Dict[str, Any]) -> bool:
        """Check for high memory usage"""
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent > 85

    def _check_cpu_usage(self, metrics: Dict[str, Any]) -> bool:
        """Check for high CPU usage"""
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        return cpu > 80

    def _check_prediction_accuracy(self, metrics: Dict[str, Any]) -> bool:
        """Check for low prediction accuracy"""
        try:
            from glue.adapters.gauntlet_adapter.monitoring.metrics import get_metrics_collector
            collector = get_metrics_collector()
            ml_metrics = collector.get_ml_metrics()

            accuracy = ml_metrics.get("average_prediction_accuracy", 1.0)
            return accuracy < 0.6
        except Exception:
            pass

        return False

    # ========== Rule Management ==========

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        with self._lock:
            self._rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if rule was removed
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.name == rule_name:
                    del self._rules[i]
                    logger.info(f"Removed alert rule: {rule_name}")
                    return True
            return False

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        with self._lock:
            return self._rules.copy()

    # ========== Notification Channels ==========

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """
        Add a notification channel.

        Args:
            channel: Notification channel to add
        """
        with self._lock:
            self._notification_channels.append(channel)
        logger.info(f"Added notification channel: {type(channel).__name__}")

    def remove_notification_channel(self, channel: NotificationChannel) -> None:
        """Remove a notification channel"""
        with self._lock:
            if channel in self._notification_channels:
                self._notification_channels.remove(channel)

    # ========== Alert Evaluation ==========

    def evaluate(self, metrics: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """
        Evaluate all alert rules.

        Args:
            metrics: Current metrics (will collect if None)

        Returns:
            List of triggered alerts
        """
        if metrics is None:
            metrics = {}

        # Collect current metrics if not provided
        try:
            from glue.adapters.gauntlet_adapter.monitoring.metrics import get_metrics_collector
            collector = get_metrics_collector()
            summary = collector.get_metric_summary()
            metrics.update(summary)
        except Exception as e:
            logger.warning(f"Failed to collect metrics for alerting: {e}")

        triggered_alerts = []
        now = time.time()

        with self._lock:
            for rule in self._rules:
                if not rule.enabled:
                    continue

                # Check cooldown
                if rule.name in self._last_triggered:
                    if now - self._last_triggered[rule.name] < rule.cooldown_seconds:
                        continue

                try:
                    # Evaluate condition
                    if rule.condition_fn(metrics):
                        # Create alert
                        self._alert_counter += 1
                        alert_id = f"{rule.name}_{int(now)}_{self._alert_counter}"

                        # Format message
                        message = rule.message_template.format(
                            value=rule.threshold,  # Will be updated below
                            threshold=rule.threshold
                        )

                        # Try to get actual value
                        actual_value = self._extract_value_for_rule(rule, metrics)
                        if actual_value is not None:
                            message = rule.message_template.format(
                                value=actual_value,
                                threshold=rule.threshold
                            )

                        alert = Alert(
                            alert_id=alert_id,
                            name=rule.name,
                            severity=rule.severity,
                            message=message,
                            condition=rule.name,
                            value=actual_value,
                            threshold=rule.threshold,
                            labels=rule.labels.copy(),
                            metadata={"metrics": metrics}
                        )

                        self._alerts[alert_id] = alert
                        self._last_triggered[rule.name] = now
                        triggered_alerts.append(alert)

                        # Send notifications
                        self._send_notifications(alert)

                except Exception as e:
                    logger.error(f"Alert rule evaluation failed: {rule.name}: {e}")

        return triggered_alerts

    def _extract_value_for_rule(
        self,
        rule: AlertRule,
        metrics: Dict[str, Any]
    ) -> Optional[float]:
        """Extract actual metric value for a rule"""
        # Rule-specific value extraction
        if rule.name == "high_error_rate":
            total = metrics.get("total_executions", 0)
            failed = metrics.get("total_failures", 0)
            if total > 0:
                return failed / total

        elif rule.name == "high_latency":
            return metrics.get("gauntlet_last_duration_ms", None)

        elif rule.name == "low_pass_rate":
            return metrics.get("global_pass_rate", None)

        elif rule.name == "high_memory_usage":
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent / 100.0

        elif rule.name == "high_cpu_usage":
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            return cpu / 100.0

        elif rule.name == "low_prediction_accuracy":
            ml_metrics = metrics.get("ml_metrics", {})
            return ml_metrics.get("average_prediction_accuracy", None)

        return None

    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert"""
        for channel in self._notification_channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Notification channel failed: {type(channel).__name__}: {e}")

    # ========== Alert Management ==========

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if alert was acknowledged
        """
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if alert was resolved
        """
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False

    def silence_alert(self, alert_id: str) -> bool:
        """Silence an alert"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.SILENCED
                logger.info(f"Alert silenced: {alert_id}")
                return True
            return False

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID"""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

    def get_alerts_by_status(self, status: AlertStatus) -> List[Alert]:
        """Get alerts by status"""
        with self._lock:
            return [a for a in self._alerts.values() if a.status == status]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        with self._lock:
            return [a for a in self._alerts.values() if a.severity == severity]

    def get_all_alerts(self, limit: int = 100) -> List[Alert]:
        """
        Get all alerts, most recent first.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        with self._lock:
            alerts = sorted(
                self._alerts.values(),
                key=lambda a: a.created_at,
                reverse=True
            )
            return alerts[:limit]

    def clear_old_alerts(self, max_age_hours: int = 24) -> int:
        """
        Clear old resolved alerts.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of alerts cleared
        """
        with self._lock:
            now = time.time()
            max_age_seconds = max_age_hours * 3600

            to_remove = []
            for alert_id, alert in self._alerts.items():
                if alert.status == AlertStatus.RESOLVED:
                    if now - alert.resolved_at > max_age_seconds:
                        to_remove.append(alert_id)

            for alert_id in to_remove:
                del self._alerts[alert_id]

            if to_remove:
                logger.info(f"Cleared {len(to_remove)} old alerts")

            return len(to_remove)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self._lock:
            total = len(self._alerts)
            active = len([a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE])
            acknowledged = len([a for a in self._alerts.values() if a.status == AlertStatus.ACKNOWLEDGED])
            resolved = len([a for a in self._alerts.values() if a.status == AlertStatus.RESOLVED])

            # By severity
            critical = len([a for a in self._alerts.values() if a.severity == AlertSeverity.CRITICAL and a.status == AlertStatus.ACTIVE])
            warning = len([a for a in self._alerts.values() if a.severity == AlertSeverity.WARNING and a.status == AlertStatus.ACTIVE])
            info = len([a for a in self._alerts.values() if a.severity == AlertSeverity.INFO and a.status == AlertStatus.ACTIVE])

            return {
                "total_alerts": total,
                "active_alerts": active,
                "acknowledged_alerts": acknowledged,
                "resolved_alerts": resolved,
                "active_by_severity": {
                    "critical": critical,
                    "warning": warning,
                    "info": info
                },
                "total_rules": len(self._rules),
                "enabled_rules": len([r for r in self._rules if r.enabled]),
                "notification_channels": len(self._notification_channels)
            }


# Global alerting engine instance
_alerting_engine = AlertingEngine()


def get_alerting_engine() -> AlertingEngine:
    """Get the global alerting engine"""
    return _alerting_engine


def evaluate_alerts(metrics: Optional[Dict[str, Any]] = None) -> List[Alert]:
    """Evaluate alert rules and return triggered alerts"""
    return get_alerting_engine().evaluate(metrics)


def get_active_alerts() -> List[Alert]:
    """Get all active alerts"""
    return get_alerting_engine().get_active_alerts()


def acknowledge_alert(alert_id: str) -> bool:
    """Acknowledge an alert"""
    return get_alerting_engine().acknowledge_alert(alert_id)


def resolve_alert(alert_id: str) -> bool:
    """Resolve an alert"""
    return get_alerting_engine().resolve_alert(alert_id)
