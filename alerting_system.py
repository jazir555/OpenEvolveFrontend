"""
Alerting System for OpenEvolve Frontend

Provides comprehensive alerting capabilities with multiple notification channels:
- In-memory alert storage
- Persistent alert storage (JSON file)
- Email notifications (via SMTP)
- Slack notifications (via webhook)
- Generic webhook notifications
- Alert aggregation and deduplication
- Alert severity levels and routing

Features:
- Alert persistence across restarts
- Multiple notification channels
- Alert history and analytics
- Alert escalation rules
- Graceful degradation when services unavailable
"""

import json
import logging
import smtplib
import hashlib
import threading
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import requests
from functools import wraps

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status tracking."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class Alert:
    """Represents a single alert."""
    id: str
    title: str
    description: str
    severity: str
    status: str
    source: str
    component: str
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    occurrences: int = 1
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create Alert from dictionary."""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'updated_at', 'acknowledged_at', 'resolved_at', 'first_seen', 'last_seen']:
            if key in data and isinstance(data[key], str):
                try:
                    data[key] = datetime.fromisoformat(data[key])
                except (ValueError, TypeError):
                    data[key] = None
        return cls(**data)


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    # Email configuration
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Slack configuration
    slack_enabled: bool = False
    slack_webhook_url: str = ""

    # Generic webhook configuration
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Console logging
    console_enabled: bool = True

    # Alert deduplication
    deduplication_window: int = 300  # 5 minutes in seconds


class AlertStore:
    """Base class for alert storage."""

    def save_alert(self, alert: Alert) -> bool:
        """Save an alert."""
        raise NotImplementedError

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        raise NotImplementedError

    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        raise NotImplementedError

    def update_alert(self, alert: Alert) -> bool:
        """Update an alert."""
        raise NotImplementedError

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        raise NotImplementedError

    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get alerts by severity."""
        raise NotImplementedError

    def get_alerts_by_component(self, component: str) -> List[Alert]:
        """Get alerts by component."""
        raise NotImplementedError


class InMemoryAlertStore(AlertStore):
    """In-memory alert storage with thread safety."""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.lock = threading.RLock()

    def save_alert(self, alert: Alert) -> bool:
        with self.lock:
            self.alerts[alert.id] = alert
            return True

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self.lock:
            return self.alerts.get(alert_id)

    def get_all_alerts(self) -> List[Alert]:
        with self.lock:
            return list(self.alerts.values())

    def update_alert(self, alert: Alert) -> bool:
        with self.lock:
            if alert.id in self.alerts:
                self.alerts[alert.id] = alert
                return True
            return False

    def delete_alert(self, alert_id: str) -> bool:
        with self.lock:
            if alert_id in self.alerts:
                del self.alerts[alert_id]
                return True
            return False

    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        with self.lock:
            return [a for a in self.alerts.values() if a.severity == severity]

    def get_alerts_by_component(self, component: str) -> List[Alert]:
        with self.lock:
            return [a for a in self.alerts.values() if a.component == component]


class PersistentAlertStore(AlertStore):
    """Persistent alert storage with JSON file backend."""

    def __init__(self, storage_path: str = "alerts.json"):
        self.storage_path = Path(storage_path)
        self.in_memory_store = InMemoryAlertStore()
        self.lock = threading.RLock()
        self._load()

    def _load(self):
        """Load alerts from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                for alert_data in data:
                    alert = Alert.from_dict(alert_data)
                    self.in_memory_store.alerts[alert.id] = alert
                logger.info(f"Loaded {len(self.in_memory_store.alerts)} alerts from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load alerts: {e}")

    def _save(self):
        """Save alerts to disk."""
        try:
            alerts = [alert.to_dict() for alert in self.in_memory_store.get_all_alerts()]
            with open(self.storage_path, 'w') as f:
                json.dump(alerts, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def save_alert(self, alert: Alert) -> bool:
        result = self.in_memory_store.save_alert(alert)
        if result:
            self._save()
        return result

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        return self.in_memory_store.get_alert(alert_id)

    def get_all_alerts(self) -> List[Alert]:
        return self.in_memory_store.get_all_alerts()

    def update_alert(self, alert: Alert) -> bool:
        result = self.in_memory_store.update_alert(alert)
        if result:
            self._save()
        return result

    def delete_alert(self, alert_id: str) -> bool:
        result = self.in_memory_store.delete_alert(alert_id)
        if result:
            self._save()
        return result

    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        return self.in_memory_store.get_alerts_by_severity(severity)

    def get_alerts_by_component(self, component: str) -> List[Alert]:
        return self.in_memory_store.get_alerts_by_component(component)


class NotificationService:
    """Handles sending notifications to various channels."""

    def __init__(self, config: NotificationConfig):
        self.config = config

    def send_notification(self, alert: Alert, channels: List[NotificationChannel]) -> bool:
        """Send notification through specified channels."""
        success = True

        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL and self.config.email_enabled:
                    self._send_email(alert)
                elif channel == NotificationChannel.SLACK and self.config.slack_enabled:
                    self._send_slack(alert)
                elif channel == NotificationChannel.WEBHOOK and self.config.webhook_enabled:
                    self._send_webhook(alert)
                elif channel == NotificationChannel.CONSOLE and self.config.console_enabled:
                    self._log_to_console(alert)
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")
                success = False

        return success

    def _send_email(self, alert: Alert):
        """Send email notification."""
        if not self.config.smtp_username or not self.config.smtp_password:
            logger.warning("Email credentials not configured")
            return

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
        msg['From'] = self.config.email_from
        msg['To'] = ', '.join(self.config.email_to)

        # Create email body
        body = f"""
Alert: {alert.title}
Severity: {alert.severity}
Component: {alert.component}
Source: {alert.source}
Description: {alert.description}
Created: {alert.created_at.isoformat()}
Tags: {', '.join(alert.tags)}
"""

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)

        logger.info(f"Email notification sent for alert {alert.id}")

    def _send_slack(self, alert: Alert):
        """Send Slack notification via webhook."""
        if not self.config.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return

        # Color mapping for severity
        colors = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'error': '#ff0000',
            'critical': '#990000'
        }

        attachment = {
            'color': colors.get(alert.severity, '#808080'),
            'title': f"[{alert.severity.upper()}] {alert.title}",
            'text': alert.description,
            'fields': [
                {'title': 'Component', 'value': alert.component, 'short': True},
                {'title': 'Source', 'value': alert.source, 'short': True},
                {'title': 'Created', 'value': alert.created_at.strftime('%Y-%m-%d %H:%M:%S'), 'short': True},
                {'title': 'Occurrences', 'value': str(alert.occurrences), 'short': True},
            ],
            'footer': 'OpenEvolve Alerting System',
            'ts': int(alert.created_at.timestamp())
        }

        payload = {'attachments': [attachment]}

        response = requests.post(self.config.slack_webhook_url, json=payload)
        response.raise_for_status()

        logger.info(f"Slack notification sent for alert {alert.id}")

    def _send_webhook(self, alert: Alert):
        """Send generic webhook notification."""
        if not self.config.webhook_url:
            logger.warning("Webhook URL not configured")
            return

        payload = {
            'alert_id': alert.id,
            'title': alert.title,
            'description': alert.description,
            'severity': alert.severity,
            'status': alert.status,
            'source': alert.source,
            'component': alert.component,
            'created_at': alert.created_at.isoformat(),
            'tags': alert.tags,
            'metadata': alert.metadata
        }

        response = requests.post(
            self.config.webhook_url,
            json=payload,
            headers=self.config.webhook_headers
        )
        response.raise_for_status()

        logger.info(f"Webhook notification sent for alert {alert.id}")

    def _log_to_console(self, alert: Alert):
        """Log alert to console."""
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        emoji = severity_emoji.get(alert.severity, '📢')

        logger.info(
            f"{emoji} [{alert.severity.upper()}] {alert.title} "
            f"(Component: {alert.component}, Source: {alert.source})"
        )
        logger.info(f"  Description: {alert.description}")
        logger.info(f"  Tags: {', '.join(alert.tags)}")


class AlertManager:
    """
    Main alert management system.

    Features:
    - Create and track alerts
    - Alert deduplication
    - Alert escalation
    - Multi-channel notifications
    - Alert history and analytics
    """

    def __init__(
        self,
        storage: Optional[AlertStore] = None,
        notification_config: Optional[NotificationConfig] = None
    ):
        """
        Initialize alert manager.

        Args:
            storage: Alert storage backend (defaults to PersistentAlertStore)
            notification_config: Notification channel configuration
        """
        self.storage = storage or PersistentAlertStore()
        self.notification_config = notification_config or NotificationConfig()
        self.notification_service = NotificationService(self.notification_config)
        self.alert_dedup_cache: Dict[str, Tuple[Alert, datetime]] = {}

    def create_alert(
        self,
        title: str,
        description: str,
        severity: str = "warning",
        source: str = "system",
        component: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        notify_channels: Optional[List[NotificationChannel]] = None
    ) -> Alert:
        """
        Create a new alert.

        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity (info, warning, error, critical)
            source: Alert source
            component: Component that generated the alert
            tags: Optional tags for categorization
            metadata: Optional additional metadata
            notify_channels: Notification channels to use

        Returns:
            Created Alert object
        """
        alert_id = self._generate_alert_id(title, component)

        now = datetime.now()

        # Check for duplicate alert
        dedup_key = self._deduplication_key(title, component, severity)
        if dedup_key in self.alert_dedup_cache:
            existing_alert, first_seen = self.alert_dedup_cache[dedup_key]
            time_since_first = (now - first_seen).total_seconds()

            if time_since_first < self.notification_config.deduplication_window:
                # Update existing alert instead of creating new one
                existing_alert.occurrences += 1
                existing_alert.last_seen = now
                existing_alert.updated_at = now
                self.storage.update_alert(existing_alert)
                logger.info(f"Updated existing alert {existing_alert.id} (occurrence #{existing_alert.occurrences})")
                return existing_alert

        # Create new alert
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.OPEN.value,
            source=source,
            component=component,
            created_at=now,
            updated_at=now,
            first_seen=now,
            last_seen=now,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Save alert
        self.storage.save_alert(alert)

        # Add to deduplication cache
        self.alert_dedup_cache[dedup_key] = (alert, now)

        # Send notifications
        if notify_channels:
            self.notification_service.send_notification(alert, notify_channels)
        elif self.notification_config.console_enabled:
            self.notification_service.send_notification(alert, [NotificationChannel.CONSOLE])

        logger.info(f"Created alert {alert.id}: {title}")
        return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        alert = self.storage.get_alert(alert_id)
        if not alert:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.now()

        return self.storage.update_alert(alert)

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        alert = self.storage.get_alert(alert_id)
        if not alert:
            return False

        alert.status = AlertStatus.RESOLVED.value
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by
        alert.updated_at = datetime.now()

        # Remove from deduplication cache
        dedup_key = self._deduplication_key(alert.title, alert.component, alert.severity)
        self.alert_dedup_cache.pop(dedup_key, None)

        return self.storage.update_alert(alert)

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.storage.get_alert(alert_id)

    def get_all_alerts(
        self,
        severity: Optional[str] = None,
        component: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Alert]:
        """Get all alerts with optional filtering."""
        alerts = self.storage.get_all_alerts()

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if component:
            alerts = [a for a in alerts if a.component == component]
        if status:
            alerts = [a for a in alerts if a.status == status]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        alerts = self.storage.get_all_alerts()

        severity_counts = {}
        status_counts = {}
        component_counts = {}

        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            status_counts[alert.status] = status_counts.get(alert.status, 0) + 1
            component_counts[alert.component] = component_counts.get(alert.component, 0) + 1

        return {
            'total': len(alerts),
            'by_severity': severity_counts,
            'by_status': status_counts,
            'by_component': component_counts
        }

    def _generate_alert_id(self, title: str, component: str) -> str:
        """Generate unique alert ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        hash_input = f"{title}:{component}:{timestamp}".encode()
        hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]
        return f"alert-{timestamp}-{hash_suffix}"

    def _deduplication_key(self, title: str, component: str, severity: str) -> str:
        """Generate deduplication key for alert."""
        return f"{component}:{severity}:{title.lower()}"


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager(
    storage: Optional[AlertStore] = None,
    config: Optional[NotificationConfig] = None
) -> AlertManager:
    """Get or create global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager(storage, config)
    return _global_alert_manager


def reset_alert_manager():
    """Reset global alert manager instance."""
    global _global_alert_manager
    _global_alert_manager = None


__all__ = [
    'Alert',
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'NotificationConfig',
    'AlertStore',
    'InMemoryAlertStore',
    'PersistentAlertStore',
    'NotificationService',
    'AlertManager',
    'get_alert_manager',
    'reset_alert_manager',
]
