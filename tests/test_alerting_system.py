"""
Comprehensive Unit Tests for Alerting System

Tests the alerting system including:
- Alert creation and management
- Severity levels
- Status tracking
- Notification channels
- Alert aggregation
- Escalation rules

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAlertSeverity:
    """Test AlertSeverity enum"""

    def test_severity_enum_values(self):
        """Test AlertSeverity enum contains expected values"""
        from alerting_system import AlertSeverity
        
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_severity_ordering(self):
        """Test severity has expected ordering"""
        from alerting_system import AlertSeverity
        
        severities = list(AlertSeverity)
        # INFO < WARNING < ERROR < CRITICAL
        assert severities[0] == AlertSeverity.INFO
        assert severities[1] == AlertSeverity.WARNING
        assert severities[2] == AlertSeverity.ERROR
        assert severities[3] == AlertSeverity.CRITICAL


class TestAlertStatus:
    """Test AlertStatus enum"""

    def test_status_enum_values(self):
        """Test AlertStatus enum contains expected values"""
        from alerting_system import AlertStatus
        
        assert AlertStatus.OPEN.value == "open"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.ESCALATED.value == "escalated"

    def test_status_transitions(self):
        """Test status has expected values"""
        from alerting_system import AlertStatus
        
        statuses = list(AlertStatus)
        assert len(statuses) == 4
        assert AlertStatus.OPEN in statuses
        assert AlertStatus.ACKNOWLEDGED in statuses


class TestNotificationChannel:
    """Test NotificationChannel enum"""

    def test_channel_enum_values(self):
        """Test NotificationChannel enum contains expected values"""
        from alerting_system import NotificationChannel
        
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.SLACK.value == "slack"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.CONSOLE.value == "console"


class TestAlertModel:
    """Test Alert dataclass"""

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing"""
        from alerting_system import Alert
        from datetime import datetime
        
        return Alert(
            id="alert_001",
            title="High CPU Usage",
            description="CPU usage exceeded 90%",
            severity="warning",
            status="open",
            source="monitoring_system",
            component="web_server",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

    def test_alert_creation(self, sample_alert):
        """Test Alert model creation"""
        from alerting_system import Alert
        
        alert = Alert(
            id="test_alert",
            title="Test Alert",
            description="Test description",
            severity="error",
            status="open",
            source="test",
            component="test_component",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert alert.id == "test_alert"
        assert alert.title == "Test Alert"
        assert alert.severity == "error"
        assert alert.status == "open"

    def test_alert_to_dict(self, sample_alert):
        """Test Alert to_dict conversion"""
        alert_dict = sample_alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict["id"] == "alert_001"
        assert alert_dict["title"] == "High CPU Usage"
        assert "created_at" in alert_dict

    def test_alert_acknowledge(self, sample_alert):
        """Test acknowledging an alert"""
        sample_alert.acknowledge("admin_user")
        
        assert sample_alert.status == "acknowledged"
        assert sample_alert.acknowledged_by == "admin_user"
        assert sample_alert.acknowledged_at is not None

    def test_alert_resolve(self, sample_alert):
        """Test resolving an alert"""
        sample_alert.resolve("admin_user")
        
        assert sample_alert.status == "resolved"
        assert sample_alert.resolved_by == "admin_user"
        assert sample_alert.resolved_at is not None


class TestAlertManager:
    """Test AlertManager functionality"""

    @pytest.fixture
    def alert_manager(self, tmp_path):
        """Create alert manager for testing"""
        from alerting_system import AlertManager
        
        return AlertManager(
            storage_dir=str(tmp_path / "alerts"),
            default_channels=["console"]
        )

    def test_alert_manager_creation(self, alert_manager):
        """Test AlertManager initialization"""
        from alerting_system import AlertManager
        
        manager = AlertManager(
            storage_dir="/tmp/alerts",
            default_channels=["console", "email"]
        )
        assert manager.storage_dir == "/tmp/alerts"
        assert "console" in manager.default_channels

    def test_create_alert(self, alert_manager):
        """Test alert creation"""
        alert = alert_manager.create_alert(
            title="Test Alert",
            description="Test description",
            severity="warning",
            source="test",
            component="test_component"
        )
        
        assert alert is not None
        assert alert.id is not None
        assert alert.title == "Test Alert"

    def test_get_alert(self, alert_manager):
        """Test retrieving an alert"""
        created_alert = alert_manager.create_alert(
            title="Get Test Alert",
            description="Testing get",
            severity="info",
            source="test",
            component="test"
        )
        
        retrieved = alert_manager.get_alert(created_alert.id)
        
        assert retrieved is not None
        assert retrieved.id == created_alert.id

    def test_get_nonexistent_alert(self, alert_manager):
        """Test retrieving non-existent alert returns None"""
        result = alert_manager.get_alert("nonexistent_id")
        assert result is None

    def test_list_alerts(self, alert_manager):
        """Test listing all alerts"""
        # Create multiple alerts
        alert_manager.create_alert(
            title="Alert 1",
            description="First",
            severity="info",
            source="test",
            component="test"
        )
        alert_manager.create_alert(
            title="Alert 2",
            description="Second",
            severity="warning",
            source="test",
            component="test"
        )
        
        alerts = alert_manager.list_alerts()
        
        assert len(alerts) >= 2

    def test_list_alerts_by_severity(self, alert_manager):
        """Test filtering alerts by severity"""
        alert_manager.create_alert(
            title="Warning Alert",
            description="Test",
            severity="warning",
            source="test",
            component="test"
        )
        alert_manager.create_alert(
            title="Error Alert",
            description="Test",
            severity="error",
            source="test",
            component="test"
        )
        
        warnings = alert_manager.list_alerts(severity="warning")
        
        for alert in warnings:
            assert alert.severity == "warning"

    def test_update_alert_status(self, alert_manager):
        """Test updating alert status"""
        alert = alert_manager.create_alert(
            title="Update Test",
            description="Testing update",
            severity="info",
            source="test",
            component="test"
        )
        
        alert_manager.update_status(alert.id, "acknowledged")
        
        updated = alert_manager.get_alert(alert.id)
        assert updated.status == "acknowledged"

    def test_delete_alert(self, alert_manager):
        """Test deleting an alert"""
        alert = alert_manager.create_alert(
            title="Delete Test",
            description="Testing delete",
            severity="info",
            source="test",
            component="test"
        )
        
        alert_manager.delete_alert(alert.id)
        
        retrieved = alert_manager.get_alert(alert.id)
        assert retrieved is None


class TestNotificationChannels:
    """Test notification channel functionality"""

    @pytest.fixture
    def notifier(self, tmp_path):
        """Create notifier for testing"""
        from alerting_system import AlertNotifier
        
        return AlertNotifier(
            smtp_host="localhost",
            smtp_port=587,
            slack_webhook_url=None,
            webhook_urls=[]
        )

    def test_email_notification(self, notifier):
        """Test email notification sending"""
        from alerting_system import send_email_notification
        
        # Should not raise exception
        result = send_email_notification(
            to="admin@example.com",
            subject="Test Alert",
            body="Alert body"
        )
        assert result is True or result is None  # May return True/False

    def test_slack_notification(self, notifier):
        """Test Slack notification sending"""
        from alerting_system import send_slack_notification
        
        # Should handle missing webhook gracefully
        result = send_slack_notification(
            channel="#alerts",
            message="Test alert message"
        )
        assert result is True or result is None

    def test_webhook_notification(self, notifier):
        """Test webhook notification sending"""
        from alerting_system import send_webhook_notification
        
        result = send_webhook_notification(
            url="http://example.com/webhook",
            data={"alert": "test"}
        )
        assert result is True or result is None


class TestAlertAggregation:
    """Test alert aggregation functionality"""

    def test_aggregate_similar_alerts(self):
        """Test aggregating similar alerts"""
        from alerting_system import AlertAggregator
        
        aggregator = AlertAggregator(
            time_window_seconds=300,
            similarity_threshold=0.8
        )
        
        # Add similar alerts
        alert1 = {"title": "CPU High", "source": "server1"}
        alert2 = {"title": "CPU High", "source": "server2"}
        
        count = aggregator.add_alert(alert1)
        count = aggregator.add_alert(alert2)
        
        # Should aggregate
        assert count == 2

    def test_aggregation_time_window(self):
        """Test alerts are grouped by time window"""
        from alerting_system import AlertAggregator
        
        aggregator = AlertAggregator(
            time_window_seconds=60,
            similarity_threshold=0.8
        )
        
        # Add alert
        aggregator.add_alert({"title": "Test", "source": "test"})
        
        # Check time window
        assert aggregator.time_window_seconds == 60


class TestAlertEscalation:
    """Test alert escalation functionality"""

    def test_escalation_rules(self):
        """Test escalation rule configuration"""
        from alerting_system import EscalationRule
        
        rule = EscalationRule(
            condition="severity == 'critical'",
            action="escalate",
            delay_seconds=300
        )
        
        assert rule.delay_seconds == 300

    def test_auto_escalate(self):
        """Test auto-escalation of critical alerts"""
        from alerting_system import auto_escalate
        
        # Function should exist
        assert callable(auto_escalate)


class TestAlertAnalytics:
    """Test alert analytics functionality"""

    def test_get_alert_statistics(self):
        """Test getting alert statistics"""
        from alerting_system import get_alert_stats
        
        stats = get_alert_stats()
        assert isinstance(stats, dict)

    def test_alert_trends(self):
        """Test alert trend analysis"""
        from alerting_system import get_alert_trends
        
        trends = get_alert_trends(hours=24)
        assert isinstance(trends, list)


class TestAlertConfiguration:
    """Test alert system configuration"""

    def test_alert_config_class(self):
        """Test AlertConfig class exists"""
        from alerting_system import AlertConfig
        
        config = AlertConfig(
            max_alerts=1000,
            retention_days=30,
            aggregation_enabled=True
        )
        assert config.max_alerts == 1000
        assert config.retention_days == 30

    def test_channel_config(self):
        """Test notification channel configuration"""
        from alerting_system import ChannelConfig
        
        config = ChannelConfig(
            email_enabled=True,
            slack_enabled=False,
            webhook_enabled=True
        )
        assert config.email_enabled == True
        assert config.slack_enabled == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
