"""Tests for Monitoring Module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import time

from adaptive_mdap.monitoring.health import (
    HealthChecker,
    HealthCheckResult,
    ComponentStatus,
)
from adaptive_mdap.monitoring.dashboard import (
    DashboardGenerator,
)
from adaptive_mdap.monitoring.alerts import (
    AlertingEngine,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
)


class TestHealthChecker:
    """Tests for HealthChecker."""
    
    def test_checker_initialization(self):
        """Test health checker can be initialized."""
        checker = HealthChecker()
        
        assert checker._check_results == {}
    
    def test_get_uptime(self):
        """Test uptime calculation."""
        checker = HealthChecker()
        
        # Initial uptime should be very small
        uptime = checker.get_uptime_seconds()
        assert uptime >= 0
    
    def test_get_overall_status_no_checks(self):
        """Test overall status with no checks."""
        checker = HealthChecker()
        
        status = checker.get_overall_status()
        assert status == ComponentStatus.UNKNOWN
    
    def test_check_all(self):
        """Test running all health checks."""
        checker = HealthChecker()
        
        results = checker.check_all()
        
        # Should have checks for memory, cpu, disk, cache, metrics
        assert len(results) >= 3  # At least memory, cpu, disk


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""
    
    def test_result_creation(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            component="test",
            status=ComponentStatus.HEALTHY,
            message="Test passed",
            details={"key": "value"},
            timestamp=time.time(),
        )
        
        assert result.component == "test"
        assert result.status == ComponentStatus.HEALTHY
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = HealthCheckResult(
            component="test",
            status=ComponentStatus.DEGRADED,
            message="Test degraded",
            details={"metric": 0.5},
            timestamp=time.time(),
        )
        
        as_dict = result.to_dict()
        
        assert as_dict["component"] == "test"
        assert as_dict["status"] == "degraded"
        assert as_dict["details"]["metric"] == 0.5


class TestDashboardGenerator:
    """Tests for DashboardGenerator."""
    
    def test_generator_initialization(self):
        """Test dashboard generator can be initialized."""
        generator = DashboardGenerator()
        
        assert generator.metrics is not None
    
    def test_generate_summary(self):
        """Test generating summary dashboard."""
        generator = DashboardGenerator()
        
        summary = generator.generate_summary()
        
        assert "generated_at" in summary
        assert "summary" in summary
        assert "performance" in summary
    
    def test_generate_execution_metrics(self):
        """Test generating execution metrics."""
        generator = DashboardGenerator()
        
        metrics = generator.generate_execution_metrics()
        
        assert "generated_at" in metrics
        assert "strategies" in metrics
    
    def test_generate_cost_dashboard(self):
        """Test generating cost dashboard."""
        generator = DashboardGenerator()
        
        costs = generator.generate_cost_dashboard()
        
        assert "generated_at" in costs
        assert "costs" in costs
    
    def test_generate_full_dashboard(self):
        """Test generating full dashboard."""
        generator = DashboardGenerator()
        
        dashboard = generator.generate_full_dashboard()
        
        assert "generated_at" in dashboard
        assert "summary" in dashboard
        assert "execution" in dashboard
        assert "costs" in dashboard


class TestAlertingEngine:
    """Tests for AlertingEngine."""
    
    def test_engine_initialization(self):
        """Test alerting engine can be initialized."""
        engine = AlertingEngine()
        
        assert engine._alerts == {}
        assert engine._rules == []
    
    def test_add_rule(self):
        """Test adding alert rule."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return metrics.get("error_rate", 0) > 0.1
        
        rule = AlertRule(
            name="high_error_rate",
            severity=AlertSeverity.WARNING,
            condition_fn=condition,
            message_template="Error rate is {error_rate:.2%}",
        )
        
        engine.add_rule(rule)
        
        assert len(engine._rules) == 1
        assert engine._rules[0].name == "high_error_rate"
    
    def test_remove_rule(self):
        """Test removing alert rule."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return False
        
        rule = AlertRule(
            name="test_rule",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
        )
        
        engine.add_rule(rule)
        result = engine.remove_rule("test_rule")
        
        assert result is True
        assert len(engine._rules) == 0
    
    def test_remove_nonexistent_rule(self):
        """Test removing nonexistent rule."""
        engine = AlertingEngine()
        
        result = engine.remove_rule("nonexistent")
        
        assert result is False
    
    def test_evaluate_no_rules_triggered(self):
        """Test evaluation with no rules triggered."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return False  # Never triggers
        
        rule = AlertRule(
            name="test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
        )
        engine.add_rule(rule)
        
        alerts = engine.evaluate_all({"value": 1.0})
        
        assert len(alerts) == 0
    
    def test_evaluate_rule_triggered(self):
        """Test evaluation with rule triggered."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return metrics.get("error_rate", 0) > 0.1
        
        rule = AlertRule(
            name="high_error",
            severity=AlertSeverity.WARNING,
            condition_fn=condition,
            message_template="Error rate: {error_rate:.2%}",
        )
        engine.add_rule(rule)
        
        alerts = engine.evaluate_all({"error_rate": 0.2})
        
        assert len(alerts) == 1
        assert alerts[0].name == "high_error"
        assert alerts[0].severity == AlertSeverity.WARNING
    
    def test_evaluate_with_cooldown(self):
        """Test evaluation with cooldown."""
        engine = AlertingEngine()
        
        trigger_count = [0]
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return True
        
        rule = AlertRule(
            name="test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
            cooldown_seconds=1,  # 1 second cooldown
        )
        engine.add_rule(rule)
        
        # First evaluation
        alerts1 = engine.evaluate_all({})
        assert len(alerts1) == 1
        
        # Immediate second evaluation (should be suppressed)
        alerts2 = engine.evaluate_all({})
        assert len(alerts2) == 0
        
        # Wait for cooldown
        time.sleep(1.1)
        
        # Third evaluation (should trigger again)
        alerts3 = engine.evaluate_all({})
        assert len(alerts3) == 1
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return True
        
        engine.add_rule(AlertRule(
            name="test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
        ))
        
        alerts = engine.evaluate_all({})
        alert_id = alerts[0].alert_id
        
        result = engine.acknowledge_alert(alert_id)
        
        assert result is True
        assert engine._alerts[alert_id].status == AlertStatus.ACKNOWLEDGED
    
    def test_resolve_alert(self):
        """Test resolving an alert."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return True
        
        engine.add_rule(AlertRule(
            name="test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
        ))
        
        alerts = engine.evaluate_all({})
        alert_id = alerts[0].alert_id
        
        result = engine.resolve_alert(alert_id)
        
        assert result is True
        assert engine._alerts[alert_id].status == AlertStatus.RESOLVED
        assert alert_id not in engine._active_alerts
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return True
        
        engine.add_rule(AlertRule(
            name="test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
        ))
        
        engine.evaluate_all({})
        
        active = engine.get_active_alerts()
        
        assert len(active) == 1
        assert active[0].status == AlertStatus.ACTIVE
    
    def test_get_alerts_by_status(self):
        """Test getting alerts by status."""
        engine = AlertingEngine()
        
        def condition(metrics: Dict[str, Any]) -> bool:
            return True
        
        engine.add_rule(AlertRule(
            name="test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Test",
        ))
        
        alerts = engine.evaluate_all({})
        alert_id = alerts[0].alert_id
        
        engine.acknowledge_alert(alert_id)
        
        active = engine.get_alerts_by_status(AlertStatus.ACTIVE)
        acknowledged = engine.get_alerts_by_status(AlertStatus.ACKNOWLEDGED)
        
        assert len(active) == 0
        assert len(acknowledged) == 1


class TestAlert:
    """Tests for Alert."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            alert_id="alert-1",
            name="test_alert",
            severity=AlertSeverity.WARNING,
            condition="error_rate > 0.1",
            message="Error rate is 0.2",
        )
        
        assert alert.alert_id == "alert-1"
        assert alert.status == AlertStatus.ACTIVE
        assert alert.created_at is not None
    
    def test_alert_to_dict(self):
        """Test converting alert to dict."""
        alert = Alert(
            alert_id="alert-1",
            name="test",
            severity=AlertSeverity.CRITICAL,
            condition="condition",
            message="message",
            metadata={"key": "value"},
        )
        
        as_dict = alert.to_dict()
        
        assert as_dict["alert_id"] == "alert-1"
        assert as_dict["severity"] == "critical"
        assert as_dict["metadata"]["key"] == "value"


class TestDefaultRules:
    """Tests for default alert rules."""
    
    def test_high_error_rate_rule(self):
        """Test high error rate rule."""
        from adaptive_mdap.monitoring.alerts import create_default_rules
        
        rules = create_default_rules()
        
        # Find high_error_rate rule
        rule = next((r for r in rules if r.name == "high_error_rate"), None)
        assert rule is not None
        
        # Should trigger with high error rate
        metrics_high = {"counters": {"classification_success": 9, "classification_failure": 1}}
        assert rule.condition_fn(metrics_high) is True
        
        # Should not trigger with low error rate
        metrics_low = {"counters": {"classification_success": 99, "classification_failure": 1}}
        assert rule.condition_fn(metrics_low) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
