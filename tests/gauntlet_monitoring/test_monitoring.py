"""
Tests for Gauntlet Monitoring System

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import pytest
import time
from glue.adapters.gauntlet_adapter.monitoring import (
    GauntletMetricsCollector,
    HealthChecker,
    AlertingEngine,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    HealthStatus,
    CheckType,
    get_metrics_collector,
    get_health_checker,
    get_alerting_engine
)


class TestMetricsCollector:
    """Tests for metrics collector"""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        collector = GauntletMetricsCollector()
        assert collector is not None

        summary = collector.get_metric_summary()
        assert "uptime_seconds" in summary
        assert "total_executions" in summary

    def test_record_execution(self):
        """Test recording gauntlet execution"""
        collector = GauntletMetricsCollector()

        collector.record_execution(
            domain="finance",
            passed=True,
            duration_ms=1234.5,
            score=0.85,
            rounds_completed=3
        )

        summary = collector.get_metric_summary()
        assert summary["total_executions"] == 1
        assert summary["total_passes"] == 1

    def test_execution_stats_by_domain(self):
        """Test domain-specific execution statistics"""
        collector = GauntletMetricsCollector()

        # Record executions for different domains
        collector.record_execution("finance", True, 1000, 0.8)
        collector.record_execution("finance", False, 1200, 0.0)
        collector.record_execution("science", True, 800, 0.9)

        stats = collector.get_execution_stats()

        assert "finance" in stats
        assert stats["finance"]["total_executions"] == 2
        assert stats["finance"]["pass_rate"] == 0.5

        assert "science" in stats
        assert stats["science"]["total_executions"] == 1
        assert stats["science"]["pass_rate"] == 1.0

    def test_ml_metrics(self):
        """Test ML component metrics"""
        collector = GauntletMetricsCollector()

        # Record optimization iteration
        collector.record_optimization_iteration(
            strategy="q_learning",
            iteration=10,
            score=0.75,
            improvement=0.15
        )

        # Record prediction
        collector.record_prediction(
            success_probability=0.80,
            confidence=0.85,
            actual_outcome=True,
            domain="finance"
        )

        # Record training metrics
        collector.record_training_metrics(
            loss=0.123,
            converged=True,
            epoch=50
        )

        ml_metrics = collector.get_ml_metrics()
        assert ml_metrics["optimization_iterations"] == 1
        assert ml_metrics["predictions_made"] == 1
        assert ml_metrics["average_prediction_accuracy"] == 1.0

    def test_prometheus_export(self):
        """Test Prometheus format export"""
        collector = GauntletMetricsCollector()

        collector.record_execution("test", True, 1000, 0.8)

        prometheus = collector.export_prometheus()

        assert "gauntlet_executions_total" in prometheus
        assert "# TYPE" in prometheus
        assert "counter" in prometheus

    def test_json_export(self):
        """Test JSON format export"""
        collector = GauntletMetricsCollector()

        collector.record_execution("test", True, 1000, 0.8)

        json_export = collector.export_json()

        import json
        data = json.loads(json_export)

        assert "counters" in data
        assert "gauges" in data
        assert "executions_by_domain" in data


class TestHealthChecker:
    """Tests for health checker"""

    def test_health_checker_initialization(self):
        """Test health checker initialization"""
        checker = HealthChecker()
        assert checker is not None

        uptime = checker.get_uptime_seconds()
        assert uptime >= 0

    def test_memory_check(self):
        """Test memory health check"""
        checker = HealthChecker()
        results = checker.check_all()

        assert "memory" in results
        result = results["memory"]
        assert result.component == "memory"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

    def test_cpu_check(self):
        """Test CPU health check"""
        checker = HealthChecker()
        results = checker.check_all()

        assert "cpu" in results
        result = results["cpu"]
        assert result.component == "cpu"

    def test_disk_check(self):
        """Test disk health check"""
        checker = HealthChecker()
        results = checker.check_all()

        assert "disk" in results
        result = results["disk"]
        assert result.component == "disk"

    def test_overall_status(self):
        """Test overall status calculation"""
        checker = HealthChecker()
        status = checker.get_overall_status()

        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

    def test_is_healthy(self):
        """Test is_healthy convenience method"""
        checker = HealthChecker()
        is_healthy = checker.is_healthy()

        assert isinstance(is_healthy, bool)

    def test_is_ready(self):
        """Test is_ready convenience method"""
        checker = HealthChecker()
        is_ready = checker.is_ready()

        assert isinstance(is_ready, bool)

    def test_health_report(self):
        """Test health report generation"""
        checker = HealthChecker()
        report = checker.get_health_report()

        assert "overall_status" in report
        assert "is_healthy" in report
        assert "is_ready" in report
        assert "uptime_seconds" in report
        assert "components" in report

    def test_custom_health_check(self):
        """Test custom health check registration"""
        checker = HealthChecker()

        def custom_check():
            from glue.adapters.gauntlet_adapter.monitoring import HealthCheckResult, HealthStatus
            return HealthCheckResult(
                component="custom",
                status=HealthStatus.HEALTHY,
                message="Custom check passed"
            )

        checker.register_custom_check("custom", custom_check)
        results = checker.check_all()

        assert "custom" in results


class TestAlertingEngine:
    """Tests for alerting engine"""

    def test_alerting_engine_initialization(self):
        """Test alerting engine initialization"""
        engine = AlertingEngine()
        assert engine is not None

        rules = engine.get_rules()
        assert len(rules) > 0  # Should have default rules

    def test_add_custom_rule(self):
        """Test adding custom alert rule"""
        engine = AlertingEngine()

        def condition(metrics):
            return metrics.get("test_value", 0) > 10

        rule = AlertRule(
            name="test_rule",
            severity=AlertSeverity.WARNING,
            condition_fn=condition,
            message_template="Test value is {value}",
            threshold=10
        )

        engine.add_rule(rule)

        rules = engine.get_rules()
        rule_names = [r.name for r in rules]
        assert "test_rule" in rule_names

    def test_remove_rule(self):
        """Test removing alert rule"""
        engine = AlertingEngine()

        def condition(metrics):
            return True

        rule = AlertRule(
            name="temp_rule",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Temporary rule"
        )

        engine.add_rule(rule)
        assert engine.remove_rule("temp_rule") is True
        assert engine.remove_rule("nonexistent") is False

    def test_evaluate_alerts(self):
        """Test alert evaluation"""
        engine = AlertingEngine()

        def condition(metrics):
            return metrics.get("trigger", False)

        rule = AlertRule(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            condition_fn=condition,
            message_template="Test alert triggered",
            cooldown_seconds=0  # No cooldown for testing
        )

        engine.add_rule(rule)

        # Trigger alert
        alerts = engine.evaluate({"trigger": True})
        assert len(alerts) > 0
        assert alerts[0].name == "test_alert"

    def test_alert_lifecycle(self):
        """Test alert acknowledge and resolve"""
        engine = AlertingEngine()

        def condition(metrics):
            return True

        rule = AlertRule(
            name="lifecycle_test",
            severity=AlertSeverity.WARNING,
            condition_fn=condition,
            message_template="Lifecycle test",
            cooldown_seconds=0
        )

        engine.add_rule(rule)
        alerts = engine.evaluate({"trigger": True})

        if alerts:
            alert_id = alerts[0].alert_id

            # Acknowledge
            assert engine.acknowledge_alert(alert_id) is True
            alert = engine.get_alert(alert_id)
            assert alert.status == AlertStatus.ACKNOWLEDGED

            # Resolve
            assert engine.resolve_alert(alert_id) is True
            alert = engine.get_alert(alert_id)
            assert alert.status == AlertStatus.RESOLVED

    def test_get_active_alerts(self):
        """Test getting active alerts"""
        engine = AlertingEngine()

        def condition(metrics):
            return True

        rule = AlertRule(
            name="active_test",
            severity=AlertSeverity.INFO,
            condition_fn=condition,
            message_template="Active test",
            cooldown_seconds=0
        )

        engine.add_rule(rule)
        engine.evaluate({"trigger": True})

        active = engine.get_active_alerts()
        assert len(active) > 0

    def test_alert_statistics(self):
        """Test alert statistics"""
        engine = AlertingEngine()
        stats = engine.get_alert_statistics()

        assert "total_alerts" in stats
        assert "active_alerts" in stats
        assert "total_rules" in stats
        assert "enabled_rules" in stats


class TestIntegration:
    """Integration tests for monitoring system"""

    def test_end_to_end_workflow(self):
        """Test complete monitoring workflow"""
        # Get components
        metrics = get_metrics_collector()
        health = get_health_checker()
        alerts = get_alerting_engine()

        # Record execution
        metrics.record_execution(
            domain="integration_test",
            passed=True,
            duration_ms=1000,
            score=0.85
        )

        # Check health
        is_healthy = health.is_healthy()
        assert isinstance(is_healthy, bool)

        # Evaluate alerts
        triggered = alerts.evaluate()
        assert isinstance(triggered, list)

        # Get summary
        summary = metrics.get_metric_summary()
        assert summary["total_executions"] >= 1

    def test_concurrent_access(self):
        """Test thread-safe concurrent access"""
        import threading

        metrics = get_metrics_collector()
        threads = []

        def record_executions():
            for i in range(10):
                metrics.record_execution(
                    domain="concurrent_test",
                    passed=i % 2 == 0,
                    duration_ms=1000 + i * 100,
                    score=0.5 + (i % 5) * 0.1
                )

        # Create multiple threads
        for _ in range(5):
            thread = threading.Thread(target=record_executions)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all executions were recorded
        summary = metrics.get_metric_summary()
        assert summary["total_executions"] >= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
