"""
Gauntlet Monitoring System

Comprehensive production monitoring, metrics collection, and alerting
for the gauntlet evaluation system.

Components:
- metrics: Prometheus-compatible metrics collection
- health_checks: Liveness and readiness probes
- alerting: Configurable alert rules and notifications

Usage:
    >>> from glue.adapters.gauntlet_adapter.monitoring import (
    ...     get_metrics_collector,
    ...     get_health_checker,
    ...     get_alerting_engine
    ... )
    >>>
    >>> # Record execution metrics
    >>> metrics = get_metrics_collector()
    >>> metrics.record_execution(
    ...     domain="finance",
    ...     passed=True,
    ...     duration_ms=1234.5,
    ...     score=0.85
    ... )
    >>>
    >>> # Check health
    >>> health = get_health_checker()
    >>> if health.is_ready():
    ...     print("System is ready")
    >>>
    >>> # Export Prometheus metrics
    >>> prometheus_metrics = metrics.export_prometheus()

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

from .metrics import (
    GauntletMetricsCollector,
    MetricType,
    MetricValue,
    Histogram,
    HistogramBucket,
    get_metrics_collector,
    record_execution,
    export_prometheus,
    export_json
)

from .health_checks import (
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
    CheckType,
    DependencyHealth,
    get_health_checker,
    check_liveness,
    check_readiness,
    is_healthy,
    is_ready
)

from .alerting import (
    AlertingEngine,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    LogNotificationChannel,
    WebhookNotificationChannel,
    get_alerting_engine,
    evaluate_alerts,
    get_active_alerts,
    acknowledge_alert,
    resolve_alert
)

__all__ = [
    # Metrics
    "GauntletMetricsCollector",
    "MetricType",
    "MetricValue",
    "Histogram",
    "HistogramBucket",
    "get_metrics_collector",
    "record_execution",
    "export_prometheus",
    "export_json",

    # Health Checks
    "HealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "CheckType",
    "DependencyHealth",
    "get_health_checker",
    "check_liveness",
    "check_readiness",
    "is_healthy",
    "is_ready",

    # Alerting
    "AlertingEngine",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "NotificationChannel",
    "LogNotificationChannel",
    "WebhookNotificationChannel",
    "get_alerting_engine",
    "evaluate_alerts",
    "get_active_alerts",
    "acknowledge_alert",
    "resolve_alert"
]
