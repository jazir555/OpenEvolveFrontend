"""
Monitoring module for Adaptive MDAP.

Provides health checks, dashboards, and alerting capabilities.
"""

from adaptive_mdap.monitoring.health import (
    HealthChecker,
    HealthCheckResult,
    ComponentStatus,
    get_health_checker,
    check_health,
)

from adaptive_mdap.monitoring.dashboard import (
    DashboardGenerator,
    DashboardPanel,
    DashboardConfig,
    get_dashboard,
    get_summary,
    get_full_dashboard,
    get_prometheus_metrics,
)

from adaptive_mdap.monitoring.alerts import (
    AlertingEngine,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    get_alerting_engine,
    check_and_alert,
    get_active_alerts,
)

__all__ = [
    # Health checks
    "HealthChecker",
    "HealthCheckResult",
    "ComponentStatus",
    "get_health_checker",
    "check_health",
    
    # Dashboard
    "DashboardGenerator",
    "DashboardPanel",
    "DashboardConfig",
    "get_dashboard",
    "get_summary",
    "get_full_dashboard",
    "get_prometheus_metrics",
    
    # Alerts
    "AlertingEngine",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "get_alerting_engine",
    "check_and_alert",
    "get_active_alerts",
]
