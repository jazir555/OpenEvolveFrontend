"""
Health Monitor for OpenEvolve Integrations

This module provides health monitoring capabilities for all integrations.
It tracks integration status, performance metrics, and provides alerts.

Author: Agent 8 (Integration Orchestrator)
Created: 2026-01-02
Status: ✅ Complete
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from integrations.registry import IntegrationRegistry, IntegrationStatus


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """
    Single health metric measurement.

    Attributes:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
        timestamp: When the metric was recorded
        metadata: Additional metadata
    """
    name: str
    value: Any
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class HealthAlert:
    """
    Health alert notification.

    Attributes:
        integration: Integration name
        level: Alert severity level
        message: Alert message
        timestamp: When alert was generated
        metrics: Related metrics
        resolved: Whether alert has been resolved
        resolved_at: When alert was resolved
    """
    integration: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "integration": self.integration,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class IntegrationHealth:
    """
    Health status of an integration.

    Attributes:
        integration: Integration name
        status: Overall health status
        last_check: Last health check timestamp
        metrics: List of health metrics
        alerts: Active alerts
        uptime: Uptime in seconds
        error_rate: Error rate (0-1)
        avg_response_time: Average response time in ms
        last_error: Last error message
    """
    integration: str
    status: HealthStatus
    last_check: datetime
    metrics: List[HealthMetric] = field(default_factory=list)
    alerts: List[HealthAlert] = field(default_factory=list)
    uptime: float = 0.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "integration": self.integration,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "metrics": [m.to_dict() for m in self.metrics],
            "alerts": [a.to_dict() for a in self.alerts if not a.resolved],
            "uptime": self.uptime,
            "error_rate": self.error_rate,
            "avg_response_time": self.avg_response_time,
            "last_error": self.last_error
        }


class HealthMonitor:
    """
    Health monitoring system for integrations.

    Features:
    - Periodic health checks
    - Performance metrics tracking
    - Alert generation and management
    - Historical health data
    - Graceful degradation monitoring
    """

    def __init__(
        self,
        registry: IntegrationRegistry,
        check_interval: float = 60.0,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize the health monitor.

        Args:
            registry: Integration registry to monitor
            check_interval: Health check interval in seconds
            alert_callbacks: Optional list of callback functions for alerts
        """
        self.registry = registry
        self.check_interval = check_interval
        self.alert_callbacks = alert_callbacks or []

        self._health_history: Dict[str, List[IntegrationHealth]] = {}
        self._current_health: Dict[str, IntegrationHealth] = {}
        self._alerts: List[HealthAlert] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Performance tracking
        self._response_times: Dict[str, List[float]] = {}
        self._error_counts: Dict[str, int] = {}
        self._request_counts: Dict[str, int] = {}

        # Start time for uptime tracking
        self._start_times: Dict[str, datetime] = {}

    async def start_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Periodic health check loop."""
        while self._monitoring:
            try:
                await self.check_all_health()
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            await asyncio.sleep(self.check_interval)

    async def check_all_health(self) -> Dict[str, IntegrationHealth]:
        """
        Check health of all integrations.

        Returns:
            Dictionary mapping integration names to health status
        """
        health_status = {}

        for integration_info in self.registry.list_integrations():
            name = integration_info.name
            health = await self.check_integration_health(name)
            health_status[name] = health

        return health_status

    async def check_integration_health(self, name: str) -> IntegrationHealth:
        """
        Check health of a specific integration.

        Args:
            name: Integration name

        Returns:
            IntegrationHealth object
        """
        start_time = time.time()

        try:
            # Get integration instance
            instance = await self.registry.get_instance(name)

            if instance is None:
                # Integration unavailable
                status = HealthStatus.UNHEALTHY
                metrics = []
                error_message = "Integration instance unavailable"
            else:
                # Perform validation if available
                if hasattr(instance, 'validate'):
                    validation_result = await instance.validate()

                    # Determine health status
                    if validation_result.get('is_valid', False):
                        status = HealthStatus.HEALTHY
                        error_message = None
                    else:
                        status = HealthStatus.UNHEALTHY
                        error_message = validation_result.get('message', 'Validation failed')

                    # Extract metrics
                    metrics = self._extract_metrics(validation_result)
                else:
                    status = HealthStatus.HEALTHY
                    metrics = []
                    error_message = None

            # Track response time
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            self._track_response_time(name, response_time)

            # Get error rate
            error_rate = self._get_error_rate(name)

            # Get uptime
            uptime = self._get_uptime(name)

            # Create health object
            health = IntegrationHealth(
                integration=name,
                status=status,
                last_check=datetime.now(),
                metrics=metrics,
                alerts=self._get_active_alerts(name),
                uptime=uptime,
                error_rate=error_rate,
                avg_response_time=self._get_avg_response_time(name),
                last_error=error_message
            )

            # Update state
            self._current_health[name] = health
            self._add_to_history(name, health)

            # Check for alerts
            await self._check_for_alerts(health)

            # Resolve old alerts if healthy
            if status == HealthStatus.HEALTHY:
                self._resolve_alerts(name)

            return health

        except Exception as e:
            logger.error(f"Error checking health for {name}: {e}")

            # Create unhealthy health object
            health = IntegrationHealth(
                integration=name,
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                last_error=str(e)
            )

            self._current_health[name] = health
            self._add_to_history(name, health)
            self._track_error(name)

            # Generate alert
            await self._generate_alert(
                integration=name,
                level=AlertLevel.ERROR,
                message=f"Health check failed: {str(e)}"
            )

            return health

    def _extract_metrics(self, validation_result: Dict[str, Any]) -> List[HealthMetric]:
        """Extract health metrics from validation result."""
        metrics = []

        # Add performance metrics if available
        if 'performance' in validation_result:
            perf = validation_result['performance']
            for key, value in perf.items():
                metrics.append(HealthMetric(
                    name=f"performance.{key}",
                    value=value,
                    unit="",
                    timestamp=datetime.now()
                ))

        # Add custom metrics
        if 'metrics' in validation_result:
            for key, value in validation_result['metrics'].items():
                metrics.append(HealthMetric(
                    name=key,
                    value=value,
                    unit="",
                    timestamp=datetime.now()
                ))

        return metrics

    def _track_response_time(self, integration: str, response_time: float) -> None:
        """Track response time for an integration."""
        if integration not in self._response_times:
            self._response_times[integration] = []

        self._response_times[integration].append(response_time)

        # Keep only last 100 measurements
        if len(self._response_times[integration]) > 100:
            self._response_times[integration] = self._response_times[integration][-100:]

        # Track request count
        self._request_counts[integration] = self._request_counts.get(integration, 0) + 1

        # Record start time if first request
        if integration not in self._start_times:
            self._start_times[integration] = datetime.now()

    def _track_error(self, integration: str) -> None:
        """Track error for an integration."""
        self._error_counts[integration] = self._error_counts.get(integration, 0) + 1

    def _get_avg_response_time(self, integration: str) -> float:
        """Get average response time for an integration."""
        if integration not in self._response_times or not self._response_times[integration]:
            return 0.0

        return sum(self._response_times[integration]) / len(self._response_times[integration])

    def _get_error_rate(self, integration: str) -> float:
        """Get error rate for an integration."""
        errors = self._error_counts.get(integration, 0)
        requests = self._request_counts.get(integration, 1)

        return errors / requests

    def _get_uptime(self, integration: str) -> float:
        """Get uptime in seconds for an integration."""
        if integration not in self._start_times:
            return 0.0

        return (datetime.now() - self._start_times[integration]).total_seconds()

    def _get_active_alerts(self, integration: str) -> List[HealthAlert]:
        """Get active (unresolved) alerts for an integration."""
        return [
            alert for alert in self._alerts
            if alert.integration == integration and not alert.resolved
        ]

    def _add_to_history(self, integration: str, health: IntegrationHealth) -> None:
        """Add health check to history."""
        if integration not in self._health_history:
            self._health_history[integration] = []

        self._health_history[integration].append(health)

        # Keep only last 1000 entries
        if len(self._health_history[integration]) > 1000:
            self._health_history[integration] = self._health_history[integration][-1000:]

    async def _check_for_alerts(self, health: IntegrationHealth) -> None:
        """Check for alert conditions and generate alerts."""
        # Check for unhealthy status
        if health.status == HealthStatus.UNHEALTHY:
            await self._generate_alert(
                integration=health.integration,
                level=AlertLevel.ERROR,
                message=f"Integration is unhealthy: {health.last_error or 'Unknown reason'}"
            )

        # Check for high error rate
        if health.error_rate > 0.1:  # 10% error rate
            await self._generate_alert(
                integration=health.integration,
                level=AlertLevel.WARNING,
                message=f"High error rate: {health.error_rate:.2%}"
            )

        # Check for slow response times
        if health.avg_response_time > 5000:  # 5 seconds
            await self._generate_alert(
                integration=health.integration,
                level=AlertLevel.WARNING,
                message=f"Slow response time: {health.avg_response_time:.2f}ms"
            )

    async def _generate_alert(
        self,
        integration: str,
        level: AlertLevel,
        message: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Generate and process an alert."""
        alert = HealthAlert(
            integration=integration,
            level=level,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics or {}
        )

        self._alerts.append(alert)

        logger.warning(f"[{level.value.upper()}] {integration}: {message}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _resolve_alerts(self, integration: str) -> None:
        """Resolve all alerts for an integration."""
        for alert in self._alerts:
            if alert.integration == integration and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Resolved alert for {integration}: {alert.message}")

    def get_current_health(self, integration: str) -> Optional[IntegrationHealth]:
        """Get current health status for an integration."""
        return self._current_health.get(integration)

    def get_all_current_health(self) -> Dict[str, IntegrationHealth]:
        """Get current health status for all integrations."""
        return self._current_health.copy()

    def get_health_history(
        self,
        integration: str,
        since: Optional[datetime] = None
    ) -> List[IntegrationHealth]:
        """
        Get health history for an integration.

        Args:
            integration: Integration name
            since: Optional start time filter

        Returns:
            List of IntegrationHealth objects
        """
        history = self._health_history.get(integration, [])

        if since:
            history = [h for h in history if h.last_check >= since]

        return history

    def get_alerts(
        self,
        integration: Optional[str] = None,
        resolved: Optional[bool] = None,
        level: Optional[AlertLevel] = None
    ) -> List[HealthAlert]:
        """
        Get alerts with optional filters.

        Args:
            integration: Optional integration name filter
            resolved: Optional resolved status filter
            level: Optional alert level filter

        Returns:
            List of HealthAlert objects
        """
        alerts = self._alerts

        if integration:
            alerts = [a for a in alerts if a.integration == integration]

        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get overall health summary.

        Returns:
            Dictionary containing health summary
        """
        summary = {
            "total_integrations": len(self._current_health),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
            "active_alerts": len([a for a in self._alerts if not a.resolved]),
            "avg_response_time": 0.0,
            "avg_error_rate": 0.0,
            "integrations": {}
        }

        total_response_time = 0.0
        total_error_rate = 0.0

        for name, health in self._current_health.items():
            # Count by status
            if health.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif health.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif health.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            else:
                summary["unknown"] += 1

            # Accumulate averages
            total_response_time += health.avg_response_time
            total_error_rate += health.error_rate

            # Add per-integration summary
            summary["integrations"][name] = {
                "status": health.status.value,
                "uptime": health.uptime,
                "error_rate": health.error_rate,
                "avg_response_time": health.avg_response_time
            }

        # Calculate overall averages
        if self._current_health:
            summary["avg_response_time"] = total_response_time / len(self._current_health)
            summary["avg_error_rate"] = total_error_rate / len(self._current_health)

        return summary

    def export_metrics(self, format: str = "json") -> str:
        """
        Export health metrics.

        Args:
            format: Export format ('json' or 'prometheus')

        Returns:
            Formatted metrics string
        """
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self) -> str:
        """Export metrics as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "health": {
                name: health.to_dict()
                for name, health in self._current_health.items()
            },
            "summary": self.get_health_summary()
        }

        return json.dumps(data, indent=2)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, health in self._current_health.items():
            # Status metric
            status_value = {
                HealthStatus.HEALTHY: 1,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0,
                HealthStatus.UNKNOWN: -1
            }.get(health.status, -1)

            lines.append(f'openevolve_integration_status{{integration="{name}"}} {status_value}')

            # Error rate metric
            lines.append(f'openevolve_integration_error_rate{{integration="{name}"}} {health.error_rate}')

            # Response time metric
            lines.append(f'openevolve_integration_response_time_ms{{integration="{name}"}} {health.avg_response_time}')

            # Uptime metric
            lines.append(f'openevolve_integration_uptime_seconds{{integration="{name}"}} {health.uptime}')

        return "\n".join(lines)
