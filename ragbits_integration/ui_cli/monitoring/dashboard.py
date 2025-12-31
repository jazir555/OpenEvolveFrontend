#!/usr/bin/env python
"""
Monitoring Dashboard

Real-time monitoring dashboard for RAGBits integration.
Tracks metrics, performance, and system health.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric with time series data"""
    name: str
    metric_type: MetricType
    description: str
    data_points: List[MetricPoint] = field(default_factory=list)
    unit: str = ""

    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add data point"""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)

    def get_current_value(self) -> Optional[float]:
        """Get most recent value"""
        if self.data_points:
            return self.data_points[-1].value
        return None

    def get_average(self, duration_minutes: int = 60) -> Optional[float]:
        """Get average over duration"""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        recent_points = [p for p in self.data_points if p.timestamp >= cutoff]

        if recent_points:
            return sum(p.value for p in recent_points) / len(recent_points)
        return None


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 100", "< 50"
    severity: AlertSeverity
    triggered: bool = False
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class SystemHealth:
    """System health status"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    components: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, float] = field(default_factory=dict)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard.

    Features:
    - Metric collection and visualization
    - Alert generation and notification
    - System health monitoring
    - Performance tracking
    - Resource usage monitoring
    """

    def __init__(self, storage_manager=None):
        """
        Initialize monitoring dashboard.

        Args:
            storage_manager: Optional storage manager
        """
        self.storage = storage_manager

        # Metrics storage
        self._metrics: Dict[str, Metric] = {}

        # Alerts
        self._alerts: Dict[str, Alert] = {}

        # Health status
        self._health_status: Optional[SystemHealth] = None

        # Initialize default metrics
        self._initialize_default_metrics()

    def _initialize_default_metrics(self):
        """Initialize default metrics to track"""
        default_metrics = [
            ("artifacts_stored_total", MetricType.COUNTER, "Total artifacts stored", "count"),
            ("artifacts_stored_rate", MetricType.GAUGE, "Artifacts stored per minute", "artifacts/min"),
            ("queries_total", MetricType.COUNTER, "Total knowledge base queries", "count"),
            ("queries_rate", MetricType.GAUGE, "Queries per minute", "queries/min"),
            ("query_latency_ms", MetricType.HISTOGRAM, "Query latency in milliseconds", "ms"),
            ("extraction_time_ms", MetricType.HISTOGRAM, "Knowledge extraction time", "ms"),
            ("vector_index_size", MetricType.GAUGE, "Vector index document count", "documents"),
            ("cache_hit_rate", MetricType.GAUGE, "Cache hit rate", "percentage"),
            ("active_review_sessions", MetricType.GAUGE, "Active review sessions", "sessions"),
            ("llm_requests_total", MetricType.COUNTER, "Total LLM requests", "count"),
            ("llm_latency_ms", MetricType.HISTOGRAM, "LLM request latency", "ms"),
            ("storage_used_mb", MetricType.GAUGE, "Storage used in MB", "MB"),
            ("memory_usage_mb", MetricType.GAUGE, "Memory usage in MB", "MB")
        ]

        for name, metric_type, description, unit in default_metrics:
            self.register_metric(name, metric_type, description, unit)

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str = ""
    ) -> Metric:
        """
        Register a new metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            unit: Optional unit

        Returns:
            Metric object
        """
        metric = Metric(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit
        )

        self._metrics[name] = metric
        return metric

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels

        Returns:
            True if recorded
        """
        if name not in self._metrics:
            return False

        self._metrics[name].add_point(value, labels)

        # Check alerts
        self._check_alerts(name, value)

        return True

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name"""
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics"""
        return self._metrics.copy()

    def define_alert(
        self,
        alert_id: str,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        severity: AlertSeverity
    ) -> Alert:
        """
        Define an alert.

        Args:
            alert_id: Alert ID
            name: Alert name
            description: Alert description
            metric_name: Metric to monitor
            condition: Condition string (e.g., "> 100", "< 50")
            severity: Alert severity

        Returns:
            Alert object
        """
        alert = Alert(
            alert_id=alert_id,
            name=name,
            description=description,
            metric_name=metric_name,
            condition=condition,
            severity=severity
        )

        self._alerts[alert_id] = alert
        return alert

    def _check_alerts(self, metric_name: str, value: float):
        """Check if any alerts should trigger"""
        for alert in self._alerts.values():
            if alert.metric_name != metric_name:
                continue

            # Evaluate condition
            triggered = self._evaluate_condition(value, alert.condition)

            if triggered and not alert.triggered:
                # Alert triggered
                alert.triggered = True
                alert.last_triggered = datetime.now()
                alert.trigger_count += 1
                self._handle_alert(alert)
            elif not triggered and alert.triggered:
                # Alert cleared
                alert.triggered = False

    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate condition string"""
        try:
            # Simple evaluation
            # In production, use safer evaluation method
            return eval(f"{value} {condition}")
        except Exception:
            return False

    def _handle_alert(self, alert: Alert):
        """Handle triggered alert"""
        # Log alert
        print(f"[ALERT] {alert.severity.value.upper()}: {alert.name}")
        print(f"  {alert.description}")
        print(f"  Condition: {alert.condition}")
        print(f"  Time: {alert.last_triggered}")

        # In production, send notifications (email, Slack, etc.)

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (triggered) alerts"""
        return [a for a in self._alerts.values() if a.triggered]

    async def update_system_health(self) -> SystemHealth:
        """
        Update system health status.

        Returns:
            Current health status
        """
        components = {}
        issues = []
        metrics_summary = {}

        # Check components
        # Vector store
        vector_size = self._metrics.get("vector_index_size", Metric("", MetricType.GAUGE, "")).get_current_value()
        if vector_size is not None and vector_size > 0:
            components["vector_store"] = "healthy"
            metrics_summary["vector_index_size"] = vector_size
        else:
            components["vector_store"] = "degraded"
            issues.append("Vector store has no indexed documents")

        # Cache
        cache_hit_rate = self._metrics.get("cache_hit_rate", Metric("", MetricType.GAUGE, "")).get_current_value()
        if cache_hit_rate is not None:
            components["cache"] = "healthy" if cache_hit_rate > 50 else "degraded"
            metrics_summary["cache_hit_rate"] = cache_hit_rate

        # Query latency
        avg_latency = self._metrics.get("query_latency_ms", Metric("", MetricType.HISTOGRAM, "")).get_average(5)
        if avg_latency is not None:
            components["query_performance"] = "healthy" if avg_latency < 1000 else "degraded"
            metrics_summary["avg_query_latency_ms"] = avg_latency

        # Determine overall status
        if all(v == "healthy" for v in components.values()):
            status = "healthy"
        elif any(v == "unhealthy" for v in components.values()):
            status = "unhealthy"
        else:
            status = "degraded"

        # Check for critical alerts
        critical_alerts = [a for a in self.get_active_alerts() if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            status = "unhealthy"
            issues.extend([a.description for a in critical_alerts])

        self._health_status = SystemHealth(
            status=status,
            timestamp=datetime.now(),
            components=components,
            issues=issues,
            metrics_summary=metrics_summary
        )

        return self._health_status

    def get_system_health(self) -> Optional[SystemHealth]:
        """Get current system health"""
        return self._health_status

    async def generate_dashboard_html(
        self,
        duration_minutes: int = 60,
        include_alerts: bool = True
    ) -> str:
        """
        Generate HTML dashboard.

        Args:
            duration_minutes: Time window for metrics
            include_alerts: Whether to include alerts

        Returns:
            HTML string
        """
        # Update health
        await self.update_system_health()

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RAGBits Monitoring Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
        }
        .header .timestamp {
            margin-top: 10px;
            opacity: 0.9;
        }
        .health-status {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .health-healthy {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .health-degraded {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .health-unhealthy {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .metric-unit {
            font-size: 16px;
            color: #666;
        }
        .metric-description {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }
        .alert-critical {
            background: #f8d7da;
            border-color: #dc3545;
        }
        .alert-error {
            background: #f8d7da;
            border-color: #dc3545;
        }
        .alert-warning {
            background: #fff3cd;
            border-color: #ffc107;
        }
        .alert-info {
            background: #d1ecf1;
            border-color: #17a2b8;
        }
        .component {
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 5px;
            font-size: 14px;
        }
        .component-healthy {
            background: #d4edda;
            color: #155724;
        }
        .component-degraded {
            background: #fff3cd;
            color: #856404;
        }
        .component-unhealthy {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RAGBits Monitoring Dashboard</h1>
        <div class="timestamp">Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    </div>
"""

        # Health status
        if self._health_status:
            health_class = f"health-{self._health_status.status}"
            html += f"""
    <div class="health-status {health_class}">
        System Status: {self._health_status.status.upper()}
    </div>
"""

            # Components
            if self._health_status.components:
                html += "<div><strong>Components:</strong>"
                for comp_name, comp_status in self._health_status.components.items():
                    comp_class = f"component-{comp_status}"
                    html += f'<span class="component {comp_class}">{comp_name}: {comp_status}</span>'
                html += "</div>"

            # Issues
            if self._health_status.issues:
                html += "<div><strong>Issues:</strong><ul>"
                for issue in self._health_status.issues:
                    html += f"<li>{issue}</li>"
                html += "</ul></div>"

        # Metrics grid
        html += '<div class="grid">'

        for metric_name, metric in self._metrics.items():
            current_value = metric.get_current_value()
            if current_value is None:
                continue

            html += f"""
    <div class="card">
        <h3>{metric.name.replace('_', ' ').title()}</h3>
        <div class="metric-value">{current_value:.2f} <span class="metric-unit">{metric.unit}</span></div>
        <div class="metric-description">{metric.description}</div>
    </div>
"""

        html += "</div>"

        # Alerts
        if include_alerts:
            active_alerts = self.get_active_alerts()
            if active_alerts:
                html += '<div class="card"><h3>Active Alerts</h3>'
                for alert in active_alerts:
                    alert_class = f"alert-{alert.severity.value}"
                    html += f"""
    <div class="alert {alert_class}">
        <strong>{alert.name}</strong> ({alert.severity.value.upper()})<br>
        {alert.description}<br>
        <small>Triggered: {alert.last_triggered}</small>
    </div>
"""
                html += "</div>"
            else:
                html += '<div class="card"><h3>Active Alerts</h3><p>No active alerts</p></div>'

        html += """
</body>
</html>
"""

        return html

    def export_metrics_json(self, duration_minutes: int = 60) -> str:
        """
        Export metrics as JSON.

        Args:
            duration_minutes: Time window to export

        Returns:
            JSON string
        """
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "metrics": {}
        }

        for name, metric in self._metrics.items():
            # Filter data points within duration
            recent_points = [
                p for p in metric.data_points
                if p.timestamp >= cutoff
            ]

            if recent_points:
                export_data["metrics"][name] = {
                    "name": metric.name,
                    "type": metric.metric_type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "current_value": metric.get_current_value(),
                    "data_points": [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "value": p.value,
                            "labels": p.labels
                        }
                        for p in recent_points
                    ]
                }

        return json.dumps(export_data, indent=2)


__all__ = ["MonitoringDashboard", "Metric", "MetricPoint", "MetricType", "Alert", "AlertSeverity", "SystemHealth"]
