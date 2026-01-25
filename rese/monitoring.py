"""
RESE Monitoring and Metrics System

Comprehensive monitoring for RESE pipeline with:
- Real-time metrics collection
- Performance monitoring
- ACI tracking
- Error tracking
- Dashboard generation

Author: Agent Z1 (Integration Specialist)
Created: 2025-12-31
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import json

from config import RESEConfig, get_config, MonitoringConfig


# =============================================================================
# Monitoring Data Structures
# =============================================================================

class MetricType(Enum):
    """Types of metrics"""
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
class Metric:
    """A single metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """An alert event"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a phase or pipeline"""
    total_duration_seconds: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class ACITracking:
    """ACI (Algorithmic Complexity of Information) tracking"""
    pipeline_id: str
    baseline_aci: float = 0.0
    current_aci: float = 0.0
    aci_reduction: float = 0.0
    reduction_percentage: float = 0.0
    history: List[float] = field(default_factory=list)
    phase_aci: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Collects and stores metrics from RESE pipeline execution.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        self._lock = threading.Lock()

    def record_metric(self, metric: Metric) -> None:
        """
        Record a metric data point.

        Args:
            metric: Metric to record
        """
        with self._lock:
            self.metrics[metric.name].append(metric)

            # Update counters/gauges
            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric.name] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric.name] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.histograms[metric.name].append(metric.value)

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=MetricType.COUNTER
        )
        self.record_metric(metric)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self.record_metric(metric)

    def record_timing(self, name: str, duration_seconds: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timing metric.

        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            labels: Optional labels
        """
        metric = Metric(
            name=name,
            value=duration_seconds,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=MetricType.HISTOGRAM
        )
        self.record_metric(metric)

    def get_metric(self, name: str) -> Optional[List[Metric]]:
        """
        Get all data points for a metric.

        Args:
            name: Metric name

        Returns:
            List of metric data points
        """
        with self._lock:
            if name in self.metrics:
                return list(self.metrics[name])
        return None

    def get_counter(self, name: str) -> float:
        """Get current counter value"""
        with self._lock:
            return self.counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        """Get current gauge value"""
        with self._lock:
            return self.gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """
        Get histogram statistics.

        Args:
            name: Histogram name

        Returns:
            Statistics (count, sum, avg, min, max, p50, p95, p99)
        """
        with self._lock:
            if name not in self.histograms:
                return {}

            values = sorted(self.histograms[name])
            if not values:
                return {}

            count = len(values)
            sum_val = sum(values)

            return {
                'count': count,
                'sum': sum_val,
                'avg': sum_val / count if count > 0 else 0,
                'min': values[0],
                'max': values[-1],
                'p50': values[int(count * 0.5)],
                'p95': values[int(count * 0.95)],
                'p99': values[int(count * 0.99)]
            }

    def clear(self) -> None:
        """Clear all metrics"""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """
    Monitors system performance during pipeline execution.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)

        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            self.psutil = None
            self.process = None
            print("Warning: psutil not available. Install with: pip install psutil")

    def start_monitoring(self, pipeline_id: str) -> None:
        """
        Start monitoring for a pipeline.

        Args:
            pipeline_id: Pipeline identifier
        """
        self.metrics_collector.set_gauge(
            f"pipeline.{pipeline_id}.running",
            1.0
        )

    def stop_monitoring(self, pipeline_id: str) -> None:
        """
        Stop monitoring for a pipeline.

        Args:
            pipeline_id: Pipeline identifier
        """
        self.metrics_collector.set_gauge(
            f"pipeline.{pipeline_id}.running",
            0.0
        )

    def record_phase_start(self, pipeline_id: str, phase_name: str) -> None:
        """Record phase start"""
        self.metrics_collector.set_gauge(
            f"pipeline.{pipeline_id}.phase.{phase_name}.running",
            1.0
        )

    def record_phase_end(
        self,
        pipeline_id: str,
        phase_name: str,
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record phase completion"""
        self.metrics_collector.set_gauge(
            f"pipeline.{pipeline_id}.phase.{phase_name}.running",
            0.0
        )

        self.metrics_collector.record_timing(
            f"pipeline.{pipeline_id}.phase.{phase_name}.duration",
            duration_seconds
        )

        self.metrics_collector.increment(
            f"pipeline.{pipeline_id}.phase.{phase_name}.completion",
            1.0 if success else 0.0
        )

    def get_current_performance(self) -> PerformanceMetrics:
        """
        Get current system performance metrics.

        Returns:
            PerformanceMetrics
        """
        if self.psutil and self.process:
            # CPU and Memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            # Disk I/O
            io_counters = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
            disk_io_mb = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024) if io_counters else 0.0

            # Network I/O
            network_io_mb = 0.0  # Process-level network I/O not available

            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                disk_io_mb=disk_io_mb,
                network_io_mb=network_io_mb
            )
        else:
            return PerformanceMetrics()

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        lines = []

        # Counters
        for name, value in self.metrics_collector.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in self.metrics_collector.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Histograms
        for name, values in self.metrics_collector.histograms.items():
            if values:
                stats = self.metrics_collector.get_histogram_stats(name)
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats.get('count', 0)}")
                lines.append(f"{name}_sum {stats.get('sum', 0)}")

        return "\n".join(lines)


# =============================================================================
# ACI Tracker
# =============================================================================

class ACITracker:
    """
    Tracks ACI (Algorithmic Complexity of Information) through pipeline phases.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.tracked_pipelines: Dict[str, ACITracking] = {}

    def start_tracking(self, pipeline_id: str, baseline_aci: float) -> None:
        """
        Start tracking ACI for a pipeline.

        Args:
            pipeline_id: Pipeline identifier
            baseline_aci: Initial ACI value
        """
        tracking = ACITracking(
            pipeline_id=pipeline_id,
            baseline_aci=baseline_aci,
            current_aci=baseline_aci
        )

        self.tracked_pipelines[pipeline_id] = tracking

    def update_aci(self, pipeline_id: str, phase: str, aci_value: float) -> None:
        """
        Update ACI value for a phase.

        Args:
            pipeline_id: Pipeline identifier
            phase: Phase name
            aci_value: Current ACI value
        """
        if pipeline_id not in self.tracked_pipelines:
            return

        tracking = self.tracked_pipelines[pipeline_id]
        tracking.current_aci = aci_value
        tracking.history.append(aci_value)
        tracking.phase_aci[phase] = aci_value

        # Calculate reduction
        tracking.aci_reduction = tracking.baseline_aci - tracking.current_aci
        tracking.reduction_percentage = (
            tracking.aci_reduction / tracking.baseline_aci * 100
            if tracking.baseline_aci > 0 else 0
        )

    def get_tracking(self, pipeline_id: str) -> Optional[ACITracking]:
        """Get ACI tracking for a pipeline"""
        return self.tracked_pipelines.get(pipeline_id)

    def get_aci_reduction_summary(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get ACI reduction summary.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Summary dictionary
        """
        tracking = self.tracked_pipelines.get(pipeline_id)
        if not tracking:
            return None

        return {
            'pipeline_id': pipeline_id,
            'baseline_aci': tracking.baseline_aci,
            'final_aci': tracking.current_aci,
            'absolute_reduction': tracking.aci_reduction,
            'relative_reduction_percent': tracking.reduction_percentage,
            'phase_aci': tracking.phase_aci,
            'history': tracking.history,
            'meets_threshold': tracking.reduction_percentage >= 20.0  # 20% threshold
        }


# =============================================================================
# Error Tracker
# =============================================================================

class ErrorTracker:
    """
    Tracks errors and exceptions during pipeline execution.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)

    def record_error(
        self,
        pipeline_id: str,
        phase: str,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None
    ) -> None:
        """
        Record an error.

        Args:
            pipeline_id: Pipeline identifier
            phase: Phase where error occurred
            error_type: Type of error
            error_message: Error message
            traceback: Optional traceback
        """
        error = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_id': pipeline_id,
            'phase': phase,
            'error_type': error_type,
            'error_message': error_message,
            'traceback': traceback
        }

        self.errors.append(error)
        self.error_counts[error_type] += 1

    def get_errors(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get errors, optionally filtered by pipeline.

        Args:
            pipeline_id: Optional pipeline filter

        Returns:
            List of errors
        """
        if pipeline_id:
            return [e for e in self.errors if e['pipeline_id'] == pipeline_id]
        return self.errors

    def get_error_summary(self) -> Dict[str, int]:
        """Get error summary by type"""
        return dict(self.error_counts)

    def clear(self) -> None:
        """Clear all errors"""
        self.errors.clear()
        self.error_counts.clear()


# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """
    Manages alert generation and notification.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add callback for alert notifications.

        Args:
            callback: Function to call on alert
        """
        self.alert_callbacks.append(callback)

    def check_thresholds(
        self,
        metrics_collector: MetricsCollector,
        aci_tracker: ACITracker,
        error_tracker: ErrorTracker
    ) -> None:
        """
        Check all thresholds and generate alerts.

        Args:
            metrics_collector: Metrics collector
            aci_tracker: ACI tracker
            error_tracker: Error tracker
        """
        # Check ACI threshold
        if hasattr(self.config, 'alert_threshold_aci'):
            for pipeline_id, tracking in aci_tracker.tracked_pipelines.items():
                if tracking.reduction_percentage < self.config.alert_threshold_aci * 100:
                    self._create_alert(
                        severity=AlertSeverity.WARNING,
                        title=f"Low ACI Reduction: {pipeline_id}",
                        description=f"ACI reduction {tracking.reduction_percentage:.1f}% below threshold {self.config.alert_threshold_aci * 100}%",
                        metric_name="aci_reduction",
                        current_value=tracking.reduction_percentage,
                        threshold=self.config.alert_threshold_aci * 100
                    )

        # Check error rate
        if hasattr(self.config, 'alert_threshold_error_rate'):
            error_summary = error_tracker.get_error_summary()
            total_errors = sum(error_summary.values())

            if total_errors > 0:
                error_rate = total_errors / max(len(error_tracker.errors), 1)
                if error_rate > self.config.alert_threshold_error_rate:
                    self._create_alert(
                        severity=AlertSeverity.ERROR,
                        title="High Error Rate",
                        description=f"Error rate {error_rate:.2%} exceeds threshold {self.config.alert_threshold_error_rate:.2%}",
                        metric_name="error_rate",
                        current_value=error_rate,
                        threshold=self.config.alert_threshold_error_rate
                    )

    def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        metric_name: str,
        current_value: float,
        threshold: float
    ) -> None:
        """Create and process an alert"""
        import uuid

        alert = Alert(
            id=f"alert_{uuid.uuid4().hex[:8]}",
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )

        self.alerts.append(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [a for a in self.alerts if not a.resolved]

    def resolve_alert(self, alert_id: str) -> None:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()


# =============================================================================
# Monitoring System
# =============================================================================

class MonitoringSystem:
    """
    Main monitoring system that integrates all monitoring components.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize monitoring system.

        Args:
            config: Optional monitoring configuration
        """
        from config import get_config

        rese_config = get_config()
        self.config = config or rese_config.monitoring

        self.performance_monitor = PerformanceMonitor(self.config)
        self.aci_tracker = ACITracker(self.config)
        self.error_tracker = ErrorTracker(self.config)
        self.alert_manager = AlertManager(self.config)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = self.config.log_file
        log_path = Path(log_file) if log_file else None

        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file if log_path else None
        )

        self.logger = logging.getLogger('RESE')

    def setup_pipeline_monitoring(self, pipeline_id: str, baseline_aci: float) -> None:
        """
        Setup monitoring for a new pipeline.

        Args:
            pipeline_id: Pipeline identifier
            baseline_aci: Initial ACI value
        """
        self.performance_monitor.start_monitoring(pipeline_id)
        self.aci_tracker.start_tracking(pipeline_id, baseline_aci)

        self.logger.info(f"Started monitoring for pipeline {pipeline_id}")

    def record_phase_completion(
        self,
        pipeline_id: str,
        phase: str,
        duration_seconds: float,
        success: bool,
        aci_value: Optional[float] = None
    ) -> None:
        """
        Record phase completion metrics.

        Args:
            pipeline_id: Pipeline identifier
            phase: Phase name
            duration_seconds: Duration of phase
            success: Whether phase succeeded
            aci_value: Optional ACI value after phase
        """
        self.performance_monitor.record_phase_end(
            pipeline_id,
            phase,
            duration_seconds,
            success
        )

        if aci_value is not None:
            self.aci_tracker.update_aci(pipeline_id, phase, aci_value)

        self.logger.info(
            f"Phase {phase} completed for pipeline {pipeline_id}: "
            f"duration={duration_seconds:.2f}s, success={success}"
        )

    def record_error(
        self,
        pipeline_id: str,
        phase: str,
        error: Exception
    ) -> None:
        """
        Record an error.

        Args:
            pipeline_id: Pipeline identifier
            phase: Phase where error occurred
            error: Exception
        """
        import traceback

        self.error_tracker.record_error(
            pipeline_id=pipeline_id,
            phase=phase,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc()
        )

        self.logger.error(
            f"Error in pipeline {pipeline_id}, phase {phase}: {error}"
        )

    def check_alerts(self) -> None:
        """Check all alert thresholds"""
        self.alert_manager.check_thresholds(
            self.performance_monitor.metrics_collector,
            self.aci_tracker,
            self.error_tracker
        )

    def get_dashboard_data(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get dashboard data for a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Dashboard data dictionary
        """
        performance = self.performance_monitor.get_current_performance()
        aci_tracking = self.aci_tracker.get_tracking(pipeline_id)
        aci_summary = self.aci_tracker.get_aci_reduction_summary(pipeline_id) if aci_tracking else None
        errors = self.error_tracker.get_errors(pipeline_id)
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            'pipeline_id': pipeline_id,
            'performance': {
                'cpu_percent': performance.cpu_percent,
                'memory_mb': performance.memory_mb,
                'disk_io_mb': performance.disk_io_mb
            },
            'aci': aci_summary,
            'errors': errors[-10:],  # Last 10 errors
            'active_alerts': len(active_alerts),
            'timestamp': datetime.now().isoformat()
        }

    def generate_metrics_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.

        Returns:
            Metrics report dictionary
        """
        return {
            'performance': self.performance_monitor.get_current_performance(),
            'prometheus_metrics': self.performance_monitor.get_prometheus_metrics(),
            'error_summary': self.error_tracker.get_error_summary(),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'MonitoringSystem',
    'PerformanceMonitor',
    'ACITracker',
    'ErrorTracker',
    'AlertManager',
    'MetricsCollector',

    # Data structures
    'Metric',
    'Alert',
    'PerformanceMetrics',
    'ACITracking',

    # Enums
    'MetricType',
    'AlertSeverity',
]
