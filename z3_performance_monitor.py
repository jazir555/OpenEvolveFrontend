"""
Z3 Performance Monitoring and Analytics

Comprehensive monitoring for Z3 operations:
- Execution time tracking
- Memory usage monitoring
- Solver statistics collection
- Performance bottleneck identification
- Trend analysis
- Alert generation
- Reporting dashboard data

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
import threading
import time
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"  # Accumulating value
    GAUGE = "gauge"      # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution
    TIMER = "timer"      # Time duration


class Severity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    std_dev: float = 0.0
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    
    # Detailed timing
    times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_execution(self, duration: float, success: bool = True, timeout: bool = False):
        """Record operation execution."""
        self.call_count += 1
        self.total_time += duration
        self.times.append(duration)
        
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        
        if len(self.times) > 0:
            self.avg_time = statistics.mean(self.times)
            if len(self.times) > 1:
                self.std_dev = statistics.stdev(self.times)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        if timeout:
            self.timeout_count += 1
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation_name,
            "calls": self.call_count,
            "success_rate": f"{self.success_rate():.1%}",
            "avg_time_ms": f"{self.avg_time * 1000:.2f}",
            "min_time_ms": f"{self.min_time * 1000:.2f}",
            "max_time_ms": f"{self.max_time * 1000:.2f}",
            "std_dev_ms": f"{self.std_dev * 1000:.2f}",
            "errors": self.error_count,
            "timeouts": self.timeout_count
        }


@dataclass
class Alert:
    """Performance alert."""
    alert_id: str
    severity: Severity
    message: str
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.alert_id,
            "severity": self.severity.value,
            "message": self.message,
            "metric": self.metric_name,
            "threshold": self.threshold,
            "actual": self.actual_value,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance."""
    timestamp: float
    operations: Dict[str, OperationMetrics]
    active_solvers: int
    queue_depth: int
    memory_usage_mb: float
    cpu_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "operations": {
                name: metrics.to_dict()
                for name, metrics in self.operations.items()
            },
            "active_solvers": self.active_solvers,
            "queue_depth": self.queue_depth,
            "memory_mb": f"{self.memory_usage_mb:.2f}",
            "cpu_percent": f"{self.cpu_percent:.1f}%"
        }


# =============================================================================
# Performance Monitor
# =============================================================================

class Z3PerformanceMonitor:
    """
    Comprehensive performance monitoring for Z3 operations.
    
    Features:
    - Real-time metric collection
    - Historical data analysis
    - Automatic alerting
    - Performance trending
    - Resource usage tracking
    """
    
    def __init__(self, history_window: int = 3600):
        self._operation_metrics: Dict[str, OperationMetrics] = defaultdict(
            lambda: OperationMetrics(operation_name="unknown")
        )
        self._metrics_history: deque = deque(maxlen=10000)
        self._alerts: List[Alert] = []
        self._snapshots: deque = deque(maxlen=1000)
        
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Alert thresholds
        self._thresholds: Dict[str, Tuple[float, Severity]] = {
            "solve_time": (30.0, Severity.WARNING),
            "solve_time_critical": (60.0, Severity.ERROR),
            "error_rate": (0.1, Severity.WARNING),
            "error_rate_critical": (0.25, Severity.ERROR),
            "memory_mb": (1024, Severity.WARNING),
            "queue_depth": (10, Severity.WARNING)
        }
        
        # Callbacks
        self._alert_handlers: List[Callable] = []
    
    # =====================================================================
    # Metric Recording
    # =====================================================================
    
    def record_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool = True,
        timeout: bool = False,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record operation execution metrics."""
        with self._lock:
            if operation_name not in self._operation_metrics:
                self._operation_metrics[operation_name] = OperationMetrics(
                    operation_name=operation_name
                )
            
            metrics = self._operation_metrics[operation_name]
            metrics.add_execution(duration, success, timeout)
            
            # Check thresholds
            self._check_thresholds(operation_name, metrics)
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self._metrics_history.append(metric)
    
    # =====================================================================
    # Threshold Checking and Alerting
    # =====================================================================
    
    def _check_thresholds(self, operation_name: str, metrics: OperationMetrics):
        """Check if metrics exceed thresholds."""
        # Check solve time
        if metrics.avg_time > self._thresholds["solve_time_critical"][0]:
            self._create_alert(
                f"Operation {operation_name} average time is {metrics.avg_time:.2f}s",
                "solve_time_critical",
                self._thresholds["solve_time_critical"][0],
                metrics.avg_time,
                self._thresholds["solve_time_critical"][1]
            )
        elif metrics.avg_time > self._thresholds["solve_time"][0]:
            self._create_alert(
                f"Operation {operation_name} is slow: {metrics.avg_time:.2f}s avg",
                "solve_time",
                self._thresholds["solve_time"][0],
                metrics.avg_time,
                self._thresholds["solve_time"][1]
            )
        
        # Check error rate
        error_rate = metrics.error_count / max(metrics.call_count, 1)
        if error_rate > self._thresholds["error_rate_critical"][0]:
            self._create_alert(
                f"Operation {operation_name} error rate is {error_rate:.1%}",
                "error_rate_critical",
                self._thresholds["error_rate_critical"][0],
                error_rate,
                self._thresholds["error_rate_critical"][1]
            )
        elif error_rate > self._thresholds["error_rate"][0]:
            self._create_alert(
                f"Operation {operation_name} elevated errors: {error_rate:.1%}",
                "error_rate",
                self._thresholds["error_rate"][0],
                error_rate,
                self._thresholds["error_rate"][1]
            )
    
    def _create_alert(
        self,
        message: str,
        metric_name: str,
        threshold: float,
        actual_value: float,
        severity: Severity
    ):
        """Create and store alert."""
        alert_id = f"alert_{int(time.time())}_{len(self._alerts)}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            actual_value=actual_value
        )
        
        self._alerts.append(alert)
        
        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Performance alert: {message}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler callback."""
        self._alert_handlers.append(handler)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
        return False
    
    def set_threshold(self, metric: str, threshold: float, severity: Severity):
        """Set alert threshold."""
        self._thresholds[metric] = (threshold, severity)
    
    # =====================================================================
    # System Monitoring
    # =====================================================================
    
    def start_monitoring(self, interval: float = 10.0):
        """Start background monitoring thread."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started performance monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self._running:
            try:
                snapshot = self._collect_snapshot()
                
                with self._lock:
                    self._snapshots.append(snapshot)
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(interval)
    
    def _collect_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot."""
        # Get resource usage
        memory_mb = 0.0
        cpu_percent = 0.0
        
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
        except ImportError:
            pass
        
        with self._lock:
            return PerformanceSnapshot(
                timestamp=time.time(),
                operations=dict(self._operation_metrics),
                active_solvers=0,  # Would be populated by integration
                queue_depth=0,     # Would be populated by integration
                memory_usage_mb=memory_mb,
                cpu_percent=cpu_percent
            )
    
    # =====================================================================
    # Analytics
    # =====================================================================
    
    def get_operation_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of operation metrics."""
        with self._lock:
            if operation_name:
                metrics = self._operation_metrics.get(operation_name)
                return metrics.to_dict() if metrics else {}
            
            return {
                name: metrics.to_dict()
                for name, metrics in self._operation_metrics.items()
            }
    
    def get_trends(self, operation_name: str, window: int = 100) -> Dict[str, Any]:
        """Get performance trends for an operation."""
        with self._lock:
            metrics = self._operation_metrics.get(operation_name)
            if not metrics or len(metrics.times) < 2:
                return {"error": "Insufficient data"}
            
            times = list(metrics.times)[-window:]
            
            return {
                "operation": operation_name,
                "samples": len(times),
                "trend": "improving" if times[-1] < times[0] else "degrading",
                "avg_recent": statistics.mean(times[-10:]) if len(times) >= 10 else statistics.mean(times),
                "avg_overall": statistics.mean(times),
                "volatility": statistics.stdev(times) if len(times) > 1 else 0
            }
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        with self._lock:
            operations = sorted(
                self._operation_metrics.values(),
                key=lambda m: m.avg_time,
                reverse=True
            )
            
            return [
                {
                    "operation": op.operation_name,
                    "avg_time_s": op.avg_time,
                    "total_time_s": op.total_time,
                    "call_count": op.call_count,
                    "impact_score": op.avg_time * op.call_count
                }
                for op in operations[:top_n]
            ]
    
    def get_alerts(
        self,
        severity: Optional[Severity] = None,
        unacknowledged_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered."""
        with self._lock:
            alerts = self._alerts
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if unacknowledged_only:
                alerts = [a for a in alerts if not a.acknowledged]
            
            return [a.to_dict() for a in alerts]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        with self._lock:
            recent_alerts = [
                a.to_dict() for a in self._alerts[-10:]
            ]
            
            bottlenecks = self.get_bottlenecks(5)
            
            # Recent snapshots
            recent_snapshots = [
                s.to_dict() for s in list(self._snapshots)[-10:]
            ]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_operations": len(self._operation_metrics),
                    "active_alerts": len([a for a in self._alerts if not a.acknowledged]),
                    "total_calls": sum(m.call_count for m in self._operation_metrics.values()),
                    "overall_success_rate": self._calculate_overall_success_rate()
                },
                "operations": self.get_operation_summary(),
                "bottlenecks": bottlenecks,
                "recent_alerts": recent_alerts,
                "recent_snapshots": recent_snapshots
            }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all operations."""
        total_calls = sum(m.call_count for m in self._operation_metrics.values())
        total_success = sum(m.success_count for m in self._operation_metrics.values())
        
        if total_calls == 0:
            return 0.0
        return total_success / total_calls
    
    # =====================================================================
    # Reporting
    # =====================================================================
    
    def generate_report(self, period: str = "1h") -> Dict[str, Any]:
        """Generate performance report."""
        period_seconds = {
            "1h": 3600,
            "24h": 86400,
            "7d": 604800
        }.get(period, 3600)
        
        cutoff_time = time.time() - period_seconds
        
        with self._lock:
            # Filter recent data
            recent_snapshots = [
                s for s in self._snapshots
                if s.timestamp > cutoff_time
            ]
            
            recent_alerts = [
                a for a in self._alerts
                if a.timestamp > cutoff_time
            ]
            
            return {
                "period": period,
                "generated_at": datetime.utcnow().isoformat(),
                "executive_summary": {
                    "total_operations": len(self._operation_metrics),
                    "total_calls": sum(m.call_count for m in self._operation_metrics.values()),
                    "success_rate": f"{self._calculate_overall_success_rate():.1%}",
                    "alert_count": len(recent_alerts)
                },
                "operation_performance": self.get_operation_summary(),
                "top_bottlenecks": self.get_bottlenecks(10),
                "alerts_by_severity": self._count_alerts_by_severity(recent_alerts),
                "recommendations": self._generate_recommendations()
            }
    
    def _count_alerts_by_severity(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by severity."""
        counts = defaultdict(int)
        for alert in alerts:
            counts[alert.severity.value] += 1
        return dict(counts)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        with self._lock:
            # Check for slow operations
            slow_ops = [
                m for m in self._operation_metrics.values()
                if m.avg_time > self._thresholds["solve_time"][0]
            ]
            
            if slow_ops:
                recommendations.append(
                    f"Consider optimizing {len(slow_ops)} slow operations: "
                    + ", ".join(op.operation_name for op in slow_ops[:3])
                )
            
            # Check for high error rates
            error_prone = [
                m for m in self._operation_metrics.values()
                if m.error_count / max(m.call_count, 1) > 0.1
            ]
            
            if error_prone:
                recommendations.append(
                    f"Review error handling for: "
                    + ", ".join(op.operation_name for op in error_prone[:3])
                )
            
            # Check for frequent timeouts
            timeout_ops = [
                m for m in self._operation_metrics.values()
                if m.timeout_count > 0
            ]
            
            if timeout_ops:
                recommendations.append(
                    f"Consider increasing timeout or optimizing: "
                    + ", ".join(op.operation_name for op in timeout_ops[:3])
                )
        
        return recommendations


# =============================================================================
# Global Instance
# =============================================================================

_performance_monitor: Optional[Z3PerformanceMonitor] = None


def get_z3_performance_monitor() -> Z3PerformanceMonitor:
    """Get global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = Z3PerformanceMonitor()
    return _performance_monitor


# =============================================================================
# Decorator for Monitoring
# =============================================================================

def monitored(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        op_name = operation_name or func.__name__
        monitor = get_z3_performance_monitor()
        
        def wrapper(*args, **kwargs):
            start = time.time()
            success = True
            timeout = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                timeout = True
                success = False
                raise
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start
                monitor.record_operation(op_name, duration, success, timeout)
        
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            success = True
            timeout = False
            
            try:
                result = await func(*args, **kwargs)
                return result
            except TimeoutError:
                timeout = True
                success = False
                raise
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start
                monitor.record_operation(op_name, duration, success, timeout)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async_wrapper.__name__ = func.__name__
            return async_wrapper
        else:
            wrapper.__name__ = func.__name__
            return wrapper
    
    return decorator


# =============================================================================
# Example Usage
# =============================================================================

def example_monitoring():
    """Example: Performance monitoring."""
    monitor = get_z3_performance_monitor()
    
    # Record some operations
    for i in range(10):
        monitor.record_operation("solve", 0.5 + i * 0.1, success=True)
        monitor.record_operation("prove", 1.0 + i * 0.2, success=(i % 3 != 0))
    
    # Get summary
    summary = monitor.get_operation_summary()
    print("Operation Summary:")
    for name, data in summary.items():
        print(f"  {name}: {data}")
    
    # Get bottlenecks
    bottlenecks = monitor.get_bottlenecks(3)
    print("\nTop Bottlenecks:")
    for b in bottlenecks:
        print(f"  {b['operation']}: {b['avg_time_s']:.3f}s")
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print("\nDashboard Summary:")
    print(f"  Total calls: {dashboard['summary']['total_calls']}")
    print(f"  Success rate: {dashboard['summary']['overall_success_rate']}")


if __name__ == "__main__":
    print("Z3 Performance Monitor")
    print("=" * 50)
    example_monitoring()
