"""
Sovereign-Grade Problem Decomposition System - Monitoring and Observability
Implements comprehensive metrics collection, distributed tracing, and observability.
"""
from __future__ import annotations


import time
import threading
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from contextlib import contextmanager
from collections import defaultdict, deque
import psutil
import sqlite3
from contextlib import contextmanager

# **ACTUAL INTEGRATION**: Alerting and knowledge for Monitoring System
try:
    from alerting_system import get_alert_manager as get_global_alert_manager, AlertSeverity
    GLOBAL_ALERTING_AVAILABLE = True
except ImportError:
    GLOBAL_ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# **ACTUAL INTEGRATION**: Adaptive MDAP monitoring
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.utils.metrics import get_metrics as get_adaptive_metrics
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: Union[int, float]
    type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    description: str = ""


class MetricsCollector:
    """Collects and stores system and application metrics"""
    
    def __init__(self, retention_minutes: int = 1440):  # 24 hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=retention_minutes * 60))  # Assuming 1 sample per second
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        self.start_time = datetime.now()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        labels = labels or {}
        metric_key = f"{name}_" + "_".join([f"{k}={v}" for k, v in sorted(labels.items())])
        
        with self._lock:
            self.counters[metric_key] += value
            metric = Metric(
                name=name,
                value=self.counters[metric_key],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=datetime.now()
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        labels = labels or {}
        metric_key = f"{name}_" + "_".join([f"{k}={v}" for k, v in sorted(labels.items())])
        
        with self._lock:
            self.gauges[metric_key] = value
            metric = Metric(
                name=name,
                value=value,
                type=MetricType.GAUGE,
                labels=labels,
                timestamp=datetime.now()
            )
            self.metrics[name].append(metric)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram value"""
        labels = labels or {}
        metric = Metric(
            name=name,
            value=value,
            type=MetricType.HISTOGRAM,
            labels=labels,
            timestamp=datetime.now()
        )
        with self._lock:
            self.metrics[name].append(metric)
    
    def get_metric(self, name: str, labels: Dict[str, str] = None, 
                   minutes_back: int = 5) -> List[Metric]:
        """Get metric values for a specific time window"""
        cutoff = datetime.now() - timedelta(minutes=minutes_back)
        labels = labels or {}
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            filtered_metrics = [
                m for m in self.metrics[name]
                if m.timestamp >= cutoff
                and all(m.labels.get(k) == v for k, v in labels.items())
            ]
            
            return filtered_metrics
    
    def get_current_gauge_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the current value of a gauge"""
        labels = labels or {}
        metric_key = f"{name}_" + "_".join([f"{k}={v}" for k, v in sorted(labels.items())])
        with self._lock:
            return self.gauges.get(metric_key)
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the current value of a counter"""
        labels = labels or {}
        metric_key = f"{name}_" + "_".join([f"{k}={v}" for k, v in sorted(labels.items())])
        with self._lock:
            return self.counters.get(metric_key)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        output = []
        
        with self._lock:
            for name, metric_list in self.metrics.items():
                if not metric_list:
                    continue
                
                latest = metric_list[-1]
                label_str = ",".join([f'{k}="{v}"' for k, v in latest.labels.items()])
                if label_str:
                    label_str = f"{{{label_str}}}"
                
                output.append(f"# HELP {name} {latest.description or name}")
                output.append(f"# TYPE {name} {latest.type.value}")
                output.append(f"{name}{label_str} {latest.value} {int(latest.timestamp.timestamp() * 1000)}")
        
        return "\n".join(output)


class TraceSpan:
    """Represents a single span in a distributed trace"""
    
    def __init__(self, trace_id: str, span_id: str, name: str, start_time: float):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.parent_span_id: Optional[str] = None
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.status: str = "UNSET"  # UNSET, OK, ERROR
    
    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the span"""
        self.attributes[key] = value
    
    def add_event(self, name: str, timestamp: float = None, attributes: Dict[str, Any] = None):
        """Add an event to the span"""
        if timestamp is None:
            timestamp = time.time()
        
        event = {
            'name': name,
            'timestamp': timestamp,
            'attributes': attributes or {}
        }
        self.events.append(event)
    
    def end(self, status: str = "OK"):
        """End the span"""
        self.end_time = time.time()
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'parent_span_id': self.parent_span_id,
            'attributes': self.attributes,
            'events': self.events,
            'status': self.status,
            'duration': (self.end_time - self.start_time) if self.end_time else None
        }


class DistributedTracer:
    """Implements distributed tracing across system components"""
    
    def __init__(self):
        self.spans: Dict[str, TraceSpan] = {}
        self.traces: Dict[str, List[str]] = defaultdict(list)  # trace_id -> [span_ids]
        self._lock = threading.Lock()
    
    def start_span(self, name: str, parent_span: Optional[TraceSpan] = None) -> TraceSpan:
        """Start a new span"""
        trace_id = parent_span.trace_id if parent_span else str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            start_time=time.time()
        )
        
        if parent_span:
            span.parent_span_id = parent_span.span_id
        
        with self._lock:
            self.spans[span_id] = span
            self.traces[trace_id].append(span_id)
        
        return span
    
    def end_span(self, span: TraceSpan, status: str = "OK"):
        """End a span"""
        span.end(status)
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace"""
        with self._lock:
            span_ids = self.traces.get(trace_id, [])
            return [self.spans[span_id] for span_id in span_ids if span_id in self.spans]
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent traces"""
        with self._lock:
            # Get the most recent traces by looking at recent span timestamps
            recent_trace_ids = list(self.traces.keys())[-limit:]
            return {
                trace_id: [span.to_dict() for span in self.get_trace(trace_id)]
                for trace_id in recent_trace_ids
            }
    
    def export_spans(self) -> List[Dict[str, Any]]:
        """Export all spans as dictionaries"""
        with self._lock:
            return [span.to_dict() for span in self.spans.values()]


class AlertRule:
    """Defines an alert rule for monitoring"""
    
    def __init__(self, name: str, metric_name: str, condition: Callable[[float], bool], 
                 description: str, severity: str = "warning"):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition  # Function that takes value and returns True if alert should fire
        self.description = description
        self.severity = severity  # info, warning, critical
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0


class AlertManager:
    """Manages alert rules and firing"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self._lock:
            self.rules.append(rule)
    
    def check_metrics(self, metrics_collector: MetricsCollector):
        """Check all metrics against alert rules"""
        with self._lock:
            for rule in self.rules:
                # Get the most recent value for the metric
                metric_values = metrics_collector.get_metric(rule.metric_name, minutes_back=1)
                if metric_values:
                    latest_value = metric_values[-1].value
                    if rule.condition(latest_value):
                        alert = {
                            'rule_name': rule.name,
                            'metric_name': rule.metric_name,
                            'metric_value': latest_value,
                            'timestamp': datetime.now(),
                            'severity': rule.severity,
                            'description': rule.description
                        }
                        
                        self.active_alerts[rule.name].append(alert)
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1
                        
                        # Log the alert
                        logger.warning(f"ALERT: {rule.name} - {rule.description} (value: {latest_value})")
    
    def get_active_alerts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all active alerts"""
        with self._lock:
            return dict(self.active_alerts)
    
    def clear_alert(self, rule_name: str):
        """Clear alerts for a specific rule"""
        with self._lock:
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]


class PerformanceProfiler:
    """Performance profiling for debugging and optimization"""
    
    def __init__(self):
        self.function_timings: Dict[str, List[float]] = defaultdict(list)
        self.function_calls: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    @contextmanager
    def profile_function(self, name: str):
        """Context manager to profile a function call"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self._lock:
                self.function_timings[name].append(duration)
                self.function_calls[name] += 1
    
    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics"""
        with self._lock:
            stats = {}
            for func_name in self.function_timings:
                timings = self.function_timings[func_name]
                stats[func_name] = {
                    'calls': self.function_calls[func_name],
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings) if timings else 0,
                    'min_time': min(timings) if timings else 0,
                    'max_time': max(timings) if timings else 0
                }
            return stats
    
    def reset(self):
        """Reset profiling data"""
        with self._lock:
            self.function_timings.clear()
            self.function_calls.clear()


class MonitoringDashboard:
    """Real-time dashboard for monitoring key metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._dashboard_data: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def update_dashboard(self):
        """Update dashboard with current metrics"""
        with self._lock:
            # System metrics
            self._dashboard_data['system'] = self.metrics_collector.get_system_metrics()['system']
            
            # Recent metrics for key performance indicators
            self._dashboard_data['workflow_metrics'] = {
                'active_workflows': self.metrics_collector.get_current_gauge_value('active_workflows') or 0,
                'completed_workflows': self.metrics_collector.get_counter_value('completed_workflows') or 0,
                'failed_workflows': self.metrics_collector.get_counter_value('failed_workflows') or 0,
            }
            
            self._dashboard_data['problem_metrics'] = {
                'problems_analyzed': self.metrics_collector.get_counter_value('problems_analyzed') or 0,
                'decomposition_plans_created': self.metrics_collector.get_counter_value('decomposition_plans_created') or 0,
                'solutions_generated': self.metrics_collector.get_counter_value('solutions_generated') or 0,
            }
            
            self._dashboard_data['performance_metrics'] = {
                'avg_problem_analysis_time': self._get_average_metric('problem_analysis_duration'),
                'avg_solution_generation_time': self._get_average_metric('solution_generation_duration'),
                'avg_decomposition_time': self._get_average_metric('decomposition_duration'),
            }
    
    def _get_average_metric(self, metric_name: str, minutes_back: int = 5) -> float:
        """Get average value for a metric over time period"""
        metrics = self.metrics_collector.get_metric(metric_name, minutes_back=minutes_back)
        if metrics:
            return sum(m.value for m in metrics) / len(metrics)
        return 0.0
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        with self._lock:
            return dict(self._dashboard_data)
    
    def get_trend_data(self, metric_name: str, minutes_back: int = 60) -> List[Dict[str, Any]]:
        """Get trend data for a metric"""
        metrics = self.metrics_collector.get_metric(metric_name, minutes_back=minutes_back)
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'value': m.value
            }
            for m in metrics
        ]


class ObservabilityManager:
    """Main observability management class"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.tracer = DistributedTracer()
        self.alert_manager = AlertManager()
        self.profiler = PerformanceProfiler()
        self.dashboard = MonitoringDashboard(self.metrics_collector)
        self._monitoring = False
        self._monitor_thread = None
        
        # Set up default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Set up default alert rules"""
        # CPU usage alert
        cpu_rule = AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_percent",
            condition=lambda x: x > 90,
            description="CPU usage is above 90%",
            severity="warning"
        )
        self.alert_manager.add_rule(cpu_rule)
        
        # Memory usage alert
        memory_rule = AlertRule(
            name="high_memory_usage",
            metric_name="system_memory_percent",
            condition=lambda x: x > 90,
            description="Memory usage is above 90%",
            severity="warning"
        )
        self.alert_manager.add_rule(memory_rule)

        # Workflow-specific alerts
        gauntlet_failure_rule = AlertRule(
            name="gauntlet_failures_detected",
            metric_name="gauntlet_failures_total",
            condition=lambda x: x > 0,
            description="One or more gauntlet runs have failed",
            severity="critical"
        )
        self.alert_manager.add_rule(gauntlet_failure_rule)

        retry_rule = AlertRule(
            name="workflow_retries_detected",
            metric_name="workflow_retries_total",
            condition=lambda x: x > 0,
            description="Workflow retries have been triggered",
            severity="warning"
        )
        self.alert_manager.add_rule(retry_rule)

        timeout_rule = AlertRule(
            name="workflow_timeouts_detected",
            metric_name="workflow_timeouts_total",
            condition=lambda x: x > 0,
            description="Workflow timeouts have been detected",
            severity="critical"
        )
        self.alert_manager.add_rule(timeout_rule)
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started observability monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Stopped observability monitoring")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Update system metrics
                system_metrics = self.metrics_collector.get_system_metrics()
                for key, value in system_metrics['system'].items():
                    if isinstance(value, (int, float)):
                        self.metrics_collector.set_gauge(f"system_{key}", value)

                # Update dashboard
                self.dashboard.update_dashboard()

                # Check alert rules
                triggered_alerts = self.alert_manager.check_metrics(self.metrics_collector)

                # **ACTUAL INTEGRATION**: Extract knowledge and trigger alerts for critical monitoring events
                if triggered_alerts:
                    # Get dashboard data for knowledge extraction
                    dashboard_data = self.dashboard.get_dashboard_data()
                    self._extract_monitoring_knowledge("monitoring_loop", {
                        "alerts_triggered": len(triggered_alerts),
                        "system_metrics": system_metrics,
                        "dashboard_data": dashboard_data
                    })

                    # Trigger global alerts for critical issues
                    critical_alerts = [a for a in triggered_alerts if a.get('severity') == 'critical']
                    if critical_alerts:
                        self._trigger_monitoring_alerts(
                            "monitoring_loop",
                            False,
                            None,
                            f"Critical alerts triggered: {len(critical_alerts)}",
                            {"alerts": critical_alerts}
                        )

                time.sleep(interval)
            except (OSError, IOError, RuntimeError, ValueError) as e:
                logger.error(f"Monitoring loop error: {e}")
                # **ACTUAL INTEGRATION**: Trigger alert for monitoring loop error
                self._trigger_monitoring_alerts("monitoring_loop", False, None, str(e))
                time.sleep(interval)
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Context manager for tracing operations"""
        span = self.tracer.start_span(operation_name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
            # **ACTUAL INTEGRATION**: Track successful operation
            self._track_monitoring_performance(f"trace_{operation_name}", True)
            span.end("OK")
        except (OSError, IOError, RuntimeError, ValueError, TypeError) as e:
            span.set_attribute("error", str(e))
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_monitoring_alerts(f"trace_{operation_name}", False, operation_name, str(e))
            self._track_monitoring_performance(f"trace_{operation_name}", False)
            span.end("ERROR")
            raise
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType, 
                         labels: Dict[str, str] = None):
        """Add a custom metric"""
        if metric_type == MetricType.COUNTER:
            self.metrics_collector.increment_counter(name, value, labels)
        elif metric_type == MetricType.GAUGE:
            self.metrics_collector.set_gauge(name, value, labels)
        elif metric_type == MetricType.HISTOGRAM:
            self.metrics_collector.observe_histogram(name, value, labels)
    
    def get_metrics(self, name: str, labels: Dict[str, str] = None, 
                   minutes_back: int = 5) -> List[Metric]:
        """Get metrics for a specific name and labels"""
        return self.metrics_collector.get_metric(name, labels, minutes_back)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return self.metrics_collector.get_system_metrics()
    
    def get_traces(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get traces for a specific trace ID"""
        return [span.to_dict() for span in self.tracer.get_trace(trace_id)]
    
    def get_recent_traces(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent traces"""
        return self.tracer.get_recent_traces(limit)
    
    def get_alerts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()
    
    def get_profile_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance profiling statistics"""
        return self.profiler.get_profile_stats()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return self.dashboard.get_dashboard_data()
    
    def get_trend_data(self, metric_name: str, minutes_back: int = 60) -> List[Dict[str, Any]]:
        """Get trend data for a metric"""
        return self.dashboard.get_trend_data(metric_name, minutes_back)
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return self.metrics_collector.export_prometheus_format()

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Monitoring System
    # =========================================================================

    def _trigger_monitoring_alerts(
        self,
        operation: str,
        success: bool,
        metric_name: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for monitoring system failures."""
        if not GLOBAL_ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_global_alert_manager()

            if not success:
                severity = AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Monitoring System Alert: {operation}",
                    description=f"Monitoring operation '{operation}' failed" +
                                 (f" for metric '{metric_name}'" if metric_name else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="monitoring_system",
                    component="observability",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Monitoring alert: {e}")

    def _extract_monitoring_knowledge(
        self,
        operation: str,
        metrics_data: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract monitoring knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"monitoring_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="monitoring_metrics",
                source_component="monitoring_system",
                title=f"Monitoring Metrics: {operation}",
                content={
                    "operation": operation,
                    "metrics_summary": metrics_data,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "num_metrics": len(metrics_data.get("metrics", {})),
                    "has_alerts": len(metrics_data.get("alerts", {})) > 0
                },
                tags=["monitoring", "observability", operation, "metrics"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Monitoring knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Monitoring knowledge: {e}")
            return False

    def _track_monitoring_performance(
        self,
        operation: str,
        success: bool
    ):
        """**ACTUAL INTEGRATION**: Track monitoring operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"monitoring_system_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Monitoring performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Monitoring performance: {e}")


# Global observability manager instance
_observability_manager = None


def get_observability_manager() -> ObservabilityManager:
    """Get the observability manager instance"""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
        _observability_manager.start_monitoring()
    return _observability_manager


def start_observing():
    """Start the observability system"""
    manager = get_observability_manager()
    manager.start_monitoring()


def stop_observing():
    """Stop the observability system"""
    manager = get_observability_manager()
    manager.stop_monitoring()


def add_metric(name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
    """Add a metric"""
    get_observability_manager().add_custom_metric(name, value, metric_type, labels)


def trace_operation(operation_name: str, attributes: Dict[str, Any] = None):
    """Get a context manager for tracing an operation"""
    return get_observability_manager().trace_operation(operation_name, attributes)


def get_current_metrics() -> Dict[str, Any]:
    """Get current metrics"""
    return get_observability_manager().get_dashboard_data()


def get_alerts() -> Dict[str, List[Dict[str, Any]]]:
    """Get active alerts"""
    return get_observability_manager().get_alerts()


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get performance statistics"""
    return get_observability_manager().get_profile_stats()


def get_trend_data(metric_name: str, minutes_back: int = 60) -> List[Dict[str, Any]]:
    """Get trend data"""
    return get_observability_manager().get_trend_data(metric_name, minutes_back)


# =============================================================================
# ADAPTIVE MDAP MONITORING
# =============================================================================

def record_adaptive_classification(
    subproblem_id: str,
    complexity_score: float,
    latency_ms: float,
    success: bool = True
):
    """
    Record Adaptive MDAP classification metrics.
    
    Args:
        subproblem_id: Sub-problem ID
        complexity_score: Computed complexity score
        latency_ms: Classification latency in milliseconds
        success: Whether classification succeeded
    """
    add_metric(
        "adaptive_classification_total",
        1,
        MetricType.COUNTER,
        {"subproblem_id": subproblem_id, "success": str(success)}
    )
    
    add_metric(
        "adaptive_complexity_score",
        complexity_score,
        MetricType.GAUGE,
        {"subproblem_id": subproblem_id}
    )
    
    add_metric(
        "adaptive_classification_latency_ms",
        latency_ms,
        MetricType.HISTOGRAM,
        {"subproblem_id": subproblem_id}
    )


def record_adaptive_allocation(
    subproblem_id: str,
    strategy: str,
    n_agents: int,
    k_ahead: int,
    latency_ms: float,
    success: bool = True
):
    """
    Record Adaptive MDAP allocation metrics.
    
    Args:
        subproblem_id: Sub-problem ID
        strategy: Allocated strategy name
        n_agents: Number of agents allocated
        k_ahead: K-ahead value
        latency_ms: Allocation latency in milliseconds
        success: Whether allocation succeeded
    """
    add_metric(
        "adaptive_allocation_total",
        1,
        MetricType.COUNTER,
        {"subproblem_id": subproblem_id, "strategy": strategy, "success": str(success)}
    )
    
    add_metric(
        "adaptive_allocated_agents",
        n_agents,
        MetricType.GAUGE,
        {"subproblem_id": subproblem_id, "strategy": strategy}
    )
    
    add_metric(
        "adaptive_allocation_latency_ms",
        latency_ms,
        MetricType.HISTOGRAM,
        {"subproblem_id": subproblem_id}
    )


def get_adaptive_metrics() -> Dict[str, Any]:
    """
    Get Adaptive MDAP specific metrics.
    
    Returns:
        Dictionary with Adaptive MDAP metrics
    """
    if not ADAPTIVE_MDAP_AVAILABLE:
        return {"adaptive_mdap_available": False}
    
    try:
        from adaptive_mdap.utils.metrics import get_metrics
        from adaptive_mdap.utils.cache import get_cache_stats
        
        metrics = get_metrics()
        cache_stats = get_cache_stats()
        
        return {
            "adaptive_mdap_available": True,
            "classifications": metrics.get("classifications", {}),
            "allocations": metrics.get("allocations", {}),
            "cache_stats": cache_stats,
            "performance": {
                "avg_classification_latency_ms": metrics.get("avg_classification_latency_ms", 0),
                "avg_allocation_latency_ms": metrics.get("avg_allocation_latency_ms", 0)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get Adaptive MDAP metrics: {e}")
        return {"adaptive_mdap_available": True, "error": str(e)}


# Example usage
if __name__ == "__main__":
    import uuid
    
    # Start observability
    obs_manager = get_observability_manager()
    
    # Add some metrics
    add_metric("active_workflows", 5, MetricType.GAUGE)
    add_metric("completed_workflows", 1, MetricType.COUNTER)
    add_metric("problems_analyzed", 3, MetricType.COUNTER)
    add_metric("problem_analysis_duration", 2.5, MetricType.HISTOGRAM)
    
    # Trace an operation
    with trace_operation("problem_analysis", {"problem_id": "prob_123"}) as span:
        span.add_event("data_loaded", attributes={"size": "large"})
        time.sleep(0.1)  # Simulate work
        span.set_attribute("result", "success")
    
    # Get dashboard data
    dashboard_data = get_current_metrics()
    print("Dashboard data:", json.dumps(dashboard_data, indent=2)[:500])
    
    # Get alerts
    alerts = get_alerts()
    print(f"Active alerts: {len(alerts)}")
    
    # Get performance stats
    perf_stats = get_performance_stats()
    print(f"Performance stats keys: {list(perf_stats.keys())}")
    
    # Get trend data
    trend_data = get_trend_data("system_cpu_percent", minutes_back=1)
    print(f"Trend data: {trend_data}")
    
    print("Monitoring and observability system implemented successfully!")
