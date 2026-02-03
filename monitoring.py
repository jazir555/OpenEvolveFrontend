"""
Sovereign-Grade Problem Decomposition System - Metrics Collection & Monitoring
Implements comprehensive metrics collection, observability, and monitoring capabilities.
"""

import time
import threading
import queue
from typing import Dict, Any, List, Callable, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from threading import Lock
import psutil
import os
import sqlite3
from contextlib import contextmanager
import atexit

# **ACTUAL INTEGRATION**: Adaptive MDAP for monitoring complexity analysis
try:
    from adaptive_mdap import TaskComplexityClassifier
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    SubProblem = None


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"        # Monotonically increasing values
    GAUGE = "gauge"            # Current state values
    HISTOGRAM = "histogram"    # Distribution of values
    SUMMARY = "summary"        # Quantiles and summaries
    TIMER = "timer"            # Timing measurements


@dataclass
class Metric:
    """Data class for metrics."""
    name: str
    value: Union[int, float]
    type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.type.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description
        }


class MetricsCollector:
    """Collects and stores metrics for the system."""
    
    def __init__(self, db_path: str = "metrics.db", retention_days: int = 30):
        """
        Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database for storing metrics
            retention_days: Number of days to retain metrics
        """
        self.db_path = db_path
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)
        self.lock = Lock()
        self._init_db()
        
        # Initialize system metrics tracking
        self.start_time = datetime.now()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def _init_db(self):
        """Initialize metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    type TEXT NOT NULL,
                    labels TEXT,  -- JSON string
                    timestamp TEXT NOT NULL,
                    description TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name_timestamp ON metrics(name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)")
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        metric_type: MetricType, 
        labels: Dict[str, str] = None,
        description: str = None
    ) -> bool:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels for the metric
            description: Optional description
            
        Returns:
            True if recorded successfully
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO metrics (name, value, type, labels, timestamp, description)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        value,
                        metric_type.value,
                        json.dumps(labels or {}),
                        datetime.now().isoformat(),
                        description
                    ))
                    conn.commit()
                    return True
        except (sqlite3.Error, OSError, IOError, TypeError) as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
            return False
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None, description: str = None):
        """Increment a counter metric."""
        # For counters, we'll record the current count in a separate tracking table
        # For now, just record a value of 1 to indicate an event occurred
        return self.record_metric(name, 1, MetricType.COUNTER, labels, description)
    
    def set_gauge(self, name: str, value: Union[int, float], labels: Dict[str, str] = None, description: str = None):
        """Set a gauge metric to a specific value."""
        return self.record_metric(name, value, MetricType.GAUGE, labels, description)
    
    def observe_histogram(self, name: str, value: Union[int, float], labels: Dict[str, str] = None, description: str = None):
        """Record a value in a histogram."""
        return self.record_metric(name, value, MetricType.HISTOGRAM, labels, description)
    
    def time_function(self, name: str, labels: Dict[str, str] = None, description: str = None) -> Callable:
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.record_metric(f"{name}_duration_seconds", execution_time, MetricType.TIMER, labels, description)
                    return result
                except (ValueError, TypeError, RuntimeError, OSError, IOError) as e:
                    self.record_metric(f"{name}_error_total", 1, MetricType.COUNTER, labels, f"Error in {name}")
                    raise e
            return wrapper
        return decorator
    
    def get_metrics(self, name: str = None, start_time: datetime = None, end_time: datetime = None) -> List[Metric]:
        """Retrieve metrics from the database."""
        query = "SELECT name, value, type, labels, timestamp, description FROM metrics WHERE 1=1"
        params = []
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT 1000"  # Limit results
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                results.append(Metric(
                    name=row[0],
                    value=row[1],
                    type=MetricType(row[2]),
                    labels=json.loads(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                    description=row[5]
                ))
        
        return results
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get a summary of a specific metric."""
        with sqlite3.connect(self.db_path) as conn:
            # Get basic stats
            cursor = conn.execute("""
                SELECT 
                    MIN(value) as min_val,
                    MAX(value) as max_val,
                    AVG(value) as avg_val,
                    COUNT(*) as count
                FROM metrics 
                WHERE name = ?
            """, (name,))
            
            min_val, max_val, avg_val, count = cursor.fetchone()
            
            # Get the latest value
            cursor = conn.execute("""
                SELECT value, timestamp 
                FROM metrics 
                WHERE name = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (name,))
            
            latest_row = cursor.fetchone()
            latest_value = latest_row[0] if latest_row else None
            latest_time = datetime.fromisoformat(latest_row[1]) if latest_row else None
            
            return {
                'name': name,
                'count': count,
                'min': min_val,
                'max': max_val,
                'average': avg_val,
                'latest_value': latest_value,
                'latest_timestamp': latest_time.isoformat() if latest_time else None
            }
    
    def _cleanup_old_metrics(self):
        """Periodically clean up old metrics."""
        while True:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM metrics WHERE timestamp < ?",
                        (cutoff_date.isoformat(),)
                    )
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        self.logger.info(f"Cleaned up {deleted_count} old metrics")
                
                # Sleep for 1 hour before next cleanup
                time.sleep(3600)
                
            except (sqlite3.Error, OSError, IOError) as e:
                self.logger.error(f"Error during metrics cleanup: {e}")
                time.sleep(3600)  # Wait before retrying
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            import psutil
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_available = memory_info.available
            memory_total = memory_info.total
            
            # Process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss
            process_cpu_percent = process.cpu_percent()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            
            return {
                'timestamp': current_time.isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_available_bytes': memory_available,
                    'memory_total_bytes': memory_total,
                    'process_memory_bytes': process_memory,
                    'process_cpu_percent': process_cpu_percent,
                    'disk_percent': disk_usage.percent,
                    'uptime_seconds': (current_time - self.start_time).total_seconds()
                }
            }
        except ImportError:
            # If psutil is not available, return basic info
            return {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_count': os.cpu_count(),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
                }
            }


class WorkflowMetricsCollector:
    """Collects metrics specific to decomposition workflows."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
    
    def record_problem_analysis(self, problem_complexity: float, analysis_time: float, success: bool):
        """Record metrics for problem analysis."""
        labels = {'success': str(success)}
        
        self.metrics_collector.observe_histogram(
            'problem_analysis_complexity',
            problem_complexity,
            labels,
            'Complexity of problems being analyzed'
        )
        
        self.metrics_collector.observe_histogram(
            'problem_analysis_duration_seconds',
            analysis_time,
            labels,
            'Time taken to analyze problems'
        )
        
        if success:
            self.metrics_collector.increment_counter(
                'problem_analysis_success_total',
                labels,
                'Successful problem analyses'
            )
        else:
            self.metrics_collector.increment_counter(
                'problem_analysis_error_total',
                labels,
                'Failed problem analyses'
            )
    
    def record_decomposition_execution(self, problem_id: str, subproblem_count: int, execution_time: float, success: bool):
        """Record metrics for decomposition execution."""
        labels = {
            'problem_id': problem_id,
            'success': str(success)
        }
        
        self.metrics_collector.observe_histogram(
            'decomposition_subproblem_count',
            subproblem_count,
            labels,
            'Number of subproblems created during decomposition'
        )
        
        self.metrics_collector.observe_histogram(
            'decomposition_execution_duration_seconds',
            execution_time,
            labels,
            'Time taken to execute decomposition'
        )
        
        if success:
            self.metrics_collector.increment_counter(
                'decomposition_success_total',
                labels,
                'Successful decompositions'
            )
        else:
            self.metrics_collector.increment_counter(
                'decomposition_error_total',
                labels,
                'Failed decompositions'
            )
    
    def record_gauntlet_execution(self, gauntlet_name: str, execution_time: float, success: bool):
        """Record metrics for gauntlet execution."""
        labels = {
            'gauntlet_name': gauntlet_name,
            'success': str(success)
        }
        
        self.metrics_collector.observe_histogram(
            'gauntlet_execution_duration_seconds',
            execution_time,
            labels,
            f'Time taken to execute {gauntlet_name}'
        )
        
        if success:
            self.metrics_collector.increment_counter(
                'gauntlet_success_total',
                labels,
                f'Successful {gauntlet_name} executions'
            )
        else:
            self.metrics_collector.increment_counter(
                'gauntlet_error_total',
                labels,
                f'Failed {gauntlet_name} executions'
            )
    
    def record_solution_attempt(self, subproblem_id: str, solution_time: float, success: bool):
        """Record metrics for solution attempts."""
        labels = {
            'subproblem_id': subproblem_id,
            'success': str(success)
        }
        
        self.metrics_collector.observe_histogram(
            'solution_attempt_duration_seconds',
            solution_time,
            labels,
            'Time taken to attempt solution'
        )
        
        if success:
            self.metrics_collector.increment_counter(
                'solution_attempt_success_total',
                labels,
                'Successful solution attempts'
            )
        else:
            self.metrics_collector.increment_counter(
                'solution_attempt_error_total',
                labels,
                'Failed solution attempts'
            )
    
    def record_integration_result(self, plan_id: str, solution_count: int, integration_time: float, success: bool):
        """Record metrics for solution integration."""
        labels = {
            'plan_id': plan_id,
            'success': str(success)
        }
        
        self.metrics_collector.observe_histogram(
            'integration_solution_count',
            solution_count,
            labels,
            'Number of solutions integrated'
        )
        
        self.metrics_collector.observe_histogram(
            'integration_duration_seconds',
            integration_time,
            labels,
            'Time taken to integrate solutions'
        )
        
        if success:
            self.metrics_collector.increment_counter(
                'integration_success_total',
                labels,
                'Successful integrations'
            )
        else:
            self.metrics_collector.increment_counter(
                'integration_error_total',
                labels,
                'Failed integrations'
            )


class ResourceMetricsCollector:
    """Collects resource utilization metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        self.collection_thread = None
        self.collection_active = False
    
    def start_collection(self, interval: int = 30):
        """Start periodic resource metric collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collect_resources_periodically,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()
    
    def stop_collection(self):
        """Stop resource metric collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _collect_resources_periodically(self, interval: int):
        """Periodically collect resource metrics."""
        while self.collection_active:
            try:
                metrics = self._collect_current_resources()
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.metrics_collector.set_gauge(
                            f"system_{metric_name}",
                            metric_value,
                            description=f"System {metric_name} metric"
                        )
                
                time.sleep(interval)
                
            except (OSError, IOError, ImportError) as e:
                self.logger.error(f"Error collecting resource metrics: {e}")
                time.sleep(interval)
    
    def _collect_current_resources(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count(logical=True)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Process metrics
            process = psutil.Process(os.getpid())
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
                'memory_percent': memory.percent,
                'memory_available_bytes': memory.available,
                'memory_used_bytes': memory.used,
                'memory_total_bytes': memory.total,
                'process_memory_bytes': process.memory_info().rss,
                'process_cpu_percent': process.cpu_percent(),
                'disk_percent': disk.percent,
                'disk_used_bytes': disk.used,
                'disk_total_bytes': disk.total,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
        except ImportError:
            return {}


class HealthCheck:
    """Represents a health check for the system."""
    
    def __init__(self, name: str, check_func: Callable[[], bool], timeout: int = 5):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.last_check = None
        self.last_result = None
        self.logger = logging.getLogger(__name__)
    
    def execute(self) -> Dict[str, Any]:
        """Execute the health check."""
        start_time = time.time()
        
        try:
            result = self.check_func()
            execution_time = time.time() - start_time
            
            check_result = {
                'name': self.name,
                'status': 'healthy' if result else 'unhealthy',
                'healthy': result,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.last_check = datetime.now()
            self.last_result = result
            
            return check_result
            
        except (OSError, IOError, RuntimeError, ValueError) as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            check_result = {
                'name': self.name,
                'status': 'error',
                'healthy': False,
                'error': error_msg,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.warning(f"Health check {self.name} failed: {error_msg}")
            
            return check_result


class HealthMonitor:
    """Monitors system health and provides health check endpoints."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.logger = logging.getLogger(__name__)
        self.system_healthy = True
        self.uptime_start = datetime.now()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], timeout: int = 5):
        """Register a health check function."""
        self.health_checks[name] = HealthCheck(name, check_func, timeout)
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        all_healthy = True
        
        for name, check in self.health_checks.items():
            result = check.execute()
            results[name] = result
            if not result['healthy']:
                all_healthy = False
        
        # Include system metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        
        overall_result = {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'healthy': all_healthy,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.uptime_start).total_seconds(),
            'checks': results,
            'system_metrics': system_metrics
        }
        
        # Update system health status
        self.system_healthy = all_healthy
        
        # Record health status metrics
        self.metrics_collector.set_gauge(
            'system_healthy',
            1 if all_healthy else 0,
            description='System health status (1=healthy, 0=unhealthy)'
        )
        
        return overall_result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status without running checks."""
        return {
            'status': 'healthy' if self.system_healthy else 'unhealthy',
            'healthy': self.system_healthy,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.uptime_start).total_seconds()
        }


class AlertManager:
    """Manages alerting based on metrics and health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: List[Dict] = []
        self.logger = logging.getLogger(__name__)
        self.alert_handlers: List[Callable[[Dict[str, Any]], None]] = []
    
    def add_alert_rule(self, name: str, metric_name: str, condition: str, threshold: float, description: str = ""):
        """Add an alert rule."""
        rule = {
            'name': name,
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'ge', 'le', 'eq', 'ne'
            'threshold': threshold,
            'description': description,
            'active': True
        }
        self.alert_rules.append(rule)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check alert conditions against recent metrics."""
        active_alerts = []
        
        for rule in self.alert_rules:
            if not rule['active']:
                continue
            
            # Get recent metrics for this metric name
            recent_metrics = self.metrics_collector.get_metrics(rule['metric_name'])
            
            if recent_metrics:
                latest_value = recent_metrics[0].value  # Most recent value
                
                # Check condition
                alert_triggered = False
                if rule['condition'] == 'gt' and latest_value > rule['threshold']:
                    alert_triggered = True
                elif rule['condition'] == 'lt' and latest_value < rule['threshold']:
                    alert_triggered = True
                elif rule['condition'] == 'ge' and latest_value >= rule['threshold']:
                    alert_triggered = True
                elif rule['condition'] == 'le' and latest_value <= rule['threshold']:
                    alert_triggered = True
                elif rule['condition'] == 'eq' and latest_value == rule['threshold']:
                    alert_triggered = True
                elif rule['condition'] == 'ne' and latest_value != rule['threshold']:
                    alert_triggered = True
                
                if alert_triggered:
                    alert = {
                        'rule_name': rule['name'],
                        'metric_name': rule['metric_name'],
                        'value': latest_value,
                        'threshold': rule['threshold'],
                        'condition': rule['condition'],
                        'timestamp': datetime.now().isoformat(),
                        'description': rule['description']
                    }
                    active_alerts.append(alert)
        
        return active_alerts
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def process_alerts(self):
        """Process active alerts and notify handlers."""
        active_alerts = self.check_alerts()
        
        for alert in active_alerts:
            self.logger.warning(f"ALERT: {alert['rule_name']} - {alert['description']}")
            
            # Call all registered handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                    self.logger.error(f"Error in alert handler: {e}")


class MonitoringDashboard:
    """Provides a simple dashboard interface for monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_monitor: HealthMonitor):
        self.metrics_collector = metrics_collector
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(__name__)
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display."""
        # Get system metrics
        system_metrics = self.metrics_collector.get_system_metrics()
        
        # Get health status
        health_status = self.health_monitor.get_health_status()
        
        # Get workflow metrics
        workflow_metrics = {
            'problem_analysis_success_rate': self._get_success_rate('problem_analysis_success_total', 'problem_analysis_error_total'),
            'decomposition_success_rate': self._get_success_rate('decomposition_success_total', 'decomposition_error_total'),
            'gauntlet_success_rate': self._get_success_rate('gauntlet_success_total', 'gauntlet_error_total'),
            'solution_success_rate': self._get_success_rate('solution_attempt_success_total', 'solution_attempt_error_total'),
            'integration_success_rate': self._get_success_rate('integration_success_total', 'integration_error_total')
        }
        
        # Get recent metrics
        recent_metrics = {}
        
        # Get execution time metrics
        for metric_name in [
            'problem_analysis_duration_seconds',
            'decomposition_execution_duration_seconds', 
            'gauntlet_execution_duration_seconds',
            'solution_attempt_duration_seconds',
            'integration_duration_seconds'
        ]:
            summary = self.metrics_collector.get_metric_summary(metric_name)
            if summary['count'] > 0:
                recent_metrics[metric_name] = summary
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': system_metrics,
            'health': health_status,
            'workflow': workflow_metrics,
            'recent_metrics': recent_metrics
        }
    
    def _get_success_rate(self, success_metric: str, error_metric: str) -> float:
        """Calculate success rate from success and error counters."""
        success_summary = self.metrics_collector.get_metric_summary(success_metric)
        error_summary = self.metrics_collector.get_metric_summary(error_metric)
        
        success_count = success_summary['count'] if success_summary['count'] else 0
        error_count = error_summary['count'] if error_summary['count'] else 0
        
        total = success_count + error_count
        return (success_count / total) if total > 0 else 0.0


# Global instances
metrics_collector = MetricsCollector()
workflow_metrics = WorkflowMetricsCollector(metrics_collector)
resource_metrics = ResourceMetricsCollector(metrics_collector)
health_monitor = HealthMonitor(metrics_collector)
alert_manager = AlertManager(metrics_collector)
monitoring_dashboard = MonitoringDashboard(metrics_collector, health_monitor)

# Start resource metrics collection
resource_metrics.start_collection(interval=30)

# Register default health checks
def check_database_connection():
    """Check if database connection is healthy."""
    try:
        with sqlite3.connect(metrics_collector.db_path) as conn:
            conn.execute("SELECT 1")
        return True
    except (sqlite3.Error, OSError, IOError):
        return False

def check_decomposition_engine():
    """Check if decomposition engine is responsive."""
    # This would check if the decomposition engine can process a simple request
    return True

def check_llm_availability():
    """Check if LLM service is available."""
    # This would check if the LLM service is available
    return True

health_monitor.register_health_check("database", check_database_connection)
health_monitor.register_health_check("decomposition_engine", check_decomposition_engine)
health_monitor.register_health_check("llm_service", check_llm_availability)

# Add default alert rules
alert_manager.add_alert_rule(
    "high_cpu_usage", 
    "system_cpu_percent", 
    "gt", 
    80.0, 
    "CPU usage is above 80%"
)
alert_manager.add_alert_rule(
    "high_memory_usage", 
    "system_memory_percent", 
    "gt", 
    85.0, 
    "Memory usage is above 85%"
)
alert_manager.add_alert_rule(
    "system_unhealthy", 
    "system_healthy", 
    "eq", 
    0.0, 
    "System is unhealthy"
)

# Add a simple alert handler that logs alerts
def log_alert(alert: Dict[str, Any]):
    logging.getLogger(__name__).warning(f"ALERT: {alert['rule_name']} - {alert['description']} "
                                      f"(Value: {alert['value']}, Threshold: {alert['threshold']})")

alert_manager.add_alert_handler(log_alert)

# Start alert processing in a background thread
def run_alert_processing():
    while True:
        try:
            alert_manager.process_alerts()
            time.sleep(60)  # Check alerts every minute
        except (OSError, IOError, RuntimeError) as e:
            logging.getLogger(__name__).error(f"Error in alert processing: {e}")
            time.sleep(60)

alert_thread = threading.Thread(target=run_alert_processing, daemon=True)
alert_thread.start()


def integrate_with_system():
    """
    Helper function to integrate monitoring with existing system components.
    This would typically be called during system initialization.
    """
    from problem_analyzer import ProblemAnalyzer
    from decomposition_engine import DecompositionEngine
    from sovereign_solution_orchestration import SolutionOrchestrator
    from sovereign_gauntlets import GauntletSystem
    
    # Example: Add decorators to existing methods to collect metrics
    original_analyze_problem = ProblemAnalyzer.analyze_problem
    ProblemAnalyzer.analyze_problem = metrics_collector.time_function(
        'problem_analyzer_analyze_problem'
    )(original_analyze_problem)
    
    original_decompose = DecompositionEngine.decompose
    DecompositionEngine.decompose = metrics_collector.time_function(
        'decomposition_engine_decompose'
    )(original_decompose)
    
    original_integrate = SolutionOrchestrator.integrate_solutions
    SolutionOrchestrator.integrate_solutions = metrics_collector.time_function(
        'solution_orchestrator_integrate_solutions'
    )(original_integrate)
    
    original_gauntlet_run = GauntletSystem.run_decomposition_gauntlets
    GauntletSystem.run_decomposition_gauntlets = metrics_collector.time_function(
        'gauntlet_system_run_decomposition_gauntlets'
    )(original_gauntlet_run)


def example_usage():
    """Example of how to use the monitoring system."""
    
    # Example 1: Record various metrics
    workflow_metrics.record_problem_analysis(7.5, 2.34, True)
    workflow_metrics.record_decomposition_execution("problem_123", 5, 4.67, True)
    workflow_metrics.record_gauntlet_execution("coherence_gauntlet", 1.23, True)
    workflow_metrics.record_solution_attempt("subprob_456", 3.45, True)
    workflow_metrics.record_integration_result("plan_789", 5, 2.11, True)
    
    # Example 2: Run health checks
    health_result = health_monitor.run_health_checks()
    print(f"Health check result: {health_result['status']}")
    print(f"Checks: {list(health_result['checks'].keys())}")
    
    # Example 3: Get dashboard metrics
    dashboard_metrics = monitoring_dashboard.get_dashboard_metrics()
    print(f"System health: {dashboard_metrics['health']['status']}")
    print(f"Uptime: {dashboard_metrics['health']['uptime_seconds']:.2f}s")
    
    # Example 4: Query specific metrics
    analysis_metrics = metrics_collector.get_metrics('problem_analysis_duration_seconds')
    print(f"Found {len(analysis_metrics)} analysis duration metrics")
    
    # Example 5: Check system metrics
    system_metrics = metrics_collector.get_system_metrics()
    print(f"CPU Usage: {system_metrics['system'].get('cpu_percent', 'N/A')}%")
    print(f"Memory Usage: {system_metrics['system'].get('memory_percent', 'N/A')}%")
    
    # Example 6: Try to trigger alerts (this would require specific conditions)
    # For example, if we set a memory metric above the threshold:
    metrics_collector.set_gauge('system_memory_percent', 90.0)  # This should trigger an alert
    time.sleep(2)  # Give time for alert processing
    
    return dashboard_metrics


if __name__ == "__main__":
    example_usage()