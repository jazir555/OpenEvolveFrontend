"""
Comprehensive Metrics and Monitoring System for OpenEvolve Gauntlet

Provides detailed metrics collection, aggregation, and reporting
for all aspects of the Gauntlet system.

Key Features:
- Real-time metrics collection
- Performance metrics tracking
- Success/failure rate monitoring
- Resource usage tracking
- Team performance analytics
- Checkpoint health monitoring
- Cache performance metrics
- Fuzzing vulnerability tracking
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import logging
import time
import psutil
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary


@dataclass
class MetricValue:
    """A single metric value"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceMetric:
    """Performance-related metric"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamPerformanceMetric:
    """Team performance metric"""
    team_id: str
    problem_id: str
    domain: str
    difficulty: int
    success: bool
    score: float
    execution_time: float
    timestamp: datetime


@dataclass
class CacheMetric:
    """Cache performance metric"""
    operation: str  # hit, miss, set, invalidate
    cache_type: str
    key: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointMetric:
    """Checkpoint-related metric"""
    operation: str  # create, load, delete, cleanup
    problem_id: str
    checkpoint_id: Optional[str]
    success: bool
    size_bytes: int
    duration_ms: float
    timestamp: datetime


@dataclass
class FuzzingMetric:
    """Fuzzing-related metric"""
    problem_id: str
    iterations: int
    crashes_found: int
    vulnerabilities_found: int
    duration_seconds: float
    timestamp: datetime


class MetricsCollector:
    """
    Collects and manages metrics for the Gauntlet system.
    """

    def __init__(self):
        # Metric storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._metric_history: List[MetricValue] = []

        # Specialized metrics
        self._performance_history: List[PerformanceMetric] = []
        self._team_metrics: List[TeamPerformanceMetric] = []
        self._cache_metrics: List[CacheMetric] = []
        self._checkpoint_metrics: List[CheckpointMetric] = []
        self._fuzzing_metrics: List[FuzzingMetric] = []

        # Resource monitoring
        self._resource_monitoring_enabled = False
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._resource_history: List[Dict[str, float]] = []

        # Locks for thread safety
        self._lock = threading.RLock()

    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value

            metric = MetricValue(
                name=name,
                value=self._counters[key],
                timestamp=datetime.utcnow(),
                labels=labels or {},
                metric_type=MetricType.COUNTER
            )
            self._metric_history.append(metric)

            logger.debug(f"Counter {name} incremented by {value} to {self._counters[key]}")

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value

            metric = MetricValue(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {},
                metric_type=MetricType.GAUGE
            )
            self._metric_history.append(metric)

            logger.debug(f"Gauge {name} set to {value}")

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram"""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)

            metric = MetricValue(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            )
            self._metric_history.append(metric)

            logger.debug(f"Histogram {name} recorded value {value}")

    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ):
        """Record a performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
                success=success,
                metadata=metadata or {}
            )
            self._performance_history.append(metric)

            # Also record in histogram
            self.record_histogram(f"{operation}_duration_ms", duration_ms)
            self.increment(f"{operation}_total")

            if not success:
                self.increment(f"{operation}_failures")

            logger.debug(f"Performance: {operation} took {duration_ms}ms, success={success}")

    def record_team_performance(
        self,
        team_id: str,
        problem_id: str,
        domain: str,
        difficulty: int,
        success: bool,
        score: float,
        execution_time: float
    ):
        """Record team performance metric"""
        with self._lock:
            metric = TeamPerformanceMetric(
                team_id=team_id,
                problem_id=problem_id,
                domain=domain,
                difficulty=difficulty,
                success=success,
                score=score,
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
            self._team_metrics.append(metric)

            # Update counters
            self.increment("team_attempts_total", labels={"team": team_id, "domain": domain})
            if success:
                self.increment("team_successes_total", labels={"team": team_id, "domain": domain})
            else:
                self.increment("team_failures_total", labels={"team": team_id, "domain": domain})

            logger.debug(f"Team performance: {team_id} - success={success}, score={score}")

    def record_cache_operation(
        self,
        operation: str,
        cache_type: str,
        key: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record cache operation"""
        with self._lock:
            metric = CacheMetric(
                operation=operation,
                cache_type=cache_type,
                key=key,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            self._cache_metrics.append(metric)

            # Update counters
            self.increment(f"cache_{operation}_total", labels={"type": cache_type})

            logger.debug(f"Cache: {operation} on {cache_type}")

    def record_checkpoint_operation(
        self,
        operation: str,
        problem_id: str,
        checkpoint_id: Optional[str],
        success: bool,
        size_bytes: int,
        duration_ms: float
    ):
        """Record checkpoint operation"""
        with self._lock:
            metric = CheckpointMetric(
                operation=operation,
                problem_id=problem_id,
                checkpoint_id=checkpoint_id,
                success=success,
                size_bytes=size_bytes,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow()
            )
            self._checkpoint_metrics.append(metric)

            # Update counters
            self.increment(f"checkpoint_{operation}_total")
            if not success:
                self.increment(f"checkpoint_{operation}_failures")

            # Record size
            self.set_gauge(f"checkpoint_size_bytes", size_bytes, labels={"operation": operation})

            logger.debug(f"Checkpoint: {operation} for {problem_id} - success={success}")

    def record_fuzzing_results(
        self,
        problem_id: str,
        iterations: int,
        crashes_found: int,
        vulnerabilities_found: int,
        duration_seconds: float
    ):
        """Record fuzzing results"""
        with self._lock:
            metric = FuzzingMetric(
                problem_id=problem_id,
                iterations=iterations,
                crashes_found=crashes_found,
                vulnerabilities_found=vulnerabilities_found,
                duration_seconds=duration_seconds,
                timestamp=datetime.utcnow()
            )
            self._fuzzing_metrics.append(metric)

            # Update counters
            self.increment("fuzzing_iterations_total", iterations)
            self.increment("fuzzing_crashes_total", crashes_found)
            self.increment("fuzzing_vulnerabilities_total", vulnerabilities_found)

            logger.debug(f"Fuzzing: {problem_id} - {crashes_found} crashes, {vulnerabilities_found} vulnerabilities")

    def start_resource_monitoring(self, interval_seconds: float = 1.0):
        """Start monitoring system resources"""
        with self._lock:
            if self._resource_monitoring_enabled:
                logger.warning("Resource monitoring already enabled")
                return

            self._resource_monitoring_enabled = True
            self._resource_monitor_thread = threading.Thread(
                target=self._monitor_resources,
                args=(interval_seconds,),
                daemon=True
            )
            self._resource_monitor_thread.start()
            logger.info(f"Started resource monitoring with {interval_seconds}s interval")

    def stop_resource_monitoring(self):
        """Stop monitoring system resources"""
        with self._lock:
            if not self._resource_monitoring_enabled:
                return

            self._resource_monitoring_enabled = False
            if self._resource_monitor_thread:
                self._resource_monitor_thread.join(timeout=5.0)
            logger.info("Stopped resource monitoring")

    def _monitor_resources(self, interval_seconds: float):
        """Internal method to monitor resources"""
        process = psutil.Process()

        while self._resource_monitoring_enabled:
            try:
                # CPU and memory
                cpu_percent = process.cpu_percent(interval=None)
                memory_info = process.memory_info()

                # Disk I/O
                io_counters = process.io_counters() if hasattr(process, 'io_counters') else None

                # Network (if available)
                net_io = psutil.net_io_counters() if hasattr(psutil, 'net_io_counters') else None

                resource_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_rss_bytes': memory_info.rss,
                    'memory_vms_bytes': memory_info.vms,
                    'memory_percent': process.memory_percent(),
                }

                if io_counters:
                    resource_data.update({
                        'io_read_count': io_counters.read_count,
                        'io_write_count': io_counters.write_count,
                        'io_read_bytes': io_counters.read_bytes,
                        'io_write_bytes': io_counters.write_bytes,
                    })

                if net_io:
                    resource_data.update({
                        'net_bytes_sent': net_io.bytes_sent,
                        'net_bytes_recv': net_io.bytes_recv,
                    })

                with self._lock:
                    self._resource_history.append(resource_data)

                    # Update gauges
                    self.set_gauge('process_cpu_percent', cpu_percent)
                    self.set_gauge('process_memory_rss_bytes', memory_info.rss)
                    self.set_gauge('process_memory_percent', process.memory_percent())

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(interval_seconds)

    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get current counter value"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current gauge value"""
        with self._lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key)

    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            key = self._make_key(name, labels)
            values = self._histograms.get(key, [])

            if not values:
                return {}

            sorted_values = sorted(values)
            count = len(values)
            total = sum(values)

            return {
                'count': count,
                'sum': total,
                'avg': total / count,
                'min': sorted_values[0],
                'max': sorted_values[-1],
                'p50': sorted_values[int(count * 0.5)],
                'p90': sorted_values[int(count * 0.9)],
                'p95': sorted_values[int(count * 0.95)],
                'p99': sorted_values[int(count * 0.99)],
            }

    def get_performance_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary for operations"""
        with self._lock:
            metrics = self._performance_history

            if operation:
                metrics = [m for m in metrics if m.operation == operation]

            if not metrics:
                return {}

            durations = [m.duration_ms for m in metrics]
            success_count = sum(1 for m in metrics if m.success)
            total_count = len(metrics)

            sorted_durations = sorted(durations)

            return {
                'operation': operation or 'all',
                'total_requests': total_count,
                'successful_requests': success_count,
                'failed_requests': total_count - success_count,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'avg_duration_ms': sum(durations) / total_count,
                'min_duration_ms': sorted_durations[0],
                'max_duration_ms': sorted_durations[-1],
                'p50_duration_ms': sorted_durations[int(total_count * 0.5)],
                'p90_duration_ms': sorted_durations[int(total_count * 0.9)],
                'p95_duration_ms': sorted_durations[int(total_count * 0.95)],
                'p99_duration_ms': sorted_durations[int(total_count * 0.99)],
            }

    def get_team_performance_summary(self, team_id: str = None) -> Dict[str, Any]:
        """Get team performance summary"""
        with self._lock:
            metrics = self._team_metrics

            if team_id:
                metrics = [m for m in metrics if m.team_id == team_id]

            if not metrics:
                return {}

            total_count = len(metrics)
            success_count = sum(1 for m in metrics if m.success)
            avg_score = sum(m.score for m in metrics) / total_count if total_count > 0 else 0
            avg_time = sum(m.execution_time for m in metrics) / total_count if total_count > 0 else 0

            # By domain
            by_domain = defaultdict(lambda: {'count': 0, 'successes': 0})
            for m in metrics:
                by_domain[m.domain]['count'] += 1
                if m.success:
                    by_domain[m.domain]['successes'] += 1

            domain_success_rates = {
                domain: data['successes'] / data['count']
                for domain, data in by_domain.items()
            }

            return {
                'team_id': team_id or 'all',
                'total_problems': total_count,
                'successful_problems': success_count,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'avg_score': avg_score,
                'avg_execution_time': avg_time,
                'domain_success_rates': domain_success_rates,
            }

    def get_cache_summary(self, cache_type: str = None) -> Dict[str, Any]:
        """Get cache performance summary"""
        with self._lock:
            metrics = self._cache_metrics

            if cache_type:
                metrics = [m for m in metrics if m.cache_type == cache_type]

            if not metrics:
                return {}

            hits = sum(1 for m in metrics if m.operation == 'hit')
            misses = sum(1 for m in metrics if m.operation == 'miss')
            total = hits + misses

            return {
                'cache_type': cache_type or 'all',
                'total_requests': total,
                'hits': hits,
                'misses': misses,
                'hit_rate': hits / total if total > 0 else 0,
                'miss_rate': misses / total if total > 0 else 0,
            }

    def get_checkpoint_summary(self, problem_id: str = None) -> Dict[str, Any]:
        """Get checkpoint summary"""
        with self._lock:
            metrics = self._checkpoint_metrics

            if problem_id:
                metrics = [m for m in metrics if m.problem_id == problem_id]

            if not metrics:
                return {}

            total_count = len(metrics)
            success_count = sum(1 for m in metrics if m.success)
            avg_size = sum(m.size_bytes for m in metrics) / total_count if total_count > 0 else 0
            avg_duration = sum(m.duration_ms for m in metrics) / total_count if total_count > 0 else 0

            # By operation
            by_operation = defaultdict(lambda: {'count': 0, 'successes': 0, 'total_size': 0})
            for m in metrics:
                by_operation[m.operation]['count'] += 1
                if m.success:
                    by_operation[m.operation]['successes'] += 1
                by_operation[m.operation]['total_size'] += m.size_bytes

            return {
                'problem_id': problem_id or 'all',
                'total_operations': total_count,
                'successful_operations': success_count,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'avg_size_bytes': avg_size,
                'avg_duration_ms': avg_duration,
                'operations_by_type': {
                    op: {
                        'count': data['count'],
                        'successes': data['successes'],
                        'avg_size_bytes': data['total_size'] / data['count'] if data['count'] > 0 else 0,
                    }
                    for op, data in by_operation.items()
                },
            }

    def get_fuzzing_summary(self, problem_id: str = None) -> Dict[str, Any]:
        """Get fuzzing summary"""
        with self._lock:
            metrics = self._fuzzing_metrics

            if problem_id:
                metrics = [m for m in metrics if m.problem_id == problem_id]

            if not metrics:
                return {}

            total_iterations = sum(m.iterations for m in metrics)
            total_crashes = sum(m.crashes_found for m in metrics)
            total_vulnerabilities = sum(m.vulnerabilities_found for m in metrics)
            total_duration = sum(m.duration_seconds for m in metrics)

            return {
                'problem_id': problem_id or 'all',
                'total_fuzzing_runs': len(metrics),
                'total_iterations': total_iterations,
                'total_crashes': total_crashes,
                'total_vulnerabilities': total_vulnerabilities,
                'crash_rate': total_crashes / total_iterations if total_iterations > 0 else 0,
                'vulnerability_rate': total_vulnerabilities / total_iterations if total_iterations > 0 else 0,
                'total_duration_seconds': total_duration,
                'avg_duration_seconds': total_duration / len(metrics),
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics report"""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'performance': {
                    'all': self.get_performance_summary(),
                },
                'team_performance': self.get_team_performance_summary(),
                'cache': self.get_cache_summary(),
                'checkpoint': self.get_checkpoint_summary(),
                'fuzzing': self.get_fuzzing_summary(),
                'timestamp': datetime.utcnow().isoformat(),
            }

    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._metric_history.clear()
            self._performance_history.clear()
            self._team_metrics.clear()
            self._cache_metrics.clear()
            self._checkpoint_metrics.clear()
            self._fuzzing_metrics.clear()
            self._resource_history.clear()
            logger.info("All metrics reset")

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a key for metric storage"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector():
    """Reset the global metrics collector"""
    global _global_collector
    _global_collector = None


# Decorators for automatic metric collection
def track_performance(operation_name: str = None):
    """Decorator to track function performance"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            op_name = operation_name or func.__name__
            start_time = time.time()
            success = False

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                logger.error(f"Error in {op_name}: {e}")
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                collector.record_performance(op_name, duration_ms, success)

        return wrapper
    return decorator


def track_cache_operation(cache_type: str):
    """Decorator to track cache operations"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            operation = func.__name__

            try:
                result = func(*args, **kwargs)

                # Determine operation type
                if operation == 'get':
                    op_type = 'hit' if result is not None else 'miss'
                elif operation == 'set':
                    op_type = 'set'
                elif operation == 'invalidate':
                    op_type = 'invalidate'
                else:
                    op_type = operation

                collector.record_cache_operation(op_type, cache_type)
                return result

            except Exception as e:
                logger.error(f"Error in cache operation {operation}: {e}")
                raise

        return wrapper
    return decorator


# Example usage
async def demo_metrics():
    """Demonstration of metrics system"""

    print("\n" + "=" * 60)
    print("Gauntlet Metrics System Demo")
    print("=" * 60)

    collector = MetricsCollector()

    # Example 1: Counter metrics
    print("\n1. Counter Metrics:")
    collector.increment("problems_solved", labels={"domain": "web"})
    collector.increment("problems_solved", labels={"domain": "ml"})
    collector.increment("problems_solved", labels={"domain": "web"})
    print(f"   Web problems solved: {collector.get_counter('problems_solved', {'domain': 'web'})}")
    print(f"   ML problems solved: {collector.get_counter('problems_solved', {'domain': 'ml'})}")

    # Example 2: Gauge metrics
    print("\n2. Gauge Metrics:")
    collector.set_gauge("active_problems", 5)
    collector.set_gauge("queue_size", 12)
    print(f"   Active problems: {collector.get_gauge('active_problems')}")
    print(f"   Queue size: {collector.get_gauge('queue_size')}")

    # Example 3: Histogram metrics
    print("\n3. Histogram Metrics:")
    for duration in [100, 150, 200, 120, 180, 90, 210]:
        collector.record_histogram("solve_duration_ms", duration)
    stats = collector.get_histogram_stats("solve_duration_ms")
    print(f"   Average duration: {stats['avg']:.1f}ms")
    print(f"   P95 duration: {stats['p95']:.1f}ms")

    # Example 4: Performance metrics
    print("\n4. Performance Metrics:")
    collector.record_performance("solve_problem", 150.5, True)
    collector.record_performance("solve_problem", 200.3, True)
    collector.record_performance("solve_problem", 100.2, False)
    summary = collector.get_performance_summary("solve_problem")
    print(f"   Total requests: {summary['total_requests']}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Average duration: {summary['avg_duration_ms']:.1f}ms")

    # Example 5: Team performance
    print("\n5. Team Performance:")
    collector.record_team_performance(
        team_id="blue_team_1",
        problem_id="problem_123",
        domain="web",
        difficulty=3,
        success=True,
        score=0.85,
        execution_time=150.0
    )
    team_summary = collector.get_team_performance_summary("blue_team_1")
    print(f"   Team success rate: {team_summary['success_rate']:.1%}")
    print(f"   Average score: {team_summary['avg_score']:.2f}")

    # Example 6: Cache metrics
    print("\n6. Cache Metrics:")
    for _ in range(8):
        collector.record_cache_operation("hit", "memory")
    for _ in range(2):
        collector.record_cache_operation("miss", "memory")
    cache_summary = collector.get_cache_summary("memory")
    print(f"   Hit rate: {cache_summary['hit_rate']:.1%}")

    # Example 7: Complete metrics report
    print("\n7. Complete Metrics Report:")
    all_metrics = collector.get_all_metrics()
    print(f"   Timestamp: {all_metrics['timestamp']}")
    print(f"   Counters: {len(all_metrics['counters'])} entries")
    print(f"   Gauges: {len(all_metrics['gauges'])} entries")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_metrics())
