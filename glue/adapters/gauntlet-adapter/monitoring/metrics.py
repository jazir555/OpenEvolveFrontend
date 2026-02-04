"""
Production Metrics Collection for Gauntlet System

Provides comprehensive metrics collection with Prometheus export format.
Tracks execution metrics, ML component metrics, and system metrics.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from collections import defaultdict
from threading import Lock
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"      # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Similar to histogram with quantiles


@dataclass
class MetricValue:
    """A single metric value"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    metric_type: MetricType = MetricType.GAUGE

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format"""
        # Format labels
        if self.labels:
            label_str = "{" + ", ".join(f'{k}="{v}"' for k, v in self.labels.items()) + "}"
        else:
            label_str = ""

        # Metric type line
        type_line = f"# TYPE {self.name} {self.metric_type.value}\n"

        # Metric value line
        value_line = f"{self.name}{label_str} {self.value} {int(self.timestamp * 1000)}\n"

        return type_line + value_line


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution tracking"""
    upper_bound: float
    count: int = 0


@dataclass
class Histogram:
    """Histogram metric for tracking distributions"""
    name: str
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default buckets if not provided"""
        if not self.buckets:
            # Default exponential buckets
            self.buckets = [
                HistogramBucket(0.005),
                HistogramBucket(0.01),
                HistogramBucket(0.025),
                HistogramBucket(0.05),
                HistogramBucket(0.1),
                HistogramBucket(0.25),
                HistogramBucket(0.5),
                HistogramBucket(1.0),
                HistogramBucket(2.5),
                HistogramBucket(5.0),
                HistogramBucket(10.0),
                HistogramBucket(float('inf')),
            ]

    def observe(self, value: float) -> None:
        """Observe a value"""
        self.sum += value
        self.count += 1

        # Increment appropriate buckets
        for bucket in self.buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format"""
        # Format labels
        if self.labels:
            label_str = "{" + ", ".join(f'{k}="{v}"' for k, v in self.labels.items()) + "}"
        else:
            label_str = ""

        lines = []

        # Type line
        lines.append(f"# TYPE {self.name} histogram")

        # Bucket lines
        for bucket in self.buckets:
            if bucket.upper_bound == float('inf'):
                le_str = "+Inf"
            else:
                le_str = str(bucket.upper_bound)

            lines.append(f"{self.name}_bucket{label_str} {{le=\"{le_str}\"}} {bucket.count}")

        # Sum and count
        lines.append(f"{self.name}_sum{label_str} {self.sum}")
        lines.append(f"{self.name}_count{label_str} {self.count}")

        return "\n".join(lines) + "\n"


class GauntletMetricsCollector:
    """
    Comprehensive metrics collector for gauntlet system.

    Features:
    - Execution metrics (total runs, pass/fail, duration)
    - ML component metrics (optimization, predictions, training)
    - System metrics (CPU, memory, WebSocket connections)
    - Prometheus export format
    - Thread-safe operations

    Example:
        >>> collector = GauntletMetricsCollector()
        >>>
        >>> # Record execution
        >>> collector.record_execution(
        ...     domain="finance",
        ...     passed=True,
        ...     duration_ms=1234.5,
        ...     score=0.85
        ... )
        >>>
        >>> # Export to Prometheus
        >>> prometheus_metrics = collector.export_prometheus()
    """

    def __init__(self):
        """Initialize metrics collector"""
        self._lock = Lock()

        # Counters
        self._counters: Dict[str, float] = defaultdict(float)

        # Gauges
        self._gauges: Dict[str, float] = defaultdict(float)

        # Histograms
        self._histograms: Dict[str, Histogram] = {}

        # Execution tracking
        self._executions_by_domain: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "total_score": 0.0,
            "total_duration_ms": 0.0
        })

        # ML metrics
        self._ml_metrics: Dict[str, Any] = {
            "optimization_iterations": 0,
            "predictions_made": 0,
            "prediction_accuracy_total": 0.0,
            "prediction_accuracy_count": 0,
            "training_loss": 0.0,
            "model_convergence_count": 0
        }

        # System metrics
        self._system_metrics: Dict[str, Any] = {
            "websocket_connections": 0,
            "active_gauntlets": 0
        }

        # Start time
        self._start_time = time.time()

        logger.info("Gauntlet Metrics Collector initialized")

    # ========== Execution Metrics ==========

    def record_execution(
        self,
        domain: str,
        passed: bool,
        duration_ms: float,
        score: float,
        rounds_completed: int = 3,
        artifact_id: Optional[str] = None
    ) -> None:
        """
        Record a gauntlet execution.

        Args:
            domain: Problem domain (e.g., "finance", "science")
            passed: Whether the gauntlet was passed
            duration_ms: Execution time in milliseconds
            score: Final score (0.0 to 1.0)
            rounds_completed: Number of rounds completed
            artifact_id: Optional artifact identifier
        """
        with self._lock:
            # Update counters
            self._counters["gauntlet_executions_total"] += 1
            self._counters[f"gauntlet_executions_total{{domain=\"{domain}\"}}"] += 1

            if passed:
                self._counters["gauntlet_passes_total"] += 1
                self._counters[f"gauntlet_passes_total{{domain=\"{domain}\"}}"] += 1
            else:
                self._counters["gauntlet_failures_total"] += 1
                self._counters[f"gauntlet_failures_total{{domain=\"{domain}\"}}"] += 1

            # Update domain-specific stats
            domain_stats = self._executions_by_domain[domain]
            domain_stats["total"] += 1
            domain_stats["total_duration_ms"] += duration_ms
            domain_stats["total_score"] += score

            if passed:
                domain_stats["passed"] += 1
            else:
                domain_stats["failed"] += 1

            # Update gauges
            self._gauges["gauntlet_last_duration_ms"] = duration_ms
            self._gauges[f"gauntlet_last_duration_ms{{domain=\"{domain}\"}}"] = duration_ms
            self._gauges["gauntlet_last_score"] = score
            self._gauges[f"gauntlet_last_score{{domain=\"{domain}\"}}"] = score

            # Record duration histogram
            hist_key = f"gauntlet_duration_seconds{{domain=\"{domain}\"}}"
            if hist_key not in self._histograms:
                self._histograms[hist_key] = Histogram(name=hist_key, labels={"domain": domain})
            self._histograms[hist_key].observe(duration_ms / 1000.0)

            logger.debug(
                f"Recorded execution: domain={domain}, passed={passed}, "
                f"duration_ms={duration_ms:.2f}, score={score:.3f}"
            )

    def get_execution_stats(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics.

        Args:
            domain: Optional domain filter

        Returns:
            Execution statistics dictionary
        """
        with self._lock:
            if domain:
                stats = self._executions_by_domain.get(domain, {})
                return self._calculate_domain_stats(domain, stats)

            # Return stats for all domains
            all_stats = {}
            for dom, stats in self._executions_by_domain.items():
                all_stats[dom] = self._calculate_domain_stats(dom, stats)

            return all_stats

    def _calculate_domain_stats(self, domain: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for a domain"""
        total = stats.get("total", 0)
        if total == 0:
            return {
                "domain": domain,
                "total_executions": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
                "average_duration_ms": 0.0
            }

        return {
            "domain": domain,
            "total_executions": total,
            "passed": stats.get("passed", 0),
            "failed": stats.get("failed", 0),
            "pass_rate": stats.get("passed", 0) / total,
            "average_score": stats.get("total_score", 0.0) / total,
            "average_duration_ms": stats.get("total_duration_ms", 0.0) / total
        }

    # ========== ML Component Metrics ==========

    def record_optimization_iteration(
        self,
        strategy: str,
        iteration: int,
        score: float,
        improvement: float
    ) -> None:
        """
        Record an optimization iteration.

        Args:
            strategy: Optimization strategy (e.g., "q_learning", "genetic")
            iteration: Iteration number
            score: Current best score
            improvement: Improvement over baseline
        """
        with self._lock:
            self._ml_metrics["optimization_iterations"] += 1
            self._counters[f"optimization_iterations_total{{strategy=\"{strategy}\"}}"] += 1

            self._gauges[f"optimization_best_score{{strategy=\"{strategy}\"}}"] = score
            self._gauges[f"optimization_improvement{{strategy=\"{strategy}\"}}"] = improvement

            logger.debug(
                f"Recorded optimization iteration: strategy={strategy}, "
                f"iteration={iteration}, score={score:.3f}"
            )

    def record_prediction(
        self,
        success_probability: float,
        confidence: float,
        actual_outcome: bool,
        domain: str
    ) -> None:
        """
        Record a prediction result.

        Args:
            success_probability: Predicted probability of success
            confidence: Prediction confidence
            actual_outcome: Whether prediction was correct
            domain: Problem domain
        """
        with self._lock:
            self._ml_metrics["predictions_made"] += 1
            self._counters[f"predictions_total{{domain=\"{domain}\"}}"] += 1

            # Calculate accuracy
            predicted_success = success_probability > 0.5
            accuracy = 1.0 if (predicted_success == actual_outcome) else 0.0

            self._ml_metrics["prediction_accuracy_total"] += accuracy
            self._ml_metrics["prediction_accuracy_count"] += 1

            # Update gauges
            avg_accuracy = (self._ml_metrics["prediction_accuracy_total"] /
                          self._ml_metrics["prediction_accuracy_count"])
            self._gauges["prediction_accuracy"] = avg_accuracy
            self._gauges[f"prediction_accuracy{{domain=\"{domain}\"}}"] = avg_accuracy

            logger.debug(
                f"Recorded prediction: domain={domain}, accuracy={accuracy:.2f}, "
                f"confidence={confidence:.2f}"
            )

    def record_training_metrics(
        self,
        loss: float,
        converged: bool,
        epoch: int
    ) -> None:
        """
        Record training metrics.

        Args:
            loss: Training loss
            converged: Whether model converged
            epoch: Training epoch number
        """
        with self._lock:
            self._ml_metrics["training_loss"] = loss

            if converged:
                self._ml_metrics["model_convergence_count"] += 1
                self._counters["model_convergence_total"] += 1

            self._gauges["training_loss"] = loss
            self._gauges["training_epoch"] = float(epoch)

            logger.debug(f"Recorded training metrics: loss={loss:.4f}, converged={converged}")

    def get_ml_metrics(self) -> Dict[str, Any]:
        """Get ML component metrics"""
        with self._lock:
            metrics = self._ml_metrics.copy()

            # Calculate average accuracy
            if metrics["prediction_accuracy_count"] > 0:
                metrics["average_prediction_accuracy"] = (
                    metrics["prediction_accuracy_total"] / metrics["prediction_accuracy_count"]
                )
            else:
                metrics["average_prediction_accuracy"] = 0.0

            return metrics

    # ========== System Metrics ==========

    def update_system_metrics(self) -> None:
        """Update system resource metrics"""
        with self._lock:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._gauges["system_cpu_usage_percent"] = cpu_percent

            # Memory
            memory = psutil.virtual_memory()
            self._gauges["system_memory_usage_percent"] = memory.percent
            self._gauges["system_memory_used_bytes"] = memory.used
            self._gauges["system_memory_available_bytes"] = memory.available

            # Disk
            disk = psutil.disk_usage("/")
            self._gauges["system_disk_usage_percent"] = disk.percent
            self._gauges["system_disk_used_bytes"] = disk.used
            self._gauges["system_disk_free_bytes"] = disk.free

            # Process-specific
            process = psutil.Process()
            self._gauges["process_memory_usage_bytes"] = process.memory_info().rss
            self._gauges["process_num_threads"] = float(process.num_threads())
            self._gauges["process_cpu_percent"] = process.cpu_percent()

            logger.debug("System metrics updated")

    def set_websocket_connections(self, count: int) -> None:
        """Set WebSocket connection count"""
        with self._lock:
            self._system_metrics["websocket_connections"] = count
            self._gauges["websocket_connections_active"] = float(count)

    def set_active_gauntlets(self, count: int) -> None:
        """Set active gauntlet count"""
        with self._lock:
            self._system_metrics["active_gauntlets"] = count
            self._gauges["gauntlets_active"] = float(count)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        with self._lock:
            return self._system_metrics.copy()

    # ========== Prometheus Export ==========

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        # Update system metrics before export
        self.update_system_metrics()

        lines = []

        with self._lock:
            # Counters
            for name, value in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value} {int(time.time() * 1000)}")

            # Gauges
            for name, value in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value} {int(time.time() * 1000)}")

            # Histograms
            for histogram in self._histograms.values():
                lines.append(histogram.to_prometheus())

            # Add uptime metric
            uptime = time.time() - self._start_time
            lines.append(f"# TYPE gauntlet_uptime_seconds gauge")
            lines.append(f"gauntlet_uptime_seconds {uptime} {int(time.time() * 1000)}")

        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        """
        Export all metrics as JSON.

        Returns:
            JSON-formatted metrics string
        """
        self.update_system_metrics()

        with self._lock:
            metrics = {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "executions_by_domain": {
                    domain: self._calculate_domain_stats(domain, stats)
                    for domain, stats in self._executions_by_domain.items()
                },
                "ml_metrics": self.get_ml_metrics(),
                "system_metrics": self.get_system_metrics()
            }

        return json.dumps(metrics, indent=2)

    # ========== Utility Methods ==========

    def reset_metrics(self) -> None:
        """Reset all metrics (use with caution)"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._executions_by_domain.clear()
            self._ml_metrics = {
                "optimization_iterations": 0,
                "predictions_made": 0,
                "prediction_accuracy_total": 0.0,
                "prediction_accuracy_count": 0,
                "training_loss": 0.0,
                "model_convergence_count": 0
            }
            self._start_time = time.time()

        logger.warning("All metrics have been reset")

    def get_metric_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        self.update_system_metrics()

        with self._lock:
            return {
                "uptime_seconds": time.time() - self._start_time,
                "total_executions": self._counters.get("gauntlet_executions_total", 0),
                "total_passes": self._counters.get("gauntlet_passes_total", 0),
                "total_failures": self._counters.get("gauntlet_failures_total", 0),
                "global_pass_rate": (
                    self._counters.get("gauntlet_passes_total", 0) /
                    max(1, self._counters.get("gauntlet_executions_total", 1))
                ),
                "ml_metrics": self.get_ml_metrics(),
                "system_metrics": self.get_system_metrics(),
                "domains": list(self._executions_by_domain.keys())
            }


# Global metrics collector instance
_metrics_collector = GauntletMetricsCollector()


def get_metrics_collector() -> GauntletMetricsCollector:
    """Get the global metrics collector"""
    return _metrics_collector


def record_execution(
    domain: str,
    passed: bool,
    duration_ms: float,
    score: float,
    rounds_completed: int = 3,
    artifact_id: Optional[str] = None
) -> None:
    """Record a gauntlet execution (convenience function)"""
    get_metrics_collector().record_execution(
        domain=domain,
        passed=passed,
        duration_ms=duration_ms,
        score=score,
        rounds_completed=rounds_completed,
        artifact_id=artifact_id
    )


def export_prometheus() -> str:
    """Export metrics in Prometheus format (convenience function)"""
    return get_metrics_collector().export_prometheus()


def export_json() -> str:
    """Export metrics as JSON (convenience function)"""
    return get_metrics_collector().export_json()
