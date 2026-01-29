"""Metrics collection for Adaptive MDAP."""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock
import statistics


@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    value: int = 0
    
    def increment(self, amount: int = 1) -> None:
        self.value += amount
    
    def get(self) -> int:
        return self.value


@dataclass
class Histogram:
    """Histogram metric for tracking distributions."""
    name: str
    values: List[float] = field(default_factory=list)
    max_size: int = 10000
    
    def observe(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values = self.values[-self.max_size:]
    
    def get_stats(self) -> Dict[str, float]:
        if not self.values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_vals = sorted(self.values)
        n = len(sorted_vals)
        
        return {
            "count": n,
            "min": min(sorted_vals),
            "max": max(sorted_vals),
            "mean": statistics.mean(sorted_vals),
            "p50": sorted_vals[int(n * 0.5)],
            "p95": sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
            "p99": sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
        }


@dataclass
class Gauge:
    """Gauge metric for tracking current values."""
    name: str
    value: float = 0.0
    
    def set(self, value: float) -> None:
        self.value = value
    
    def get(self) -> float:
        return self.value


@dataclass
class Timer:
    """Timer metric for tracking durations."""
    name: str
    durations: List[float] = field(default_factory=list)
    max_size: int = 10000
    
    def time(self, duration_ms: float) -> None:
        self.durations.append(duration_ms)
        if len(self.durations) > self.max_size:
            self.durations = self.durations[-self.max_size:]
    
    def get_stats(self) -> Dict[str, float]:
        if not self.durations:
            return {"count": 0, "min_ms": 0, "max_ms": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
        
        sorted_durations = sorted(self.durations)
        n = len(sorted_durations)
        
        return {
            "count": n,
            "min_ms": min(sorted_durations),
            "max_ms": max(sorted_durations),
            "mean_ms": statistics.mean(sorted_durations),
            "p50_ms": sorted_durations[int(n * 0.5)],
            "p95_ms": sorted_durations[int(n * 0.95)] if n >= 20 else sorted_durations[-1],
            "p99_ms": sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1],
        }


class MetricsCollector:
    """Collects and aggregates metrics for Adaptive MDAP."""
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._timers: Dict[str, Timer] = {}
        self._lock = Lock()
    
    def counter(self, name: str) -> Counter:
        """Get or create counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name)
            return self._counters[name]
    
    def histogram(self, name: str) -> Histogram:
        """Get or create histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name)
            return self._histograms[name]
    
    def gauge(self, name: str) -> Gauge:
        """Get or create gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name)
            return self._gauges[name]
    
    def timer(self, name: str) -> Timer:
        """Get or create timer."""
        with self._lock:
            if name not in self._timers:
                self._timers[name] = Timer(name)
            return self._timers[name]
    
    def record_classification(self, duration_ms: float, success: bool) -> None:
        """Record classification metrics."""
        self.timer("classification_latency_ms").time(duration_ms)
        if success:
            self.counter("classification_success").increment()
        else:
            self.counter("classification_failure").increment()
    
    def record_allocation(
        self,
        strategy: str,
        complexity_score: float,
        duration_ms: float
    ) -> None:
        """Record allocation metrics."""
        self.counter(f"allocation_{strategy}").increment()
        self.histogram("complexity_score").observe(complexity_score)
        self.timer("allocation_latency_ms").time(duration_ms)
    
    def record_execution(
        self,
        strategy: str,
        success: bool,
        duration_ms: float,
        cost: float
    ) -> None:
        """Record execution metrics."""
        self.counter(f"execution_{strategy}").increment()
        if success:
            self.counter(f"execution_{strategy}_success").increment()
        else:
            self.counter(f"execution_{strategy}_failure").increment()
        self.timer(f"execution_{strategy}_latency_ms").time(duration_ms)
        self.histogram(f"execution_{strategy}_cost").observe(cost)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            return {
                "counters": {name: counter.get() for name, counter in self._counters.items()},
                "gauges": {name: gauge.get() for name, gauge in self._gauges.items()},
                "histograms": {name: hist.get_stats() for name, hist in self._histograms.items()},
                "timers": {name: timer.get_stats() for name, timer in self._timers.items()},
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, counter in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {counter.get()}")
            
            for name, gauge in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {gauge.get()}")
            
            for name, hist in self._histograms.items():
                stats = hist.get_stats()
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['mean'] * stats['count']}")
            
            for name, timer in self._timers.items():
                stats = timer.get_stats()
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['mean_ms'] * stats['count']}")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._timers.clear()


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics
