# Gauntlet Metrics and Monitoring Guide

This guide explains how to monitor and observe the OpenEvolve Gauntlet system using the comprehensive metrics framework.

## Table of Contents

1. [Overview](#overview)
2. [Metric Types](#metric-types)
3. [Collecting Metrics](#collecting-metrics)
4. [Performance Monitoring](#performance-monitoring)
5. [Team Performance Analytics](#team-performance-analytics)
6. [Cache Monitoring](#cache-monitoring)
7. [Checkpoint Monitoring](#checkpoint-monitoring)
8. [Fuzzing Metrics](#fuzzing-metrics)
9. [System Resource Monitoring](#system-resource-monitoring)
10. [Exporting Metrics](#exporting-metrics)

---

## Overview

The Gauntlet metrics system provides comprehensive observability for all aspects of problem-solving operations.

### Key Features

- **Real-time metrics collection** for all operations
- **Performance tracking** with detailed timing data
- **Team analytics** for performance comparison
- **Cache statistics** for hit rates and efficiency
- **Checkpoint health** monitoring
- **Fuzzing results** and vulnerability tracking
- **System resource** monitoring (CPU, memory, I/O)

### Getting Started

```python
from bubblelabs_nodes import get_metrics_collector

# Get the global metrics collector
collector = get_metrics_collector()

# Start monitoring
collector.start_resource_monitoring()

# Your operations here...

# Get metrics report
metrics = collector.get_all_metrics()
print(metrics)
```

---

## Metric Types

### Counters

Monotonically increasing values for counting events.

```python
from bubblelabs_nodes import get_metrics_collector

collector = get_metrics_collector()

# Increment counter
collector.increment("problems_solved")
collector.increment("problems_solved", value=5)

# Increment with labels
collector.increment(
    "team_attempts",
    value=1,
    labels={"team": "blue_team_1", "domain": "web"}
)

# Get counter value
value = collector.get_counter("problems_solved")
print(f"Problems solved: {value}")
```

**Use cases:**
- Total problems solved
- Total attempts per team
- Total cache hits/misses
- Total checkpoint operations

### Gauges

Point-in-time values that can go up or down.

```python
# Set gauge value
collector.set_gauge("active_problems", 5)
collector.set_gauge("queue_size", 12)

# Set gauge with labels
collector.set_gauge(
    "team_active_problems",
    3,
    labels={"team": "blue_team_1"}
)

# Get gauge value
active = collector.get_gauge("active_problems")
print(f"Active problems: {active}")
```

**Use cases:**
- Current active problems
- Queue sizes
- Memory usage
- CPU percentage

### Histograms

Distributions of values (durations, sizes, etc.).

```python
# Record histogram value
collector.record_histogram("solve_duration_ms", 150.5)
collector.record_histogram("solve_duration_ms", 200.3)
collector.record_histogram("solve_duration_ms", 120.1)

# Get histogram statistics
stats = collector.get_histogram_stats("solve_duration_ms")
print(f"Average: {stats['avg']:.1f}ms")
print(f"P95: {stats['p95']:.1f}ms")
print(f"P99: {stats['p99']:.1f}ms")
```

**Use cases:**
- Operation durations
- Solution scores
- Problem sizes
- Team performance scores

**Available statistics:**
- `count`: Total number of values
- `sum`: Sum of all values
- `avg`: Average value
- `min`: Minimum value
- `max`: Maximum value
- `p50`: 50th percentile (median)
- `p90`: 90th percentile
- `p95`: 95th percentile
- `p99`: 99th percentile

---

## Collecting Metrics

### Performance Metrics

Track operation performance automatically.

```python
from bubblelabs_nodes import get_metrics_collector, track_performance

collector = get_metrics_collector()

# Manual recording
collector.record_performance(
    operation="solve_problem",
    duration_ms=150.5,
    success=True,
    metadata={"problem_id": "problem_123", "team": "blue_team_1"}
)

# Automatic recording with decorator
@track_performance("solve_problem")
async def solve_problem(problem):
    # Your implementation
    return result

# Get performance summary
summary = collector.get_performance_summary("solve_problem")
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Average duration: {summary['avg_duration_ms']:.1f}ms")
```

### Team Performance Metrics

Track team performance over time.

```python
# Record team performance
collector.record_team_performance(
    team_id="blue_team_1",
    problem_id="problem_123",
    domain="web",
    difficulty=3,
    success=True,
    score=0.85,
    execution_time=150.0
)

# Get team performance summary
summary = collector.get_team_performance_summary("blue_team_1")
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Average score: {summary['avg_score']:.2f}")
print(f"Average time: {summary['avg_execution_time']:.1f}s")
print(f"Domain success rates: {summary['domain_success_rates']}")
```

**Output:**
```
Success rate: 85.0%
Average score: 0.82
Average time: 145.3s
Domain success rates: {'web': 0.85, 'ml': 0.80, 'data': 0.90}
```

### Cache Metrics

Monitor cache performance.

```python
# Record cache operations
collector.record_cache_operation(
    operation="hit",  # or "miss", "set", "invalidate"
    cache_type="memory",
    key="problem_hash_123",
    metadata={"retrieval_time_ms": 0.5}
)

# Get cache summary
summary = collector.get_cache_summary("memory")
print(f"Hit rate: {summary['hit_rate']:.1%}")
print(f"Total requests: {summary['total_requests']}")
print(f"Hits: {summary['hits']}")
print(f"Misses: {summary['misses']}")
```

**Output:**
```
Hit rate: 75.0%
Total requests: 100
Hits: 75
Misses: 25
```

### Checkpoint Metrics

Monitor checkpoint health and performance.

```python
# Record checkpoint operation
collector.record_checkpoint_operation(
    operation="create",  # or "load", "delete", "cleanup"
    problem_id="problem_123",
    checkpoint_id="cp_123_456",
    success=True,
    size_bytes=1024000,  # 1 MB
    duration_ms=150.5
)

# Get checkpoint summary
summary = collector.get_checkpoint_summary("problem_123")
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Average size: {summary['avg_size_bytes'] / 1024:.1f} KB")
print(f"Average duration: {summary['avg_duration_ms']:.1f}ms")
print(f"Operations: {summary['operations_by_type']}")
```

**Output:**
```
Success rate: 95.0%
Average size: 950.5 KB
Average duration: 145.2ms
Operations: {
    'create': {'count': 5, 'successes': 5, 'avg_size_bytes': 1024000},
    'load': {'count': 3, 'successes': 3, 'avg_size_bytes': 1024000}
}
```

### Fuzzing Metrics

Track vulnerability discovery.

```python
# Record fuzzing results
collector.record_fuzzing_results(
    problem_id="problem_123",
    iterations=1000,
    crashes_found=2,
    vulnerabilities_found=1,
    duration_seconds=30.5
)

# Get fuzzing summary
summary = collector.get_fuzzing_summary("problem_123")
print(f"Crash rate: {summary['crash_rate']:.2%}")
print(f"Vulnerability rate: {summary['vulnerability_rate']:.2%}")
print(f"Total crashes: {summary['total_crashes']}")
print(f"Total vulnerabilities: {summary['total_vulnerabilities']}")
```

**Output:**
```
Crash rate: 0.20%
Vulnerability rate: 0.10%
Total crashes: 2
Total vulnerabilities: 1
```

---

## Performance Monitoring

### Operation Duration Tracking

```python
import time
from bubblelabs_nodes import get_metrics_collector

collector = get_metrics_collector()

async def monitored_solve(problem):
    start_time = time.time()

    try:
        result = await solve_internal(problem)
        duration_ms = (time.time() - start_time) * 1000

        collector.record_performance(
            operation="solve_problem",
            duration_ms=duration_ms,
            success=True
        )

        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        collector.record_performance(
            operation="solve_problem",
            duration_ms=duration_ms,
            success=False
        )

        raise
```

### Performance Summary

```python
# Get summary for specific operation
summary = collector.get_performance_summary("solve_problem")

print(f"Total requests: {summary['total_requests']}")
print(f"Successful: {summary['successful_requests']}")
print(f"Failed: {summary['failed_requests']}")
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Average duration: {summary['avg_duration_ms']:.1f}ms")
print(f"P50 duration: {summary['p50_duration_ms']:.1f}ms")
print(f"P90 duration: {summary['p90_duration_ms']:.1f}ms")
print(f"P95 duration: {summary['p95_duration_ms']:.1f}ms")
print(f"P99 duration: {summary['p99_duration_ms']:.1f}ms")
```

**Output:**
```
Total requests: 100
Successful: 85
Failed: 15
Success rate: 85.0%
Average duration: 150.5ms
P50 duration: 145.0ms
P90 duration: 180.0ms
P95 duration: 195.0ms
P99 duration: 220.0ms
```

---

## Team Performance Analytics

### Comparing Team Performance

```python
# Record performance for multiple teams
for team_id in ["blue_team_1", "blue_team_2", "red_team_1"]:
    collector.record_team_performance(
        team_id=team_id,
        problem_id="problem_123",
        domain="web",
        difficulty=3,
        success=True,
        score=0.85,
        execution_time=150.0
    )

# Compare teams
teams = ["blue_team_1", "blue_team_2", "red_team_1"]
for team_id in teams:
    summary = collector.get_team_performance_summary(team_id)
    print(f"\n{team_id}:")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Average score: {summary['avg_score']:.2f}")
    print(f"  Average time: {summary['avg_execution_time']:.1f}s")
```

### Domain-Specific Performance

```python
summary = collector.get_team_performance_summary("blue_team_1")

print("Domain-specific success rates:")
for domain, rate in summary['domain_success_rates'].items():
    print(f"  {domain}: {rate:.1%}")
```

**Output:**
```
Domain-specific success rates:
  web: 85.0%
  ml: 80.0%
  data: 90.0%
  security: 75.0%
```

---

## Cache Monitoring

### Cache Efficiency

```python
# Monitor cache hit rate over time
summary = collector.get_cache_summary("memory")

if summary['hit_rate'] < 0.5:
    print("WARNING: Low cache hit rate!")
    print("Consider:")
    print("  - Increasing cache size")
    print("  - Adjusting TTL")
    print("  - Analyzing cache key patterns")

print(f"\nCache statistics:")
print(f"  Hit rate: {summary['hit_rate']:.1%}")
print(f"  Miss rate: {summary['miss_rate']:.1%}")
print(f"  Total requests: {summary['total_requests']}")
```

### Cache Optimization

```python
from bubblelabs_nodes import create_config

summary = collector.get_cache_summary()

# Optimize based on metrics
if summary['hit_rate'] < 0.5:
    # Increase cache size
    config = create_config()
    config.cache.max_size = 2000  # Double the size

if summary['miss_rate'] > 0.8:
    # Adjust TTL
    config.cache.ttl_seconds = 7200  # 2 hours
```

---

## Checkpoint Monitoring

### Checkpoint Health

```python
# Monitor checkpoint health
summary = collector.get_checkpoint_summary()

if summary['success_rate'] < 0.95:
    print("WARNING: Checkpoint failures detected!")
    print("Review:")
    print("  - Disk space")
    print("  - File permissions")
    print("  - Storage path")

print(f"\nCheckpoint health:")
print(f"  Success rate: {summary['success_rate']:.1%}")
print(f"  Average size: {summary['avg_size_bytes'] / 1024:.1f} KB")
print(f"  Average duration: {summary['avg_duration_ms']:.1f}ms")
```

### Checkpoint Size Monitoring

```python
# Monitor checkpoint size growth
summary = collector.get_checkpoint_summary()

avg_size_mb = summary['avg_size_bytes'] / (1024 * 1024)

if avg_size_mb > 10:
    print("WARNING: Large checkpoints!")
    print("Consider:")
    print("  - Enabling compression")
    print("  - Reducing checkpoint frequency")
    print("  - Cleaning context before checkpointing")
```

---

## Fuzzing Metrics

### Vulnerability Discovery Rate

```python
# Track vulnerability discovery
summary = collector.get_fuzzing_summary()

print(f"Fuzzing results:")
print(f"  Total iterations: {summary['total_iterations']}")
print(f"  Crashes found: {summary['total_crashes']}")
print(f"  Vulnerabilities found: {summary['total_vulnerabilities']}")
print(f"  Crash rate: {summary['crash_rate']:.2%}")
print(f"  Vulnerability rate: {summary['vulnerability_rate']:.2%}")

# Alert on high crash rates
if summary['crash_rate'] > 0.05:  # 5%
    print("ALERT: High crash rate detected!")
    print("Immediate investigation required!")
```

### Fuzzing Efficiency

```python
# Calculate fuzzing efficiency
iterations_per_vulnerability = (
    summary['total_iterations'] / summary['total_vulnerabilities']
    if summary['total_vulnerabilities'] > 0
    else float('inf')
)

print(f"Fuzzing efficiency:")
print(f"  Iterations per vulnerability: {iterations_per_vulnerability:.0f}")

if iterations_per_vulnerability < 100:
    print("  EXCELLENT: Highly efficient fuzzing")
elif iterations_per_vulnerability < 1000:
    print("  GOOD: Normal fuzzing efficiency")
else:
    print("  WARNING: Low fuzzing efficiency")
```

---

## System Resource Monitoring

### Enable Resource Monitoring

```python
from bubblelabs_nodes import get_metrics_collector

collector = get_metrics_collector()

# Start monitoring (runs in background thread)
collector.start_resource_monitoring(interval_seconds=5.0)

# Do work...

# Stop monitoring
collector.stop_resource_monitoring()
```

### Resource Metrics

```python
# Resource metrics are automatically collected as gauges
cpu_percent = collector.get_gauge('process_cpu_percent')
memory_rss = collector.get_gauge('process_memory_rss_bytes')
memory_percent = collector.get_gauge('process_memory_percent')

print(f"Resource usage:")
print(f"  CPU: {cpu_percent:.1f}%")
print(f"  Memory: {memory_rss / (1024**3):.2f} GB")
print(f"  Memory: {memory_percent:.1f}%")
```

### Resource Alerts

```python
# Check for resource issues
cpu = collector.get_gauge('process_cpu_percent')
memory = collector.get_gauge('process_memory_percent')

if cpu > 80:
    print("WARNING: High CPU usage!")
    print("Consider:")
    print("  - Reducing parallelism")
    print("  - Optimizing algorithms")

if memory > 80:
    print("WARNING: High memory usage!")
    print("Consider:")
    print("  - Reducing cache size")
    print("  - Cleaning up old checkpoints")
    print("  - Optimizing data structures")
```

---

## Exporting Metrics

### Complete Metrics Report

```python
# Get all metrics
all_metrics = collector.get_all_metrics()

print("Complete metrics report:")
print(f"Timestamp: {all_metrics['timestamp']}")
print(f"Counters: {len(all_metrics['counters'])}")
print(f"Gauges: {len(all_metrics['gauges'])}")
print(f"\nPerformance:")
for op, summary in all_metrics['performance'].items():
    print(f"  {op}: {summary['success_rate']:.1%} success")
```

### JSON Export

```python
import json

# Export metrics to JSON
all_metrics = collector.get_all_metrics()

with open('metrics_report.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

print("Metrics exported to metrics_report.json")
```

### Prometheus Integration

```python
# Export metrics in Prometheus format
def export_prometheus(collector):
    lines = []

    # Counters
    for key, value in collector._counters.items():
        lines.append(f"# TYPE {key} counter")
        lines.append(f"{key} {value}")

    # Gauges
    for key, value in collector._gauges.items():
        lines.append(f"# TYPE {key} gauge")
        lines.append(f"{key} {value}")

    # Histograms
    for key, values in collector._histograms.items():
        lines.append(f"# TYPE {key} histogram")
        for i, value in enumerate(values):
            lines.append(f'{key}{{le="{i}"}} {value}')

    return '\n'.join(lines)

# Export
prometheus_metrics = export_prometheus(collector)
print(prometheus_metrics)
```

### Grafana Dashboard

Create a Grafana dashboard with these panels:

**Panel 1: Success Rate**
```
Metric: solve_problem_success_rate
Type: Gauge
Query: avg(success_rate) by (operation)
```

**Panel 2: Operation Duration**
```
Metric: solve_operation_duration_ms
Type: Histogram
Query: histogram_quantile(0.95, duration_ms)
```

**Panel 3: Cache Hit Rate**
```
Metric: cache_hit_rate
Type: Gauge
Query: avg(hit_rate) by (cache_type)
```

**Panel 4: Resource Usage**
```
Metric: process_cpu_percent, process_memory_percent
Type: Gauge
Query: avg(cpu_percent), avg(memory_percent)
```

---

## Best Practices

### 1. Start Monitoring Early

```python
# Enable monitoring at application startup
collector = get_metrics_collector()
collector.start_resource_monitoring()
```

### 2. Use Labels Effectively

```python
# Good: Descriptive labels
collector.increment("problems_solved", labels={
    "domain": "web",
    "difficulty": "3",
    "team": "blue_team_1"
})

# Bad: No labels
collector.increment("problems_solved")
```

### 3. Set Up Alerts

```python
# Define alert thresholds
ALERT_THRESHOLDS = {
    "success_rate": 0.7,
    "cpu_percent": 80.0,
    "memory_percent": 80.0,
    "cache_hit_rate": 0.5,
}

# Check thresholds
summary = collector.get_performance_summary()
if summary['success_rate'] < ALERT_THRESHOLDS["success_rate"]:
    send_alert("Low success rate!")
```

### 4. Monitor Trends

```python
# Track metrics over time
import time

while True:
    metrics = collector.get_all_metrics()
    store_metrics(metrics)  # Your storage function
    time.sleep(60)  # Every minute
```

### 5. Clean Up Old Metrics

```python
# Reset metrics periodically to prevent memory growth
if len(collector._metric_history) > 10000:
    collector.reset_metrics()
```

---

## Summary

The Gauntlet metrics system provides:
- ✅ Comprehensive metrics collection (counters, gauges, histograms)
- ✅ Performance monitoring with percentiles
- ✅ Team analytics and comparison
- ✅ Cache, checkpoint, and fuzzing metrics
- ✅ System resource monitoring
- ✅ Easy export to JSON, Prometheus, Grafana

For more information:
- See `bubblelabs_nodes/gauntlet_metrics.py` for implementation
- See `CONFIGURATION_GUIDE.md` for configuration options
- See `CHECKPOINTING_GUIDE.md` for checkpointing details
