# Gauntlet Monitoring System

Comprehensive production monitoring, metrics collection, and alerting for the Gauntlet evaluation system.

## Overview

The monitoring system provides:

1. **Metrics Collection** (`metrics.py`)
   - Execution metrics (total runs, pass/fail rates, duration)
   - ML component metrics (optimization, predictions, training)
   - System metrics (CPU, memory, WebSocket connections)
   - Prometheus-compatible export format

2. **Health Checks** (`health_checks.py`)
   - Liveness probes (is the service running?)
   - Readiness probes (can it handle traffic?)
   - Dependency health checks
   - Resource monitoring

3. **Alerting** (`alerting.py`)
   - Configurable alert rules with thresholds
   - Multiple severity levels (CRITICAL, WARNING, INFO)
   - Alert lifecycle management
   - Multiple notification channels

## Quick Start

### Basic Usage

```python
from glue.adapters.gauntlet_adapter.monitoring import (
    get_metrics_collector,
    get_health_checker,
    get_alerting_engine
)

# Get instances
metrics = get_metrics_collector()
health = get_health_checker()
alerts = get_alerting_engine()

# Record a gauntlet execution
metrics.record_execution(
    domain="finance",
    passed=True,
    duration_ms=1234.5,
    score=0.85,
    rounds_completed=3
)

# Check system health
if health.is_ready():
    print("System is ready to handle traffic")

# Export metrics for Prometheus
prometheus_metrics = metrics.export_prometheus()
print(prometheus_metrics)

# Evaluate alerts
triggered_alerts = alerts.evaluate()
for alert in triggered_alerts:
    print(f"ALERT: {alert.message}")
```

### Integration with Gauntlet Executor

```python
from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import PredictiveGauntletExecutor
from glue.adapters.gauntlet_adapter.monitoring import get_metrics_collector

executor = PredictiveGauntletExecutor()
metrics = get_metrics_collector()

# Run gauntlet with monitoring
result = executor.execute_with_prediction(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance"
)

# Record metrics
metrics.record_execution(
    domain="finance",
    passed=result.actual_outcome.get("passed", False),
    duration_ms=result.execution_time * 1000,
    score=result.actual_outcome.get("score", 0.0)
)
```

## Metrics Reference

### Execution Metrics

| Metric Name | Type | Description |
|------------|------|-------------|
| `gauntlet_executions_total` | Counter | Total number of gauntlet executions |
| `gauntlet_passes_total` | Counter | Total number of passed gauntlets |
| `gauntlet_failures_total` | Counter | Total number of failed gauntlets |
| `gauntlet_last_duration_ms` | Gauge | Last execution duration in milliseconds |
| `gauntlet_last_score` | Gauge | Last execution score |
| `gauntlet_duration_seconds` | Histogram | Execution duration distribution |

### ML Component Metrics

| Metric Name | Type | Description |
|------------|------|-------------|
| `optimization_iterations_total` | Counter | Total optimization iterations |
| `optimization_best_score` | Gauge | Best optimization score found |
| `prediction_accuracy` | Gauge | ML prediction accuracy |
| `training_loss` | Gauge | Current training loss |
| `model_convergence_total` | Counter | Total model convergences |

### System Metrics

| Metric Name | Type | Description |
|------------|------|-------------|
| `system_cpu_usage_percent` | Gauge | CPU usage percentage |
| `system_memory_usage_percent` | Gauge | Memory usage percentage |
| `system_disk_usage_percent` | Gauge | Disk usage percentage |
| `websocket_connections_active` | Gauge | Active WebSocket connections |
| `gauntlets_active` | Gauge | Active gauntlet executions |
| `gauntlet_uptime_seconds` | Gauge | System uptime in seconds |

## Health Check Endpoints

### Liveness Probe

```python
from glue.adapters.gauntlet_adapter.monitoring import check_liveness

health_status = check_liveness()
print(health_status)
# {
#     "overall_status": "healthy",
#     "is_healthy": true,
#     "uptime_seconds": 1234.56,
#     "components": {...}
# }
```

### Readiness Probe

```python
from glue.adapters.gauntlet_adapter.monitoring import check_readiness

readiness_status = check_readiness()
print(readiness_status)
# {
#     "overall_status": "healthy",
#     "is_ready": true,
#     "components": {
#         "gauntlet_executor": {...},
#         "ml_components": {...},
#         "websocket_server": {...}
#     }
# }
```

## Alerting Configuration

### Default Alert Rules

The system includes pre-configured alert rules:

1. **High Error Rate** - Triggers when error rate > 10%
2. **High Latency** - Triggers when execution time > 5 seconds
3. **Low Pass Rate** - Triggers when pass rate < 50%
4. **High Memory Usage** - Triggers when memory > 85%
5. **High CPU Usage** - Triggers when CPU > 80%
6. **Low Prediction Accuracy** - Triggers when accuracy < 60%

### Adding Custom Alert Rules

```python
from glue.adapters.gauntlet_adapter.monitoring import get_alerting_engine, AlertRule, AlertSeverity

alerts = get_alerting_engine()

# Define custom condition
def custom_condition(metrics):
    return metrics.get("custom_metric", 0) > 100

# Add rule
alerts.add_rule(AlertRule(
    name="custom_alert",
    severity=AlertSeverity.WARNING,
    condition_fn=custom_condition,
    message_template="Custom metric is {value} (threshold: {threshold})",
    threshold=100,
    cooldown_seconds=300
))
```

### Notification Channels

```python
from glue.adapters.gauntlet_adapter.monitoring import (
    get_alerting_engine,
    WebhookNotificationChannel
)

alerts = get_alerting_engine()

# Add webhook notification
webhook = WebhookNotificationChannel(
    url="https://your-webhook-url.com/alerts",
    timeout=5
)
alerts.add_notification_channel(webhook)
```

### Alert Management

```python
# Get active alerts
active_alerts = alerts.get_active_alerts()
for alert in active_alerts:
    print(f"{alert.severity.value}: {alert.message}")

# Acknowledge an alert
alerts.acknowledge_alert(alert.alert_id)

# Resolve an alert
alerts.resolve_alert(alert.alert_id)

# Get alert statistics
stats = alerts.get_alert_statistics()
print(stats)
```

## Prometheus Integration

### 1. Export Metrics Endpoint

Create an HTTP endpoint to expose metrics:

```python
from flask import Flask, Response
from glue.adapters.gauntlet_adapter.monitoring import export_prometheus

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(
        export_prometheus(),
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(port=9090)
```

### 2. Configure Prometheus

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'gauntlet'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

### 3. Load Alert Rules

Add to `prometheus.yml`:

```yaml
rule_files:
  - 'gauntlet_alerts.yml'
```

See `PROMETHEUS_ALERTS.yml` for example alert rules.

## Grafana Dashboard

Import the provided dashboard configuration:

1. Open Grafana
2. Go to Dashboards -> Import
3. Upload `GRAFANA_DASHBOARD.json`
4. Select your Prometheus data source

The dashboard includes panels for:
- Execution rates and pass/fail ratios
- Duration percentiles (p50, p95, p99)
- Score trends by domain
- System resource usage
- ML prediction accuracy
- Training loss and optimization progress

## Advanced Usage

### Custom Health Checks

```python
from glue.adapters.gauntlet_adapter.monitoring import get_health_checker, HealthCheckResult, HealthStatus

health = get_health_checker()

def custom_check() -> HealthCheckResult:
    # Your custom health check logic
    try:
        # Check something
        is_healthy = check_something()

        return HealthCheckResult(
            component="custom_component",
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            message="Custom check passed" if is_healthy else "Custom check failed"
        )
    except Exception as e:
        return HealthCheckResult(
            component="custom_component",
            status=HealthStatus.UNHEALTHY,
            message=f"Check failed: {str(e)}"
        )

health.register_custom_check("custom", custom_check)
```

### Dependency Monitoring

```python
from glue.adapters.gauntlet_adapter.monitoring import get_health_checker

health = get_health_checker()

# Add external dependency to monitor
health.add_dependency(
    name="ml_service",
    url="http://ml-service:8000",
    health_check_url="http://ml-service:8000/health"
)

# Run dependency checks
results = health.check_all()
for name, result in results.items():
    print(f"{name}: {result.status.value}")
```

### ML Metrics Tracking

```python
from glue.adapters.gauntlet_adapter.monitoring import get_metrics_collector

metrics = get_metrics_collector()

# Record optimization iteration
metrics.record_optimization_iteration(
    strategy="q_learning",
    iteration=10,
    score=0.85,
    improvement=0.15
)

# Record prediction
metrics.record_prediction(
    success_probability=0.75,
    confidence=0.80,
    actual_outcome=True,
    domain="finance"
)

# Record training metrics
metrics.record_training_metrics(
    loss=0.123,
    converged=True,
    epoch=50
)

# Get ML metrics summary
ml_metrics = metrics.get_ml_metrics()
print(ml_metrics)
```

## Best Practices

1. **Record Metrics Early**: Record metrics as soon as possible after execution
2. **Use Labels**: Add labels to metrics for better filtering (e.g., domain, strategy)
3. **Set Appropriate Thresholds**: Adjust alert thresholds based on your workload
4. **Monitor Dependencies**: Keep track of external service health
5. **Review Alerts Regularly**: Tune alert rules to reduce noise
6. **Export Metrics Frequently**: Set Prometheus scrape interval to 15s or less

## Troubleshooting

### Metrics Not Appearing

- Check that metrics are being recorded with `metrics.get_metric_summary()`
- Verify Prometheus scrape configuration
- Check firewall rules allow Prometheus to reach metrics endpoint

### Health Checks Failing

- Check component-specific error messages in health check results
- Verify dependencies are reachable
- Check system resource availability

### Alerts Not Triggering

- Verify alert rules are enabled
- Check cooldown periods (won't alert if recently triggered)
- Verify condition function is returning True
- Check alert logs for errors

## License

MIT License - See LICENSE file for details
