# BubbleLab Monitoring Troubleshooting Guide

Common issues and solutions for BubbleLab monitoring infrastructure.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Prometheus Issues](#prometheus-issues)
3. [Grafana Issues](#grafana-issues)
4. [Alertmanager Issues](#alertmanager-issues)
5. [Metrics Collection Issues](#metrics-collection-issues)
6. [Alerting Issues](#alerting-issues)
7. [Performance Issues](#performance-issues)
8. [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### Health Check Script

```bash
#!/bin/bash
# monitoring/health-check.sh

echo "Checking BubbleLab Monitoring Stack..."

# Check Prometheus
echo -n "Prometheus: "
if curl -sf http://localhost:9090/-/healthy > /dev/null; then
  echo "✓ Healthy"
else
  echo "✗ Unhealthy"
fi

# Check Grafana
echo -n "Grafana: "
if curl -sf http://localhost:3000/api/health > /dev/null; then
  echo "✓ Healthy"
else
  echo "✗ Unhealthy"
fi

# Check Alertmanager
echo -n "Alertmanager: "
if curl -sf http://localhost:9093/-/healthy > /dev/null; then
  echo "✓ Healthy"
else
  echo "✗ Unhealthy"
fi

# Check application metrics endpoint
echo -n "Application Metrics: "
if curl -sf http://localhost:8000/metrics > /dev/null; then
  echo "✓ Healthy"
else
  echo "✗ Unhealthy"
fi

echo "Health check complete!"
```

### Service Status Check

```bash
cd monitoring
docker-compose ps
```

Expected output:
```
NAME                        STATUS
bubblelab-prometheus        Up (healthy)
bubblelab-grafana           Up (healthy)
bubblelab-alertmanager      Up (healthy)
bubblelab-node-exporter     Up (healthy)
```

## Prometheus Issues

### Issue: Prometheus Not Starting

**Symptoms**:
- Container exits immediately
- Logs show "permission denied"
- Can't access http://localhost:9090

**Solutions**:

1. **Check permissions**:
```bash
chmod -R 777 monitoring/prometheus/data
```

2. **Check configuration**:
```bash
docker-compose config | grep prometheus -A 20
```

3. **View logs**:
```bash
docker-compose logs prometheus
```

4. **Validate config**:
```bash
docker run --rm -v $(pwd)/prometheus:/etc/prometheus \
  prom/prometheus:latest promtool check config /etc/prometheus/prometheus.yml
```

### Issue: Targets Showing as "DOWN"

**Symptoms**:
- Targets page shows red "DOWN" status
- No metrics being collected

**Solutions**:

1. **Verify application is running**:
```bash
curl http://localhost:8000/health
```

2. **Check metrics endpoint**:
```bash
curl http://localhost:8000/metrics
```

Expected: Prometheus metrics in text format

3. **Check network connectivity**:
```bash
# From within Prometheus container
docker exec -it bubblelab-prometheus sh
ping host.docker.internal
curl http://host.docker.internal:8000/metrics
```

4. **Verify Prometheus configuration**:
```yaml
scrape_configs:
  - job_name: 'bubblelab-api'
    static_configs:
      - targets: ['host.docker.internal:8000']  # Use host.docker.internal for Mac/Windows
    metrics_path: '/metrics'
    scrape_interval: 15s
```

5. **Check firewall rules**:
```bash
# Linux
sudo iptables -L | grep 8000

# Mac
sudo pfctl -s rules | grep 8000

# Windows
netsh advfirewall firewall show rule name=all | findstr 8000
```

### Issue: High Memory Usage

**Symptoms**:
- Prometheus consuming > 4GB RAM
- Container being killed by OOM killer

**Solutions**:

1. **Reduce retention time**:
```yaml
command:
  - '--storage.tsdb.retention.time=7d'  # Reduce from 15d
```

2. **Add memory limit**:
```yaml
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 4G
```

3. **Reduce scrape interval**:
```yaml
global:
  scrape_interval: 30s  # Increase from 15s
```

4. **Drop unnecessary metrics**:
```yaml
scrape_configs:
  - job_name: 'bubblelab-api'
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'go_.*|process_.*'
        action: drop
```

### Issue: No Metrics in Query Results

**Symptoms**:
- Queries return "No data"
- Dashboard panels show "No Data"

**Solutions**:

1. **Check time range**:
   - Ensure you're querying the right time period
   - Try "Last 5 minutes" to see recent data

2. **Verify metric name**:
```bash
curl http://localhost:8000/metrics | grep bubble_operation_total
```

3. **Check PromQL syntax**:
   - Use metric autocomplete
   - Validate in Prometheus UI

4. **Verify data collection**:
```bash
# Check if metrics are being scraped
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

## Grafana Issues

### Issue: Can't Login to Grafana

**Symptoms**:
- Login fails with "Invalid username or password"
- Forgot admin password

**Solutions**:

1. **Check environment variables**:
```bash
docker-compose exec grafana env | grep ADMIN
```

2. **Reset admin password**:
```bash
docker-compose exec grafana grafana-cli admin reset-admin-password admin newpassword
```

3. **Restart Grafana**:
```bash
docker-compose restart grafana
```

### Issue: Dashboards Not Loading

**Symptoms**:
- Dashboard shows "Dashboard not found"
- Panels show "Data source not found"

**Solutions**:

1. **Verify data source**:
   - Go to **Configuration > Data Sources**
   - Click "Test" on Prometheus data source
   - Should show "Data source is working"

2. **Check dashboard provisioning**:
```bash
ls -la monitoring/grafana/dashboards/
```

3. **Import dashboard manually**:
   - Go to **Dashboards > Import**
   - Upload JSON file from `monitoring/grafana/dashboards/`
   - Select Prometheus data source

4. **Check Grafana logs**:
```bash
docker-compose logs grafana | grep -i error
```

### Issue: Timezone Issues

**Symptoms**:
- Timestamps are incorrect
- Data appears in wrong time window

**Solutions**:

1. **Set Grafana timezone**:
   - Go to **Configuration > Preferences**
   - Set "Timezone" to your local timezone

2. **Set dashboard timezone**:
   - Click dashboard settings (gear icon)
   - Set "Timezone" to "Local browser time"

3. **Verify system timezone**:
```bash
date
timedatectl  # Linux
```

## Alertmanager Issues

### Issue: Alerts Not Firing

**Symptoms**:
- Alerts not appearing in Alertmanager
- Expected alerts are missing

**Solutions**:

1. **Check alert rules**:
   - Go to Prometheus UI: **Status > Rules**
   - Verify rules are loaded
   - Check evaluation state

2. **Test alert expression**:
   - Go to Prometheus UI: **Graph**
   - Enter alert expression
   - Verify query returns data

3. **Check Alertmanager connectivity**:
```bash
curl http://localhost:9093/-/healthy
```

4. **Verify Prometheus alerting config**:
```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Issue: Not Receiving Alert Notifications

**Symptoms**:
- Alerts firing but no notifications
- Slack/Email not receiving alerts

**Solutions**:

1. **Check Alertmanager configuration**:
```bash
docker-compose exec alertmanager cat /etc/alertmanager/alertmanager.yml
```

2. **Verify Slack webhook**:
```bash
curl -X POST https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK \
  -d '{"text":"Test message"}'
```

3. **Check Alertmanager logs**:
```bash
docker-compose logs alertmanager | grep -i error
```

4. **Verify notification channel**:
   - Go to http://localhost:9093
   - Check **Status** page
   - Verify receivers are configured

5. **Test notification**:
```bash
# Trigger test alert
curl -X POST http://localhost:9093/api/v1/alerts -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning"
  },
  "annotations": {
    "description": "This is a test alert"
  }
}]'
```

## Metrics Collection Issues

### Issue: Metrics Not Being Generated

**Symptoms**:
- `/metrics` endpoint returns empty
- No metrics in Prometheus

**Solutions**:

1. **Verify metrics are registered**:
```typescript
import { register } from '@bubblelab/bubble-core/src/metrics';

// Check if metrics are registered
console.log(await register.getMetricsAsJson());
```

2. **Check middleware is applied**:
```typescript
import { metricsMiddleware } from '@bubblelab/bubble-core/src/metrics';

app.use(metricsMiddleware('your-bubble-name'));
```

3. **Verify metrics endpoint**:
```typescript
import { metricsEndpoint } from '@bubblelab/bubble-core/src/metrics';

app.get('/metrics', metricsEndpoint);
```

4. **Test metrics generation**:
```bash
curl http://localhost:8000/metrics | grep bubble_
```

### Issue: Incorrect Metric Labels

**Symptoms**:
- Metrics missing expected labels
- Dashboard variables not working

**Solutions**:

1. **Verify label names**:
```typescript
bubbleOperationTotal.inc({
  bubble: 'your-bubble-name',
  operation: 'createPayment',
  status: 'success'
});
```

2. **Check dashboard variables**:
   - Go to dashboard settings
   - Verify variable query: `label_values(bubble_operation_total, bubble)`

3. **Query to check labels**:
```promql
# Show all label values
bubble_operation_total

# Group by labels
sum(bubble_operation_total) by (bubble, operation, status)
```

## Alerting Issues

### Issue: False Alerts

**Symptoms**:
- Alerts firing when they shouldn't
- Too many notifications

**Solutions**:

1. **Adjust threshold**:
```yaml
# Before
expr: error_rate > 0.1

# After
expr: error_rate > 0.2
```

2. **Increase duration**:
```yaml
# Before
for: 2m

# After
for: 10m
```

3. **Add hysteresis**:
```yaml
# Use separate thresholds for alerting and recovery
- alert: HighErrorRate
  expr: error_rate > 0.2
  for: 5m
  annotations:
    summary: "High error rate detected"

- alert: ErrorRateRecovered
  expr: error_rate < 0.05
  for: 10m
  annotations:
    summary: "Error rate recovered"
```

### Issue: Alert Flooding

**Symptoms**:
- Hundreds of alerts in short time
- Notification channel overwhelmed

**Solutions**:

1. **Configure alert grouping**:
```yaml
route:
  group_by: ['alertname', 'bubble']
  group_wait: 30s
  group_interval: 5m
```

2. **Set repeat interval**:
```yaml
route:
  repeat_interval: 12h
```

3. **Use inhibition rules**:
```yaml
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['bubble', 'alertname']
```

## Performance Issues

### Issue: Slow Dashboard Queries

**Symptoms**:
- Dashboards take > 10 seconds to load
- Queries timeout

**Solutions**:

1. **Reduce time range**:
   - Query smaller time windows
   - Use "Last 1h" instead of "Last 7d"

2. **Optimize queries**:
```promql
# Before: Expensive subquery
rate(bubble_operation_total[5m])[10m:]

# After: Pre-aggregated rate
rate(bubble_operation_total[10m])
```

3. **Use recording rules**:
```yaml
# Add to prometheus.yml
groups:
  - name: performance
    interval: 30s
    rules:
      - record: job:bubble_operation_rate:5m
        expr: rate(bubble_operation_total[5m])
```

4. **Add query timeout**:
   - Go to **Configuration > Data Sources**
   - Set "Timeout" to 30s

### Issue: High CPU Usage

**Symptoms**:
- Prometheus using 100% CPU
- Queries running slowly

**Solutions**:

1. **Reduce scrape interval**:
```yaml
global:
  scrape_interval: 30s
```

2. **Drop expensive metrics**:
```yaml
metric_relabel_configs:
  - source_labels: [__name__]
    regex: 'expensive_metric_.*'
    action: drop
```

3. **Increase evaluation interval**:
```yaml
global:
  evaluation_interval: 30s
```

## Advanced Debugging

### Enable Debug Logging

**Prometheus**:
```yaml
command:
  - '--log.level=debug'
```

**Grafana**:
```yaml
environment:
  - GF_LOG_LEVEL=debug
```

**Alertmanager**:
```yaml
command:
  - '--log.level=debug'
```

### Query Performance Analysis

```promql
# Show query statistics
prometheus_tsdb_symbol_table_size_bytes
prometheus_tsdb_compaction_duration_seconds

# Show target statistics
prometheus_target_interval_length_seconds
prometheus_target_scrapes_sample_duplicate
```

### Metrics Cardinality Check

```bash
# Check high cardinality metrics
curl http://localhost:9090/api/v1/label/__name__/values | \
  jq '.data[]' | \
  while read metric; do
    count=$(curl -s "http://localhost:9090/api/v1/label/__name__/values" | \
      jq ".data | length")
    echo "$metric: $count"
  done | \
  sort -t: -k2 -rn | \
  head -20
```

### Network Debugging

```bash
# Test Prometheus to application connectivity
docker exec -it bubblelab-prometheus wget -O- http://host.docker.internal:8000/metrics

# Test Grafana to Prometheus connectivity
docker exec -it bubblelab-grafana wget -O- http://prometheus:9090/api/v1/query?query=up

# Test Alertmanager to Slack connectivity
docker exec -it bubblelab-alertmanager wget -O- --post-data='{"text":"test"}' \
  https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

## Getting Help

### Collect Diagnostic Information

```bash
#!/bin/bash
# monitoring/diagnostics.sh

OUTPUT_DIR="diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Prometheus diagnostics
echo "Collecting Prometheus diagnostics..."
curl http://localhost:9090/api/v1/status/config > "$OUTPUT_DIR/prometheus-config.json"
curl http://localhost:9090/api/v1/status/flags > "$OUTPUT_DIR/prometheus-flags.json"
curl http://localhost:9090/api/v1/targets > "$OUTPUT_DIR/prometheus-targets.json"
docker-compose logs prometheus > "$OUTPUT_DIR/prometheus.log"

# Grafana diagnostics
echo "Collecting Grafana diagnostics..."
curl http://localhost:3000/api/health > "$OUTPUT_DIR/grafana-health.json"
docker-compose logs grafana > "$OUTPUT_DIR/grafana.log"

# Alertmanager diagnostics
echo "Collecting Alertmanager diagnostics..."
curl http://localhost:9093/api/v1/status > "$OUTPUT_DIR/alertmanager-status.json"
docker-compose logs alertmanager > "$OUTPUT_DIR/alertmanager.log"

# Application diagnostics
echo "Collecting application diagnostics..."
curl http://localhost:8000/metrics > "$OUTPUT_DIR/app-metrics.txt"
curl http://localhost:8000/health > "$OUTPUT_DIR/app-health.json"

echo "Diagnostics collected to: $OUTPUT_DIR"
tar -czf "$OUTPUT_DIR.tar.gz" "$OUTPUT_DIR"
```

### Log Locations

```
monitoring/prometheus/data/     # Prometheus data and logs
monitoring/grafana/data/        # Grafana data and logs
monitoring/alertmanager/data/   # Alertmanager data and logs
logs/                           # Application logs
```

### Useful Resources

- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
- Alertmanager Documentation: https://prometheus.io/docs/alerting/latest/alertmanager/
- PromQL Tips: https://promlabs.com/promql-tutorial

## Next Steps

- Run diagnostics script
- Check logs for errors
- Verify configuration files
- Test service connectivity
- Review metrics and queries
