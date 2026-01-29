# Production Monitoring Configuration

## Overview

This document describes the production monitoring setup for the BubbleLab system, including Prometheus alerts, Grafana dashboards, log aggregation, and on-call procedures.

**Monitoring Stack**:
- Prometheus: Metrics collection
- Grafana: Visualization
- Alertmanager: Alert routing
- Loki: Log aggregation
- Jaeger: Distributed tracing (optional)

---

## Table of Contents

1. [Prometheus Configuration](#prometheus-configuration)
2. [Grafana Dashboards](#grafana-dashboards)
3. [Alertmanager Configuration](#alertmanager-configuration)
4. [Log Aggregation](#log-aggregation)
5. [Uptime Monitoring](#uptime-monitoring)
6. [Anomaly Detection](#anomaly-detection)
7. [On-Call Runbook](#on-call-runbook)

---

## Prometheus Configuration

### Installation

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager-data://alertmanager
    networks:
      - monitoring

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
  alertmanager-data:
  loki-data:
```

### Prometheus Configuration File

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'bubblelab-production'
    replica: '1'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load rules once and periodically evaluate them
rule_files:
  - "alerts/*.yml"

# Scrape configurations
scrape_configs:
  # BubbleLab API
  - job_name: 'bubblelab-api'
    static_configs:
      - targets: ['bubblelab-api:3000']
    metrics_path: '/metrics'
    scrape_interval: 15s

  # Service Bubbles
  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  - job_name: 'elasticsearch'
    static_configs:
      - targets: ['elasticsearch:9200']
    metrics_path: '/_prometheus/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Node Exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor (container metrics)
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Alert Rules

```yaml
# alerts/bubblelab.yml
groups:
  - name: bubblelab_alerts
    interval: 30s
    rules:
      # Circuit Breaker Alerts
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state{state="open"} == 1
        for: 1m
        labels:
          severity: critical
          component: resilience
        annotations:
          summary: "Circuit breaker open for {{ $labels.service }}"
          description: "Circuit breaker has been open for more than 1 minute for {{ $labels.service }}"

      - alert: CircuitBreakerHalfOpen
        expr: circuit_breaker_state{state="half_open"} == 1
        for: 5m
        labels:
          severity: warning
          component: resilience
        annotations:
          summary: "Circuit breaker half-open for {{ $labels.service }}"
          description: "Circuit breaker in half-open state for 5 minutes for {{ $labels.service }}"

      # Error Rate Alerts
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "High error rate on {{ $labels.endpoint }}"
          description: "Error rate is {{ $value }} errors/sec for {{ $labels.endpoint }}"

      - alert: ElevatedErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 10m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "Elevated error rate on {{ $labels.endpoint }}"
          description: "Error rate is {{ $value }} errors/sec for {{ $labels.endpoint }}"

      # Response Time Alerts
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: critical
          component: performance
        annotations:
          summary: "High response time on {{ $labels.endpoint }}"
          description: "95th percentile response time is {{ $value }}s for {{ $labels.endpoint }}"

      - alert: ElevatedResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
          component: performance
        annotations:
          summary: "Elevated response time on {{ $labels.endpoint }}"
          description: "95th percentile response time is {{ $value }}s for {{ $labels.endpoint }}"

      # Rate Limiting Alerts
      - alert: HighRateLimitBreaches
        expr: rate(rate_limit_breaches_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
          component: security
        annotations:
          summary: "High rate limit breaches"
          description: "Rate limit breaches are {{ $value }} breaches/sec"

      # Dead Letter Queue Alerts
      - alert: DLQSizeHigh
        expr: dlq_size > 1000
        for: 5m
        labels:
          severity: warning
          component: resilience
        annotations:
          summary: "Dead Letter Queue size high"
          description: "DLQ has {{ $value }} messages for {{ $labels.service }}"

      - alert: DLQSizeCritical
        expr: dlq_size > 5000
        for: 1m
        labels:
          severity: critical
          component: resilience
        annotations:
          summary: "Dead Letter Queue size critical"
          description: "DLQ has {{ $value }} messages for {{ $labels.service }}"

      # Database Alerts
      - alert: DatabaseConnectionPoolExhausted
        expr: db_pool_active_connections / db_pool_max_connections > 0.9
        for: 5m
        labels:
          severity: critical
          component: database
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Using {{ $value | humanizePercentage }} of max connections for {{ $labels.database }}"

      - alert: DatabaseSlowQueries
        expr: rate(db_query_duration_seconds_count{le="1"}[5m]) / rate(db_query_duration_seconds_count[5m]) < 0.95
        for: 10m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "High percentage of slow database queries"
          description: "Less than 95% of queries complete in < 1s for {{ $labels.database }}"

      # Service Health Alerts
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
          component: infrastructure
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.job }} has been down for more than 1 minute"

      # Memory Alerts
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / node_memory_MemTotal > 0.9
        for: 5m
        labels:
          severity: critical
          component: infrastructure
        annotations:
          summary: "High memory usage"
          description: "{{ $labels.job }} is using {{ $value | humanizePercentage }} of total memory"

      - alert: MemoryLeakDetected
        expr: rate(process_resident_memory_bytes[1h]) > 1000000
        for: 30m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "Possible memory leak detected"
          description: "{{ $labels.job }} memory growing at > 1MB/sec for 30 minutes"

      # Disk Space Alerts
      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.1
        for: 5m
        labels:
          severity: critical
          component: infrastructure
        annotations:
          summary: "Disk space low"
          description: "Less than 10% disk space available on {{ $labels.device }}"

      - alert: DiskSpaceWarning
        expr: node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.2
        for: 10m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "Disk space warning"
          description: "Less than 20% disk space available on {{ $labels.device }}"
```

---

## Grafana Dashboards

### Dashboard 1: System Overview

**JSON Definition**: `grafana/dashboards/system-overview.json`

**Panels**:
1. Request Rate (requests/sec)
2. Error Rate (%)
3. Response Time (p50, p95, p99)
4. Active Connections
5. Memory Usage
6. CPU Usage
7. Disk Usage
8. Service Health (up/down)

**Queries**:
- Request Rate: `sum(rate(http_requests_total[5m]))`
- Error Rate: `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))`
- Response Time p95: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`

### Dashboard 2: Workflow Executions

**JSON Definition**: `grafana/dashboards/workflow-executions.json`

**Panels**:
1. Workflow Executions (total)
2. Workflow Success Rate (%)
3. Workflow Duration (p50, p95, p99)
4. Workflow Failures by Type
5. Active Workflows
6. Workflow Queue Depth

**Queries**:
- Executions: `sum(rate(workflow_executions_total[5m]))`
- Success Rate: `sum(rate(workflow_executions_total{status="success"}[5m])) / sum(rate(workflow_executions_total[5m]))`
- Duration: `histogram_quantile(0.95, rate(workflow_duration_seconds_bucket[5m]))`

### Dashboard 3: Service Bubble Health

**JSON Definition**: `grafana/dashboards/service-bubbles.json`

**Panels**:
1. Qdrant: Request Rate, Error Rate, Latency
2. Elasticsearch: Index Rate, Search Rate, Query Latency
3. Redis: Operations Rate, Hit Rate, Memory Usage
4. PostgreSQL: Query Rate, Connection Pool Usage, Slow Queries

**Queries**:
- Qdrant: `sum(rate(qdrant_requests_total[5m]))`
- Elasticsearch: `sum(rate(elasticsearch_requests_total[5m]))`
- Redis: `sum(rate(redis_commands_total[5m]))`
- PostgreSQL: `sum(rate(postgres_queries_total[5m]))`

### Dashboard 4: Error Analysis

**JSON Definition**: `grafana/dashboards/error-analysis.json`

**Panels**:
1. Errors by Endpoint
2. Errors by Status Code
3. Errors by Service
4. Error Log Messages
5. Error Rate Trend (24h)

**Queries**:
- By Endpoint: `sum by (endpoint) (rate(http_requests_total{status=~"5.."}[5m]))`
- By Status Code: `sum by (status) (rate(http_requests_total{status=~"5.."}[5m]))`

### Dashboard 5: Performance Metrics

**JSON Definition**: `grafana/dashboards/performance.json`

**Panels**:
1. Response Time Heatmap
2. Response Time Distribution
3. Throughput vs Response Time
4. Database Query Performance
5. Cache Hit Rate

---

## Alertmanager Configuration

### Configuration File

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: '${SLACK_WEBHOOK_URL}'

# Route configuration
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    # Warning alerts go to Slack
    - match:
        severity: warning
      receiver: 'slack'

    # All alerts go to email
    - match:
        severity: critical|warning
      receiver: 'email'

# Receivers configuration
receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

  - name: 'slack'
    slack_configs:
      - channel: '#alerts'
        title: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}*Alert:* {{ .Annotations.summary }}*Description:* {{ .Annotations.description }}{{ end }}'
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'

  - name: 'email'
    email_configs:
      - to: 'on-call@example.com'
        headers:
          Subject: '[ALERT] {{ .GroupLabels.alertname }}'
        html: '{{ range .Alerts }}<strong>{{ .Annotations.summary }}</strong><br/>{{ .Annotations.description }}{{ end }}'

# Inhibition rules
inhibit_rules:
  # Inhibit warning if critical is firing
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']

# Templates
templates:
  - '/etc/alertmanager/templates/*.tmpl'
```

### Alert Templates

```yaml
# /etc/alertmanager/templates/default.tmpl
{{ define "slack.default.title" }}
[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .GroupLabels.alertname }}
{{ end }}

{{ define "slack.default.text" }}
{{ range .Alerts }}
*Alert:* {{ .Labels.alertname }}
*Summary:* {{ .Annotations.summary }}
*Description:* {{ .Annotations.description }}
*Severity:* {{ .Labels.severity }}
*Service:* {{ .Labels.service }}
{{ end }}
{{ end }}
```

---

## Log Aggregation

### Loki Configuration

```yaml
# /etc/loki/local-config.yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb:
    directory: /loki/index

  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
```

### Application Logging

```typescript
// Log to Loki (via Promtail)
import { StructuredLogger } from '../security-utils';

const logger = new StructuredLogger({
  service: 'bubblelab-api',
  environment: process.env.NODE_ENV
});

logger.info('Request received', {
  correlation_id: generateCorrelationId(),
  endpoint: '/api/workflows',
  method: 'POST',
  user_id: userId,
  timestamp: new Date().toISOString()
});

// Output format (JSON)
{
  "level": "info",
  "service": "bubblelab-api",
  "environment": "production",
  "correlation_id": "abc-123",
  "endpoint": "/api/workflows",
  "method": "POST",
  "user_id": "user-123",
  "timestamp": "2026-01-18T14:30:00.000Z",
  "message": "Request received"
}
```

### Promtail Configuration

```yaml
# /etc/promtail/config.yml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: bubblelab-api
    static_configs:
      - targets:
          - localhost
        labels:
          job: bubblelab-api
          environment: production
          __path__: /var/log/bubblelab/*.log
```

---

## Uptime Monitoring

### Blackbox Exporter

```yaml
# blackbox.yml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: [200]
      method: GET
      follow_redirects: true
      preferred_ip_protocol: "ip4"

  tcp_connect:
    prober: tcp
    timeout: 5s

  icmp:
    prober: icmp
    timeout: 5s
```

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - http://bubblelab-api:3000/health
          - http://qdrant:6333/healthz
          - http://elasticsearch:9200/_cluster/health
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### Uptime Dashboard

Create Grafana dashboard showing:
1. Service uptime percentage
2. Last time service was down
3. Current service status
4. Response time from monitoring location

---

## Anomaly Detection

### Machine Learning-Based Anomaly Detection

Using Prometheus + Grafana ML:

```yaml
# alerts/anomalies.yml
groups:
  - name: anomaly_detection
    rules:
      # Anomaly: Unusual spike in error rate
      - alert: AnomalyHighErrorRate
        expr: |
          (
            rate(http_requests_total{status=~"5.."}[5m])
            > (
              avg_over_time(rate(http_requests_total{status=~"5.."}[5m])[1h:5m])
              + 3 * stddev_over_time(rate(http_requests_total{status=~"5.."}[5m])[1h:5m])
            )
          )
        for: 5m
        labels:
          severity: warning
          type: anomaly
        annotations:
          summary: "Unusual spike in error rate detected"
          description: "Error rate for {{ $labels.endpoint }} is {{ $value }} which is 3 standard deviations above 1-hour average"

      # Anomaly: Unusual drop in traffic
      - alert: AnomalyLowTraffic
        expr: |
          (
            rate(http_requests_total[5m])
            < (
              avg_over_time(rate(http_requests_total[5m])[1h:5m])
              - 3 * stddev_over_time(rate(http_requests_total[5m])[1h:5m])
            )
          )
        for: 10m
        labels:
          severity: warning
          type: anomaly
        annotations:
          summary: "Unusual drop in traffic detected"
          description: "Request rate for {{ $labels.endpoint }} is {{ $value }} which is 3 standard deviations below 1-hour average"

      # Anomaly: Memory usage growing linearly
      - alert: AnomalyMemoryGrowth
        expr: |
          predict_linear(process_resident_memory_bytes[1h], 3600) > node_memory_MemTotal * 0.9
        for: 10m
        labels:
          severity: warning
          type: anomaly
        annotations:
          summary: "Memory growing linearly - possible leak"
          description: "{{ $labels.job }} memory will exceed 90% of total in 1 hour if trend continues"
```

### Seasonal Anomaly Detection

For systems with daily/weekly patterns:

```yaml
# Compare same time last week
- alert: AnomalyComparedToLastWeek
  expr: |
    (
      rate(http_requests_total[5m])
      < (
        avg_over_time(rate(http_requests_total[5m])[7d:5m]) * 0.5
      )
    )
  for: 15m
  labels:
    severity: warning
    type: anomaly
  annotations:
    summary: "Traffic significantly lower than same time last week"
    description: "Current rate is less than 50% of rate at same time last week"
```

---

## On-Call Runbook

### On-Call Responsibilities

**Primary Responsibilities**:
1. Monitor alerts 24/7
2. Respond to critical alerts within 15 minutes
3. Acknowledge alerts
4. Diagnose issues
5. Implement fixes or escalate
6. Document incidents

**Tools**:
- PagerDuty (alert routing)
- Slack (communication)
- Grafana (dashboards)
- kubectl/docker-compose (system control)

### Alert Response Procedures

#### Critical Alerts (Severity: CRITICAL)

**Response Time**: < 15 minutes

**Procedure**:
1. **Acknowledge Alert** (immediately)
   - PagerDuty: Acknowledge
   - Slack: React with 👍

2. **Assess Impact** (5 minutes)
   - Check Grafana dashboards
   - Check error logs
   - Check service health
   - Determine user impact

3. **Diagnose Issue** (10 minutes)
   - Review recent changes
   - Check logs: `tail -f /var/log/bubblelab/app.log`
   - Check metrics: Look for spikes/drops
   - Check dependencies: Database, Redis, Qdrant, Elasticsearch

4. **Implement Fix** (15-30 minutes)
   - Restart service if hung
   - Rollback recent deployment if needed
   - Scale up if resource issue
   - Fix database connection pool

5. **Verify Fix** (5 minutes)
   - Check service health
   - Monitor metrics
   - Verify user access restored

6. **Document Incident** (post-incident)
   - Write post-mortem
   - Create action items
   - Update runbooks

#### Warning Alerts (Severity: WARNING)

**Response Time**: < 1 hour

**Procedure**:
1. **Acknowledge Alert**
2. **Investigate** (during business hours)
3. **Create Ticket** (if not immediate fix)
4. **Monitor** (watch for escalation)

### Common Issues and Solutions

#### Issue: Circuit Breaker Open

**Symptoms**:
- Alert: CircuitBreakerOpen
- High error rate
- Requests failing fast

**Diagnosis**:
```bash
# Check circuit breaker state
curl http://localhost:3000/metrics | grep circuit_breaker

# Check downstream service health
curl http://qdrant:6333/healthz
curl http://elasticsearch:9200/_cluster/health
```

**Solutions**:
1. Fix downstream service issue
2. Wait for circuit breaker to enter half-open state
3. Manually reset circuit breaker (if tool available)

#### Issue: High Memory Usage

**Symptoms**:
- Alert: HighMemoryUsage
- Service slow to respond
- OOM kills

**Diagnosis**:
```bash
# Check memory usage
docker stats

# Check for memory leaks
heapdump --write-out=/tmp/heap-$(date +%s).heapsnapshot

# Check connections
netstat -an | grep :3000 | wc -l
```

**Solutions**:
1. Restart service (quick fix)
2. Scale up (add more memory)
3. Find and fix memory leak (permanent fix)
4. Implement connection pooling limits

#### Issue: Database Connection Pool Exhausted

**Symptoms**:
- Alert: DatabaseConnectionPoolExhausted
- Slow database queries
- Timeouts

**Diagnosis**:
```bash
# Check active connections
psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Check long-running queries
psql -U postgres -c "SELECT pid, query, state, wait_event FROM pg_stat_activity WHERE state = 'active';"

# Check pool size
curl http://localhost:3000/metrics | grep db_pool
```

**Solutions**:
1. Kill long-running queries
2. Increase pool size
3. Fix slow queries (add indexes)
4. Restart application

#### Issue: High Error Rate

**Symptoms**:
- Alert: HighErrorRate
- User complaints
- Dashboard shows red

**Diagnosis**:
```bash
# Check error logs
tail -f /var/log/bubblelab/app.log | grep ERROR

# Check recent deployments
git log --oneline -10

# Check service health
curl http://localhost:3000/health
```

**Solutions**:
1. Rollback recent deployment
2. Fix bug in code
3. Restart service
4. Check configuration

### Escalation Matrix

| Time to Resolve | Escalation |
|----------------|------------|
| 30 minutes | Notify team lead |
| 1 hour | Notify engineering manager |
| 2 hours | Notify VP Engineering |
| 4 hours | Declare major incident |

### Major Incident Procedure

**Definition**: Critical system down for > 1 hour, or > 50% of users affected

**Steps**:
1. **Declare Major Incident** (Slack #incidents)
2. **Assemble Incident Team**
   - On-call engineer
   - Engineering lead
   - Product manager
   - Support lead

3. **Create War Room** (Slack channel + Zoom call)
4. **Assign Roles**
   - Incident Commander (communication)
   - Technical Lead (fixing)
   - Communications Lead (external updates)

5. **Status Updates** (every 15 minutes)
   - Internal: Slack #incidents
   - External: Status page (if available)

6. **Post-Incident** (within 24 hours)
   - Post-mortem meeting
   - Written post-mortem document
   - Action items assigned
   - Runbooks updated

### On-Call Handoff

**Procedure**:
1. **Schedule** (published in Google Calendar)
2. **Handoff Meeting** (15 minutes, weekly)
   - Review past week's incidents
   - Discuss ongoing issues
   - Review runbooks
   - Answer questions

3. **Handoff Checklist**:
   - [ ] PagerDuty handoff complete
   - [ ] No outstanding incidents
   - [ ] Documentation up to date
   - [ ] New on-call trained
   - [ ] Contact info confirmed

### Training

**New On-Call Training** (1 week):
- Day 1: System overview
- Day 2: Tools training (Grafana, PagerDuty, kubectl)
- Day 3: Runbook review
- Day 4: Shadow on-call (observe)
- Day 5: Mock incidents (practice)

**Quarterly Training**:
- Review major incidents
- Update runbooks
- Practice procedures
- Feedback session

---

## Monitoring Checklist

### Pre-Production

- [ ] Prometheus configured and scraping
- [ ] Grafana dashboards created
- [ ] Alertmanager configured
- [ ] Alert rules tested
- [ ] PagerDuty integration tested
- [ ] Slack integration tested
- [ ] Log aggregation configured
- [ ] Uptime monitoring configured
- [ ] On-call team trained
- [ ] Runbooks documented

### Post-Deployment

- [ ] Verify all targets up in Prometheus
- [ ] Verify dashboards showing data
- [ ] Verify alerts firing correctly
- [ ] Test alert routing
- [ ] Check log aggregation working
- [ ] Monitor for 1 hour after deployment
- [ ] Check error rates
- [ ] Check response times
- [ ] Verify no new anomalies

### Daily

- [ ] Check for critical alerts
- [ ] Review error logs
- [ ] Check dashboard for anomalies
- [ ] Verify backup jobs completed
- [ ] Check disk space

### Weekly

- [ ] Review all alerts
- [ ] Update runbooks if needed
- [ ] Check for memory leaks
- [ ] Review performance trends
- [ ] Test restore procedures

### Monthly

- [ ] Review major incidents
- [ ] Update runbooks
- [ ] Train on-call team
- [ ] Review alert thresholds
- [ ] Optimize dashboards
- [ ] Check for unused metrics

---

**Last Updated**: 2026-01-18
**Next Review**: 2026-02-18
**Maintained By**: DevOps Team
