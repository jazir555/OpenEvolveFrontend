# BubbleLab Alerting Guide

Comprehensive guide for configuring and managing alerts in BubbleLab monitoring infrastructure.

## Table of Contents

1. [Alert Overview](#alert-overview)
2. [Alert Severity Levels](#alert-severity-levels)
3. [Configuring Alerts](#configuring-alerts)
4. [Alert Routing](#alert-routing)
5. [Notification Channels](#notification-channels)
6. [Alert Management](#alert-management)
7. [Runbooks](#runbooks)
8. [Best Practices](#best-practices)

## Alert Overview

BubbleLab uses Prometheus Alertmanager for alert routing and notification. Alerts are organized by severity and team responsibility.

### Alert Categories

1. **Critical Alerts (P0)**: Immediate action required
2. **Warning Alerts (P1)**: Attention needed within hour
3. **Security Alerts (P0)**: Immediate security response
4. **Performance Alerts (P1)**: Performance degradation
5. **Availability Alerts (P0)**: Service downtime

## Alert Severity Levels

### Critical (P0)

**Response Time**: Within 5 minutes
**Impact**: Service degradation or outage
**Examples**:
- Error rate > 10%
- Circuit breaker open
- Service down
- Memory exhaustion
- Security breaches

### Warning (P1)

**Response Time**: Within 1 hour
**Impact**: Performance degradation or potential issues
**Examples**:
- P95 latency > 30s
- Memory usage > 80%
- Rate limit breaches
- High retry rate

## Configuring Alerts

### Alert Rule Structure

Alerts are defined in `monitoring/prometheus/alerts.yml`:

```yaml
groups:
  - name: bubblelab.critical
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(bubble_error_total[5m])) by (bubble)
          /
          sum(rate(bubble_operation_total[5m])) by (bubble)
          > 0.1
        for: 5m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Error rate above 10% for {{ $labels.bubble }}"
          description: "Bubble {{ $labels.bubble }} has an error rate of {{ $value | humanizePercentage }}"
          runbook_url: "https://docs.bubblelab.ai/runbooks/high-error-rate"
```

### Alert Rule Components

1. **expr**: PromQL expression to evaluate
2. **for**: Duration threshold must be met before alerting
3. **labels**: Metadata for alert routing
4. **annotations**: Human-readable information

### Creating New Alerts

1. Define the PromQL expression
2. Set appropriate duration (`for`)
3. Add labels for routing
4. Write clear annotations
5. Create runbook documentation

**Example: Custom Alert**

```yaml
- alert: CustomSlowQueryAlert
  expr: |
    histogram_quantile(0.99,
      sum(rate(bubble_operation_duration_seconds_bucket{operation="query"}[5m])) by (le, bubble)
    ) > 120
  for: 10m
  labels:
    severity: warning
    team: platform
  annotations:
    summary: "Slow queries detected for {{ $labels.bubble }}"
    description: "P99 query latency is {{ $value | humanizeDuration }}"
    runbook_url: "https://docs.bubblelab.ai/runbooks/slow-queries"
```

## Alert Routing

### Routing Configuration

Alertmanager routes alerts based on labels in `monitoring/prometheus/alertmanager.yml`:

```yaml
route:
  receiver: 'default'
  group_by: ['alertname', 'bubble', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

  routes:
    # Critical alerts
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 0s
      repeat_interval: 5m

      # Security critical
      - match:
          team: security
        receiver: 'security-team'
        group_wait: 0s
        repeat_interval: 2m

    # Warning alerts
    - match:
        severity: warning
      receiver: 'warning-alerts'
      group_wait: 30s
      repeat_interval: 1h
```

### Routing Best Practices

1. **Group related alerts**: Use `group_by` to bundle alerts
2. **Set appropriate wait times**: Balance noise vs responsiveness
3. **Configure repeat intervals**: Avoid alert fatigue
4. **Use label matching**: Route based on severity, team, or service

## Notification Channels

### Slack Integration

Configure Slack webhook in `alertmanager.yml`:

```yaml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

receivers:
  - name: 'critical-alerts'
    slack_configs:
      - channel: '#bubblelab-critical'
        title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
        text: >-
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Bubble:* {{ .Labels.bubble }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          {{ end }}
        send_resolved: true
        color: 'danger'
        username: 'BubbleLab AlertManager'
        icon_emoji: ':rotating_light:'
```

### Email Notifications

Configure email in `alertmanager.yml`:

```yaml
receivers:
  - name: 'email-alerts'
    email_configs:
      - to: 'platform-team@bubblelab.ai'
        from: 'alertmanager@bubblelab.ai'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alertmanager@bubblelab.ai'
        auth_password: '${SMTP_PASSWORD}'
        headers:
          Subject: 'BubbleLab Alert: {{ .GroupLabels.alertname }}'
```

### PagerDuty Integration

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}'
        severity: '{{ .Labels.severity }}'
```

### Webhook Notifications

```yaml
receivers:
  - name: 'webhook'
    webhook_configs:
      - url: 'https://your-webhook-endpoint.com/alerts'
        send_resolved: true
```

## Alert Management

### Viewing Active Alerts

**Prometheus UI**:
1. Go to http://localhost:9090
2. Navigate to **Alerts**
3. View active, pending, and inactive alerts

**Alertmanager UI**:
1. Go to http://localhost:9093
2. View alert status and silence status

### Silencing Alerts

**Temporary Silence** (via Alertmanager UI):
1. Click on alert
2. Select **Silence**
3. Set duration and comment
4. Click **Silence**

**Permanent Silence** (in alertmanager.yml):
```yaml
silence:
  - matchers:
      - name: alertname
        value: NoisyAlert
    expires: 2024-12-31T23:59:59Z
    createdBy: admin@bubblelab.ai
    comment: "Alert is too noisy, under investigation"
```

### Testing Alerts

**Test Alert Expression**:
1. Go to Prometheus UI: http://localhost:9090
2. Enter expression in query bar
3. Verify query returns data
4. Check evaluation time

**Force Alert**:
```bash
# Generate load to trigger alert
for i in {1..1000}; do
  curl http://localhost:8000/api/test
done
```

## Runbooks

Each alert should have a corresponding runbook. Runbooks are located at `docs/runbooks/`.

### Runbook Template

```markdown
# Runbook: [Alert Name]

## Alert Description
Brief description of what the alert means

## Impact
What happens when this alert fires

## Immediate Actions (5-15 minutes)
1. Check [dashboard](http://grafana:3000/d/...)
2. Verify service health
3. Check logs
4. ...

## Investigation Steps
1. ...
2. ...

## Resolution Steps
1. ...
2. ...

## Prevention
How to prevent this from happening again

## Escalation
When and who to escalate to
```

### Example Runbook

**Runbook: HighErrorRate**

```markdown
# Runbook: High Error Rate

## Alert Description
Error rate has exceeded 10% for the specified bubble

## Impact
Users may experience failed operations and degraded service quality

## Immediate Actions (5-15 minutes)
1. Check [Overview Dashboard](http://grafana:3000/d/bubblelab-overview)
2. Identify which bubble is affected
3. Check [Per-Bubble Dashboard](http://grafana:3000/d/bubblelab-per-bubble)
4. Review error types and rates

## Investigation Steps
1. **Check Dependencies**
   - Database: `curl http://localhost:8000/health`
   - External APIs: Check status pages
   - Network: Check connectivity

2. **Review Logs**
   ```bash
   tail -f logs/combined.log | grep ERROR
   ```

3. **Analyze Errors**
   ```promql
   topk(10, sum(rate(bubble_error_total[5m])) by (error_type))
   ```

## Resolution Steps
1. If database error: Check DB connection pool, restart if needed
2. If network error: Check network connectivity, DNS, firewall
3. If validation error: Check recent code changes
4. If external API error: Check rate limits, API status

## Prevention
1. Implement circuit breakers for external dependencies
2. Add retry logic with exponential backoff
3. Improve error handling and logging
4. Set up synthetic monitoring

## Escalation
- If unresolved after 30 minutes: Contact platform team lead
- If customer impact: Notify customer support
- If security related: Contact security team immediately
```

## Best Practices

### Alert Design

1. **Actionable**: Every alert should require action
2. **Clear**: Alert name and description should be self-explanatory
3. **Runbook**: Every alert must have a runbook
4. **Testing**: Test alert expressions before deploying

### Alert Fatigue Prevention

1. **Threshold tuning**: Adjust thresholds based on baseline metrics
2. **Grouping**: Group related alerts to reduce noise
3. **Scheduling**: Use time-based routing for off-hours
4. **Silencing**: Silence known issues during maintenance

### Maintenance Windows

Configure maintenance windows in `alertmanager.yml`:

```yaml
mute_time_intervals:
  - name: 'scheduled-maintenance'
    time_intervals:
      - start_time: '2024-01-15T22:00:00Z'
        end_time: '2024-01-16T06:00:00Z'
```

### Alert Quality Metrics

Track these metrics to ensure alert quality:

1. **False Positive Rate**: Should be < 5%
2. **Mean Time to Acknowledge (MTTA)**: Target < 5 minutes
3. **Mean Time to Resolve (MTTR)**: Target < 30 minutes
4. **Alert Frequency**: Should not exceed 10 per day per service

## Alert Tuning

### Reducing False Positives

**Increase duration threshold**:
```yaml
# Before
for: 2m

# After
for: 10m
```

**Adjust threshold**:
```yaml
# Before
expr: rate(bubble_error_total[5m]) > 0.1

# After
expr: rate(bubble_error_total[5m]) > 0.2
```

### Improving Alert Sensitivity

**Decrease duration threshold**:
```yaml
# Before
for: 10m

# After
for: 2m
```

**Lower threshold**:
```yaml
# Before
expr: memory_usage > 0.8

# After
expr: memory_usage > 0.7
```

## Troubleshooting

### Alerts Not Firing

1. Check PromQL expression in Prometheus UI
2. Verify alert rules are loaded: **Status > Rules**
3. Check alert evaluation interval
4. Verify threshold and duration

### Alerts Not Sending

1. Check Alertmanager status: http://localhost:9093
2. Verify notification channel configuration
3. Check Alertmanager logs: `docker-compose logs alertmanager`
4. Test notification channel (e.g., send test Slack message)

### Duplicate Alerts

1. Check alert grouping configuration
2. Review label matching in routing
3. Verify alert inhibition rules

## Next Steps

- Review available alerts in `monitoring/prometheus/alerts.yml`
- Set up notification channels
- Create runbooks for your alerts
- Test alert configuration

## Additional Resources

- [Prometheus Alerting Documentation](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Alerting Best Practices](https://www.robustperception.io/on-the-heuristics-of-alerting/)
