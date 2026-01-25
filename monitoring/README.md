# BubbleLab Monitoring Infrastructure

Complete monitoring and alerting infrastructure for BubbleLab bubbles.

## Quick Start

```bash
# Navigate to monitoring directory
cd monitoring

# Start monitoring stack
docker-compose up -d

# Access services
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/changeme)
# Alertmanager: http://localhost:9093
```

## Directory Structure

```
monitoring/
├── docker-compose.yml                    # Docker Compose configuration
├── .env.example                          # Environment variables template
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/                  # Grafana datasource configs
│   │   └── dashboards/                   # Dashboard provider configs
│   ├── dashboards/                       # Dashboard JSON files
│   │   ├── overview-dashboard.json
│   │   ├── per-bubble-dashboard.json
│   │   ├── security-dashboard.json
│   │   ├── performance-dashboard.json
│   │   └── business-metrics-dashboard.json
│   └── data/                            # Grafana persistent data
├── prometheus/
│   ├── prometheus.yml                   # Prometheus configuration
│   ├── alerts.yml                       # Alert rules
│   └── data/                            # Prometheus data
├── alertmanager/
│   ├── alertmanager.yml                 # Alertmanager configuration
│   └── data/                            # Alertmanager data
└── docs/                                # Documentation
    ├── MONITORING_SETUP.md
    ├── METRICS_REFERENCE.md
    ├── ALERTING_GUIDE.md
    ├── DASHBOARD_TOUR.md
    └── TROUBLESHOOTING.md
```

## Components

### Prometheus

Metrics collection and storage:
- Scrapes metrics from BubbleLab bubbles
- Evaluates alert rules
- Stores 15 days of data (configurable)
- Exposes metrics at `http://localhost:9090`

### Grafana

Visualization and dashboards:
- 5 pre-configured dashboards
- Customizable dashboards
- Real-time metrics visualization
- Access at `http://localhost:3000`

### Alertmanager

Alert routing and notifications:
- Routes alerts to appropriate channels
- Supports Slack, Email, PagerDuty, Webhooks
- Groups and deduplicates alerts
- Access at `http://localhost:9093`

## Metrics

### Operation Metrics
- `bubble_operation_duration_seconds`: Operation latency
- `bubble_operation_total`: Operation count
- `bubble_operation_retry_total`: Retry count

### Circuit Breaker Metrics
- `circuit_breaker_state`: Circuit breaker state
- `circuit_breaker_failure_total`: Failure count
- `circuit_breaker_success_total`: Success count

### Error Metrics
- `bubble_error_total`: Error count by type
- `bubble_validation_error_total`: Validation errors
- `bubble_authentication_error_total`: Auth failures

### Security Metrics
- `sql_injection_blocked_total`: SQL injection attempts blocked
- `xss_blocked_total`: XSS attempts blocked
- `unauthorized_access_total`: Unauthorized access attempts

### Performance Metrics
- `bubble_request_size_bytes`: Request size distribution
- `bubble_response_size_bytes`: Response size distribution
- `bubble_memory_usage_bytes`: Memory usage

### Business Metrics
- `bubble_active_operations`: Active operation count
- `bubble_throughput_per_second`: Operations per second
- `active_workflows`: Active workflow count

## Dashboards

### Overview Dashboard
System-wide metrics and health status

### Per-Bubble Dashboard
Individual bubble metrics with template variable selection

### Security Dashboard
Security threats and authentication metrics

### Performance Dashboard
Latency and resource usage metrics

### Business Metrics Dashboard
Operational KPIs and business metrics

## Alerts

### Critical Alerts (P0)
- **HighErrorRate**: Error rate > 10%
- **CircuitBreakerOpen**: Circuit breaker is open
- **HighAuthFailureRate**: Auth failures > 50/min
- **ServiceDown**: Service is not responding
- **MemoryExhaustion**: Memory usage > 95%

### Warning Alerts (P1)
- **SlowOperations**: P95 latency > 30s
- **HighMemoryUsage**: Memory usage > 80%
- **RateLimitBreach**: Rate limit violations
- **HighRetryRate**: Retry rate > 5%

### Security Alerts (P0)
- **SQLInjectionAttempts**: SQL injection attempts detected
- **XSSAttempts**: XSS attempts detected
- **UnauthorizedAccessSpike**: Unauthorized access spike

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Prometheus
PROMETHEUS_RETENTION_TIME=15d

# Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=changeme
GF_INSTALL_PLUGINS=

# Alertmanager
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SMTP_PASSWORD=your-smtp-password

# Application
LOG_LEVEL=info
NODE_ENV=production
```

### Customizing Dashboards

1. Export dashboard from Grafana UI
2. Save to `monitoring/grafana/dashboards/`
3. Restart Grafana: `docker-compose restart grafana`

### Customizing Alerts

1. Edit `monitoring/prometheus/alerts.yml`
2. Restart Prometheus: `docker-compose restart prometheus`
3. Verify in Prometheus UI: http://localhost:9090

## Integration with Application

### Enable Metrics

```typescript
import express from 'express';
import {
  metricsEndpoint,
  metricsMiddleware
} from '@bubblelab/bubble-core/src/metrics';

const app = express();

// Add metrics middleware
app.use(metricsMiddleware('your-bubble-name'));

// Expose metrics endpoint
app.get('/metrics', metricsEndpoint);
```

### Enable Health Checks

```typescript
import {
  healthCheckEndpoint,
  readinessCheckEndpoint,
  livenessCheckEndpoint
} from '@bubblelab/bubble-core/src/health';

app.get('/health', healthCheckEndpoint);
app.get('/health/ready', readinessCheckEndpoint(checks));
app.get('/health/live', livenessCheckEndpoint);
```

### Enable Logging

```typescript
import { getLogger, requestLoggingMiddleware } from '@bubblelab/bubble-core/src/logging';

const logger = getLogger();
app.use(requestLoggingMiddleware(logger));
```

## Documentation

- **[Monitoring Setup Guide](MONITORING_SETUP.md)**: Complete setup instructions
- **[Metrics Reference](METRICS_REFERENCE.md)**: Complete metrics catalog
- **[Alerting Guide](ALERTING_GUIDE.md)**: Alert configuration and management
- **[Dashboard Tour](DASHBOARD_TOUR.md)**: Dashboard walkthrough
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Common issues and solutions

## Maintenance

### Backup Data

```bash
# Backup Prometheus data
tar -czf prometheus-backup.tar.gz prometheus/data/

# Backup Grafana data
tar -czf grafana-backup.tar.gz grafana/data/

# Backup Alertmanager data
tar -czf alertmanager-backup.tar.gz alertmanager/data/
```

### Restore Data

```bash
# Restore Prometheus data
tar -xzf prometheus-backup.tar.gz -C prometheus/

# Restore Grafana data
tar -xzf grafana-backup.tar.gz -C grafana/

# Restore Alertmanager data
tar -xzf alertmanager-backup.tar.gz -C alertmanager/

# Restart services
docker-compose restart
```

### Update Monitoring Stack

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d
```

### Cleanup

```bash
# Stop services
docker-compose down

# Remove volumes (WARNING: Deletes all data)
docker-compose down -v

# Remove old data
rm -rf prometheus/data/* grafana/data/* alertmanager/data/*
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose logs -f

# Check port conflicts
netstat -tuln | grep -E ':(3000|9090|9093)'

# Verify configuration
docker-compose config
```

### No Metrics in Dashboards

1. Verify Prometheus is scraping targets: http://localhost:9090/targets
2. Check metrics endpoint: `curl http://localhost:8000/metrics`
3. Verify Grafana datasource: http://localhost:3000/datasources
4. Check dashboard time range

### Alerts Not Firing

1. Check alert rules: http://localhost:9090/rules
2. Test alert expression in Prometheus UI
3. Verify Alertmanager connectivity: http://localhost:9093
4. Check notification channels

## Support

- Documentation: See `docs/` directory
- GitHub Issues: https://github.com/bubblelabai/BubbleLab/issues
- Slack: #bubblelab-monitoring

## License

Apache-2.0
