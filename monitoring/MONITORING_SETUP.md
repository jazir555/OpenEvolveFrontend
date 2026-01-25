# BubbleLab Monitoring Setup Guide

This guide provides comprehensive instructions for setting up and configuring the monitoring infrastructure for BubbleLab.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running Services](#running-services)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## Overview

The BubbleLab monitoring stack consists of:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notifications
- **Winston**: Structured logging infrastructure

## Prerequisites

### Required Software

- Docker and Docker Compose
- Node.js 18+ and pnpm
- At least 4GB RAM available
- 20GB free disk space

### Environment Variables

Create a `.env` file in the monitoring directory:

```bash
# Prometheus Configuration
PROMETHEUS_RETENTION_TIME=15d
PROMETHEUS_STORAGE_SIZE=10GB

# Grafana Configuration
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=changeme
GF_INSTALL_PLUGINS=grafana-piechart-panel

# Alertmanager Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SMTP_PASSWORD=your-smtp-password

# Application Configuration
LOG_LEVEL=info
NODE_ENV=production
```

## Installation

### 1. Install Dependencies

```bash
# Install monitoring dependencies
cd BubbleLab/packages/bubble-core
pnpm install prom-client winston winston-elasticsearch
```

### 2. Create Monitoring Directories

```bash
mkdir -p monitoring/{prometheus,grafana,alertmanager}
mkdir -p monitoring/prometheus/data
mkdir -p monitoring/grafana/data
mkdir -p monitoring/alertmanager/data
mkdir -p logs
```

### 3. Set Permissions

```bash
chmod -R 777 monitoring/prometheus/data
chmod -R 777 monitoring/grafana/data
chmod -R 777 monitoring/alertmanager/data
```

## Configuration

### 1. Prometheus Configuration

Create `monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'bubblelab'
    environment: 'production'

# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load rules
rule_files:
  - 'alerts.yml'

# Scrape configurations
scrape_configs:
  # BubbleLab API
  - job_name: 'bubblelab-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/metrics'

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter (system metrics)
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node_exporter:9100']
```

### 2. Grafana Configuration

Create `monitoring/grafana/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### 3. Docker Compose Configuration

Create `monitoring/docker-compose.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: bubblelab-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - ./prometheus/data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:latest
    container_name: bubblelab-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=${GF_SECURITY_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GF_SECURITY_ADMIN_PASSWORD:-changeme}
      - GF_INSTALL_PLUGINS=${GF_INSTALL_PLUGINS:-}
    volumes:
      - ./grafana/data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:latest
    container_name: bubblelab-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - ./alertmanager/data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'

  node_exporter:
    image: prom/node-exporter:latest
    container_name: bubblelab-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
```

## Running Services

### Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### Check Service Status

```bash
docker-compose ps
```

Expected output:
```
NAME                        STATUS
bubblelab-prometheus        Up
bubblelab-grafana           Up
bubblelab-alertmanager      Up
bubblelab-node-exporter     Up
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f prometheus
docker-compose logs -f grafana
docker-compose logs -f alertmanager
```

## Verification

### 1. Verify Prometheus

1. Open http://localhost:9090
2. Check **Status > Targets** - all targets should be "UP"
3. Query a metric: `bubble_operation_total`
4. Verify data is being collected

### 2. Verify Grafana

1. Open http://localhost:3000
2. Login with admin credentials
3. Check **Configuration > Data Sources** - Prometheus should be healthy
4. Import dashboards from `monitoring/grafana/dashboards/`
5. Verify dashboards display data

### 3. Verify Alertmanager

1. Open http://localhost:9093
2. Check **Status** - should show alert rules
3. Verify alert receivers are configured

### 4. Verify Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

Expected: Prometheus metrics in text format

### 5. Verify Health Checks

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/health/ready

# Liveness check
curl http://localhost:8000/health/live
```

## Integration with Application

### 1. Enable Metrics in Your Application

```typescript
import express from 'express';
import {
  register,
  metricsEndpoint,
  metricsMiddleware
} from '@bubblelab/bubble-core/src/metrics';

const app = express();

// Add metrics middleware
app.use(metricsMiddleware('your-bubble-name'));

// Expose metrics endpoint
app.get('/metrics', metricsEndpoint);

// Start server
app.listen(8000);
```

### 2. Enable Health Checks

```typescript
import {
  healthCheckEndpoint,
  readinessCheckEndpoint,
  livenessCheckEndpoint,
  registerHealthCheck
} from '@bubblelab/bubble-core/src/health';

// Register custom health checks
registerHealthCheck('database', async () => {
  // Your database check logic
});

// Add endpoints
app.get('/health', healthCheckEndpoint);
app.get('/health/ready', readinessCheckEndpoint(checks));
app.get('/health/live', livenessCheckEndpoint);
```

### 3. Enable Logging

```typescript
import { getLogger, requestLoggingMiddleware } from '@bubblelab/bubble-core/src/logging';

const logger = getLogger();

// Add request logging middleware
app.use(requestLoggingMiddleware(logger));

// Use logger
logger.info('Application started');
logger.error('Error occurred', error);
```

## Troubleshooting

### Prometheus Not Collecting Metrics

**Problem**: Targets show as "DOWN" in Prometheus

**Solutions**:
1. Verify the application is running: `curl http://localhost:8000/health`
2. Check metrics endpoint: `curl http://localhost:8000/metrics`
3. Verify Prometheus configuration
4. Check firewall rules
5. Review Prometheus logs: `docker-compose logs prometheus`

### Grafana Dashboards Not Showing Data

**Problem**: Dashboards display "No Data"

**Solutions**:
1. Verify Prometheus is collecting metrics
2. Check Grafana data source configuration
3. Verify time range in dashboard
4. Check query syntax in dashboard panels
5. Review Grafana logs: `docker-compose logs grafana`

### Alertmanager Not Sending Alerts

**Problem**: Alerts not being delivered

**Solutions**:
1. Verify Alertmanager configuration
2. Check Slack webhook URL
3. Review Alertmanager logs: `docker-compose logs alertmanager`
4. Verify alert rules are firing in Prometheus UI
5. Check alert routing configuration

### High Memory Usage

**Problem**: Monitoring services consuming too much memory

**Solutions**:
1. Reduce Prometheus retention time
2. Adjust scrape intervals
3. Limit metrics retention
4. Add memory limits to docker-compose.yml

### Logs Not Being Generated

**Problem**: Application logs not appearing

**Solutions**:
1. Verify log directory exists and has correct permissions
2. Check LOG_LEVEL environment variable
3. Review logger configuration
4. Check Winston transport configuration

## Maintenance

### Updating Dashboards

1. Export updated dashboards from Grafana
2. Replace JSON files in `monitoring/grafana/dashboards/`
3. Restart Grafana: `docker-compose restart grafana`

### Updating Alert Rules

1. Edit `monitoring/prometheus/alerts.yml`
2. Restart Prometheus: `docker-compose restart prometheus`
3. Verify rules in Prometheus UI: **Status > Rules**

### Backup Configuration

```bash
# Backup Grafana dashboards
docker exec bubblelab-grafana grafana-cli admin export-dashboard > backup.json

# Backup Prometheus data
tar -czf prometheus-backup.tar.gz monitoring/prometheus/data/

# Backup Grafana data
tar -czf grafana-backup.tar.gz monitoring/grafana/data/
```

## Next Steps

- Review [Metrics Reference](METRICS_REFERENCE.md)
- Configure [Alerting](ALERTING_GUIDE.md)
- Take the [Dashboard Tour](DASHBOARD_TOUR.md)
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/bubblelabai/BubbleLab/issues
- Slack: #bubblelab-monitoring
- Email: platform-team@bubblelab.ai
