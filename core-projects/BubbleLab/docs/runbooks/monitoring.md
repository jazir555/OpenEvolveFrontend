# Monitoring and Alerting Runbook

## Table of Contents

- [Overview](#overview)
- [Monitoring Architecture](#monitoring-architecture)
- [Metrics Collection](#metrics-collection)
- [Logging](#logging)
- [Distributed Tracing](#distributed-tracing)
- [Dashboards](#dashboards)
- [Alerting](#alerting)
- [Performance Monitoring](#performance-monitoring)

---

## Overview

This runbook covers monitoring and alerting for BubbleLab deployments, ensuring observability and proactive issue detection.

### Monitoring Stack

```mermaid
graph TB
    subgraph "Applications"
        API[BubbleLab API]
        Studio[Bubble Studio]
        DB[(PostgreSQL)]
        Redis[(Redis)]
    end

    subgraph "Telemetry"
        Metrics[OpenTelemetry Metrics]
        Logs[Structured Logs]
        Traces[OpenTelemetry Traces]
    end

    subgraph "Collection"
        Collector[OTel Collector]
        Prom[Prometheus]
        Loki[Loki]
        Jaeger[Jaeger]
    end

    subgraph "Visualization"
        Grafana[Grafana]
    end

    subgraph "Alerting"
        Alertmgr[AlertManager]
        PagerDuty[PagerDuty]
        Slack[Slack]
    end

    API --> Metrics
    API --> Logs
    API --> Traces

    Metrics --> Collector
    Logs --> Loki
    Traces --> Jaeger

    Collector --> Prom
    Prom --> Alertmgr

    Prom --> Grafana
    Loki --> Grafana
    Jaeger --> Grafana

    Alertmgr --> PagerDuty
    Alertmgr --> Slack
```

---

## Monitoring Architecture

### OpenTelemetry Setup

**API Configuration:**

```typescript
// apps/bubblelab-api/src/monitoring/otel.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { AwsInstrumentation } from '@opentelemetry/instrumentation-aws-sdk';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { PgInstrumentation } from '@opentelemetry/instrumentation-pg';
import { RedisInstrumentation } from '@opentelemetry/instrumentation-redis';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-grpc';

const sdk = new NodeSDK({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'bubblelab-api',
    [SemanticResourceAttributes.SERVICE_VERSION]: process.env.npm_package_version,
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.BUBBLE_ENV,
  }),
  traceExporter: new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://jaeger:4317',
  }),
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({
      url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://jaeger:4317',
    }),
    exportIntervalMillis: 15000,
  }),
  instrumentations: [
    new AwsInstrumentation(),
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
    new PgInstrumentation(),
    new RedisInstrumentation(),
  ],
});

sdk.start();
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'bubblelab-production'
    environment: 'production'

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

rule_files:
  - '/etc/prometheus/rules/*.yml'

scrape_configs:
  # API Server
  - job_name: 'bubblelab-api'
    static_configs:
      - targets: ['bubblelab-api:3001']
        labels:
          service: 'api'
          component: 'backend'

  # Studio Frontend
  - job_name: 'bubble-studio'
    static_configs:
      - targets: ['bubble-studio:3000']
        labels:
          service: 'studio'
          component: 'frontend'

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
        labels:
          service: 'database'
          component: 'postgresql'

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
        labels:
          service: 'cache'
          component: 'redis'

  # Node Exporter
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
        labels:
          service: 'infrastructure'
          component: 'node'
```

---

## Metrics Collection

### Key Metrics

**Application Metrics:**

```typescript
// Custom metrics
import { Meter } from '@opentelemetry/api';

const meter = Meter.getMeter('bubblelab-api');

// Request counter
const requestCounter = meter.createCounter('http_requests_total', {
  description: 'Total number of HTTP requests',
});

// Request duration histogram
const requestDuration = meter.createHistogram('http_request_duration_ms', {
  description: 'HTTP request duration in milliseconds',
  boundaries: [10, 50, 100, 500, 1000, 5000],
});

// Active workflows gauge
const activeWorkflows = meter.createGauge('active_workflows', {
  description: 'Number of currently executing workflows',
});

// Execution counter
const executionCounter = meter.createCounter('workflow_executions_total', {
  description: 'Total number of workflow executions',
});

// Error counter
const errorCounter = meter.createCounter('errors_total', {
  description: 'Total number of errors',
});

// Usage in code
app.use((req, res, next) => {
  const start = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - start;
    requestCounter.add(1, {
      method: req.method,
      route: req.route?.path || req.path,
      status: res.statusCode,
    });
    requestDuration.record(duration, {
      method: req.method,
      route: req.route?.path || req.path,
    });
  });

  next();
});
```

### Database Metrics

**PostgreSQL Exporter:**

```yaml
# postgres-exporter deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-exporter
  namespace: bubblelab
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-exporter
  template:
    metadata:
      labels:
        app: postgres-exporter
    spec:
      containers:
      - name: postgres-exporter
        image: prometheuscommunity/postgres-exporter:latest
        env:
        - name: DATA_SOURCE_NAME
          value: "postgresql://postgres:password@postgres:5432/bubblelab?sslmode=disable"
        ports:
        - containerPort: 9187
```

**Key DB Metrics:**
- `pg_stat_database_*` - Database statistics
- `pg_stat_statements_*` - Query performance
- `pg_replication_*` - Replication lag
- Connection pool metrics

---

## Logging

### Structured Logging

**Logger Configuration:**

```typescript
// apps/bubblelab-api/src/lib/logger.ts
import pino from 'pino';

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => {
      return { level: label };
    },
  },
  serializers: {
    err: pino.stdSerializers.err,
    req: pino.stdSerializers.req,
    res: pino.stdSerializers.res,
  },
  redact: {
    paths: [
      'req.headers.authorization',
      'req.headers.cookie',
      'req.body.password',
      'req.body.token',
      'req.body.apiKey',
    ],
    remove: true,
  },
  // Add context
  mixin() {
    return {
      environment: process.env.BUBBLE_ENV,
      service: 'bubblelab-api',
      version: process.env.npm_package_version,
    };
  },
});

// Usage
logger.info({
  userId: user.id,
  workflowId: workflow.id,
  action: 'workflow_execution_started',
});

logger.error({
  err: error,
  workflowId: workflow.id,
  step: 'ai_agent',
}, 'Workflow execution failed');
```

### Loki Configuration

```yaml
# loki-config.yaml
server:
  http_listen_port: 3100

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
          namespace: bubblelab
          __path__: /var/log/pods/*_bubblelab-api_*/*.log

    pipeline_stages:
      - json:
          expressions:
            level: level
            message: message
            timestamp: time
            userId: userId
            workflowId: workflowId
      - labels:
          level:
          userId:
          workflowId:
```

---

## Distributed Tracing

### Jaeger Setup

**Docker Compose:**

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"  # UI
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

**Tracing Context:**

```typescript
import { trace } from '@opentelemetry/api';

// Start a span
const tracer = trace.getTracer('bubblelab-api');
const span = tracer.startSpan('workflow_execution');

try {
  // Add attributes
  span.setAttribute('workflow.id', workflowId);
  span.setAttribute('workflow.name', workflowName);

  // Execute workflow
  const result = await executeWorkflow(workflow);

  span.setStatus({ code: SpanStatusCode.OK });
  return result;
} catch (error) {
  span.recordException(error);
  span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
  throw error;
} finally {
  span.end();
}
```

---

## Dashboards

### Grafana Dashboards

**API Performance Dashboard:**

```json
{
  "dashboard": {
    "title": "BubbleLab API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{service=\"api\"}[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{service=\"api\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{service=\"api\"}[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_ms_bucket{service=\"api\"}[5m])) by (le))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Workflows",
        "targets": [
          {
            "expr": "active_workflows"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

**Database Dashboard:**

```json
{
  "dashboard": {
    "title": "PostgreSQL Performance",
    "panels": [
      {
        "title": "Connections",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends{datname=\"bubblelab\"}"
          }
        ]
      },
      {
        "title": "Transaction Rate",
        "targets": [
          {
            "expr": "sum(rate(pg_stat_database_xact_commit{datname=\"bubblelab\"}[5m]))"
          }
        ]
      },
      {
        "title": "Cache Hit Ratio",
        "targets": [
          {
            "expr": "sum(pg_stat_database_blks_hit{datname=\"bubblelab\"}) / (sum(pg_stat_database_blks_hit{datname=\"bubblelab\"}) + sum(pg_stat_database_blks_read{datname=\"bubblelab\"}))"
          }
        ]
      }
    ]
  }
}
```

---

## Alerting

### Alert Rules

**Prometheus Alert Rules:**

```yaml
# alerts.yml
groups:
  - name: bubblelab-api
    interval: 30s
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{service="api",status=~"5.."}[5m]))
          / sum(rate(http_requests_total{service="api"}[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on API"
          description: "Error rate is {{ $value | humanizePercentage }} for last 5 minutes"

      # High Latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_ms_bucket{service="api"}[5m])) by (le)
          ) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
          description: "P95 latency is {{ $value }}ms"

      # Service Down
      - alert: ServiceDown
        expr: up{job="bubblelab-api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "BubbleLab API is down"
          description: "{{ $labels.instance }} has been down for more than 2 minutes"

      # Database Connection Issues
      - alert: DatabaseConnectionIssues
        expr: pg_stat_database_numbackends{datname="bubblelab"} > 180
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connection count"
          description: "{{ $value }} connections to bubblelab database"

  - name: bubblelab-workflows
    interval: 30s
    rules:
      # Failed Workflow Executions
      - alert: HighWorkflowFailureRate
        expr: |
          sum(rate(workflow_executions_total{status="failed"}[10m]))
          / sum(rate(workflow_executions_total[10m])) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High workflow failure rate"
          description: "Workflow failure rate is {{ $value | humanizePercentage }}"
```

### AlertManager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  receiver: 'default-receiver'
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  group_by: ['alertname', 'severity']

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'default-receiver'
    slack_configs:
      - api_url: 'SLACK_WEBHOOK_URL'
        channel: '#alerts'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

  - name: 'slack'
    slack_configs:
      - api_url: 'SLACK_WEBHOOK_URL'
        channel: '#bubblelab-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

---

## Performance Monitoring

### Synthetic Monitoring

**Uptime Monitoring:**

```yaml
# uptime-monitor deployment
apiVersion: batch/v1
kind: CronJob
metadata:
  name: uptime-monitor
  namespace: bubblelab
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: uptime-monitor
            image: curlimages/curl:latest
            command:
            - /bin/sh
            - -c
            - |
              # Check API health
              curl -f https://api.bubblelab.ai/health || exit 1

              # Check API response time
              TIME=$(curl -w "%{time_total}" -o /dev/null -s https://api.bubblelab.ai/health)
              echo "API response time: ${TIME}s"

              # Check Studio
              curl -f https://app.bubblelab.ai/ || exit 1

              # Check database connectivity
              kubectl exec -it postgres-0 -n bubblelab -- pg_isready || exit 1

              echo "All checks passed"
          restartPolicy: OnFailure
```

### Real User Monitoring (RUM)

**Frontend Monitoring:**

```typescript
// Bubble Studio RUM
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: process.env.VITE_SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 0.1,
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,

  // Performance monitoring
  integrations: [
    new Sentry.BrowserTracing({
      tracingOrigins: ['api.bubblelab.ai'],
    }),
    new Sentry.Replay(),
  ],

  // Custom breadcrumbs
  beforeBreadcrumb(breadcrumb) {
    if (breadcrumb.category === 'xhr') {
      return {
        ...breadcrumb,
        data: {
          ...breadcrumb.data,
          // Add custom context
        },
      };
    }
    return breadcrumb;
  },
});
```

---

## Related Documentation

- [troubleshooting.md](./troubleshooting.md) - Troubleshooting guide
- [scaling.md](./scaling.md) - Scaling and performance
- [deployment.md](./deployment.md) - Deployment configuration

---

*Last Updated: January 2026*
