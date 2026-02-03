# Metrics and Monitoring Library

Comprehensive monitoring system for the OpenEvolve Glue Layer, following the Federation Constitution principles.

## Features

- **Prometheus Metrics Collection**: Request latency, error rates, circuit breaker states, adapter health
- **Knowledge Extraction Metrics**: Entity/relation counts, extraction duration
- **Health Checks**: HTTP health endpoints, readiness/liveness probes, dependency monitoring
- **Distributed Tracing**: OpenTelemetry integration with span correlation
- **Alert Management**: Threshold-based alerting with multiple notification channels
- **JSON Lines Logging**: Structured logging with correlation IDs

## Installation

```bash
npm install @openevolve/glue-metrics
```

## Environment Variables

### Required
None - all monitoring components have sensible defaults

### Optional
```bash
# Prometheus
METRICS_PREFIX=openevolve_           # Metric name prefix
PROMETHEUS_PORT=9090                  # Metrics endpoint port

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=your-service-name

# Service
SERVICE_NAME=your-service-name        # Service identifier
```

## Quick Start

### 1. Initialize Monitoring

```typescript
import { initializeMonitoring } from '@openevolve/glue-metrics';

const monitoring = await initializeMonitoring({
  serviceName: 'crm-adapter',
  prometheus: {
    prefix: 'openevolve_',
    port: 9090,
  },
  otel: {
    endpoint: process.env.OTEL_EXPORTER_OTLP_ENDPOINT,
  },
  health: {
    enabled: true,
  },
  alerts: {
    enabled: true,
    webhookUrl: process.env.ALERT_WEBHOOK_URL,
  },
});

const { metrics, health, tracer, alerts } = monitoring;
```

### 2. Track HTTP Requests

```typescript
import { getMetricsCollector } from '@openevolve/glue-metrics';

const metrics = getMetricsCollector();

// Record request
const start = Date.now();
try {
  const result = await makeRequest();
  const duration = (Date.now() - start) / 1000;

  metrics.recordHttpRequestDuration({
    service: 'crm-adapter',
    operation: 'fetch-users',
    status: '2xx',
  }, duration);

  metrics.incrementHttpRequests({
    service: 'crm-adapter',
    operation: 'fetch-users',
    status: '2xx',
  });

  metrics.setAdapterLastSuccess('crm-adapter');

  return result;
} catch (error) {
  metrics.recordError({
    service: 'crm-adapter',
    operation: 'fetch-users',
    error_type: error.name,
  });

  metrics.setAdapterLastFailure('crm-adapter');
  throw error;
}
```

### 3. Health Checks

```typescript
import { HealthChecker, createHttpHealthCheck } from '@openevolve/glue-metrics';

const health = new HealthChecker('crm-adapter');

// Register health checks
health.register('api', createHttpHealthCheck('http://crm-core:8000/health', {
  timeout: 5000,
  expectedStatus: 200,
}));

health.register('database', async () => {
  try {
    await database.query('SELECT 1');
    return {
      name: 'database',
      status: 'healthy',
      message: 'Database OK',
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    return {
      name: 'database',
      status: 'unhealthy',
      message: error.message,
      timestamp: new Date().toISOString(),
    };
  }
});

// Check health
const healthStatus = await health.checkHealth();
console.log(healthStatus);
```

### 4. Express Integration

```typescript
import express from 'express';
import {
  createMetricsMiddleware,
  createHealthMiddleware,
  createRequestTrackingMiddleware,
} from '@openevolve/glue-metrics';

const app = express();

// Add middleware
app.use(createRequestTrackingMiddleware('crm-adapter'));
app.use(createMetricsMiddleware());
app.use(createHealthMiddleware(health));

// Endpoints available:
// GET /metrics - Prometheus metrics
// GET /health - Overall health
// GET /health/live - Liveness probe
// GET /health/ready - Readiness probe
// GET /health/{checkName} - Specific health check

app.listen(3000);
```

### 5. Distributed Tracing

```typescript
import { getTracer } from '@openevolve/glue-metrics';

const tracer = getTracer('crm-adapter');

// Trace async operation
await tracer.traceAsync({
  name: 'fetch-users',
  correlationId: 'abc-123',
}, async (span) => {
  span.setAttributes({ user_id: '12345' });
  const users = await fetchUsers();
  return users;
});

// Trace HTTP request
await tracer.traceHttpRequest({
  method: 'GET',
  url: 'http://api:8000/users',
  correlationId: 'abc-123',
  fn: async () => {
    return await fetch('http://api:8000/users').then(r => r.json());
  }
});

// Trace database operation
await tracer.traceDatabaseOperation({
  operation: 'SELECT',
  table: 'users',
  correlationId: 'abc-123',
  fn: async () => {
    return await db.query('SELECT * FROM users');
  }
});
```

### 6. Alert Management

```typescript
import { getAlertManager, AlertRulePresets } from '@openevolve/glue-metrics';

const alerts = getAlertManager('crm-adapter');

// Register predefined rules
alerts.registerRule(AlertRulePresets.highErrorRate({
  threshold: 5, // 5%
  window: 60000, // 1 minute
  notifications: [
    { type: 'log', config: {} },
    { type: 'webhook', config: { url: 'https://hooks.example.com/alert' } },
  ],
}));

alerts.registerRule(AlertRulePresets.circuitBreakerOpen({
  service: 'crm-adapter',
  notifications: [
    { type: 'slack', config: { webhook_url: 'https://hooks.slack.com/...' } },
  ],
}));

// Register custom rule
alerts.registerRule({
  id: 'custom-rule',
  name: 'Custom Rule',
  description: 'Monitors custom condition',
  severity: 'warning',
  condition: {
    type: 'custom',
    eval: (data) => {
      return data.custom_metric > 100;
    },
  },
  notifications: [
    { type: 'log', config: {} },
  ],
  cooldown: 30000,
  enabled: true,
});

// Evaluate rules
const triggeredAlerts = await alerts.evaluateRules({
  error_rate: 7,
  latency_p95: 2500,
  health_status: 'healthy',
});
```

## Metrics Catalog

### HTTP Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_http_request_duration_seconds` | Histogram | service, operation, status | Request latency in seconds |
| `openvolve_http_requests_total` | Counter | service, operation, status | Total HTTP requests |
| `openvolve_http_requests_in_progress` | Gauge | service | Currently in-flight requests |

### Error Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_errors_total` | Counter | service, error_type | Total errors |
| `openvolve_errors_by_type_total` | Counter | service, operation, error_type | Errors by type |

### Circuit Breaker Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_circuit_breaker_state` | Gauge | service, circuit | State (0=closed, 1=half_open, 2=open) |
| `openvolve_circuit_breaker_failures_total` | Counter | service, circuit | Total failures |
| `openvolve_circuit_breaker_successes_total` | Counter | service, circuit | Total successes |
| `openvolve_circuit_breaker_rejects_total` | Counter | service, circuit | Rejected requests |

### Adapter Health Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_adapter_health` | Gauge | adapter | Health (0=unhealthy, 1=degraded, 2=healthy) |
| `openvolve_adapter_last_success_timestamp` | Gauge | adapter | Unix timestamp of last success |
| `openvolve_adapter_last_failure_timestamp` | Gauge | adapter | Unix timestamp of last failure |

### Knowledge Extraction Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_knowledge_extraction_total` | Counter | source, method, entity_type, success | Total extractions |
| `openvolve_knowledge_extraction_duration_seconds` | Histogram | source, method | Extraction duration |
| `openvolve_knowledge_extraction_entities` | Gauge | source, entity_type | Entity count |
| `openvolve_knowledge_extraction_relations` | Gauge | source | Relation count |

### Event Bus Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_events_processed_total` | Counter | event_type, status | Total events processed |
| `openvolve_event_processing_duration_seconds` | Histogram | event_type | Processing duration |
| `openvolve_events_in_queue` | Gauge | queue_name | Queue depth |

### Retry Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `openvolve_retry_attempts_total` | Counter | service, operation | Total retry attempts |
| `openvolve_retry_success_total` | Counter | service, operation | Successful retries |
| `openvolve_retry_failure_total` | Counter | service, operation | Failed retries |

## Health Check Endpoints

### `/health`
Returns overall health status including all dependencies.

**Response:**
```json
{
  "name": "crm-adapter",
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "response_time_ms": 45,
  "dependencies": [
    {
      "name": "api",
      "status": "healthy",
      "message": "HTTP 200",
      "timestamp": "2025-01-15T10:30:00.000Z",
      "response_time_ms": 42
    },
    {
      "name": "database",
      "status": "healthy",
      "message": "Database OK",
      "timestamp": "2025-01-15T10:30:00.000Z",
      "response_time_ms": 3
    }
  ]
}
```

### `/health/live`
Liveness probe - always returns 200 if the process is alive.

**Response:**
```json
{
  "name": "crm-adapter",
  "status": "healthy",
  "message": "Service is running",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### `/health/ready`
Readiness probe - returns 200 only if all critical dependencies are healthy.

**Response:**
```json
{
  "name": "crm-adapter",
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "dependencies": [...]
}
```

### `/health/{checkName}`
Execute a specific health check.

**Response:**
```json
{
  "name": "api",
  "status": "healthy",
  "message": "HTTP 200",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "response_time_ms": 42
}
```

## Alert Configuration

### Alert Severities

- `info`: Informational alerts
- `warning`: Warning conditions that may need attention
- `error`: Error conditions that should be investigated
- `critical`: Critical conditions requiring immediate attention

### Notification Channels

#### Log
```typescript
{ type: 'log', config: {} }
```

#### Webhook
```typescript
{
  type: 'webhook',
  config: {
    url: 'https://hooks.example.com/alert',
    headers: {
      'Authorization': 'Bearer token'
    }
  }
}
```

#### Slack
```typescript
{
  type: 'slack',
  config: {
    webhook_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  }
}
```

#### Email
```typescript
{
  type: 'email',
  config: {
    to: 'alerts@example.com',
    subject: 'Alert Notification'
  }
}
```

## Integration Examples

### With Circuit Breaker

```typescript
import { CircuitBreaker } from '@openevolve/glue-lib';
import { getMetricsCollector } from '@openevolve/glue-metrics';

const metrics = getMetricsCollector();

const circuitBreaker = new CircuitBreaker({
  threshold: 5,
  timeout_ms: 60000,
  onStateChange: (oldState, newState) => {
    // Update metrics when state changes
    metrics.setCircuitBreakerState('crm-adapter', 'api', newState);

    if (newState === 'open') {
      metrics.recordError({
        service: 'crm-adapter',
        operation: 'api-call',
        error_type: 'circuit_breaker_open',
      });
    }
  },
});

// Use with circuit breaker
try {
  const result = await circuitBreaker.execute(async () => {
    metrics.recordCircuitBreakerSuccess('crm-adapter', 'api');
    return await makeRequest();
  });
} catch (error) {
  if (circuitBreaker.getState() === 'open') {
    metrics.recordCircuitBreakerReject('crm-adapter', 'api');
  }
  metrics.recordCircuitBreakerFailure('crm-adapter', 'api');
  throw error;
}
```

### With Knowledge Extraction

```typescript
import { getMetricsCollector, getTracer } from '@openevolve/glue-metrics';

const metrics = getMetricsCollector();
const tracer = getTracer();

const entities = await tracer.traceKnowledgeExtraction({
  source: 'github',
  method: 'code-analysis',
  correlationId: 'abc-123',
  fn: async () => {
    const entities = await extractEntities();

    // Record metrics
    metrics.recordKnowledgeExtraction({
      source: 'github',
      method: 'code-analysis',
      entity_type: 'class',
      success: 'true',
    });

    metrics.setEntitiesExtracted('github', 'class', entities.length);
    metrics.setRelationsExtracted('github', entities.length * 2);

    return entities;
  }
});
```

### With Event Bus

```typescript
import { getMetricsCollector } from '@openevolve/glue-metrics';

const metrics = getMetricsCollector();

async function processEvent(event: Event) {
  const start = Date.now();

  try {
    await handleEvent(event);

    const duration = (Date.now() - start) / 1000;
    metrics.recordEventProcessed(event.type, 'success');
    metrics.recordEventProcessingDuration(event.type, duration);
  } catch (error) {
    const duration = (Date.now() - start) / 1000;
    metrics.recordEventProcessed(event.type, 'failure');
    metrics.recordEventProcessingDuration(event.type, duration);
    throw error;
  }
}

// Monitor queue depth
setInterval(() => {
  metrics.setEventsInQueue('main', eventQueue.length);
}, 5000);
```

## Best Practices

### 1. Correlation IDs
Always pass correlation IDs for request tracing:

```typescript
const correlationId = req.headers['x-correlation-id'] || generateId();

await tracer.traceAsync({
  name: 'operation',
  correlationId,
}, async (span) => {
  // Your code
});
```

### 2. Metric Label Cardinality
Avoid high-cardinality labels like user IDs:

```typescript
// Bad
metrics.recordError({
  service: 'crm-adapter',
  error_type: `user_${userId}`, // Too many unique values
});

// Good
metrics.recordError({
  service: 'crm-adapter',
  error_type: 'validation_error',
  operation: 'create_user',
});
```

### 3. Health Check Timeouts
Always set timeouts for health checks:

```typescript
health.register('api', createHttpHealthCheck('http://service:8000/health', {
  timeout: 5000, // Fail after 5 seconds
  expectedStatus: 200,
}));
```

### 4. Alert Cooldowns
Use cooldowns to prevent alert spam:

```typescript
alerts.registerRule({
  // ...
  cooldown: 60000, // Wait 1 minute before re-alerting
});
```

## Kubernetes Deployment

### Service Monitors

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: openevolve-glue
spec:
  selector:
    matchLabels:
      app: openevolve-glue
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s
```

### Probes

```yaml
spec:
  containers:
  - name: adapter
    livenessProbe:
      httpGet:
        path: /health/live
        port: http
      initialDelaySeconds: 5
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: http
      initialDelaySeconds: 10
      periodSeconds: 5
```

## Troubleshooting

### Metrics Not Appearing
1. Check `PROMETHEUS_PORT` is set
2. Verify `/metrics` endpoint is accessible
3. Check Prometheus configuration for scrape targets

### Health Checks Failing
1. Check timeout values are appropriate
2. Verify service URLs are correct
3. Check network connectivity between services

### Alerts Not Firing
1. Verify rule conditions match data structure
2. Check notification channel configuration
3. Review alert history for errors

## License

MIT

## Contributing

This library follows the OpenEvolve Federation Constitution. All contributions must:
1. Use environment variables for configuration
2. Include UTC timestamps
3. Provide structured JSON logging
4. Include correlation IDs
5. Handle failures gracefully
