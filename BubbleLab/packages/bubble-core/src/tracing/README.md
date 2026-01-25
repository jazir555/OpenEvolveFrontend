# Distributed Tracing for BubbleLab

Complete OpenTelemetry-based distributed tracing implementation for end-to-end request tracking across all BubbleLab bubbles and workflows.

## Features

- **Automatic Span Creation**: Automatic tracing for all bubble operations
- **Context Propagation**: Seamless trace propagation across service boundaries
- **Multiple Exporters**: Support for Jaeger, OTLP, Honeycomb, New Relic, and more
- **Performance Metrics**: Real-time performance analysis and bottleneck identification
- **Alerting**: Built-in alerting based on trace metrics
- **Production Ready**: Configurable sampling, batching, and error handling

## Installation

Install required OpenTelemetry packages:

```bash
cd packages/bubble-core
pnpm add @opentelemetry/api @opentelemetry/sdk-trace-node
pnpm add @opentelemetry/exporter-trace-jaeger
pnpm add @opentelemetry/exporter-trace-otlp-grpc
pnpm add @opentelemetry/exporter-trace-otlp-http
pnpm add @opentelemetry/context-async-hooks
pnpm add @opentelemetry/resources
pnpm add @opentelemetry/semantic-conventions
```

## Quick Start

### 1. Initialize Tracing

```typescript
import { TracingManager } from '@bubblelab/bubble-core/tracing';

const manager = TracingManager.getInstance();

await manager.initialize({
  serviceName: 'bubble-lab-api',
  enabled: true,
  sampleRate: 1.0, // 100% sampling for development
  exporter: {
    type: 'jaeger',
    options: {
      host: 'localhost',
      port: 6832,
    },
  },
});
```

### 2. Trace Your Operations

```typescript
import { traceAsync } from '@bubblelab/bubble-core/tracing';

async function myOperation() {
  return traceAsync(
    {
      name: 'my-operation',
      attributes: {
        'operation.type': 'business-logic',
      },
    },
    async (span) => {
      // Your operation code here
      const result = await doSomething();

      // Add custom attributes
      if (span) {
        span.setAttribute('result.count', result.length);
      }

      return result;
    }
  );
}
```

### 3. Start Jaeger

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Visit http://localhost:16686 to view traces.

## Core Modules

### TracingManager

Central management of OpenTelemetry tracing configuration and lifecycle.

```typescript
import { TracingManager } from '@bubblelab/bubble-core/tracing';

const manager = TracingManager.getInstance();

// Initialize
await manager.initialize(config);

// Check status
const stats = manager.getStats();

// Flush traces
await manager.flush();

// Shutdown
await manager.shutdown();
```

### BubbleTracer

Specialized tracing for Bubble operations with automatic attribute management.

```typescript
import { BubbleTracer } from '@bubblelab/bubble-core/tracing';

const tracer = new BubbleTracer();

// Trace bubble execution
const result = await tracer.traceBubbleAction(
  {
    bubbleName: 'ai-agent',
    bubbleType: 'service',
    operation: 'generate-text',
    correlationId: 'req-123',
  },
  async (span) => {
    // Your bubble operation
    return await performAction();
  }
);
```

### Context Propagation

Automatic trace context propagation across service boundaries.

```typescript
import { injectContext, extractContext } from '@bubblelab/bubble-core/tracing';

// Outgoing: Inject context into headers
const headers = {};
injectContext(headers);
fetch('http://api.example.com', { headers });

// Incoming: Extract context from headers
const ctx = extractContext(request.headers);
```

### Trace Metrics

Performance analysis and metrics from trace data.

```typescript
import { TraceMetrics } from '@bubblelab/bubble-core/tracing';

const metrics = new TraceMetrics();

// Record operation
metrics.recordOperation('my-operation', duration);

// Get metrics
const stats = metrics.getMetrics();
// { totalTraces, errorRate, avgDuration, p95Duration, p99Duration, ... }

// Analyze performance
const analysis = metrics.analyzePerformance();
// { bottlenecks, recommendations, criticalPath }
```

### Trace Alerts

Alerting based on trace metrics.

```typescript
import { TraceAlertManager, CommonAlertRules } from '@bubblelab/bubble-core/tracing';

const alertManager = new TraceAlertManager(metrics);

// Add alert rules
alertManager.addRule(CommonAlertRules.highP95Latency(30000));
alertManager.addRule(CommonAlertRules.highErrorRate(5));

// Register notification callback
alertManager.registerNotificationCallback('default', (alert) => {
  console.warn('Alert triggered:', alert);
});

// Evaluate rules
const triggers = alertManager.evaluateRules();
```

## Configuration

### Development Setup

```typescript
await manager.initialize({
  serviceName: 'bubble-lab-api',
  enabled: true,
  sampleRate: 1.0, // 100% sampling
  exporter: {
    type: 'jaeger',
    options: {
      host: 'localhost',
      port: 6832,
    },
  },
});
```

### Production Setup

```typescript
await manager.initialize({
  serviceName: 'bubble-lab-api',
  enabled: true,
  sampleRate: 0.1, // 10% sampling
  exporter: {
    type: 'collector',
    options: {
      endpoint: process.env.OTEL_EXPORTER_OTLP_ENDPOINT,
      headers: {
        'Authorization': `Bearer ${process.env.OTEL_AUTH_TOKEN}`,
      },
    },
  },
  batchExport: {
    exportIntervalMillis: 5000,
    maxQueueSize: 2048,
    maxExportBatchSize: 512,
  },
});
```

### Supported Exporters

- **Jaeger**: Local development and on-premises
- **OTLP**: OpenTelemetry Collector
- **Honeycomb**: Production observability
- **New Relic**: APM integration
- **Console**: Debugging

## Span Attributes

### Standard Attributes

```typescript
// BubbleLab-specific
'bubble.name': 'ai-agent'
'bubble.operation': 'generate-text'
'bubble.type': 'service'
'correlation.id': 'req-123'
'execution.id': 'exec-456'

// HTTP
'http.method': 'POST'
'http.url': 'https://api.example.com/data'
'http.status_code': 200

// Database
'db.system': 'postgresql'
'db.name': 'bubblelab'
'db.operation': 'SELECT'

// Error
'error.type': 'ValidationError'
'error.message': 'Invalid input'
'error.stack': '...'
```

### Custom Attributes

```typescript
if (span) {
  span.setAttribute('user.id', 'user-123');
  span.setAttribute('cache.hit', true);
  span.setAttribute('feature.name', 'advanced-analytics');
}
```

## Use Cases

### 1. Trace Bubble Flows

```typescript
import { BubbleTracer } from '@bubblelab/bubble-core/tracing';

const tracer = new BubbleTracer();

// Trace entire bubble flow
const result = await tracer.traceBubbleAction(
  {
    bubbleName: 'ai-agent',
    bubbleType: 'service',
    operation: 'complete-task',
  },
  async (span) => {
    // All nested operations are automatically traced
    const data = await fetchData();
    const processed = await processData(data);
    return processed;
  }
);
```

### 2. Trace External API Calls

```typescript
import { traceAsync, injectContext } from '@bubblelab/bubble-core/tracing';

return traceAsync(
  {
    name: 'http.request',
    attributes: {
      'http.method': 'POST',
      'http.url': 'https://api.example.com/data',
    },
  },
  async () => {
    const headers = {};
    injectContext(headers); // Propagate trace context

    const response = await fetch('https://api.example.com/data', {
      method: 'POST',
      headers,
      body: JSON.stringify({ query: 'test' }),
    });

    return response.json();
  }
);
```

### 3. Trace Database Queries

```typescript
import { traceAsync } from '@bubblelab/bubble-core/tracing';

return traceAsync(
  {
    name: 'database.query',
    attributes: {
      'db.system': 'postgresql',
      'db.name': 'bubblelab',
      'db.statement': 'SELECT * FROM users WHERE id = $1',
    },
  },
  async (span) => {
    const result = await db.query('SELECT * FROM users WHERE id = $1', [123]);

    if (span) {
      span.setAttribute('db.rows_affected', result.rowCount);
    }

    return result;
  }
);
```

### 4. Monitor Performance

```typescript
import { TraceMetrics } from '@bubblelab/bubble-core/tracing';

const metrics = new TraceMetrics();

// Record operation performance
const startTime = Date.now();
try {
  await performOperation();
  metrics.recordOperation('my-operation', Date.now() - startTime);
} catch (error) {
  metrics.recordError('my-operation');
}

// Analyze performance
const analysis = metrics.analyzePerformance();
console.log('Bottlenecks:', analysis.bottlenecks);
console.log('Recommendations:', analysis.recommendations);
```

### 5. Set Up Alerts

```typescript
import { TraceAlertManager, CommonAlertRules } from '@bubblelab/bubble-core/tracing';

const alertManager = new TraceAlertManager(metrics);

// Add rules
alertManager.addRule(CommonAlertRules.highP95Latency(30000));  // P95 > 30s
alertManager.addRule(CommonAlertRules.highErrorRate(5));       // Error rate > 5%
alertManager.addRule(CommonAlertRules.missingSpans(300));      // No traces for 5min

// Register notifications
alertManager.registerNotificationCallback('slack', (alert) => {
  sendToSlack(alert);
});

// Evaluate periodically
setInterval(() => alertManager.evaluateRules(), 60000);
```

## Environment Variables

```bash
# OpenTelemetry Configuration
OTEL_SERVICE_NAME=bubble-lab-api
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=0.1

# Jaeger Configuration
JAEGER_HOST=jaeger
JAEGER_PORT=6832

# OTLP Collector Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_AUTH_TOKEN=your-token

# Honeycomb Configuration
HONEYCOMB_API_KEY=your-api-key
HONEYCOMB_DATASET=bubble-lab

# Debugging
OTEL_DEBUG=true
NODE_ENV=production
```

## Documentation

- [OpenTelemetry Setup Guide](./docs/OPENTELEMETRY_SETUP.md)
- [Trace Visualization with Jaeger](./docs/TRACE_VISUALIZATION.md)
- [Performance Analysis](./docs/PERFORMANCE_ANALYSIS.md)
- [Troubleshooting Traces](./docs/TROUBLESHOOTING_TRACES.md)

## Examples

See [examples.ts](./examples.ts) for comprehensive usage examples.

## Best Practices

1. **Sampling Rate**: Use 1.0 (100%) in development, 0.1 (10%) in production
2. **Batch Export**: Always enable batch export in production
3. **Error Handling**: Always record exceptions with context
4. **Context Propagation**: Inject/extract context for all external calls
5. **Custom Attributes**: Add meaningful attributes for filtering
6. **Performance**: Monitor overhead and adjust sampling accordingly

## Performance Overhead

| Sampling Rate | Overhead |
|--------------|----------|
| 100% (1.0) | ~5-10% |
| 10% (0.1) | ~1% |
| 1% (0.01) | < 1% |

## Troubleshooting

### Traces Not Appearing

1. Check if tracing is initialized: `manager.getStats()`
2. Verify Jaeger is running: `docker ps | grep jaeger`
3. Check sample rate (if < 1.0, not all traces are exported)
4. Manually flush: `await manager.flush()`

### High Memory Usage

1. Reduce sample rate: `sampleRate: 0.01`
2. Reduce batch size: `maxQueueSize: 512`
3. Shorter export intervals: `exportIntervalMillis: 1000`

### Context Not Propagating

1. Ensure you're using `injectContext()` for outgoing calls
2. Use `extractContext()` for incoming calls
3. Verify headers are being sent/received correctly

## License

Apache-2.0

## Support

For issues and questions:
- Check the [troubleshooting guide](./docs/TROUBLESHOOTING_TRACES.md)
- Review [OpenTelemetry documentation](https://opentelemetry.io/docs/)
- Open an issue on GitHub
