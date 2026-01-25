# Troubleshooting Distributed Traces

Guide for diagnosing and fixing common issues with distributed tracing in BubbleLab.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Missing Traces](#missing-traces)
3. [Incomplete Traces](#incomplete-traces)
4. [Performance Issues](#performance-issues)
5. [Export Problems](#export-problems)
6. [Debugging Tips](#debugging-tips)
7. [FAQ](#faq)

## Common Issues

### Issue: Traces Not Appearing in Jaeger

**Symptoms**:
- Jaeger UI shows no traces
- Service is running but no data visible
- Empty trace list

**Diagnosis**:

1. Check if tracing is initialized:
```typescript
const manager = TracingManager.getInstance();
console.log('Tracing stats:', manager.getStats());
// Should show: { initialized: true, enabled: true }
```

2. Check if spans are being created:
```typescript
const currentSpan = trace.getSpan(context.active());
console.log('Current span:', currentSpan);
```

3. Check exporter connection:
```typescript
// For Jaeger
curl http://localhost:16686/api/services

// Should return your service name
```

**Solutions**:

1. **Ensure tracing is initialized**:
```typescript
await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 1.0,  // Try 100% sampling
  exporter: {
    type: 'jaeger',
    options: {
      host: 'localhost',
      port: 6832,
    },
  },
});
```

2. **Check Jaeger is running**:
```bash
docker ps | grep jaeger
# Should show jaeger container running

# Check Jaeger logs
docker logs jaeger
```

3. **Verify network connectivity**:
```bash
# Test Jaeger agent connection
nc -zv localhost 6832
# Should return: Connection to localhost 6832 port [udp/*] succeeded!
```

4. **Flush traces manually**:
```typescript
await tracingManager.flush();
```

### Issue: Sampling Too Aggressive

**Symptoms**:
- Only few traces appear
- Inconsistent trace visibility

**Solution**:

Increase sample rate:
```typescript
await tracingManager.initialize({
  sampleRate: 1.0,  // 100% for development
});
```

Or use force recording for important operations:
```typescript
const span = tracer.startSpan('important-operation', {
  forceRecording: true,  // Always record this span
});
```

## Missing Traces

### Issue: Traces Stop After Certain Point

**Symptoms**:
- Root span exists but child spans missing
- Trace ends prematurely

**Diagnosis**:

1. Check for async context loss:
```typescript
// Bad: Context not propagated
async function operation() {
  const span = tracer.startSpan('operation');
  setTimeout(() => {
    // This won't be in the trace - context lost!
    doSomething();
  }, 100);
  span.end();
}

// Good: Context preserved
async function operation() {
  const span = tracer.startSpan('operation');
  await context.with(
    trace.setSpan(context.active(), span),
    async () => {
      await new Promise(resolve => {
        setTimeout(() => {
          // This WILL be in the trace - context preserved
          doSomething();
          resolve();
        }, 100);
      });
    }
  );
  span.end();
}
```

2. Check for context propagation in HTTP calls:
```typescript
import { injectHeadersIntoRequest } from '@bubblelab/bubble-core/tracing';

// Bad: No trace context
axios.get('http://api.example.com/data');

// Good: Trace context injected
const headers = injectHeadersIntoRequest({});
axios.get('http://api.example.com/data', { headers });
```

**Solution**:

Always propagate context in async operations:
```typescript
import { withTracePropagation } from '@bubblelab/bubble-core/tracing';

async function operation() {
  return withTracePropagation(async () => {
    // Your async code here
    // Context is automatically preserved
  });
}
```

### Issue: Cross-Service Traces Broken

**Symptoms**:
- Traces don't connect across services
- Separate trace IDs for each service

**Solution**:

Ensure trace context is propagated in HTTP headers:
```typescript
// Service A (caller)
import { injectContext } from '@bubblelab/bubble-core/tracing';

const headers = {};
injectContext(headers);

fetch('http://service-b/api', {
  headers: {
    ...headers,
    'Content-Type': 'application/json',
  },
});

// Service B (receiver)
import { extractContext, propagateContext } from '@bubblelab/bubble-core/tracing';

app.use((req, res, next) => {
  const ctx = extractContext(req.headers);
  // Use this context for subsequent operations
  next();
});
```

## Incomplete Traces

### Issue: Spans Not Closing

**Symptoms**:
- Spans show as "in progress" indefinitely
- Missing span durations

**Diagnosis**:

```typescript
// Bad: Span not closed
const span = tracer.startSpan('operation');
// Missing: span.end();

// Good: Span properly closed
const span = tracer.startSpan('operation');
try {
  // Do work
} finally {
  span.end();
}
```

**Solution**:

Always close spans:
```typescript
import { traceAsync } from '@bubblelab/bubble-core/tracing';

// Automatically handles span lifecycle
await traceAsync({
  name: 'operation',
}, async (span) => {
  // Your code here
  // Span is automatically closed
});
```

### Issue: Error Information Missing

**Symptoms**:
- Failed operations don't show error details
- No stack traces in spans

**Solution**:

Always record errors:
```typescript
try {
  await riskyOperation();
} catch (error) {
  if (span) {
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message,
    });
  }
  throw error;
}
```

Or use the helper:
```typescript
import { recordException } from '@bubblelab/bubble-core/tracing';

try {
  await riskyOperation();
} catch (error) {
  recordException(error);  // Automatically records and sets status
  throw error;
}
```

## Performance Issues

### Issue: High Memory Usage

**Symptoms**:
- Memory usage increases over time
- Out of memory errors

**Diagnosis**:

```typescript
// Check span queue size
const stats = manager.getStats();
console.log('Tracing stats:', stats);
```

**Solution**:

1. **Reduce sample rate**:
```typescript
sampleRate: 0.01,  // Only 1% sampling
```

2. **Reduce batch size**:
```typescript
batchExport: {
  maxQueueSize: 512,      // Reduce from 2048
  maxExportBatchSize: 128, // Reduce from 512
}
```

3. **Use shorter export intervals**:
```typescript
batchExport: {
  exportIntervalMillis: 1000,  // Export every 1 second
}
```

### Issue: High CPU Usage

**Symptoms**:
- CPU usage high when tracing is enabled
- Slow application performance

**Solution**:

1. **Reduce sampling**:
```typescript
sampleRate: 0.1,  // Only 10% sampling
```

2. **Reduce attributes**:
```typescript
// Bad: Too many attributes
span.setAttribute('huge.data', giganticJsonString);

// Good: Minimal attributes
span.setAttribute('data.size', giganticJsonString.length);
```

3. **Disable auto-instrumentation if not needed**:
```typescript
// Only instrument what you need
// Don't use @opentelemetry/auto-instrumentations in production
```

## Export Problems

### Issue: Jaeger Connection Refused

**Symptoms**:
- Error logs showing "ECONNREFUSED"
- No traces in Jaeger UI

**Solution**:

1. **Check Jaeger is running**:
```bash
docker ps | grep jaeger
```

2. **Start Jaeger if not running**:
```bash
docker run -d --name jaeger \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

3. **Check firewall rules**:
```bash
# Allow Jaeger ports
sudo ufw allow 6831/udp
sudo ufw allow 6832/udp
```

### Issue: OTLP Export Fails

**Symptoms**:
- Authentication errors
- Connection timeout

**Solution**:

1. **Check authentication**:
```typescript
exporter: {
  type: 'otlp',
  options: {
    url: 'https://otel-collector.example.com:4317',
    headers: {
      'Authorization': `Bearer ${process.env.OTEL_AUTH_TOKEN}`,
    },
  },
}
```

2. **Test connectivity**:
```bash
# Test OTLP endpoint
curl -X POST https://otel-collector.example.com:4317/v1/traces \
  -H "Authorization: Bearer $TOKEN"
```

3. **Check collector logs**:
```bash
kubectl logs -l app=opentelemetry-collector -n monitoring
```

## Debugging Tips

### Enable Debug Logging

```typescript
import { TraceLogger } from '@bubblelab/bubble-core/tracing';

const logger = new TraceLogger(true);  // Enable debug mode
```

Or set environment variable:
```bash
export OTEL_DEBUG=true
```

### Inspect Trace Context

```typescript
import { getCurrentTraceContext } from '@bubblelab/bubble-core/tracing';

const ctx = getCurrentTraceContext();
console.log('Trace ID:', ctx?.traceId);
console.log('Span ID:', ctx?.spanId);
```

### Manually Export Spans

```typescript
const manager = TracingManager.getInstance();

// Force immediate flush
await manager.flush();
```

### Use Console Exporter

For debugging, use console exporter:
```typescript
await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  exporter: {
    type: 'console',
    options: {
      colors: true,
      format: 'pretty',
    },
  },
});
```

### Trace Metadata

Add metadata to help debugging:
```typescript
span.setAttributes({
  'debug.environment': process.env.NODE_ENV,
  'debug.version': process.env.npm_package_version,
  'debug.hostname': os.hostname(),
  'debug.pid': process.pid,
});
```

## FAQ

### Q: Why do I see duplicate traces?

**A**: This can happen when:
1. Multiple tracer instances are created (use singleton)
2. Spans are manually exported multiple times
3. Batch export is configured incorrectly

**Solution**:
```typescript
// Always use singleton
const manager = TracingManager.getInstance();

// Only initialize once
if (!manager.getStats().initialized) {
  await manager.initialize(config);
}
```

### Q: Traces appear delayed

**A**: This is normal due to:
1. Batch export interval (default 5 seconds)
2. Network latency to exporter
3. Jaeger processing time

**Solution**: Reduce export interval for testing:
```typescript
batchExport: {
  exportIntervalMillis: 1000,  // 1 second
}
```

### Q: How do I trace database queries?

**A**: Use database instrumentation:
```typescript
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { PgInstrumentation } from '@opentelemetry/instrumentation-pg';

registerInstrumentations({
  instrumentations: [
    new PgInstrumentation(),
  ],
});
```

### Q: Can I trace third-party libraries?

**A**: Yes, use auto-instrumentations:
```bash
pnpm add -D @opentelemetry/auto-instrumentations
```

```typescript
import { registerInstrumentations } from '@opentelemetry/auto-instrumentations';

registerInstrumentations({
  instrumentations: {
    '@opentelemetry/instrumentation-http': true,
    '@opentelemetry/instrumentation-express': true,
    // Add more as needed
  },
});
```

### Q: How do I filter sensitive data?

**A**: Use span processors to filter:
```typescript
import { SpanProcessor } from '@opentelemetry/sdk-trace-base';

class SensitiveDataFilter implements SpanProcessor {
  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  onStart(span: ReadableSpan): void {
    // Filter sensitive attributes
    const attributes = span.attributes;
    delete attributes['password'];
    delete attributes['api_key'];

    // Redact sensitive values
    if (attributes['authorization']) {
      attributes['authorization'] = 'REDACTED';
    }
  }

  onEnd(): void {}
  // ... other methods
}
```

### Q: What's the performance overhead?

**A**: Typically:
- **100% sampling**: ~5-10% overhead
- **10% sampling**: ~1% overhead
- **1% sampling**: < 1% overhead

Use lower sampling rates in production.

### Q: How do I correlate traces with logs?

**A**: Use trace ID in logs:
```typescript
import { getCurrentTraceContext } from '@bubblelab/bubble-core/tracing';

const traceContext = getCurrentTraceContext();
logger.info('Processing request', {
  trace_id: traceContext?.traceId,
  span_id: traceContext?.spanId,
});
```

## Getting Help

If you're still having issues:

1. **Check logs**: Enable debug logging
2. **Validate config**: Use configuration validator
3. **Test connectivity**: Ping exporter endpoints
4. **Review docs**: Check official OpenTelemetry docs
5. **Create issue**: Report bugs with trace samples

## Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Troubleshooting](https://opentelemetry.io/docs/instrumentation/js/troubleshooting/)

## Next Steps

- [OpenTelemetry Setup](./OPENTELEMETRY_SETUP.md)
- [Trace Visualization](./TRACE_VISUALIZATION.md)
- [Performance Analysis](./PERFORMANCE_ANALYSIS.md)
