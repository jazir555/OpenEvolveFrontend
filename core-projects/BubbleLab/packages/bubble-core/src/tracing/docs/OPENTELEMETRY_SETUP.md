# OpenTelemetry Setup Guide

Complete guide for setting up distributed tracing with OpenTelemetry in BubbleLab.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Exporters](#exporters)
5. [Environment Variables](#environment-variables)
6. [Production Deployment](#production-deployment)

## Installation

Install the required OpenTelemetry packages:

```bash
cd packages/bubble-core
pnpm add @opentelemetry/api @opentelemetry/sdk-trace-node
pnpm add @opentelemetry/exporter-trace-jaeger
pnpm add @opentelemetry/exporter-trace-otlp-grpc
pnpm add @opentelemetry/exporter-trace-otlp-http
pnpm add @opentelemetry/context-async-hooks
pnpm add @opentelemetry/resources
pnpm add @opentelemetry/semantic-conventions
pnpm add -D @opentelemetry/auto-instrumentations
```

## Quick Start

### 1. Initialize Tracing

```typescript
import { TracingManager } from '@bubblelab/bubble-core/tracing';

// For local development with Jaeger
const tracingManager = TracingManager.getInstance();

await tracingManager.initialize({
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
  resourceAttributes: {
    'environment': 'development',
    'version': '1.0.0',
  },
});
```

### 2. Add Tracing to Your Code

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

### 3. Start Jaeger (Docker)

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest
```

Visit http://localhost:16686 to view traces in Jaeger UI.

## Configuration

### Basic Configuration

```typescript
interface TracingConfig {
  serviceName: string;              // Required: Service name
  enabled: boolean;                 // Enable/disable tracing
  sampleRate: number;               // 0.0 to 1.0 (sampling rate)
  exporter: ExporterConfig;         // Exporter configuration
  batchExport?: BatchExportConfig;  // Batch export settings
  resourceAttributes?: Record<string, string>;
}
```

### Sampling Strategies

```typescript
// Development: 100% sampling
const devConfig = {
  sampleRate: 1.0,
};

// Production: 10% sampling
const prodConfig = {
  sampleRate: 0.1,
};

// High-traffic production: 1% sampling
const highTrafficConfig = {
  sampleRate: 0.01,
};
```

### Batch Export Configuration

```typescript
const tracingManager = TracingManager.getInstance();

await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 0.1,
  exporter: {
    type: 'jaeger',
    options: { host: 'jaeger', port: 6832 },
  },
  batchExport: {
    exportIntervalMillis: 5000,    // Export every 5 seconds
    maxQueueSize: 2048,            // Max 2048 spans in queue
    maxExportBatchSize: 512,       // Max 512 spans per batch
    exportTimeoutMillis: 30000,    // 30 second timeout
  },
});
```

## Exporters

### Jaeger (Local Development)

```typescript
import { JaegerConfigHelper } from '@bubblelab/bubble-core/tracing';

const config = JaegerConfigHelper.forLocalDevelopment({
  host: 'localhost',
  port: 6832,
});

await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 1.0,
  ...config,
});
```

### Jaeger (Production)

```typescript
const config = JaegerConfigHelper.forProduction({
  host: process.env.JAEGER_HOST || 'jaeger',
  port: 6832,
});
```

### OpenTelemetry Collector

```typescript
import { OtlpConfigHelper } from '@bubblelab/bubble-core/tracing';

const config = OtlpConfigHelper.forCollector(
  process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://otel-collector:4317',
  {
    'Authorization': `Bearer ${process.env.OTEL_AUTH_TOKEN}`,
  }
);

await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 0.1,
  ...config,
});
```

### Honeycomb

```typescript
const config = OtlpConfigHelper.forHoneycomb(
  process.env.HONEYCOMB_API_KEY,
  'bubble-lab-production'
);

await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 0.1,
  ...config,
});
```

### New Relic

```typescript
const config = OtlpConfigHelper.forNewRelic(
  process.env.NEW_RELIC_API_KEY
);

await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 0.1,
  ...config,
});
```

### Console (Debugging)

```typescript
await tracingManager.initialize({
  serviceName: 'bubble-lab',
  enabled: true,
  sampleRate: 1.0,
  exporter: {
    type: 'console',
    options: {
      colors: true,
      format: 'pretty', // or 'json'
    },
  },
});
```

## Environment Variables

Create a `.env` file or set environment variables:

```bash
# OpenTelemetry Configuration
OTEL_SERVICE_NAME=bubble-lab
OTEL_ENABLED=true
OTEL_SAMPLE_RATE=0.1

# Jaeger Configuration
JAEGER_HOST=jaeger
JAEGER_PORT=6832

# OTLP Collector Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <token>

# Honeycomb Configuration
HONEYCOMB_API_KEY=your-api-key
HONEYCOMB_DATASET=bubble-lab

# Debugging
OTEL_DEBUG=true
NODE_ENV=production
```

## Production Deployment

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  bubble-lab-api:
    build: .
    environment:
      - OTEL_SERVICE_NAME=bubble-lab-api
      - OTEL_ENABLED=true
      - OTEL_SAMPLE_RATE=0.1
      - JAEGER_HOST=jaeger
      - JAEGER_PORT=6832
    depends_on:
      - jaeger

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bubble-lab-config
data:
  OTEL_SERVICE_NAME: bubble-lab-api
  OTEL_ENABLED: "true"
  OTEL_SAMPLE_RATE: "0.1"
  JAEGER_HOST: jaeger
  JAEGER_PORT: "6832"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bubble-lab-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bubble-lab-api
  template:
    metadata:
      labels:
        app: bubble-lab-api
    spec:
      containers:
      - name: bubble-lab-api
        image: bubble-lab-api:latest
        envFrom:
        - configMapRef:
            name: bubble-lab-config
```

### Initialization in Production

```typescript
// src/tracing-init.ts
import { TracingManager } from '@bubblelab/bubble-core/tracing';

export async function initializeTracing() {
  const manager = TracingManager.getInstance();

  const config: TracingConfig = {
    serviceName: process.env.OTEL_SERVICE_NAME || 'bubble-lab',
    enabled: process.env.OTEL_ENABLED === 'true',
    sampleRate: parseFloat(process.env.OTEL_SAMPLE_RATE || '0.1'),
    exporter: {
      type: process.env.OTEL_EXPORTER_TYPE || 'jaeger',
      options: {
        host: process.env.JAEGER_HOST || 'localhost',
        port: parseInt(process.env.JAEGER_PORT || '6832'),
      },
    },
    batchExport: {
      exportIntervalMillis: 5000,
      maxQueueSize: 2048,
      maxExportBatchSize: 512,
      exportTimeoutMillis: 30000,
    },
    resourceAttributes: {
      'environment': process.env.NODE_ENV || 'production',
      'version': process.env.npm_package_version || '1.0.0',
    },
  };

  await manager.initialize(config);
}

// Call this in your application startup
await initializeTracing();
```

### Graceful Shutdown

```typescript
// Graceful shutdown handler
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully...');

  const manager = TracingManager.getInstance();
  await manager.flush();
  await manager.shutdown();

  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('SIGINT received, shutting down gracefully...');

  const manager = TracingManager.getInstance();
  await manager.flush();
  await manager.shutdown();

  process.exit(0);
});
```

## Best Practices

1. **Sampling Rate**: Use lower sampling rates (0.01-0.1) in production to reduce overhead
2. **Batch Export**: Always use batch export in production for better performance
3. **Resource Attributes**: Include environment, version, and region for better filtering
4. **Graceful Shutdown**: Always flush traces before shutting down
5. **Error Handling**: Initialize tracing early in application startup
6. **Development**: Use 100% sampling in development for complete visibility

## Troubleshooting

### Traces Not Appearing

1. Check if tracing is enabled
2. Verify exporter connection
3. Check sample rate (if < 1.0, not all traces are exported)
4. Review logs for initialization errors

### High Memory Usage

1. Reduce `maxQueueSize` in batch export config
2. Reduce sample rate
3. Decrease `maxExportBatchSize`

### Slow Performance

1. Reduce sample rate
2. Use async exporters
3. Adjust batch export intervals

## Next Steps

- [Trace Visualization Guide](./TRACE_VISUALIZATION.md)
- [Performance Analysis](./PERFORMANCE_ANALYSIS.md)
- [Troubleshooting Traces](./TROUBLESHOOTING_TRACES.md)
