# Performance Analysis with Traces

Guide for analyzing performance using distributed traces to identify bottlenecks and optimize operations.

## Table of Contents

1. [Metrics from Traces](#metrics-from-traces)
2. [Identifying Bottlenecks](#identifying-bottlenecks)
3. [Performance Patterns](#performance-patterns)
4. [Optimization Strategies](#optimization-strategies)
5. [Dashboards](#dashboards)
6. [Common Performance Issues](#common-performance-issues)

## Metrics from Traces

### Key Performance Indicators

Distributed traces provide valuable metrics:

```typescript
import { TraceMetrics } from '@bubblelab/bubble-core/tracing';

const metrics = new TraceMetrics();

// Get overall metrics
const overallMetrics = metrics.getMetrics();
console.log(overallMetrics);
// {
//   totalTraces: 1000,
//   totalSpans: 5000,
//   errorRate: 0.02,  // 2% error rate
//   avgDuration: 1234.5,
//   p50Duration: 800,
//   p95Duration: 2500,
//   p99Duration: 5000,
//   throughput: 45.2  // operations per second
// }
```

### Operation-Specific Metrics

```typescript
// Get metrics for specific operation
const bubbleMetrics = metrics.getOperationMetrics('ai-agent');

console.log(bubbleMetrics);
// {
//   count: 500,
//   avgDuration: 2500,
//   p95Duration: 5000,
//   p99Duration: 8000,
//   errorRate: 0.03
// }
```

### Calculated Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Throughput** | Operations per second | `1000 / avg_duration` |
| **Error Rate** | Percentage of failed operations | `errors / total * 100` |
| **P50 Latency** | Median latency | 50th percentile |
| **P95 Latency** | 95th percentile latency | 95th percentile |
| **P99 Latency** | 99th percentile latency | 99th percentile |
| **Saturation** | How busy the system is | `avg_duration / max_duration` |

## Identifying Bottlenecks

### Automatic Bottleneck Detection

```typescript
import { analyzePerformance } from '@bubblelab/bubble-core/tracing';

const analysis = analyzePerformance();

console.log(analysis.bottlenecks);
// [
//   {
//     operation: 'bubble.execution:ai-agent',
//     avgDuration: 5000,
//     impact: 'critical',
//     frequency: 100,
//     suggestedAction: 'Consider implementing response caching'
//   }
// ]
```

### Manual Bottleneck Analysis

1. **Identify Slow Operations**
   ```typescript
   const metrics = new TraceMetrics();
   const slowest = metrics.getMetrics().slowestOperations;

   // Top 10 slowest operations
   slowest.forEach(op => {
     console.log(`${op.name}: ${op.duration}ms`);
   });
   ```

2. **Analyze Critical Path**
   ```typescript
   const analysis = analyzePerformance();
   const criticalPath = analysis.criticalPath;

   criticalPath.forEach(path => {
     console.log(`Path: ${path.operations.join(' → ')}`);
     console.log(`Total: ${path.totalDuration}ms`);
     console.log(`Impact: ${path.percentageOfTotal.toFixed(1)}%`);
   });
   ```

3. **Compare Operation Performance**
   ```typescript
   // Compare database vs API calls
   const dbMetrics = metrics.getOperationMetrics('bubble.database.query');
   const apiMetrics = metrics.getOperationMetrics('bubble.api.call');

   console.log('DB avg:', dbMetrics?.avgDuration);
   console.log('API avg:', apiMetrics?.avgDuration);
   ```

### Bottleneck Categories

| Category | Symptoms | Common Causes |
|----------|----------|---------------|
| **Database** | High latency in `db.query` spans | Missing indexes, N+1 queries, large result sets |
| **External API** | High latency in `http.request` spans | Slow APIs, network issues, rate limiting |
| **AI/LLM** | High latency in `ai-agent` spans | Large prompts, complex models, token limits |
| **Computation** | High CPU usage, long processing | Inefficient algorithms, data transformation |
| **I/O** | Slow file/network operations | Large file sizes, disk contention |

## Performance Patterns

### Common Patterns

#### 1. N+1 Query Problem

```
❌ Bad Pattern:
├─ Get User (10ms)
├─ Get User Posts (15ms) × 100
└─ Total: 1510ms

✅ Good Pattern:
├─ Get User (10ms)
├─ Get All Posts (50ms)
└─ Total: 60ms
```

**Detection**: Look for many identical database operations in one trace

#### 2. Sequential vs Parallel

```
❌ Sequential:
├─ API Call 1 (500ms)
├─ API Call 2 (500ms)
└─ API Call 3 (500ms)
Total: 1500ms

✅ Parallel:
├─ API Call 1 (500ms) ┐
├─ API Call 2 (500ms) ├→ Parallel execution
└─ API Call 3 (500ms) ┘
Total: 500ms
```

**Detection**: Look for sequential spans that could run in parallel

#### 3. Cache Misses

```
❌ Cache Miss:
├─ Check Cache (1ms) → miss
├─ Fetch from DB (500ms)
└─ Total: 501ms

✅ Cache Hit:
├─ Check Cache (1ms) → hit
└─ Total: 1ms
```

**Detection**: Compare cache hit vs miss patterns

#### 4. Chatty Services

```
❌ Too Many Calls:
├─ Get User (10ms)
├─ Get Profile (10ms)
├─ Get Settings (10ms)
├─ Get Preferences (10ms)
└─ ... (10 more calls)
Total: 130ms

✅ Batched Call:
├─ Get All User Data (20ms)
└─ Total: 20ms
```

**Detection**: Many small calls to same service

## Optimization Strategies

### 1. Database Optimization

**Identify Slow Queries**
```typescript
// Find slow database operations
const dbMetrics = metrics.getOperationMetrics('bubble.database.query');
if (dbMetrics && dbMetrics.p95Duration > 1000) {
  console.warn('Database queries are slow (P95 > 1s)');
}
```

**Solutions**:
- Add indexes on frequently queried columns
- Use query optimization (EXPLAIN ANALYZE)
- Implement query result caching
- Use connection pooling
- Batch multiple queries into one

**Example Improvement**:
```typescript
// Before: N+1 queries
for (const userId of userIds) {
  const user = await db.query('SELECT * FROM users WHERE id = $1', [userId]);
}

// After: Single query with IN clause
const users = await db.query(
  'SELECT * FROM users WHERE id = ANY($1)',
  [userIds]
);
```

### 2. Caching Strategy

**Identify Caching Opportunities**
```typescript
// Find operations with high repetition
const operationNames = new Set();
traces.forEach(trace => {
  trace.spans.forEach(span => {
    operationNames.add(span.operationName);
  });
});

// If same operation appears frequently, consider caching
```

**Cache Layers**:
1. **In-memory cache** (Redis, Memcached)
2. **Application cache** (LRU cache)
3. **CDN cache** (for static content)
4. **Browser cache** (HTTP caching)

**Example**:
```typescript
import { Redis } from 'ioredis';

const redis = new Redis();

async function getCachedUser(userId: string) {
  // Check cache first
  const cached = await redis.get(`user:${userId}`);
  if (cached) {
    return JSON.parse(cached);
  }

  // Cache miss - fetch from database
  const user = await db.query('SELECT * FROM users WHERE id = $1', [userId]);

  // Store in cache for 5 minutes
  await redis.setex(`user:${userId}`, 300, JSON.stringify(user));

  return user;
}
```

### 3. Parallel Execution

**Identify Parallelizable Operations**
```typescript
// Look for sequential independent operations
const sequentialSpans = trace.spans.filter(span =>
  span.parentSpanId &&
  !span.overlapsWithSiblings()
);
```

**Example Improvement**:
```typescript
// Before: Sequential
const user = await getUser(userId);
const posts = await getUserPosts(userId);
const comments = await getUserComments(userId);

// After: Parallel
const [user, posts, comments] = await Promise.all([
  getUser(userId),
  getUserPosts(userId),
  getUserComments(userId),
]);
```

### 4. AI/LLM Optimization

**Reduce Token Usage**
```typescript
// Before: Long prompt
const systemPrompt = `
  You are a helpful assistant with extensive knowledge
  about many topics. Please provide detailed responses...
`;

// After: Concise prompt
const systemPrompt = 'You are a helpful assistant. Be concise.';
```

**Cache AI Responses**
```typescript
async function getAIResponse(prompt: string) {
  const cacheKey = `ai:${hash(prompt)}`;

  // Check cache
  const cached = await redis.get(cacheKey);
  if (cached) {
    return cached;
  }

  // Call AI
  const response = await ai.complete(prompt);

  // Cache for 1 hour
  await redis.setex(cacheKey, 3600, response);

  return response;
}
```

**Use Smaller Models**
```typescript
// For simple tasks, use smaller/faster models
const model = taskComplexity === 'high'
  ? 'gpt-4'        // Slower, more capable
  : 'gpt-3.5-turbo'; // Faster, sufficient for simple tasks
```

### 5. External API Optimization

**Batch Requests**
```typescript
// Before: Individual requests
for (const id of ids) {
  const result = await api.get(`/items/${id}`);
}

// After: Batch request
const results = await api.post('/items/batch', { ids });
```

**Implement Request Coalescing**
```typescript
import { pLimit } from 'p-limit';

// Limit concurrent requests
const limit = pLimit(10); // Max 10 concurrent requests

const promises = ids.map(id =>
  limit(() => api.get(`/items/${id}`))
);

const results = await Promise.all(promises);
```

**Use HTTP/2**
```typescript
// HTTP/2 supports multiplexing
const agent = new http2.Agent();

const response = await fetch(url, {
  // Use HTTP/2 for better performance
  dispatcher: agent,
});
```

## Dashboards

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "BubbleLab Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(bubble_operation_total[5m])"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, bubble_operation_duration_seconds)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(bubble_operation_error_total[5m]) / rate(bubble_operation_total[5m])"
          }
        ]
      },
      {
        "title": "Slow Operations",
        "targets": [
          {
            "expr": "bubble_operation_duration_seconds{quantile='0.99'} > 1"
          }
        ]
      }
    ]
  }
}
```

### Key Metrics to Monitor

| Metric | Query | Alert Threshold |
|--------|-------|-----------------|
| **Request Rate** | `rate(bubble_operation_total[5m])` | < 0.1 ops/sec (WARNING) |
| **P95 Latency** | `histogram_quantile(0.95, duration)` | > 30s (WARNING) |
| **P99 Latency** | `histogram_quantile(0.99, duration)` | > 60s (CRITICAL) |
| **Error Rate** | `rate(errors[5m]) / rate(total[5m])` | > 5% (ERROR) |
| **Memory Usage** | `process_resident_memory_bytes` | > 1GB (WARNING) |

## Common Performance Issues

### 1. Memory Leaks

**Symptoms**:
- Increasing memory usage over time
- Gradually slowing performance
- Out of memory errors

**Detection**:
```typescript
// Monitor memory in spans
span.setAttribute('memory.used.mb', process.memoryUsage().heapUsed / 1024 / 1024);
```

**Solution**:
- Fix circular references
- Clear caches appropriately
- Use weak references for large objects

### 2. Connection Pool Exhaustion

**Symptoms**:
- Timeouts waiting for database connections
- High latency under load

**Detection**:
```typescript
// Monitor pool usage
span.setAttribute('db.pool.active', pool.activeConnections);
span.setAttribute('db.pool.idle', pool.idleConnections);
```

**Solution**:
```typescript
// Increase pool size
const pool = new Pool({
  max: 20,  // Increase from default
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});
```

### 3. Synchronous Blocking

**Symptoms**:
- Event loop blocked
- No requests processed during blocking operation

**Detection**:
```typescript
// Monitor event loop lag
const start = Date.now();
setImmediate(() => {
  const lag = Date.now() - start;
  span.setAttribute('eventloop.lag.ms', lag);
});
```

**Solution**:
- Use async operations
- Break up CPU-intensive work
- Use worker threads for heavy computation

### 4. Unbounded Growth

**Symptoms**:
- Arrays/maps growing indefinitely
- Increasing memory usage

**Detection**:
```typescript
span.setAttribute('cache.size', cache.size);
span.setAttribute('queue.length', queue.length);
```

**Solution**:
```typescript
// Use LRU cache with max size
const LRU = require('lru-cache');
const cache = new LRU({ max: 1000 });
```

## Performance Optimization Checklist

- [ ] Identify slow operations (P95 > 3s)
- [ ] Check for N+1 query problems
- [ ] Implement caching for hot data
- [ ] Parallelize independent operations
- [ ] Optimize database queries (indexes, batch)
- [ ] Reduce AI/LLM token usage
- [ ] Implement request batching
- [ ] Monitor memory usage
- [ ] Check connection pool settings
- [ ] Review event loop lag
- [ ] Set up performance alerts
- [ ] Create performance dashboards

## Next Steps

- [Trace Visualization Guide](./TRACE_VISUALIZATION.md)
- [OpenTelemetry Setup](./OPENTELEMETRY_SETUP.md)
- [Troubleshooting Traces](./TROUBLESHOOTING_TRACES.md)
