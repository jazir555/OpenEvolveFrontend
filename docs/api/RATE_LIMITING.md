# Rate Limiting

Complete guide to rate limiting in BubbleLab.

**Table of Contents:**
- [Overview](#overview)
- [Rate Limit Tiers](#rate-limit-tiers)
- [Rate Limit Headers](#rate-limit-headers)
- [Handling Rate Limits](#handling-rate-limits)
- [Best Practices](#best-practices)
- [Rate Limit Strategies](#rate-limit-strategies)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [FAQ](#faq)

---

## Overview

BubbleLab implements rate limiting to ensure fair usage and system stability.

### Rate Limiting Philosophy

- **Fair Usage**: Prevent any single user from monopolizing resources
- **System Stability**: Protect against traffic spikes
- **Cost Management**: Control infrastructure costs
- **Predictable Performance**: Ensure consistent response times

### Rate Limit Types

1. **Request Rate Limits**: Maximum requests per time period
2. **Burst Limits**: Maximum concurrent requests
3. **Quota Limits**: Total resources per billing period
4. **Concurrent Execution Limits**: Maximum simultaneous flow executions

---

## Rate Limit Tiers

### Free Tier

| Resource | Limit | Period |
|----------|-------|--------|
| API Requests | 1,000 | per hour |
| Flow Executions | 100 | per day |
| Concurrent Executions | 3 | simultaneous |
| Storage | 1 GB | total |
| Webhooks | 50 | per day |

**Rate Limit:**
- 10 requests per second
- 1,000 requests per hour
- 10,000 requests per day

---

### Pro Tier

| Resource | Limit | Period |
|----------|-------|--------|
| API Requests | 10,000 | per hour |
| Flow Executions | 1,000 | per day |
| Concurrent Executions | 10 | simultaneous |
| Storage | 10 GB | total |
| Webhooks | 500 | per day |

**Rate Limit:**
- 50 requests per second
- 10,000 requests per hour
- 100,000 requests per day

---

### Enterprise Tier

| Resource | Limit | Period |
|----------|-------|--------|
| API Requests | Unlimited | - |
| Flow Executions | Unlimited | - |
| Concurrent Executions | 100 | simultaneous |
| Storage | 1 TB | total |
| Webhooks | Unlimited | - |

**Rate Limit:**
- 1,000 requests per second
- Custom limits available
- Dedicated resources

---

### Bubble-Specific Limits

Different bubbles have different rate limits:

**Service Bubbles:**

| Bubble | Requests/Second | Requests/Minute | Notes |
|--------|----------------|-----------------|-------|
| HTTP Bubble | 10 | 600 | Per external API |
| AI Agent | 5 | 300 | Provider-dependent |
| PostgreSQL | 50 | 3,000 | Per database |
| Slack | 1 | 60 | Slack API limit |
| Storage | 100 | 6,000 | Uploads/downloads |

**Tool Bubbles:**

| Tool | Executions/Minute | Executions/Hour | Notes |
|------|-------------------|-----------------|-------|
| Code Edit | 60 | 3,600 | Per user |
| Chart.js | 120 | 7,200 | Per user |
| Research Agent | 10 | 600 | Provider-dependent |
| Social Media | 30 | 1,800 | Per platform |

---

## Rate Limit Headers

All API responses include rate limit headers:

### Response Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1642579200
X-RateLimit-Reset-Text: Thu, 18 Jan 2024 11:00:00 GMT
Retry-After: 60
X-RateLimit-Bucket: api-requests
X-RateLimit-User: user-123
```

### Header Descriptions

| Header | Description | Example |
|--------|-------------|---------|
| `X-RateLimit-Limit` | Maximum requests per period | `1000` |
| `X-RateLimit-Remaining` | Remaining requests in period | `950` |
| `X-RateLimit-Reset` | Unix timestamp of reset | `1642579200` |
| `X-RateLimit-Reset-Text` | Human-readable reset time | `Thu, 18 Jan 2024 11:00:00 GMT` |
| `Retry-After` | Seconds to wait before retry | `60` |
| `X-RateLimit-Bucket` | Rate limit bucket name | `api-requests` |
| `X-RateLimit-User` | User identifier | `user-123` |

---

## Handling Rate Limits

### Detecting Rate Limits

```typescript
async function makeRequest(url, options) {
  const response = await fetch(url, options);

  // Check rate limit headers
  const rateLimit = {
    limit: parseInt(response.headers.get('X-RateLimit-Limit')),
    remaining: parseInt(response.headers.get('X-RateLimit-Remaining')),
    reset: parseInt(response.headers.get('X-RateLimit-Reset')),
    resetText: response.headers.get('X-RateLimit-Reset-Text'),
    retryAfter: parseInt(response.headers.get('Retry-After'))
  };

  console.log('Rate limit status:', rateLimit);

  if (response.status === 429) {
    console.log('Rate limit exceeded. Retry after:', rateLimit.retryAfter, 'seconds');
  }

  return response;
}
```

---

### Automatic Retry with Backoff

```typescript
class RateLimitedClient {
  constructor() {
    this.requestLog = [];
  }

  async request(url, options, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const response = await fetch(url, options);

        // Success
        if (response.status === 429) {
          const retryAfter = parseInt(response.headers.get('Retry-After')) || 60;
          console.log(`Rate limited. Retrying after ${retryAfter} seconds...`);

          await sleep(retryAfter * 1000);
          continue;
        }

        if (response.ok) {
          return await response.json();
        }

        // Non-retryable error
        throw new Error(`Request failed: ${response.status}`);
      } catch (error) {
        if (attempt === maxRetries - 1) {
          throw error;
        }

        // Exponential backoff
        const backoff = Math.pow(2, attempt) * 1000;
        await sleep(backoff);
      }
    }
  }
}

// Usage
const client = new RateLimitedClient();
const result = await client.request('https://api.bubblelab.io/v1/flows');
```

---

### Token Bucket Algorithm

```typescript
class TokenBucket {
  constructor(capacity, refillRate) {
    this.capacity = capacity;      // Maximum tokens
    this.refillRate = refillRate;  // Tokens per second
    this.tokens = capacity;        // Current tokens
    this.lastRefill = Date.now();
  }

  async consume(tokens = 1) {
    // Refill tokens
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(
      this.capacity,
      this.tokens + (elapsed * this.refillRate)
    );
    this.lastRefill = now;

    // Check if we have enough tokens
    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }

    // Calculate wait time
    const waitTime = (tokens - this.tokens) / this.refillRate;
    await sleep(waitTime * 1000);

    // Try again
    return this.consume(tokens);
  }
}

// Usage
const bucket = new TokenBucket(10, 1); // 10 tokens, refills at 1/second

for (let i = 0; i < 20; i++) {
  await bucket.consume(1);
  console.log(`Request ${i + 1} sent`);
}
```

---

### Sliding Window Log

```typescript
class SlidingWindowLog {
  constructor(limit, windowMs) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.requests = [];
  }

  async allowRequest() {
    const now = Date.now();

    // Remove old requests outside window
    this.requests = this.requests.filter(
      timestamp => now - timestamp < this.windowMs
    );

    // Check if under limit
    if (this.requests.length < this.limit) {
      this.requests.push(now);
      return true;
    }

    // Calculate wait time
    const oldestRequest = this.requests[0];
    const waitTime = this.windowMs - (now - oldestRequest);

    await sleep(waitTime);

    // Try again
    return this.allowRequest();
  }
}

// Usage
const limiter = new SlidingWindowLog(100, 60000); // 100 requests per minute

for (let i = 0; i < 150; i++) {
  await limiter.allowRequest();
  console.log(`Request ${i + 1} allowed`);
}
```

---

## Best Practices

### 1. Implement Exponential Backoff

```typescript
async function requestWithBackoff(url, options, maxRetries = 5) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get('Retry-After')) || 60;

        // Exponential backoff with jitter
        const backoff = Math.min(
          retryAfter * Math.pow(2, attempt),
          600 // Max 10 minutes
        );
        const jitter = Math.random() * 1000; // Random jitter

        await sleep(backoff * 1000 + jitter);
        continue;
      }

      if (response.ok) {
        return await response.json();
      }

      throw new Error(`Request failed: ${response.status}`);
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error;
      }

      const backoff = Math.pow(2, attempt) * 1000;
      await sleep(backoff);
    }
  }
}
```

---

### 2. Batch Requests

```typescript
// Instead of multiple individual requests
for (const item of items) {
  await api.createItem(item);
}

// Use batch request
await api.createItems(items); // Much more efficient
```

---

### 3. Cache Responses

```typescript
class CachedClient {
  constructor() {
    this.cache = new Map();
    this.ttl = 60000; // 1 minute
  }

  async get(url) {
    const cached = this.cache.get(url);

    if (cached && Date.now() - cached.timestamp < this.ttl) {
      console.log('Cache hit:', url);
      return cached.data;
    }

    console.log('Cache miss:', url);
    const response = await fetch(url);
    const data = await response.json();

    this.cache.set(url, {
      data: data,
      timestamp: Date.now()
    });

    return data;
  }
}

// Usage
const client = new CachedClient();
const data = await client.get('https://api.bubblelab.io/v1/flows');
```

---

### 4. Use Webhooks Instead of Polling

```typescript
// Instead of polling
while (true) {
  const result = await api.checkStatus(jobId);
  if (result.status === 'complete') {
    break;
  }
  await sleep(5000);
}

// Use webhooks
await api.createJob(jobData, {
  webhookUrl: 'https://your-app.com/webhook'
});

// When job is complete, your webhook endpoint is called
app.post('/webhook', async (req, res) => {
  const { jobId, status, result } = req.body;
  console.log('Job complete:', jobId, result);
  res.status(200).send('OK');
});
```

---

### 5. Prioritize Requests

```typescript
class PriorityQueue {
  constructor() {
    this.high = [];
    this.medium = [];
    this.low = [];
  }

  enqueue(request, priority = 'medium') {
    this[priority].push(request);
  }

  async process() {
    while (this.high.length > 0) {
      await this.execute(this.high.shift());
    }

    while (this.medium.length > 0) {
      await this.execute(this.medium.shift());
    }

    while (this.low.length > 0) {
      await this.execute(this.low.shift());
    }
  }

  async execute(request) {
    // Implement rate limiting
    await rateLimiter.consume();
    return await fetch(request.url, request.options);
  }
}

// Usage
const queue = new PriorityQueue();

queue.enqueue(
  { url: '/api/critical', options: {} },
  'high'
);

queue.enqueue(
  { url: '/api/normal', options: {} },
  'medium'
);

queue.enqueue(
  { url: '/api/background', options: {} },
  'low'
);

await queue.process();
```

---

## Rate Limit Strategies

### Strategy 1: Fixed Window Counter

**Simple but has edge cases**

```typescript
class FixedWindowCounter {
  constructor(limit, windowMs) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.count = 0;
    this.windowStart = Date.now();
  }

  async allowRequest() {
    const now = Date.now();

    // Reset window if expired
    if (now - this.windowStart >= this.windowMs) {
      this.count = 0;
      this.windowStart = now;
    }

    // Check limit
    if (this.count < this.limit) {
      this.count++;
      return true;
    }

    return false;
  }
}
```

**Pros:**
- Simple to implement
- Low memory usage

**Cons:**
- Allows bursts at window boundaries
- Can be unfair

---

### Strategy 2: Sliding Window Log

**More accurate but uses more memory**

```typescript
class SlidingWindowLog {
  constructor(limit, windowMs) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.requests = [];
  }

  async allowRequest() {
    const now = Date.now();

    // Remove old requests
    this.requests = this.requests.filter(
      timestamp => now - timestamp < this.windowMs
    );

    // Check limit
    if (this.requests.length < this.limit) {
      this.requests.push(now);
      return true;
    }

    return false;
  }
}
```

**Pros:**
- Very accurate
- Smooth rate limiting

**Cons:**
- Higher memory usage
- More complex

---

### Strategy 3: Token Bucket

**Good for burst traffic**

```typescript
class TokenBucket {
  constructor(capacity, refillRate) {
    this.capacity = capacity;
    this.refillRate = refillRate;
    this.tokens = capacity;
    this.lastRefill = Date.now();
  }

  async consume(tokens = 1) {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;

    // Refill tokens
    this.tokens = Math.min(
      this.capacity,
      this.tokens + (elapsed * this.refillRate)
    );
    this.lastRefill = now;

    // Check if we have enough tokens
    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }

    return false;
  }
}
```

**Pros:**
- Allows bursts up to capacity
- Smooth rate limiting

**Cons:**
- Requires tuning capacity and refill rate

---

### Strategy 4: Leaky Bucket

**Good for smoothing traffic**

```typescript
class LeakyBucket {
  constructor(capacity, leakRate) {
    this.capacity = capacity;
    this.leakRate = leakRate; // Requests per second
    this.queue = [];
    this.lastLeak = Date.now();
  }

  async addRequest(request) {
    const now = Date.now();

    // Leak requests
    const elapsed = (now - this.lastLeak) / 1000;
    const leakAmount = Math.floor(elapsed * this.leakRate);
    this.queue.splice(0, leakAmount);
    this.lastLeak = now;

    // Check if queue is full
    if (this.queue.length < this.capacity) {
      this.queue.push(request);
      return true;
    }

    return false;
  }
}
```

**Pros:**
- Smooths out traffic
- Prevents bursts

**Cons:**
- Adds latency
- Requires queue management

---

## Monitoring and Alerts

### Track Rate Limit Usage

```typescript
class RateLimitMonitor {
  constructor() {
    this.metrics = {
      totalRequests: 0,
      rateLimitedRequests: 0,
      retryCount: 0,
      averageWaitTime: 0
    };
  }

  recordRequest(rateLimited = false, waitTime = 0) {
    this.metrics.totalRequests++;

    if (rateLimited) {
      this.metrics.rateLimitedRequests++;
    }

    if (waitTime > 0) {
      this.metrics.retryCount++;
      this.metrics.averageWaitTime =
        (this.metrics.averageWaitTime * (this.metrics.retryCount - 1) + waitTime) /
        this.metrics.retryCount;
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      rateLimitRate: this.metrics.rateLimitedRequests / this.metrics.totalRequests,
      retryRate: this.metrics.retryCount / this.metrics.totalRequests
    };
  }

  logReport() {
    const metrics = this.getMetrics();

    console.log('Rate Limit Metrics:');
    console.log(`  Total Requests: ${metrics.totalRequests}`);
    console.log(`  Rate Limited: ${metrics.rateLimitedRequests} (${(metrics.rateLimitRate * 100).toFixed(2)}%)`);
    console.log(`  Retries: ${metrics.retryCount} (${(metrics.retryRate * 100).toFixed(2)}%)`);
    console.log(`  Avg Wait Time: ${metrics.averageWaitTime.toFixed(2)}ms`);
  }
}

// Usage
const monitor = new RateLimitMonitor();

async function makeRequest(url) {
  const start = Date.now();
  let rateLimited = false;

  try {
    const response = await fetch(url);

    if (response.status === 429) {
      rateLimited = true;
      const retryAfter = parseInt(response.headers.get('Retry-After'));
      await sleep(retryAfter * 1000);
      return await makeRequest(url);
    }

    const waitTime = Date.now() - start;
    monitor.recordRequest(rateLimited, waitTime);

    return await response.json();
  } catch (error) {
    monitor.recordRequest(rateLimited, Date.now() - start);
    throw error;
  }
}

// Log report periodically
setInterval(() => {
  monitor.logReport();
}, 60000); // Every minute
```

---

### Set Up Alerts

```typescript
class RateLimitAlert {
  constructor(thresholds) {
    this.thresholds = {
      rateLimitRate: 0.1, // Alert if 10% of requests are rate limited
      retryRate: 0.05,     // Alert if 5% of requests require retries
      ...thresholds
    };
  }

  check(metrics) {
    const alerts = [];

    if (metrics.rateLimitRate > this.thresholds.rateLimitRate) {
      alerts.push({
        severity: 'warning',
        message: `High rate limit rate: ${(metrics.rateLimitRate * 100).toFixed(2)}%`,
        threshold: this.thresholds.rateLimitRate
      });
    }

    if (metrics.retryRate > this.thresholds.retryRate) {
      alerts.push({
        severity: 'warning',
        message: `High retry rate: ${(metrics.retryRate * 100).toFixed(2)}%`,
        threshold: this.thresholds.retryRate
      });
    }

    if (metrics.averageWaitTime > 1000) {
      alerts.push({
        severity: 'critical',
        message: `High average wait time: ${metrics.averageWaitTime.toFixed(2)}ms`,
        threshold: 1000
      });
    }

    return alerts;
  }
}

// Usage
const monitor = new RateLimitMonitor();
const alerter = new RateLimitAlert({
  rateLimitRate: 0.1,
  retryRate: 0.05
});

setInterval(() => {
  const metrics = monitor.getMetrics();
  const alerts = alerter.check(metrics);

  alerts.forEach(alert => {
    console.log(`[${alert.severity.toUpperCase()}] ${alert.message}`);

    // Send to monitoring service
    sendAlert(alert);
  });
}, 60000);
```

---

## FAQ

### Q: What happens when I exceed my rate limit?

**A:** You'll receive a `429 Too Many Requests` response with a `Retry-After` header indicating how long to wait before retrying.

---

### Q: Do rate limits reset at midnight?

**A:** Not necessarily. Rate limits use sliding windows, not fixed time periods. Check the `X-RateLimit-Reset` header for the exact reset time.

---

### Q: Can I increase my rate limit?

**A:** Yes! Upgrade to a higher tier plan for increased rate limits. Contact sales for Enterprise custom limits.

---

### Q: Are webhooks rate limited?

**A:** Yes, webhooks have separate rate limits. Check your plan details for specific limits.

---

### Q: Do internal API calls count against my limit?

**A:** No, calls between your flows and bubbles don't count. Only external API requests are rate limited.

---

### Q: How are concurrent executions counted?

**A:** Each flow execution that's running simultaneously counts. The limit resets as executions complete.

---

### Q: What's the difference between requests and executions?

**A:**
- **Requests**: Individual API calls (e.g., `/api/flows`)
- **Executions**: Flow runs (which may involve multiple requests internally)

---

### Q: Can I get rate limit alerts?

**A:** Yes! Set up monitoring in your dashboard to receive alerts when approaching limits.

---

### Q: Do retries count against my limit?

**A:** Yes, all requests (including retries) count against your rate limit. Use exponential backoff to minimize retry attempts.

---

### Q: Is there a burst allowance?

**A:** Yes, most rate limits allow short bursts above the average rate. The exact burst allowance depends on your plan and the specific limit type.

---

### Q: How precise are rate limits?

**A:** Rate limits are enforced based on server time, not client time. Small discrepancies (±1 second) are normal due to network latency.

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** BubbleLab Core Team
