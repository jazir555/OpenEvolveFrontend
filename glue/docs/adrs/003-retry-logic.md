# ADR-003: Retry Logic with Exponential Backoff

## Status
**Accepted**

## Context
The OpenEvolve Frontend communicates with external services over the internet, which can experience:
- Transient network failures (packet loss, DNS blips)
- Temporary service unavailability (deployments, restarts)
- Rate limiting (429 responses)
- Load balancer hiccups

Many of these failures are **transient** - they would succeed if retried after a short delay.

## Problem
Without retry logic:
1. **False failures**: Temporary network blips cause permanent errors
2. **Poor UX**: Users see errors for issues that would self-correct
3. **Unnecessary support**: Users report issues that would resolve themselves
4. **Cascading failures**: One temporary failure causes downstream failures

## Decision
Implement **retry logic with exponential backoff and jitter** for all transient failures.

### Implementation

#### Retry Strategy
- **Max retries**: 3 attempts (initial + 3 retries = 4 total)
- **Backoff**: Exponential (100ms, 200ms, 400ms, 800ms)
- **Jitter**: Add random ±25% to prevent thundering herd
- **Timeout**: Overall timeout of 30 seconds

#### Retryable Errors
Retry on:
- Network errors (ECONNRESET, ETIMEDOUT, ENOTFOUND)
- HTTP 408 (Request Timeout)
- HTTP 429 (Rate Limited) - with respect to Retry-After header
- HTTP 500, 502, 503, 504 (Server errors)
- HTTP 599 (Network errors)

#### Non-Retryable Errors
Don't retry on:
- HTTP 400 (Bad Request) - client error, won't fix itself
- HTTP 401 (Unauthorized) - need new credentials
- HTTP 403 (Forbidden) - permission issue
- HTTP 404 (Not Found) - resource doesn't exist
- HTTP 422 (Unprocessable Entity) - validation error

#### Code
```typescript
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= config.max_retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      // Don't retry if it's a client error (4xx)
      if (isClientError(error)) {
        throw error;
      }

      // Don't retry after max attempts
      if (attempt >= config.max_retries) {
        throw error;
      }

      // Exponential backoff with jitter
      const baseDelay = config.base_delay_ms * Math.pow(2, attempt);
      const jitter = baseDelay * 0.25 * (Math.random() * 2 - 1);
      const delay = baseDelay + jitter;

      await sleep(delay);
    }
  }

  throw lastError;
}
```

### Integration
- Built into all API clients (OpenEvolve, RAGBits, Datapizza)
- Works with circuit breaker (retry before tripping circuit)
- Logs each retry attempt with correlation ID

### Example
```typescript
const result = await retryWithBackoff(
  async () => await fetch(url),
  { max_retries: 3, base_delay_ms: 100 }
);
```

## Consequences

### Positive
- ✅ **Resilient**: Handles transient failures automatically
- ✅ **Better UX**: Users don't see temporary errors
- ✅ **Jitter**: Prevents thundering herd on service recovery
- ✅ **Smart**: Doesn't retry client errors (waste of time)
- ✅ **Observable**: Logs retry attempts for debugging

### Negative
- ⚠️ **Latency**: Failed requests take longer (100ms + 200ms + 400ms = 700ms)
- ⚠️ **Masking**: Can hide real issues if everything is retried
- ⚠️ **Cost**: More API calls (4x worst case)

### Mitigations
- Limit retries to 3 attempts
- Only retry transient errors
- Log all retry attempts
- Use circuit breaker to stop retrying failing service
- Respect Retry-After header for rate limiting

## Alternatives Considered

### Alternative 1: No Retry
**Description**: Fail immediately on any error

**Pros**: Simple, fast feedback, predictable latency

**Cons**: Poor resilience, false errors, bad UX

**Rejected**: Unacceptable for production system

### Alternative 2: Fixed Delay Retry
**Description**: Retry with fixed 1 second delay

**Pros**: Simpler than exponential backoff

**Cons**: Thundering herd problem, no adaptation

**Rejected**: Can cause service overload on recovery

### Alternative 3: Unlimited Retries
**Description**: Retry forever until success

**Pros**: Maximum resilience

**Cons**: Infinite hangs, wasted resources, bad UX

**Rejected**: Violates timeout requirements

## Related Decisions
- [ADR-001: Circuit Breaker Pattern](./001-circuit-breaker.md)
- [ADR-002: Structured Logging with Correlation IDs](./002-structured-logging.md)

## Implementation Date
2026-02-15

## Author
OpenEvolve Federation Team
