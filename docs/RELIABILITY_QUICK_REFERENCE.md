# Reliability Fixes - Quick Reference

## Overview
Critical reliability fixes for timeout, retry logic, and circuit breaker protection following the Federation Constitution.

## Files Changed

### Modified
- `BubbleLab/apps/bubble-studio/src/lib/api.ts` - Timeout & Retry logic
- `BubbleLab/apps/bubble-studio/src/services/evolutionApi.ts` - Circuit breaker integration

### New
- `BubbleLab/apps/bubble-studio/src/lib/circuitBreaker.ts` - Circuit breaker implementation

## Usage Examples

### Basic API Client (No Retry)
```typescript
import { ApiClient } from '@/lib/api';

const apiClient = new ApiClient(API_BASE_URL);
// Default: 30s timeout, no retry
```

### API Client with Retry
```typescript
import { ApiClient, ApiClientConfig } from '@/lib/api';

const config: ApiClientConfig = {
  baseURL: API_BASE_URL,
  timeout: 30000,      // 30 seconds
  enableRetry: true,   // Enable retry
  maxRetries: 3,       // Max 3 retries
  retryDelay: 1000,    // 1s base delay
};

const apiClient = new ApiClient(config);
```

### Evolution API (with Circuit Breaker)
```typescript
import { evolutionApi } from '@/services/evolutionApi';

// Automatic: timeout + retry + circuit breaker
try {
  const result = await evolutionApi.startEvolution(payload);
} catch (error) {
  // Circuit breaker open: "Circuit breaker [evolution-api] is OPEN"
  // Timeout: "Request timeout after 30000ms"
  // All retries exhausted: Original error after 3 retries
}
```

### Circuit Breaker Monitoring
```typescript
import { getEvolutionApiCircuitBreakerMetrics } from '@/services/evolutionApi';

const metrics = getEvolutionApiCircuitBreakerMetrics();
console.log(metrics);
// {
//   name: 'evolution-api',
//   state: 'closed' | 'open' | 'half_open',
//   failureCount: 0,
//   successCount: 42,
//   lastFailureTime: 0,
//   timeUntilReset: 0
// }
```

### Manual Circuit Breaker Reset
```typescript
import { resetEvolutionApiCircuitBreaker } from '@/services/evolutionApi';

// Force reset (for testing/recovery)
resetEvolutionApiCircuitBreaker();
```

## Configuration

### Timeout Configuration
```typescript
timeout: number  // milliseconds
// Default: 30000 (30 seconds)
// Recommended: 30000 for APIs, 60000 for long-running operations
```

### Retry Configuration
```typescript
enableRetry: boolean  // Enable/disable retry
// Default: false
// Recommended: true for external APIs, false for internal

maxRetries: number  // Maximum retry attempts
// Default: 3
// Recommended: 3-5

retryDelay: number  // Base delay in milliseconds
// Default: 1000 (1 second)
// Recommended: 1000-2000
```

### Circuit Breaker Configuration
```typescript
{
  failureThreshold: number,  // Failures before opening circuit
  // Default: 5
  // Recommended: 5-10

  timeout: number,  // Milliseconds to wait before reset attempt
  // Default: 60000 (60 seconds)
  // Recommended: 30000-120000

  halfOpenAttempts: number  // Successful attempts to close circuit
  // Default: 3
  // Recommended: 2-5
}
```

## Retry Behavior

### Retry Timeline
```
Attempt 1: Immediate
  ↓ Failure
Attempt 2: After 1000-1300ms (1s + 0-30% jitter)
  ↓ Failure
Attempt 3: After 2000-2600ms (2s + 0-30% jitter)
  ↓ Failure
Attempt 4: After 4000-5200ms (4s + 0-30% jitter)
  ↓ Failure
  → Throw error (all retries exhausted)
```

### Retryable Errors
- Network errors: "Failed to fetch", "Network error", "timeout"
- HTTP 5xx: Server errors (500, 502, 503, 504)
- HTTP 429: Rate limit exceeded

### Non-Retryable Errors
- HTTP 4xx (except 429): Client errors (400, 401, 403, 404)
- Authentication errors
- Validation errors

## Circuit Breaker States

### CLOSED (Normal)
- Requests pass through normally
- Failures increment counter
- After `failureThreshold` failures → OPEN

### OPEN (Blocking)
- All requests immediately rejected
- Error: "Circuit breaker [name] is OPEN. Blocking request..."
- After `timeout` milliseconds → HALF_OPEN

### HALF_OPEN (Testing)
- Limited requests allowed (halfOpenAttempts)
- Success: Increment success counter
- Failure: Back to OPEN
- After `halfOpenAttempts` successes → CLOSED

## Logging

### Correlation IDs
All requests include unique correlation ID for tracking:
```json
{
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_service": "bubble-studio",
  "target_service": "evolution-api"
}
```

### Retry Logs
```json
{
  "msg": "Retry attempt 2/3",
  "correlation_id": "...",
  "attempt": 2,
  "max_retries": 3,
  "delay_ms": 2156
}
```

### Circuit Breaker Logs
```json
{
  "msg": "[CircuitBreaker:evolution-api] Transitioned from CLOSED to OPEN after 5 failures"
}
```

### Timeout Logs
```json
{
  "msg": "Request timeout",
  "correlation_id": "...",
  "timeout_ms": 30000
}
```

## Error Types

### NetworkError
```typescript
// Thrown when: Network failure, timeout, DNS resolution failed
// Contains: correlationId, originalError, baseURL
```

### AuthenticationError
```typescript
// Thrown when: 401 Unauthorized
// Contains: correlationId, message
```

### RateLimitError
```typescript
// Thrown when: 429 Too Many Requests
// Contains: correlationId, retryAfter (seconds)
```

### ApiHttpError
```typescript
// Thrown when: Other HTTP errors
// Contains: status, data, correlationId
```

## Testing

### Test Timeout
```typescript
// Slow endpoint that takes > 30s
// Expected: AbortError with "Request timeout after 30000ms"
```

### Test Retry
```typescript
// Endpoint that fails temporarily
// Expected: Up to 3 retries with exponential backoff
// Check logs for "Retry attempt X/3"
```

### Test Circuit Breaker
```typescript
// Trigger 5 failures
// Expected: Circuit opens, blocks requests
// Wait 60s
// Expected: Circuit goes to HALF_OPEN
// Make 3 successful requests
// Expected: Circuit closes
```

## Best Practices

### DO ✅
- Always use ApiClient for HTTP requests (never raw fetch)
- Enable retry for external APIs
- Monitor circuit breaker metrics
- Include correlation IDs in logs
- Handle all error types appropriately

### DON'T ❌
- Don't use raw fetch without timeout
- Don't disable retry for external services
- Don't ignore circuit breaker OPEN state
- Don't set timeout < 5000ms (too aggressive)
- Don't set maxRetries > 5 (too many retries)

## Troubleshooting

### Problem: "Circuit breaker is OPEN"
**Cause**: Service has failed 5+ times recently
**Solution**:
1. Wait 60 seconds for automatic reset
2. Or call `resetEvolutionApiCircuitBreaker()` manually
3. Check service health

### Problem: "Request timeout after 30000ms"
**Cause**: Request took > 30 seconds
**Solution**:
1. Increase timeout for specific client
2. Or optimize slow endpoint
3. Check network connectivity

### Problem: Too many retries
**Cause**: Service consistently failing
**Solution**:
1. Reduce maxRetries to 2-3
2. Check service health
3. Circuit breaker should open to prevent spam

### Problem: Retries not working
**Cause**: enableRetry is false
**Solution**:
```typescript
const config: ApiClientConfig = {
  baseURL: API_BASE_URL,
  enableRetry: true,  // Make sure this is true
  maxRetries: 3,
};
```

## Federation Constitution Compliance

✅ Law of "Runtime Truth" - Verify execution
✅ Law of Idempotency - Safe to retry operations
✅ Law of Configuration Explicitness - No magic defaults
✅ Law of UTC - All timestamps in UTC
✅ Transient Failure → Exponential Backoff Retry (Jittered)
✅ System Failure → Circuit Breaker
✅ Every HTTP request has timeout
✅ Observability - Structured logging with correlation IDs

## Support

For issues or questions:
1. Check logs for correlation ID
2. Get circuit breaker metrics
3. Review error type and message
4. See RELIABILITY_FIXES_SUMMARY.md for details
