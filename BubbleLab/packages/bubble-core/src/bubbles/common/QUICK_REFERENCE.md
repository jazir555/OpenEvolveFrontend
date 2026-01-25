# Common Utilities - Quick Reference Guide

## Overview

The `bubbles/common/` directory contains shared utilities for all bubble implementations. These utilities provide validation, error handling, retry logic, connection pooling, caching, and more.

## Module Index

| Module | Purpose | Lines | Status |
|--------|---------|-------|--------|
| `validators.ts` | Input validation & sanitization | 600+ | ✅ Complete |
| `error-handlers.ts` | Custom errors & error handling | 400+ | ✅ Complete |
| `retry.ts` | Retry logic & circuit breakers | 500+ | ✅ Complete |
| `types.ts` | Shared types & interfaces | 600+ | ✅ Complete |
| `constants.ts` | Configuration constants | 400+ | ✅ Complete |
| `connection-pool.ts` | Connection pooling | 500+ | ✅ Complete |
| `cache.ts` | Response caching | 600+ | ✅ Complete |

**Total**: ~3,700 lines of production-ready code

## Quick Import

```typescript
// Import everything from common
import * as Common from '../common/index.js';

// Or import specific modules
import {
  validateEmail,
  validateUrl,
  ValidationError,
  NetworkError,
  retryWithBackoff,
  CircuitBreaker,
  InMemoryCache,
  CONNECTION_POOL,
  CACHE_TTL
} from '../common/index.js';
```

## Validator Quick Reference

### Email Validation
```typescript
import { validateEmail, createEmailSchema } from '../common/index.js';

// Function-style
try {
  validateEmail('user@example.com');
  console.log('Valid email');
} catch (error) {
  console.error('Invalid email:', error.message);
}

// Zod schema
const emailSchema = createEmailSchema();
const result = emailSchema.parse('user@example.com');
```

### URL Validation
```typescript
import { validateUrl, createUrlSchema } from '../common/index.js';

// Function-style
try {
  validateUrl('https://api.example.com', ['http', 'https']);
  console.log('Valid URL');
} catch (error) {
  console.error('Invalid URL:', error.message);
}

// Zod schema
const urlSchema = createUrlSchema(['http', 'https']);
const result = urlSchema.parse('https://api.example.com');
```

### File Path Validation
```typescript
import { validateFilePath } from '../common/index.js';

try {
  validateFilePath('./uploads/file.txt', false); // No absolute paths
  console.log('Valid file path');
} catch (error) {
  console.error('Invalid file path:', error.message);
}
```

## Error Handling Quick Reference

### Throwing Custom Errors
```typescript
import {
  ValidationError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  NetworkError
} from '../common/index.js';

// Validation error
if (!email) {
  throw new ValidationError('Email is required', 'email');
}

// Authentication error
if (!token) {
  throw new AuthenticationError('Invalid or missing token');
}

// Not found error
if (!user) {
  throw new NotFoundError('User', userId);
}

// Rate limit error
if (rateLimitExceeded) {
  throw new RateLimitError('Too many requests', 60000); // retryAfter
}

// Network error
if (connectionFailed) {
  throw new NetworkError('Failed to connect to server');
}
```

### Categorizing Errors
```typescript
import { categorizeError, isRetryable, ErrorCategory } from '../common/index.js';

try {
  await apiCall();
} catch (error) {
  const category = categorizeError(error);

  switch (category) {
    case ErrorCategory.TRANSIENT:
      console.log('Temporary failure - will retry');
      break;
    case ErrorCategory.PERMANENT:
      console.log('Permanent failure - will not retry');
      break;
    case ErrorCategory.THROTTLED:
      console.log('Rate limited - will wait and retry');
      break;
    default:
      console.log('Unknown error type');
  }

  // Or simply check if retryable
  if (isRetryable(error)) {
    // Retry the operation
  }
}
```

### Creating Error Responses
```typescript
import { createErrorResponse, createSuccessResponse } from '../common/index.js';

// Success response
const success = createSuccessResponse(data, correlationId);

// Error response
const error = createErrorResponse(error, correlationId);
// Returns: { error, message, code, statusCode, correlationId, retryable, details }
```

## Retry Logic Quick Reference

### Simple Retry with Backoff
```typescript
import { retryWithBackoff, generateCorrelationId } from '../common/index.js';

const result = await retryWithBackoff(
  async () => await fetchData(),
  {
    maxAttempts: 3,
    baseDelayMs: 1000,
    maxDelayMs: 30000,
    backoffMultiplier: 2,
    jitter: true,
    correlationId: generateCorrelationId(),
    operation: 'Fetch Data',
    onRetry: (attempt, error) => {
      console.log(`Retry attempt ${attempt}:`, error.message);
    }
  }
);
```

### Circuit Breaker
```typescript
import { CircuitBreaker } from '../common/index.js';

const circuitBreaker = new CircuitBreaker(
  {
    failureThreshold: 5,      // Open after 5 failures
    successThreshold: 2,      // Close after 2 successes
    timeoutMs: 60000,         // Try reset after 60 seconds
    onOpen: () => console.warn('Circuit opened'),
    onClose: () => console.log('Circuit closed')
  },
  'MyServiceCircuitBreaker'
);

const result = await circuitBreaker.execute(
  async () => await apiCall(),
  'API Call'
);

// Check circuit breaker state
const state = circuitBreaker.getState(); // 'open' | 'closed' | 'half_open'

// Get statistics
const stats = circuitBreaker.getStats();
// Returns: { state, failureCount, successCount, openedAt }
```

### Combined Resilience Pattern
```typescript
import { executeWithResilience } from '../common/index.js';

const result = await executeWithResilience(
  async () => await apiCall(),
  circuitBreaker,
  {
    maxAttempts: 3,
    baseDelayMs: 1000,
    operation: 'Resilient API Call'
  }
);
```

## Caching Quick Reference

### Basic Cache Usage
```typescript
import { InMemoryCache } from '../common/index.js';

const cache = new InMemoryCache<UserData>({
  maxSize: 1000,           // Maximum 1000 entries
  defaultTtl: 300000,      // 5 minute TTL
  cleanupInterval: 60000   // Cleanup every minute
});

// Set value
cache.set('user:123', userData);

// Get value
const data = cache.get('user:123');

// Check if key exists
if (cache.has('user:123')) {
  // Key exists and is not expired
}

// Delete key
cache.delete('user:123');

// Clear all
cache.clear();

// Get statistics
const stats = cache.getStats();
// Returns: { size, hits, misses, hitRate, evictions }
console.log(`Hit rate: ${(stats.hitRate * 100).toFixed(2)}%`);
```

### Cache Key Builder
```typescript
import { CacheKeyBuilder } from '../common/index.js';

const keyBuilder = new CacheKeyBuilder();

// Simple key
const key1 = keyBuilder
  .withPrefix('api')
  .withComponent('users')
  .build(); // "api:users"

// Key with parameters
const key2 = keyBuilder
  .reset()
  .withPrefix('api')
  .withComponent('users')
  .withParams({ page: 1, limit: 10, sort: 'name' })
  .build(); // "api:users:limit=10&page=1&sort=name"
```

### Multi-Tier Cache
```typescript
import { MultiTierCache, InMemoryCache } from '../common/index.js';

const l1Cache = new InMemoryCache({ maxSize: 100, defaultTtl: 60000 });
const l2Cache = new InMemoryCache({ maxSize: 1000, defaultTtl: 3600000 });

const multiCache = new MultiTierCache(l1Cache, l2Cache);

// Stores in both L1 and L2
multiCache.set('key', value, 300000);

// Checks L1 first, then L2, promotes to L1 on hit
const data = multiCache.get('key');

// Get combined statistics
const stats = multiCache.getStats();
// Returns: { l1, l2, combined }
console.log(`Combined hit rate: ${(stats.combined.hitRate * 100).toFixed(2)}%`);
```

### Global Caches
```typescript
import { globalCaches } from '../common/index.js';

// HTTP responses
globalCaches.http.set('https://api.example.com/data', responseBody);
const cached = globalCaches.http.get('https://api.example.com/data');

// Database queries
globalCaches.database.set('SELECT * FROM users', queryResult);

// API responses
globalCaches.api.set('endpoint:key', responseData);

// User data
globalCaches.userData.set('user:123', userProfile);
```

## Connection Pool Quick Reference

### PostgreSQL Pool
```typescript
import { globalPoolRegistry } from '../common/index.js';

// Get or create pool
const pool = globalPoolRegistry.getPostgresPool(connectionString);

// Initialize (optional, happens automatically)
await pool.initialize();

// Execute query
const result = await pool.query('SELECT * FROM users WHERE id = $1', [userId]);

// Get connection
const connection = await pool.getConnection();
try {
  await connection.query('BEGIN');
  // ... multiple queries
  await connection.query('COMMIT');
} finally {
  connection.release();
}

// Get statistics
const stats = pool.getStats();
// Returns: { total, active, idle, waiting, max, min }
console.log(`Active: ${stats.active}/${stats.max}`);

// Close pool
await pool.close();
```

### HTTP Pool
```typescript
import { globalHttpPool } from '../common/index.js';

// Get pool for origin
const pool = globalHttpPool.getPool('https://api.example.com');

// Acquire connection (managed automatically)
const connection = await pool.acquire();

// Release connection when done
pool.release(connection);

// Close all pools
await pool.close();
```

### Pool Registry
```typescript
import { globalPoolRegistry } from '../common/index.js';

// Close all pools (Postgres + HTTP)
await globalPoolRegistry.closeAll();

// Get all statistics
const allStats = globalPoolRegistry.getAllStats();
// Returns: { postgres: Map, http: Map }
```

## Constants Quick Reference

### Timeouts
```typescript
import { TIMEOUT, getTimeout } from '../common/index.js';

// Direct usage
const timeout = TIMEOUT.HTTP_REQUEST; // 30000ms

// With validation
const safeTimeout = getTimeout(userTimeout, TIMEOUT.HTTP_REQUEST);
```

### Pagination
```typescript
import { PAGINATION, validateLimit } from '../common/index.js';

// Direct usage
const defaultLimit = PAGINATION.DEFAULT_LIMIT; // 50
const maxLimit = PAGINATION.MAX_LIMIT; // 1000

// Validate user input
const safeLimit = validateLimit(userLimit); // Clamps to MIN-MAX range
```

### Cache TTL
```typescript
import { CACHE_TTL } from '../common/index.js';

const ttl = CACHE_TTL.SHORT; // 5 minutes
const ttl = CACHE_TTL.MEDIUM; // 15 minutes
const ttl = CACHE_TTL.LONG; // 1 hour
const ttl = CACHE_TTL.VERY_LONG; // 24 hours
```

### Retry Configuration
```typescript
import { RETRY } from '../common/index.js';

const maxAttempts = RETRY.MAX_ATTEMPTS; // 3
const baseDelay = RETRY.BASE_DELAY_MS; // 1000ms
const maxDelay = RETRY.MAX_DELAY_MS; // 30000ms
```

### Connection Pool Configuration
```typescript
import { CONNECTION_POOL } from '../common/index.js';

const poolConfig = {
  min: CONNECTION_POOL.MIN_SIZE, // 2
  max: CONNECTION_POOL.MAX_SIZE, // 10
  acquireTimeoutMillis: CONNECTION_POOL.ACQUIRE_TIMEOUT_MS, // 10000ms
  idleTimeoutMillis: CONNECTION_POOL.IDLE_TIMEOUT_MS, // 30000ms
};
```

## Type Utilities Quick Reference

### Result Type
```typescript
import { ok, err, unwrap, isOk, isErr } from '../common/index.js';

// Create results
const success = ok(data); // { success: true, data }
const failure = err(error); // { success: false, error }

// Check results
if (isOk(result)) {
  console.log(result.data); // TypeScript knows this is safe
}

// Unwrap (throws if error)
const data = unwrap(result);

// Pattern matching
if (result.success) {
  console.log(result.data);
} else {
  console.error(result.error);
}
```

### Type Guards
```typescript
import { isResult, isPlainObject, isIsoTimestamp } from '../common/index.js';

// Check if Result type
if (isResult(value)) {
  // TypeScript knows this is a Result
}

// Check if plain object
if (isPlainObject(value)) {
  // Safe to access as object
}

// Check if ISO timestamp
if (isIsoTimestamp(value)) {
  // Valid ISO 8601 format
}
```

### Deep Clone & Merge
```typescript
import { deepClone, deepMerge } from '../common/index.js';

// Clone object
const copy = deepClone(original);

// Merge objects
const merged = deepMerge(
  { a: 1, b: { c: 2 } },
  { b: { d: 3 }, e: 4 }
);
// Result: { a: 1, b: { c: 2, d: 3 }, e: 4 }
```

## Best Practices

### 1. Always Validate Input
```typescript
// ✅ Good
validateEmail(userEmail);

// ❌ Bad
if (!userEmail.includes('@')) {
  throw new Error('Invalid email');
}
```

### 2. Use Custom Error Types
```typescript
// ✅ Good
throw new ValidationError('Email is required', 'email');

// ❌ Bad
throw new Error('Email is required');
```

### 3. Leverage Retry Logic
```typescript
// ✅ Good
return await retryWithBackoff(async () => await fetchData());

// ❌ Bad
let attempts = 0;
while (attempts < 3) {
  try {
    return await fetchData();
  } catch {
    attempts++;
  }
}
```

### 4. Cache When Appropriate
```typescript
// ✅ Good
const cached = cache.get(key);
if (cached) return cached;
const result = await fetchData();
cache.set(key, result, ttl);
return result;

// ❌ Bad
return await fetchData(); // No caching
```

### 5. Use Connection Pools
```typescript
// ✅ Good
const pool = globalPoolRegistry.getPostgresPool(connectionString);
const result = await pool.query(sql, params);

// ❌ Bad
const client = new Client(connectionString);
await client.connect();
const result = await client.query(sql, params);
await client.end();
```

## Troubleshooting

### Cache Not Working
```typescript
// Check cache statistics
const stats = cache.getStats();
console.log(`Hit rate: ${(stats.hitRate * 100).toFixed(2)}%`);

// Common issues:
// 1. TTL too short → Increase defaultTtl
// 2. Cache too small → Increase maxSize
// 3. Keys not matching → Check key generation
```

### Circuit Breaker Always Open
```typescript
// Check circuit breaker state
const state = circuitBreaker.getState();
const stats = circuitBreaker.getStats();

// Common issues:
// 1. Failures too frequent → Increase failureThreshold
// 2. Recovery too slow → Decrease timeoutMs
// 3. Service actually down → Fix underlying issue

// Manual reset
circuitBreaker.reset();
```

### Connection Pool Exhausted
```typescript
// Check pool stats
const stats = pool.getStats();
console.log(`Active: ${stats.active}, Waiting: ${stats.waiting}`);

// Common issues:
// 1. Connections not released → Ensure proper release()
// 2. Pool too small → Increase max pool size
// 3. Queries too slow → Optimize queries
```

## Additional Resources

- **Full Documentation**: See `P3_REFACTORING_COMPLETE.md`
- **Usage Examples**: See each module's JSDoc comments
- **Type Definitions**: See `types.ts` for all available interfaces
- **Constants**: See `constants.ts` for all configuration values

## Support

For issues or questions:
1. Check the JSDoc comments in each module
2. Review the examples in this guide
3. See the full documentation in `P3_REFACTORING_COMPLETE.md`
4. Check the test files for usage examples

---

**Last Updated**: 2025-01-18
**Version**: 1.0.0
**Status**: Production Ready
