# P3 Final Wave - Implementation Guide

**Quick Start Guide for Implementing P3 Tasks**

---

## Phase 1: Testing Infrastructure Setup (Day 1)

### Step 1: Install Coverage Dependencies (15 minutes)

```bash
# From BubbleLab root
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLab

# Add coverage dependency to workspace
pnpm add -D -w @vitest/coverage-v8

# Verify installation
pnpm list @vitest/coverage-v8
```

### Step 2: Configure Vitest Coverage (30 minutes)

Create or update vitest config files:

**For bubble-runtime:**
```typescript
// packages/bubble-runtime/vitest.config.ts
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'dist/',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/types/',
        '**/fixtures/',
        '**/dist/',
      ],
      // Coverage thresholds
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

**For bubble-core:**
```typescript
// packages/bubble-core/vitest.config.ts
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['./src/tests/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'dist/',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/types/',
        '**/fixtures/',
        '**/dist/',
      ],
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### Step 3: Run Baseline Coverage Report (15 minutes)

```bash
# Run coverage for runtime package
cd packages/bubble-runtime
pnpm test:coverage

# Run coverage for core package
cd ../bubble-core
pnpm test:coverage

# Run all coverage
cd ../../
pnpm test:coverage
```

**Expected Output:**
```
% Coverage report from v8
Lines          │ 45.2% (1234/5432)
Branches       │ 38.7% (567/2345)
Functions      │ 52.1% (890/3456)
Statements     │ 45.2% (1234/5432)
```

**Save this baseline!** This is your starting point.

---

## Phase 2: Common Utilities Tests (Day 1-2)

### Task 2.1: Create Validator Tests (1 hour)

**File:** `packages/bubble-runtime/src/utils/validators.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import {
  validateEmail,
  validateUrl,
  validateTimestamp,
  validatePhoneNumber,
  validateUUID
} from './validators';

describe('Validators', () => {
  describe('validateEmail', () => {
    it('should accept valid email addresses', () => {
      const validEmails = [
        'test@example.com',
        'user.name@example.com',
        'user+tag@example.co.uk',
        'user_name@test-domain.com',
      ];

      validEmails.forEach(email => {
        expect(validateEmail(email)).toBe(true);
      });
    });

    it('should reject invalid email addresses', () => {
      const invalidEmails = [
        'invalid',
        '@example.com',
        'user@',
        'user @example.com',
        'user@example..com',
        '',
        null,
        undefined,
      ];

      invalidEmails.forEach(email => {
        expect(validateEmail(email as any)).toBe(false);
      });
    });

    it('should handle edge cases', () => {
      expect(validateEmail('a@b.c')).toBe(true); // Minimal valid
      expect(validateEmail('  test@example.com  ')).toBe(false); // Spaces
      expect(validateEmail('TEST@EXAMPLE.COM')).toBe(true); // Uppercase
    });
  });

  describe('validateUrl', () => {
    it('should accept valid URLs', () => {
      const validUrls = [
        'http://example.com',
        'https://example.com',
        'https://example.com/path',
        'https://example.com/path?query=value',
        'https://example.com:8080/path',
        'ftp://example.com',
      ];

      validUrls.forEach(url => {
        expect(validateUrl(url)).toBe(true);
      });
    });

    it('should reject invalid URLs', () => {
      const invalidUrls = [
        'example.com',
        'http:/example.com',
        '://example.com',
        '',
        null,
        undefined,
      ];

      invalidUrls.forEach(url => {
        expect(validateUrl(url as any)).toBe(false);
      });
    });

    it('should support protocol filtering', () => {
      const url = 'https://example.com';
      expect(validateUrl(url, ['https'])).toBe(true);
      expect(validateUrl(url, ['http'])).toBe(false);
    });
  });

  describe('validateTimestamp', () => {
    it('should accept valid ISO-8601 timestamps', () => {
      const validTimestamps = [
        '2024-01-01T00:00:00Z',
        '2024-01-01T00:00:00.000Z',
        '2024-01-01T00:00:00+00:00',
        '2024-01-01T00:00:00.000+00:00',
      ];

      validTimestamps.forEach(timestamp => {
        expect(validateTimestamp(timestamp)).toBe(true);
      });
    });

    it('should reject invalid timestamps', () => {
      const invalidTimestamps = [
        '2024-01-01',
        '00:00:00',
        'invalid',
        '',
        null,
        undefined,
      ];

      invalidTimestamps.forEach(timestamp => {
        expect(validateTimestamp(timestamp as any)).toBe(false);
      });
    });
  });

  describe('validatePhoneNumber', () => {
    it('should accept valid phone numbers', () => {
      const validPhones = [
        '+1234567890',
        '+1 234 567 8900',
        '+44 20 7946 0958',
      ];

      validPhones.forEach(phone => {
        expect(validatePhoneNumber(phone)).toBe(true);
      });
    });

    it('should reject invalid phone numbers', () => {
      const invalidPhones = [
        '1234567890',
        '+123',
        '',
        null,
        undefined,
      ];

      invalidPhones.forEach(phone => {
        expect(validatePhoneNumber(phone as any)).toBe(false);
      });
    });
  });

  describe('validateUUID', () => {
    it('should accept valid UUIDs', () => {
      const validUUIDs = [
        '550e8400-e29b-41d4-a716-446655440000',
        '6ba7b810-9dad-11d1-80b4-00c04fd430c8',
        '6ba7b811-9dad-11d1-80b4-00c04fd430c8',
      ];

      validUUIDs.forEach(uuid => {
        expect(validateUUID(uuid)).toBe(true);
      });
    });

    it('should reject invalid UUIDs', () => {
      const invalidUUIDs = [
        '550e8400-e29b-41d4-a716', // Too short
        '550e8400-e29b-41d4-a716-44665544000g', // Invalid char
        '',
        null,
        undefined,
      ];

      invalidUUIDs.forEach(uuid => {
        expect(validateUUID(uuid as any)).toBe(false);
      });
    });
  });
});
```

### Task 2.2: Create Error Handler Tests (1 hour)

**File:** `packages/bubble-runtime/src/utils/error-handlers.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import {
  categorizeError,
  isRetryableError,
  isTransientError,
  parseErrorMessage,
  createStructuredError
} from './error-handlers';

describe('Error Handlers', () => {
  describe('categorizeError', () => {
    it('should categorize network errors as transient', () => {
      const networkError = new Error('ECONNREFUSED');
      expect(categorizeError(networkError)).toBe('transient');
    });

    it('should categorize timeout errors as transient', () => {
      const timeoutError = new Error('ETIMEDOUT');
      expect(categorizeError(timeoutError)).toBe('transient');
    });

    it('should categorize validation errors as permanent', () => {
      const validationError = new Error('Validation failed');
      expect(categorizeError(validationError)).toBe('permanent');
    });

    it('should categorize authentication errors as permanent', () => {
      const authError = new Error('Unauthorized');
      expect(categorizeError(authError)).toBe('permanent');
    });

    it('should categorize unknown errors as unknown', () => {
      const unknownError = new Error('Unknown error');
      expect(categorizeError(unknownError)).toBe('unknown');
    });
  });

  describe('isRetryableError', () => {
    it('should return true for retryable HTTP status codes', () => {
      expect(isRetryableError(429)).toBe(true); // Too Many Requests
      expect(isRetryableError(500)).toBe(true); // Internal Server Error
      expect(isRetryableError(502)).toBe(true); // Bad Gateway
      expect(isRetryableError(503)).toBe(true); // Service Unavailable
      expect(isRetryableError(504)).toBe(true); // Gateway Timeout
    });

    it('should return false for non-retryable HTTP status codes', () => {
      expect(isRetryableError(400)).toBe(false); // Bad Request
      expect(isRetryableError(401)).toBe(false); // Unauthorized
      expect(isRetryableError(403)).toBe(false); // Forbidden
      expect(isRetryableError(404)).toBe(false); // Not Found
    });

    it('should detect retryable network errors', () => {
      const retryableErrors = [
        new Error('ECONNRESET'),
        new Error('ECONNREFUSED'),
        new Error('ETIMEDOUT'),
        new Error('ENOTFOUND'),
      ];

      retryableErrors.forEach(error => {
        expect(isRetryableError(error)).toBe(true);
      });
    });

    it('should return false for non-retryable errors', () => {
      const nonRetryableErrors = [
        new Error('ValidationError'),
        new Error('AuthenticationError'),
        new Error('AuthorizationError'),
      ];

      nonRetryableErrors.forEach(error => {
        expect(isRetryableError(error)).toBe(false);
      });
    });
  });

  describe('isTransientError', () => {
    it('should return true for transient errors', () => {
      const transientErrors = [
        new Error('ECONNRESET'),
        new Error('ETIMEDOUT'),
        new Error('Temporarily unavailable'),
      ];

      transientErrors.forEach(error => {
        expect(isTransientError(error)).toBe(true);
      });
    });

    it('should return false for permanent errors', () => {
      const permanentErrors = [
        new Error('Validation failed'),
        new Error('Unauthorized'),
        new Error('Not found'),
      ];

      permanentErrors.forEach(error => {
        expect(isTransientError(error)).toBe(false);
      });
    });
  });

  describe('parseErrorMessage', () => {
    it('should extract error message from Error object', () => {
      const error = new Error('Test error message');
      expect(parseErrorMessage(error)).toBe('Test error message');
    });

    it('should handle error objects with custom message property', () => {
      const error = { message: 'Custom error message' };
      expect(parseErrorMessage(error)).toBe('Custom error message');
    });

    it('should handle string errors', () => {
      expect(parseErrorMessage('String error')).toBe('String error');
    });

    it('should handle null/undefined gracefully', () => {
      expect(parseErrorMessage(null)).toBe('Unknown error');
      expect(parseErrorMessage(undefined)).toBe('Unknown error');
    });

    it('should extract message from HTTP response errors', () => {
      const error = {
        response: {
          data: { message: 'HTTP error message' }
        }
      };
      expect(parseErrorMessage(error)).toBe('HTTP error message');
    });
  });

  describe('createStructuredError', () => {
    it('should create structured error with all fields', () => {
      const error = createStructuredError({
        message: 'Test error',
        code: 'TEST_ERROR',
        category: 'transient',
        retryable: true,
        context: { userId: '123' }
      });

      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_ERROR');
      expect(error.category).toBe('transient');
      expect(error.retryable).toBe(true);
      expect(error.context).toEqual({ userId: '123' });
      expect(error.timestamp).toBeDefined();
    });

    it('should use defaults for optional fields', () => {
      const error = createStructuredError({
        message: 'Test error'
      });

      expect(error.message).toBe('Test error');
      expect(error.code).toBe('UNKNOWN_ERROR');
      expect(error.category).toBe('unknown');
      expect(error.retryable).toBe(false);
      expect(error.context).toEqual({});
    });
  });
});
```

### Task 2.3: Create Retry Logic Tests (1.5 hours)

**File:** `packages/bubble-runtime/src/utils/retry-logic.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  calculateBackoff,
  shouldRetry,
  CircuitBreaker,
  RetryPolicy
} from './retry-logic';

describe('Retry Logic', () => {
  describe('calculateBackoff', () => {
    it('should calculate exponential backoff', () => {
      expect(calculateBackoff(1, 1000, 2)).toBe(2000);   // 2^1 * 1000
      expect(calculateBackoff(2, 1000, 2)).toBe(4000);   // 2^2 * 1000
      expect(calculateBackoff(3, 1000, 2)).toBe(8000);   // 2^3 * 1000
    });

    it('should add jitter to backoff', () => {
      const backoff = calculateBackoff(1, 1000, 2, 0.1); // 10% jitter
      expect(backoff).toBeGreaterThanOrEqual(1800);      // 2000 - 200
      expect(backoff).toBeLessThanOrEqual(2200);         // 2000 + 200
    });

    it('should respect max backoff', () => {
      const backoff = calculateBackoff(10, 1000, 2, 0, 5000);
      expect(backoff).toBe(5000);
    });

    it('should handle zero retries', () => {
      expect(calculateBackoff(0, 1000, 2)).toBe(0);
    });
  });

  describe('shouldRetry', () => {
    it('should return true if retry count below max', () => {
      const policy = new RetryPolicy({ maxRetries: 3 });
      expect(shouldRetry(1, new Error('ECONNRESET'), policy)).toBe(true);
      expect(shouldRetry(2, new Error('ECONNRESET'), policy)).toBe(true);
    });

    it('should return false if retry count exceeds max', () => {
      const policy = new RetryPolicy({ maxRetries: 3 });
      expect(shouldRetry(3, new Error('ECONNRESET'), policy)).toBe(false);
    });

    it('should check if error is retryable', () => {
      const policy = new RetryPolicy({ maxRetries: 3 });

      expect(shouldRetry(1, new Error('ECONNRESET'), policy)).toBe(true);
      expect(shouldRetry(1, new Error('ValidationError'), policy)).toBe(false);
    });

    it('should respect retryable error filter', () => {
      const policy = new RetryPolicy({
        maxRetries: 3,
        retryableIf: (error) => error.message.includes('retryable')
      });

      expect(shouldRetry(1, new Error('retryable error'), policy)).toBe(true);
      expect(shouldRetry(1, new Error('non-retryable'), policy)).toBe(false);
    });
  });

  describe('CircuitBreaker', () => {
    let breaker: CircuitBreaker;

    beforeEach(() => {
      breaker = new CircuitBreaker({
        failureThreshold: 3,
        resetTimeout: 60000,
      });
    });

    it('should start in closed state', () => {
      expect(breaker.state).toBe('closed');
    });

    it('should open after failure threshold', () => {
      // Record failures
      breaker.recordFailure();
      breaker.recordFailure();
      breaker.recordFailure();

      expect(breaker.state).toBe('open');
    });

    it('should not allow calls when open', () => {
      // Trigger open state
      breaker.recordFailure();
      breaker.recordFailure();
      breaker.recordFailure();

      expect(breaker.allowCall()).toBe(false);
    });

    it('should transition to half-open after reset timeout', async () => {
      // Open the circuit
      breaker.recordFailure();
      breaker.recordFailure();
      breaker.recordFailure();

      // Fast forward time
      vi.advanceTimersByTime(61000);

      expect(breaker.state).toBe('half-open');
    });

    it('should close on successful call in half-open state', () => {
      // Open then transition to half-open
      breaker.recordFailure();
      breaker.recordFailure();
      breaker.recordFailure();
      vi.advanceTimersByTime(61000);

      // Record success
      breaker.recordSuccess();

      expect(breaker.state).toBe('closed');
    });

    it('should reopen on failure in half-open state', () => {
      // Open then transition to half-open
      breaker.recordFailure();
      breaker.recordFailure();
      breaker.recordFailure();
      vi.advanceTimersByTime(61000);

      // Record failure
      breaker.recordFailure();

      expect(breaker.state).toBe('open');
    });

    it('should track success rate', () => {
      breaker.recordSuccess();
      breaker.recordSuccess();
      breaker.recordFailure();

      const stats = breaker.getStats();
      expect(stats.successRate).toBeCloseTo(0.667, 2);
    });

    it('should provide stats', () => {
      breaker.recordSuccess();
      breaker.recordFailure();

      const stats = breaker.getStats();
      expect(stats.totalCalls).toBe(2);
      expect(stats.successfulCalls).toBe(1);
      expect(stats.failedCalls).toBe(1);
    });

    it('should reset stats', () => {
      breaker.recordSuccess();
      breaker.recordFailure();

      breaker.reset();

      const stats = breaker.getStats();
      expect(stats.totalCalls).toBe(0);
      expect(stats.successRate).toBe(0);
    });
  });
});
```

### Task 2.4: Create Cache Tests (1 hour)

**File:** `packages/bubble-runtime/src/utils/cache.test.ts`

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Cache } from './cache';

describe('Cache', () => {
  let cache: Cache;

  beforeEach(() => {
    cache = new Cache({ maxSize: 100, ttl: 60000 });
  });

  describe('Basic Operations', () => {
    it('should set and get values', () => {
      cache.set('key1', 'value1');
      expect(cache.get('key1')).toBe('value1');
    });

    it('should return undefined for non-existent keys', () => {
      expect(cache.get('nonexistent')).toBeUndefined();
    });

    it('should delete values', () => {
      cache.set('key1', 'value1');
      cache.delete('key1');
      expect(cache.get('key1')).toBeUndefined();
    });

    it('should clear all values', () => {
      cache.set('key1', 'value1');
      cache.set('key2', 'value2');
      cache.clear();

      expect(cache.get('key1')).toBeUndefined();
      expect(cache.get('key2')).toBeUndefined();
    });

    it('should check if key exists', () => {
      cache.set('key1', 'value1');
      expect(cache.has('key1')).toBe(true);
      expect(cache.has('key2')).toBe(false);
    });
  });

  describe('TTL (Time To Live)', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    it('should expire entries after TTL', () => {
      cache.set('key1', 'value1', 1000); // 1 second TTL

      expect(cache.get('key1')).toBe('value1');

      vi.advanceTimersByTime(1100);
      expect(cache.get('key1')).toBeUndefined();
    });

    it('should use default TTL if not specified', () => {
      cache.set('key1', 'value1'); // Uses default 60s TTL

      vi.advanceTimersByTime(61000);
      expect(cache.get('key1')).toBeUndefined();
    });

    it('should refresh TTL on get', () => {
      cache.set('key1', 'value1', 1000);

      vi.advanceTimersByTime(500);
      expect(cache.get('key1')).toBe('value1'); // Refreshes TTL

      vi.advanceTimersByTime(500);
      expect(cache.get('key1')).toBe('value1'); // Still exists

      vi.advanceTimersByTime(1100);
      expect(cache.get('key1')).toBeUndefined();
    });
  });

  describe('LRU Eviction', () => {
    it('should evict least recently used when at capacity', () => {
      const smallCache = new Cache({ maxSize: 3, ttl: 60000 });

      smallCache.set('key1', 'value1');
      smallCache.set('key2', 'value2');
      smallCache.set('key3', 'value3');
      smallCache.set('key4', 'value4'); // Should evict key1

      expect(smallCache.get('key1')).toBeUndefined();
      expect(smallCache.get('key2')).toBe('value2');
      expect(smallCache.get('key3')).toBe('value3');
      expect(smallCache.get('key4')).toBe('value4');
    });

    it('should update recency on get', () => {
      const smallCache = new Cache({ maxSize: 3, ttl: 60000 });

      smallCache.set('key1', 'value1');
      smallCache.set('key2', 'value2');
      smallCache.set('key3', 'value3');
      smallCache.get('key1'); // Update recency
      smallCache.set('key4', 'value4'); // Should evict key2

      expect(smallCache.get('key1')).toBe('value1');
      expect(smallCache.get('key2')).toBeUndefined();
      expect(smallCache.get('key3')).toBe('value3');
      expect(smallCache.get('key4')).toBe('value4');
    });
  });

  describe('Statistics', () => {
    it('should track hit rate', () => {
      cache.set('key1', 'value1');

      cache.get('key1'); // Hit
      cache.get('key2'); // Miss
      cache.get('key1'); // Hit
      cache.get('key3'); // Miss

      const stats = cache.getStats();
      expect(stats.hits).toBe(2);
      expect(stats.misses).toBe(2);
      expect(stats.hitRate).toBe(0.5);
    });

    it('should track size', () => {
      expect(cache.size).toBe(0);

      cache.set('key1', 'value1');
      cache.set('key2', 'value2');

      expect(cache.size).toBe(2);

      cache.delete('key1');

      expect(cache.size).toBe(1);
    });

    it('should provide detailed stats', () => {
      cache.set('key1', 'value1');
      cache.get('key1');

      const stats = cache.getStats();
      expect(stats).toEqual({
        hits: 1,
        misses: 0,
        totalRequests: 1,
        hitRate: 1,
        size: 1,
        maxSize: 100,
      });
    });
  });

  describe('Concurrent Access', () => {
    it('should handle concurrent sets', async () => {
      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(Promise.resolve().then(() => cache.set(`key${i}`, `value${i}`)));
      }

      await Promise.all(promises);

      expect(cache.size).toBe(100);
    });

    it('should handle concurrent gets', async () => {
      cache.set('key1', 'value1');

      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(Promise.resolve().then(() => cache.get('key1')));
      }

      const results = await Promise.all(promises);
      expect(results.every(r => r === 'value1')).toBe(true);
    });
  });
});
```

### Task 2.5: Create Connection Pool Tests (1.5 hours)

**File:** `packages/bubble-runtime/src/utils/connection-pool.test.ts`

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ConnectionPool } from './connection-pool';

describe('ConnectionPool', () => {
  let pool: ConnectionPool;
  let mockConnectionFactory: any;

  beforeEach(() => {
    mockConnectionFactory = vi.fn(() => ({
      id: Math.random(),
      active: true,
      close: vi.fn(),
    }));

    pool = new ConnectionPool({
      factory: mockConnectionFactory,
      maxConnections: 5,
      idleTimeout: 30000,
    });
  });

  describe('Connection Acquisition', () => {
    it('should create new connection when pool is empty', async () => {
      const conn = await pool.acquire();
      expect(conn).toBeDefined();
      expect(mockConnectionFactory).toHaveBeenCalledTimes(1);
    });

    it('should reuse idle connections', async () => {
      const conn1 = await pool.acquire();
      pool.release(conn1);
      const conn2 = await pool.acquire();

      expect(conn1).toBe(conn2);
      expect(mockConnectionFactory).toHaveBeenCalledTimes(1);
    });

    it('should create multiple connections up to max', async () => {
      const connections = [];
      for (let i = 0; i < 5; i++) {
        connections.push(await pool.acquire());
      }

      expect(mockConnectionFactory).toHaveBeenCalledTimes(5);
      expect(pool.getStats().totalConnections).toBe(5);
    });

    it('should wait when pool is exhausted', async () => {
      // Acquire all connections
      const connections = [];
      for (let i = 0; i < 5; i++) {
        connections.push(await pool.acquire());
      }

      // Try to acquire one more (should wait)
      const acquirePromise = pool.acquire();

      // Release one connection
      pool.release(connections[0]);

      // Should now acquire
      const conn = await acquirePromise;
      expect(conn).toBeDefined();
    });

    it('should timeout if connection not available', async () => {
      const timeoutPool = new ConnectionPool({
        factory: mockConnectionFactory,
        maxConnections: 1,
        acquireTimeout: 100,
      });

      await timeoutPool.acquire();

      await expect(timeoutPool.acquire()).rejects.toThrow('Acquire timeout');
    });
  });

  describe('Connection Release', () => {
    it('should return connection to pool', async () => {
      const conn = await pool.acquire();
      pool.release(conn);

      expect(pool.getStats().idleConnections).toBe(1);
    });

    it('should close connection if pool is full', async () => {
      const connections = [];
      for (let i = 0; i < 5; i++) {
        connections.push(await pool.acquire());
      }

      const conn = await pool.acquire();
      pool.release(conn);

      expect(conn.close).toHaveBeenCalled();
    });

    it('should not close idle connections', async () => {
      const conn = await pool.acquire();
      pool.release(conn);

      expect(conn.close).not.toHaveBeenCalled();
    });
  });

  describe('Connection Health', () => {
    it('should check connection health', async () => {
      const conn = await pool.acquire();
      expect(pool.isHealthy(conn)).toBe(true);
    });

    it('should destroy unhealthy connections', async () => {
      const unhealthyConn = await pool.acquire();
      unhealthyConn.active = false;

      pool.release(unhealthyConn);

      expect(unhealthyConn.close).toHaveBeenCalled();
      expect(pool.getStats().idleConnections).toBe(0);
    });

    it('should recreate unhealthy connections on acquire', async () => {
      const unhealthyConn = await pool.acquire();
      unhealthyConn.active = false;
      pool.release(unhealthyConn);

      const newConn = await pool.acquire();
      expect(newConn.active).toBe(true);
    });
  });

  describe('Idle Timeout', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    it('should close idle connections after timeout', async () => {
      const idlePool = new ConnectionPool({
        factory: mockConnectionFactory,
        maxConnections: 5,
        idleTimeout: 1000,
      });

      const conn = await idlePool.acquire();
      idlePool.release(conn);

      vi.advanceTimersByTime(1100);

      expect(conn.close).toHaveBeenCalled();
      expect(idlePool.getStats().idleConnections).toBe(0);
    });

    it('should not close active connections', async () => {
      const idlePool = new ConnectionPool({
        factory: mockConnectionFactory,
        maxConnections: 5,
        idleTimeout: 1000,
      });

      const conn = await idlePool.acquire();

      vi.advanceTimersByTime(1100);

      expect(conn.close).not.toHaveBeenCalled();
    });
  });

  describe('Statistics', () => {
    it('should track pool statistics', async () => {
      const conn1 = await pool.acquire();
      const conn2 = await pool.acquire();
      pool.release(conn1);

      const stats = pool.getStats();
      expect(stats.totalConnections).toBe(2);
      expect(stats.activeConnections).toBe(1);
      expect(stats.idleConnections).toBe(1);
      expect(stats.waitingClients).toBe(0);
    });

    it('should track waiting clients', async () => {
      // Acquire all connections
      const connections = [];
      for (let i = 0; i < 5; i++) {
        connections.push(await pool.acquire());
      }

      // Try to acquire more (will wait)
      pool.acquire();
      pool.acquire();

      const stats = pool.getStats();
      expect(stats.waitingClients).toBe(2);
    });
  });

  describe('Pool Cleanup', () => {
    it('should close all connections on drain', async () => {
      const connections = [];
      for (let i = 0; i < 5; i++) {
        connections.push(await pool.acquire());
      }
      connections.forEach(conn => pool.release(conn));

      await pool.drain();

      connections.forEach(conn => {
        expect(conn.close).toHaveBeenCalled();
      });
      expect(pool.getStats().totalConnections).toBe(0);
    });

    it('should reject new acquisitions after drain', async () => {
      await pool.drain();

      await expect(pool.acquire()).rejects.toThrow('Pool is drained');
    });
  });
});
```

---

## Phase 3: Run Coverage Report and Verify (Day 2)

### Step 1: Run Coverage Report

```bash
# From BubbleLab root
pnpm test:coverage
```

### Step 2: Review Results

Check that you've achieved:
- Lines: 80%+
- Branches: 75%+
- Functions: 80%+
- Statements: 80%+

### Step 3: Generate HTML Report

```bash
# Open detailed HTML report
open packages/bubble-runtime/coverage/index.html
open packages/bubble-core/coverage/index.html
```

### Step 4: Create Coverage Summary

Create a summary document:

```markdown
# Test Coverage Summary

## Baseline (Before)
- Lines: XX%
- Branches: XX%
- Functions: XX%
- Statements: XX%

## Current (After Common Utilities)
- Lines: XX%
- Branches: XX%
- Functions: XX%
- Statements: XX%

## Improvements
- Lines: +XX%
- Branches: +XX%
- Functions: +XX%
- Statements: +XX%

## Next Steps
- Service bubble tests
- Tool bubble tests
- Integration tests
```

---

## Next Steps After Completing Phase 1-3

Once common utilities tests are complete and coverage is measured:

1. **Service Bubble Tests** (10 hours)
   - Prioritize: HTTP, AI Agent, Google Sheets
   - Follow test patterns from Phase 2
   - Mock external dependencies

2. **Architecture Documentation** (4 hours)
   - Create Mermaid diagrams
   - Document all subsystems
   - Update ARCHITECTURE.md

3. **Operational Runbooks** (4 hours)
   - Create 7 runbooks
   - Document procedures
   - Include troubleshooting

4. **Production Preparation** (10 hours)
   - Security checklist
   - Load testing
   - Monitoring setup

---

**Status Tracking:**
- [ ] Coverage infrastructure set up
- [ ] Common utilities tests written
- [ ] Baseline coverage measured
- [ ] Service bubble tests started
- [ ] Architecture documentation started
- [ ] Operational runbooks started
- [ ] Production preparation started

**Next Session Focus:**
1. Complete remaining utility tests
2. Start service bubble tests with HTTP bubble
3. Create first architecture diagram
