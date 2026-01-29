# Error Handling Quick Reference Guide

## 🚀 Quick Start

### Import the Utilities
```typescript
import {
  createErrorResponse,
  createSuccessResponse,
  generateCorrelationId,
  withTimeout,
  retryWithBackoff,
  CircuitBreaker,
  defaultCircuitBreakerConfig,
  categorizeError,
  sanitizeErrorMessage,
} from '../../utils/error-handler.js';
```

### Basic Error Handling Pattern
```typescript
protected async performAction(context?: BubbleContext): Promise<Result> {
  const correlationId = generateCorrelationId();
  const startTime = Date.now();

  try {
    // Your operation here
    const result = await this.doSomething();

    console.log(`[${correlationId}] Success in ${Date.now() - startTime}ms`);
    return { ...result, success: true, error: '' };
  } catch (error) {
    const errorResponse = createErrorResponse(error, correlationId);
    console.error(`[${correlationId}] Failed:`, error);

    return {
      ...this.getDefaultResult(),
      success: false,
      error: errorResponse.error.message,
    };
  }
}
```

## 📦 Key Components

### 1. Correlation IDs
```typescript
const correlationId = generateCorrelationId(); // UUID v4
console.log(`[${correlationId}] Operation started`);
```

### 2. Timeouts
```typescript
// Wrap any promise with a timeout
const result = await withTimeout(
  someAsyncOperation(),
  30000, // 30 seconds
  'OperationName'
);
```

### 3. Retry Logic
```typescript
// Retry with exponential backoff
const result = await retryWithBackoff(
  async () => {
    return await someFlakyOperation();
  },
  {
    maxAttempts: 3,        // Try 3 times total
    baseDelayMs: 1000,     // Start with 1s delay
    maxDelayMs: 16000,     // Max 16s delay
    correlationId,
    operation: 'MyOperation',
  }
);
```

### 4. Circuit Breaker
```typescript
// Initialize in constructor
this.circuitBreaker = new CircuitBreaker({
  failureThreshold: 5,      // Open after 5 failures
  successThreshold: 2,      // Close after 2 successes
  timeoutMs: 30000,         // Operation timeout
  monitoringPeriodMs: 60000, // Retry after 60s
});

// Use in performAction
const result = await this.circuitBreaker.execute(
  async () => await someOperation(),
  'OperationName'
);
```

### 5. Error Responses
```typescript
// Create error response
const errorResponse = createErrorResponse(
  error,
  correlationId,
  { additionalContext: 'value' } // optional details
);

// Create success response
const successResponse = createSuccessResponse(
  data,
  correlationId
);
```

## 🔄 Complete Pattern

### Full Implementation Example
```typescript
import {
  createErrorResponse,
  generateCorrelationId,
  withTimeout,
  retryWithBackoff,
  CircuitBreaker,
  defaultCircuitBreakerConfig,
} from '../../utils/error-handler.js';

export class MyBubble extends ServiceBubble<Params, Result> {
  private circuitBreaker: CircuitBreaker;

  constructor(params: ParamsInput, context?: BubbleContext) {
    super(params, context);

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker({
      ...defaultCircuitBreakerConfig,
      failureThreshold: 5,
      successThreshold: 2,
      timeoutMs: params.timeout || 30000,
      monitoringPeriodMs: 60000,
    });
  }

  protected async performAction(context?: BubbleContext): Promise<Result> {
    const correlationId = generateCorrelationId();
    const startTime = Date.now();

    try {
      // Step 1: Validate inputs
      this.validateInputs();

      // Step 2: Execute with circuit breaker → retry → timeout
      const result = await this.circuitBreaker.execute(
        async () => {
          return await retryWithBackoff(
            async () => {
              return await withTimeout(
                this.executeCoreOperation(),
                this.params.timeout || 30000,
                'MyOperation'
              );
            },
            {
              maxAttempts: 3,
              baseDelayMs: 1000,
              correlationId,
              operation: 'MyOperation',
            }
          );
        },
        'MyOperation'
      );

      const executionTime = Date.now() - startTime;
      console.log(`[${correlationId}] Success in ${executionTime}ms`);

      return {
        ...result,
        executionTime,
        success: true,
        error: '',
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const errorResponse = createErrorResponse(error, correlationId);

      console.error(
        `[${correlationId}] Failed after ${executionTime}ms:`,
        error
      );

      return {
        ...this.getDefaultResult(),
        executionTime,
        success: false,
        error: errorResponse.error.message,
      };
    }
  }

  private async executeCoreOperation(): Promise<PartialResult> {
    // Your core operation logic here
    return { data: 'result' };
  }

  private getDefaultResult(): Result {
    return {
      data: null,
      success: false,
      error: '',
    };
  }
}
```

## 🛡️ Error Categories

### Automatic Categorization
```typescript
const category = categorizeError(error, statusCode);

// Categories:
// ValidationError (400)      - Invalid input
// AuthenticationError (401)  - Auth failed
// AuthorizationError (403)   - Not permitted
// NotFoundError (404)        - Resource not found
// ConflictError (409)        - Resource conflict
// RateLimitError (429)       - Too many requests
// ServerError (500)          - Server error
// NetworkError (503)         - Network unavailable
// TimeoutError (504)         - Operation timed out
// UnknownError (500)         - Unknown error
```

### Retryable vs Non-Retryable
```typescript
// Automatically determined by error category
// Retryable: NetworkError, TimeoutError, ServerError (5xx), RateLimitError
// Non-retryable: ValidationError, AuthenticationError, AuthorizationError
```

## 🔒 Security

### Error Sanitization
```typescript
import { sanitizeErrorMessage } from '../../utils/error-handler.js';

const safeMessage = sanitizeErrorMessage(error.message);

// Removes:
// - Credentials (password, token, key, secret)
// - File paths (C:\..., /home/...)
// - IP addresses (192.168.1.1)
// - Stack traces in production
```

### Never Log
```typescript
// ❌ BAD
console.log('Error:', error);
console.log('Credentials:', this.params.credentials);

// ✅ GOOD
const safeError = createErrorResponse(error, correlationId);
console.error(`[${correlationId}] Operation failed:`, safeError.error.message);
```

## 📊 Monitoring

### Circuit Breaker Metrics
```typescript
const metrics = this.circuitBreaker.getMetrics();
// Returns:
// {
//   state: 'closed' | 'open' | 'half_open',
//   failureCount: 0,
//   successCount: 0,
//   lastFailureTime: timestamp,
//   nextAttemptTime: timestamp
// }
```

### Logging Pattern
```typescript
console.log(`[${correlationId}] Operation started`);
console.log(`[${correlationId}] Retry attempt ${attempt}/${maxAttempts}`);
console.log(`[${correlationId}] Success in ${duration}ms`);
console.error(`[${correlationId}] Failed:`, error);
```

## 🧪 Testing

### Test Timeouts
```typescript
test('should timeout after 30s', async () => {
  await expect(
    withTimeout(
      new Promise(resolve => setTimeout(resolve, 60000)),
      30000,
      'Test'
    )
  ).rejects.toThrow('timed out after 30000ms');
});
```

### Test Retry Logic
```typescript
test('should retry 3 times', async () => {
  let attempts = 0;
  await retryWithBackoff(
    async () => {
      attempts++;
      if (attempts < 3) throw new Error('Temporary failure');
      return 'success';
    },
    { maxAttempts: 3 }
  );
  expect(attempts).toBe(3);
});
```

### Test Circuit Breaker
```typescript
test('should open circuit after 5 failures', async () => {
  const cb = new CircuitBreaker({ failureThreshold: 5, ... });

  // Fail 5 times
  for (let i = 0; i < 5; i++) {
    try {
      await cb.execute(async () => throw new Error('Fail'), 'Test');
    } catch {}
  }

  // Circuit should be open
  expect(cb.getState()).toBe('open');

  // Should reject immediately
  await expect(
    cb.execute(async () => 'success', 'Test')
  ).rejects.toThrow('Circuit breaker is OPEN');
});
```

## 📋 Checklist

### Before Committing
- [ ] All async operations wrapped in try-catch
- [ ] Correlation ID added to all log messages
- [ ] Timeout configured (default 30s)
- [ ] Circuit breaker initialized (for external APIs)
- [ ] Input validation before operations
- [ ] Error responses use createErrorResponse
- [ ] Success responses include execution time
- [ ] No credentials logged
- [ ] File paths sanitized in errors

### Common Mistakes

❌ **Don't:**
```typescript
// Missing correlation ID
try {
  await operation();
} catch (error) {
  console.error('Error:', error);
}

// No timeout
await operation(); // Could hang forever

// No retry
await fetch(url); // Fails on network blip

// Logging credentials
console.log('Connecting with:', password);
```

✅ **Do:**
```typescript
// With correlation ID
const correlationId = generateCorrelationId();
try {
  await operation();
} catch (error) {
  console.error(`[${correlationId}] Error:`, error);
}

// With timeout
await withTimeout(operation(), 30000, 'Operation');

// With retry
await retryWithBackoff(() => operation(), { maxAttempts: 3 });

// Sanitized logging
console.log('Connecting with:', '***');
```

## 🚨 Troubleshooting

### Timeout Not Working
```typescript
// ❌ Wrong - competing timeouts
setTimeout(() => abort(), timeout);
await operation(); // Has its own timeout

// ✅ Correct - single timeout
await withTimeout(operation(), timeout, 'Operation');
```

### Retry Not Triggering
```typescript
// Check if error is retryable
const category = categorizeError(error);
// 4xx errors (except 429) are NOT retryable
// 5xx, network, timeout ARE retryable
```

### Circuit Breaker Not Opening
```typescript
// Verify error count
const metrics = cb.getMetrics();
console.log('Failures:', metrics.failureCount);
console.log('Threshold:', config.failureThreshold);

// Reset if needed
cb.reset();
```

## 📚 Additional Resources

- Full Implementation Guide: `BUBBLE_ERROR_HANDLING_IMPLEMENTATION.md`
- Implementation Summary: `ERROR_HANDLING_IMPLEMENTATION_SUMMARY.md`
- Error Handling Utilities: `bubble-core/src/utils/error-handler.ts`
- Bubble Error Handler: `bubble-core/src/utils/bubble-error-handler.ts`

## 🎯 Best Practices

1. **Always use correlation IDs** - enables request tracking
2. **Set appropriate timeouts** - prevents infinite hangs
3. **Use circuit breakers for external APIs** - prevents cascading failures
4. **Retry with exponential backoff** - handles transient failures
5. **Sanitize all error messages** - prevents information leakage
6. **Log at appropriate levels** - info for success, error for failures
7. **Monitor circuit breaker state** - enables proactive troubleshooting
8. **Test error scenarios** - ensures robustness

---

**Last Updated:** 2025-01-18
**Version:** 1.0
