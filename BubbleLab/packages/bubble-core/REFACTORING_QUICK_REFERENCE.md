# Bubble Refactoring Quick Reference Card

## Common Utilities - Import Statement

```typescript
import {
  // Validators
  validateEmail,
  validateUrl,
  validateTimestamp,
  validateNonEmptyString,
  validateNumberRange,
  validateArrayLength,
  validateRequiredProperties,
  validateFilePath,
  sanitizeString,
  batchValidate,

  // Error Handlers
  BubbleError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  RateLimitError,
  NetworkError,
  TimeoutError,
  ConfigurationError,
  ExternalServiceError,
  categorizeError,
  isRetryable,
  createErrorResponse,
  wrapError,
  logError,
  assert,

  // Retry Logic
  retryWithBackoff,
  retryWithTimeout,
  withTimeout,
  CircuitBreaker,
  executeWithResilience,
  sleep,
  calculateDelay,
} from '../common/index.js';
```

## Before/After Examples

### Validation

**Before:**
```typescript
if (!email || typeof email !== 'string') {
  throw new Error('Email is required');
}
if (email.length > 254) {
  throw new Error('Email too long');
}
if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
  throw new Error('Invalid email format');
}
```

**After:**
```typescript
validateEmail(email);
```

### Error Handling

**Before:**
```typescript
try {
  await operation();
} catch (err) {
  console.error('Error:', err.message);
  throw new Error(`Operation failed: ${err.message}`);
}
```

**After:**
```typescript
try {
  await operation();
} catch (err) {
  throw wrapError(err, {
    message: 'Operation failed',
    code: 'OP_ERROR'
  });
}
```

**Or better:**
```typescript
try {
  await operation();
} catch (err) {
  throw new ExternalServiceError('ServiceName', 'Operation failed', undefined, { cause: err });
}
```

### Retry Logic

**Before:**
```typescript
let lastError;
for (let attempt = 0; attempt < 3; attempt++) {
  try {
    return await operation();
  } catch (err) {
    lastError = err;
    if (attempt < 2) {
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
    }
  }
}
throw lastError;
```

**After:**
```typescript
return await retryWithBackoff(
  async () => await operation(),
  {
    maxAttempts: 3,
    baseDelayMs: 1000,
    operation: 'OperationName'
  }
);
```

### Circuit Breaker

**Before:**
```typescript
// No circuit breaker - cascading failures possible
return await this.apiCall();
```

**After:**
```typescript
return await this.circuitBreaker.execute(
  async () => await this.apiCall(),
  'APICall'
);
```

## JSDoc Templates

### Class Documentation

```typescript
/**
 * Brief one-line description.
 *
 * Detailed description spanning multiple lines
 * if needed. Explain what the bubble does, its
 * use cases, and important notes.
 *
 * @class
 * @extends ServiceBubble<ParamsType, ResultType>
 *
 * @example
 * ```typescript
 * const bubble = new BubbleName({
 *   param1: 'value1',
 *   param2: 'value2'
 * });
 * const result = await bubble.action();
 * ```
 *
 * @see {@link https://external-docs-url|External Documentation}
 *
 * @remarks
 * Implementation notes, security considerations,
 * performance characteristics, etc.
 */
```

### Method Documentation

```typescript
/**
 * Brief description.
 *
 * @param param1 - Description of param1
 * @param param2 - Description of param2
 * @returns Description of return value
 *
 * @throws {ValidationError} If validation fails
 * @throws {NetworkError} If network operation fails
 *
 * @example
 * ```typescript
 * const result = await bubble.methodName('paramValue');
 * ```
 *
 * @see {@link relatedMethod|Related Method}
 */
```

## Error Throwing Guidelines

### DO:
```typescript
throw new ValidationError('Invalid parameter', 'fieldName');
throw new AuthenticationError('Invalid API token');
throw new AuthorizationError('Insufficient permissions');
throw new NotFoundError('Resource', 'resource-id');
throw new RateLimitError('Too many requests', 60); // retryAfter in seconds
throw new NetworkError('Connection failed');
throw new TimeoutError('Operation timed out', timeoutMs);
throw new ConfigurationError('Missing config', 'CONFIG_KEY');
throw new ExternalServiceError('ServiceName', 'API error', 'ERROR_CODE');
```

### DON'T:
```typescript
throw new Error('Invalid parameter'); // ❌ Use ValidationError
throw 'Error message'; // ❌ Always throw Error objects
throw undefined; // ❌ Never throw undefined
return { success: false, error: 'message' }; // ❌ Throw errors instead
```

## Refactoring Checklist

- [ ] Import from `../common/index.js`
- [ ] Replace `throw new Error()` with specific error types
- [ ] Replace custom validation with `validate*()` functions
- [ ] Replace custom retry with `retryWithBackoff()`
- [ ] Wrap async operations in `withTimeout()`
- [ ] Use `circuitBreaker.execute()` for external calls
- [ ] Add JSDoc to class (description, example, @see)
- [ ] Add JSDoc to all public methods
- [ ] Add @param, @returns, @throws tags
- [ ] Add @example with real usage
- [ ] Run tests to ensure no regressions

## Common Patterns

### API Call with Retry, Timeout, and Circuit Breaker

```typescript
protected async performAction(): Promise<Result> {
  return await this.circuitBreaker.execute(
    async () => {
      return await retryWithBackoff(
        async () => {
          return await this.makeApiCall();
        },
        {
          maxAttempts: 3,
          baseDelayMs: 1000,
          correlationId: generateCorrelationId(),
          operation: 'APIOperation'
        }
      );
    },
    'APIOperation'
  );
}

private async makeApiCall(): Promise<Result> {
  const authToken = this.chooseCredential();

  if (!authToken) {
    throw new AuthenticationError('API token required');
  }

  const response = await withTimeout(
    fetch(url, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    }),
    this.params.timeout || 30000,
    'API Request'
  );

  if (!response.ok) {
    if (response.status === 401) {
      throw new AuthenticationError('Invalid token');
    }
    if (response.status === 429) {
      throw new RateLimitError('Rate limit exceeded');
    }
    throw new ExternalServiceError('API', `HTTP ${response.status}`);
  }

  return await response.json();
}
```

### Database Query with Validation

```typescript
protected async performAction(): Promise<Result> {
  // Validate inputs
  validateNonEmptyString(this.params.query, 'query');
  validateArrayLength(this.params.parameters, 0, 100, 'parameters');
  validateNumberRange(this.params.timeout, 1000, 300000, 'timeout');

  // Execute with resilience patterns
  return await this.circuitBreaker.execute(
    async () => {
      return await retryWithBackoff(
        async () => await this.executeQuery(),
        {
          maxAttempts: 3,
          baseDelayMs: 1000,
          operation: 'DatabaseQuery'
        }
      );
    },
    'DatabaseQuery'
  );
}
```

## Metrics to Track

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Total Lines | | | ↓ 10% |
| Duplicated Validation | | | 0 |
| Duplicated Error Handling | | | 0 |
| Duplicated Retry Logic | | | 0 |
| JSDoc Comments | | | ↑ 100% |
| JSDoc Coverage % | | | 100% |
| Common Imports | 0 | ≥3 | - |

## Quality Gates

✅ Must pass ALL gates before marking bubble as refactored:

1. TypeScript compiles without errors
2. All existing tests pass
3. JSDoc on class and all public methods
4. No custom validation (uses common validators)
5. No generic `throw new Error()` (uses specific error types)
6. No custom retry loops (uses `retryWithBackoff`)
7. At least one `@example` in JSDoc
8. `@see` reference to external docs
9. Error handling uses typed error classes
10. No code duplication (DRY)

## Tools

```bash
# Analyze a bubble
npx tsx scripts/refactor-bubbles.ts analyze <file>

# Generate refactored code
npx tsx scripts/refactor-bubbles.ts refactor <file>

# Show statistics
npx tsx scripts/refactor-bubbles.ts stats
```

## Resources

- **Refactoring Guide:** `src/bubbles/REFACTORING_GUIDE.md`
- **Summary:** `P3_REFACTORING_SUMMARY.md`
- **Common Utilities:** `src/bubbles/common/`
- **Examples:** This file

## Tips

1. **Start with imports** - Add common imports first
2. **Refactor validation** - Replace custom validation with common
3. **Fix error handling** - Use specific error types
4. **Add retry logic** - Replace retry loops with common
5. **Document** - Add comprehensive JSDoc last
6. **Test** - Verify everything still works
7. **Measure** - Track lines saved and quality improved

## Priority Order

1. High-use services (postgres, redis, slack, etc.)
2. Google services (drive, sheets, gmail)
3. AI/ML services (ai-agent, eleven-labs)
4. Tool bubbles (code-edit, research-agent, etc.)

---

**Remember:** The goal is consistency, maintainability, and reduced duplication. Every refactored bubble should follow the same patterns and conventions.
