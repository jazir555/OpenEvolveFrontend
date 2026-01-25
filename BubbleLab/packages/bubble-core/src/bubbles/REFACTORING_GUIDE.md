# Bubble Refactoring Guide - P3 Final Wave

## Overview

This guide provides systematic instructions for refactoring all BubbleLab bubbles to use common utilities and patterns, eliminating code duplication and improving maintainability.

## Current Status

- **Total Bubbles:** 92 files
- **Common Utilities Available:** ✅
  - Validators (`common/validators.ts`)
  - Error Handlers (`common/error-handlers.ts`)
  - Retry Logic (`common/retry.ts`)
  - Connection Pool (`common/connection-pool.ts`)
  - Cache (`common/cache.ts`)

## Refactoring Strategy

### Phase 1: High-Priority Service Bubbles (6 hours)
Priority bubbles that are commonly used:

1. ✅ `postgresql.ts` - Database operations
2. `redis.ts` - Caching operations
3. `qdrant.ts` - Vector database
4. `elasticsearch.ts` - Search engine
5. `slack.ts` - Communication
6. `sendgrid.ts` - Email
7. `twilio.ts` - SMS/Phone
8. `stripe.ts` - Payments
9. `notion.ts` - Documentation
10. `airtable.ts` - Spreadsheets

### Phase 2: Google Service Bubbles (2 hours)
11. `google-drive.ts`
12. `google-sheets.ts`
13. `google-calendar.ts`
14. `gmail.ts`

### Phase 3: AI/ML Service Bubbles (2 hours)
15. `ai-agent.ts`
16. `eleven-labs.ts`
17. `hephaestus.ts`

### Phase 4: Tool Bubbles (3 hours)
18. `code-edit-tool.ts`
19. `chart-js-tool.ts`
20. `research-agent-tool.ts`
21. `sql-query-tool.ts`
22. All other tool bubbles

## Refactoring Template

### Before (Duplicated Code)
```typescript
import { ServiceBubble } from '../../types/service-bubble-class.js';

export class SomeBubble extends ServiceBubble<Params, Result> {
  // Custom validation
  private validateEmail(email: string): void {
    if (!email || typeof email !== 'string' || email.length > 254) {
      throw new Error('Invalid email');
    }
  }

  // Custom error handling
  private async executeWithRetry(fn: () => Promise): Promise {
    try {
      return await fn();
    } catch (err) {
      console.error('Error:', err.message);
      throw err;
    }
  }
}
```

### After (Using Common Utilities)
```typescript
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { validateEmail } from '../common/validators.js';
import { NetworkError, wrapError } from '../common/error-handlers.js';
import { retryWithBackoff } from '../common/retry.js';

/**
 * Some Bubble Description
 *
 * @class
 * @extends ServiceBubble<Params, Result>
 *
 * @example
 * ```typescript
 * const bubble = new SomeBubble({ /* params * / });
 * const result = await bubble.action();
 * ```
 *
 * @see {@link https://api-docs-url|API Documentation}
 */
export class SomeBubble extends ServiceBubble<Params, Result> {
  /**
   * Main execution method
   *
   * @returns Promise resolving to operation result
   */
  protected async performAction(): Promise<Result> {
    // Use common validators
    validateEmail(this.params.email);

    // Use common retry logic
    return retryWithBackoff(
      async () => await this.executeOperation(),
      {
        maxAttempts: 3,
        operation: 'SomeOperation'
      }
    );
  }

  /**
   * Execute the core operation
   *
   * @private
   * @returns Promise resolving to operation result
   * @throws NetworkError on network failures
   */
  private async executeOperation(): Promise<Result> {
    try {
      // Operation logic here
      return result;
    } catch (error) {
      // Use common error wrapping
      throw wrapError(error, {
        message: 'Operation failed',
        code: 'OP_ERROR'
      });
    }
  }
}
```

## JSDoc Template

### Class Documentation
```typescript
/**
 * Bubble Name and Description
 *
 * Detailed description of what this bubble does, its use cases,
 * and important implementation notes.
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
 *
 * const result = await bubble.action();
 * console.log(result.data);
 * ```
 *
 * @see {@link https://external-docs-url|External Documentation}
 *
 * @remarks
 * Additional implementation notes, security considerations,
 * performance characteristics, etc.
 */
```

### Method Documentation
```typescript
/**
 * Brief description of what the method does
 *
 * Detailed description if needed, explaining the algorithm,
 * side effects, or important implementation details.
 *
 * @param paramName - Parameter description
 * @param param2 - Second parameter description
 * @returns Promise resolving to return type description
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

## Common Utilities Reference

### Validators (`common/validators.ts`)

```typescript
import {
  validateEmail,           // Email validation
  validateUrl,             // URL validation
  validateTimestamp,       // ISO 8601 timestamp
  validateNonEmptyString,  // String not empty/whitespace
  validateNumberRange,     // Number in range [min, max]
  validateArrayLength,     // Array length constraints
  validateRequiredProperties, // Object has required props
  validateFilePath,        // Safe file path (no traversal)
  sanitizeString,          // Remove dangerous chars
  batchValidate            // Multiple validations
} from '../common/validators.js';
```

### Error Handlers (`common/error-handlers.ts`)

```typescript
import {
  BubbleError,            // Base error class
  ValidationError,         // Invalid input
  AuthenticationError,    // Auth failed
  AuthorizationError,     // Access denied
  NotFoundError,          // Resource not found
  RateLimitError,         // Rate limited
  NetworkError,           // Network/transient failure
  TimeoutError,           // Operation timeout
  ConfigurationError,     // Bad configuration
  ExternalServiceError,   // Third-party service error
  categorizeError,        // Get error category
  isRetryable,            // Check if retryable
  createErrorResponse,     // Format error response
  wrapError,              // Add context to error
  logError,               // Structured error logging
  assert                  // Assertion helper
} from '../common/error-handlers.js';
```

### Retry Logic (`common/retry.ts`)

```typescript
import {
  retryWithBackoff,       // Retry with exponential backoff
  retryWithTimeout,       // Retry with timeout per attempt
  withTimeout,            // Add timeout to any promise
  CircuitBreaker,         // Circuit breaker pattern
  executeWithResilience,  // Combine retry + circuit breaker
  sleep,                  // Promise-based sleep
  calculateDelay          // Calculate retry delay
} from '../common/retry.js';
```

## Refactoring Checklist

For each bubble file:

- [ ] Add imports from `common/` utilities
- [ ] Replace custom validators with common validators
- [ ] Replace custom error handling with common error classes
- [ ] Replace custom retry logic with common retry utilities
- [ ] Add comprehensive JSDoc to class
- [ ] Add JSDoc to all public methods
- [ ] Add JSDoc to important private methods
- [ ] Add usage examples in JSDoc
- [ ] Add @see references to external docs
- [ ] Improve variable naming
- [ ] Simplify complex functions
- [ ] Remove code duplication
- [ ] Add error type hints in @throws
- [ ] Add parameter descriptions in @param
- [ ] Add return descriptions in @returns

## Refactoring Metrics

Track these metrics for each bubble:

### Before Refactoring
- Total lines of code
- Lines of duplicated validation code
- Lines of duplicated error handling code
- Lines of duplicated retry logic
- Number of JSDoc comments
- JSDoc coverage percentage

### After Refactoring
- Total lines of code (should decrease)
- Lines of common utility imports (should increase)
- Number of JSDoc comments (should increase significantly)
- JSDoc coverage percentage (target: 100% for public APIs)

## Quality Gates

Each refactored bubble must:

1. ✅ Compile without errors
2. ✅ Pass all existing tests
3. ✅ Have JSDoc on class and all public methods
4. ✅ Use common validators (no custom validation)
5. ✅ Use common error classes (no generic Error throws)
6. ✅ Use common retry logic (no custom retry)
7. ✅ Have at least one usage example
8. ✅ Have @see reference to external docs

## Automation Script

See `scripts/refactor-bubble.ts` for automated refactoring assistance.

## Progress Tracking

| # | Bubble | Status | Lines Reduced | JSDoc % |
|---|--------|--------|---------------|---------|
| 1 | postgresql | ✅ | TBD | TBD |
| 2 | redis | ⏳ | TBD | TBD |
| 3 | qdrant | ⏳ | TBD | TBD |
| 4 | elasticsearch | ⏳ | TBD | TBD |
| 5 | slack | ⏳ | TBD | TBD |

## Estimated Effort

- **Phase 1:** 6 hours (10 bubbles = 36 min/bubble)
- **Phase 2:** 2 hours (4 bubbles = 30 min/bubble)
- **Phase 3:** 2 hours (3 bubbles = 40 min/bubble)
- **Phase 4:** 3 hours (10+ tool bubbles = 18 min/bubble)

**Total Estimated Time:** 12-13 hours for full refactoring

## Next Steps

1. Start with PostgreSQL (already in progress)
2. Move to Redis, Qdrant, Elasticsearch
3. Tackle communication bubbles (Slack, SendGrid, Twilio)
4. Refactor Google services
5. Complete AI/ML bubbles
6. Finish with tool bubbles

## Notes

- Focus on high-impact, commonly-used bubbles first
- Maintain backward compatibility
- All existing tests must pass
- JSDoc should be comprehensive but concise
- Use examples from real usage where possible
- Link to official API documentation in @see tags
