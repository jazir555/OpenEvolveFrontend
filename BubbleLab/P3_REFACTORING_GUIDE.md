# Bubble Refactoring Guide

## Overview

This guide provides patterns and examples for systematically refactoring all 117+ bubbles in the BubbleLab codebase to use common utilities, reducing code duplication by ~14,200 lines.

## Common Utilities Available

### Location
`BubbleLab/packages/bubble-core/src/bubbles/common/`

### Modules

1. **validators.ts** - Input validation functions
   - `validateEmail()` - Email validation
   - `validateUrl()` - URL validation with protocol checking
   - `validateTimestamp()` - ISO 8601 timestamp validation
   - `validateNonEmptyString()` - String validation
   - `validateNumberRange()` - Numeric range validation
   - `validateArrayLength()` - Array length validation
   - `validateRequiredProperties()` - Object property validation
   - `validateFilePath()` - File path validation (prevents path traversal)
   - `sanitizeString()` - String sanitization
   - `createNonEmptyStringSchema()` - Zod schema helpers
   - `createEmailSchema()` - Email Zod schema
   - `createUrlSchema()` - URL Zod schema
   - `batchValidate()` - Batch validation

2. **error-handlers.ts** - Error handling utilities
   - `BubbleError` - Base error class
   - `AuthenticationError` - Auth failures
   - `AuthorizationError` - Permission errors
   - `ValidationError` - Input validation errors
   - `NotFoundError` - Resource not found
   - `RateLimitError` - Rate limiting
   - `NetworkError` - Network failures
   - `TimeoutError` - Timeout failures
   - `ConfigurationError` - Config issues
   - `ExternalServiceError` - Third-party service errors
   - `categorizeError()` - Error categorization
   - `isRetryable()` - Check if error is retryable
   - `createErrorResponse()` - Format error responses
   - `createSuccessResponse()` - Format success responses
   - `wrapError()` - Add context to errors
   - `logError()` - Structured error logging
   - `assert()` - Assertion helper
   - `assertNonNull()` - Non-null assertion

3. **retry.ts** - Retry logic and circuit breakers
   - `retryWithBackoff()` - Exponential backoff retry
   - `retryWithTimeout()` - Retry with per-attempt timeout
   - `withTimeout()` - Add timeout to promise
   - `CircuitBreaker` - Circuit breaker implementation
   - `executeWithResilience()` - Combine retry + circuit breaker
   - `calculateDelay()` - Calculate exponential backoff delay
   - `sleep()` - Async sleep

4. **types.ts** - Common type definitions
   - `Result<T, E>` - Result type for operations
   - `CredentialType` - Credential type enum
   - `Credential` - Credential interface
   - `RequestOptions` - HTTP request options
   - `PaginationOptions` - Pagination config
   - `PaginatedResponse<T>` - Paginated response
   - `SortOptions` - Sorting config
   - `QueryOptions` - Combined query options
   - `DateRange` - Date range
   - `TimeRange` - Time range
   - `Coordinate` - Geographic coordinate
   - `Address` - Address information
   - `Money` - Monetary amount
   - `PersonName` - Person's name
   - `ContactInfo` - Contact information
   - `UserProfile` - User profile
   - `OperationMetadata` - Operation metadata
   - `CacheEntry<T>` - Cache entry
   - `ConnectionPoolConfig` - Connection pool config
   - `RateLimitConfig` - Rate limit config
   - `RetryConfig` - Retry config
   - `CircuitBreakerConfig` - Circuit breaker config
   - `ResilienceConfig` - Combined resilience config
   - Type guards: `isResult()`, `isOk()`, `isErr()`, `isPlainObject()`, `isIsoTimestamp()`, etc.
   - Utility functions: `deepClone()`, `deepMerge()`

5. **connection-pool.ts** - Connection pool management
6. **cache.ts** - Caching utilities
7. **constants.ts** - Common constants

## Refactoring Patterns

### Pattern 1: Replace Inline Validation with Common Validators

**Before:**
```typescript
// Inline email validation
if (!email || typeof email !== 'string') {
  throw new Error('Email is required and must be a string');
}
const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
if (!emailRegex.test(email)) {
  throw new Error('Invalid email format');
}
```

**After:**
```typescript
import { validateEmail } from '../common/validators.js';

// Use common validator
validateEmail(email);
```

**Lines saved:** ~6 lines per validation

### Pattern 2: Replace Inline Error Handling with Common Error Classes

**Before:**
```typescript
if (!authToken) {
  throw new Error('Authentication token is required');
}
if (!response.ok) {
  throw new Error(`API request failed: ${response.status}`);
}
```

**After:**
```typescript
import { AuthenticationError, ExternalServiceError } from '../common/error-handlers.js';

if (!authToken) {
  throw new AuthenticationError('Authentication token is required');
}
if (!response.ok) {
  throw new ExternalServiceError('slack', `API request failed: ${response.status}`, String(response.status));
}
```

**Lines saved:** ~2-4 lines per error handling

### Pattern 3: Replace Inline Retry Logic

**Before:**
```typescript
let retries = 0;
const maxRetries = 3;
while (retries < maxRetries) {
  try {
    return await makeRequest();
  } catch (error) {
    retries++;
    if (retries >= maxRetries) throw error;
    await new Promise(resolve => setTimeout(resolve, Math.pow(2, retries) * 1000));
  }
}
```

**After:**
```typescript
import { retryWithBackoff } from '../common/retry.js';

return await retryWithBackoff(
  () => makeRequest(),
  {
    maxAttempts: 3,
    baseDelayMs: 1000,
    operation: 'Slack API Request'
  }
);
```

**Lines saved:** ~10 lines per retry logic

### Pattern 4: Replace File Path Validation

**Before:**
```typescript
// SECURITY: Block path traversal attempts
if (file_path.includes('..') || file_path.includes('~')) {
  return {
    ok: false,
    error: 'File path contains forbidden characters (.. or ~). Path traversal is not allowed.',
    success: false,
  };
}

// SECURITY: Block absolute paths
if (file_path.startsWith('/') || file_path.startsWith('\\')) {
  return {
    ok: false,
    error: 'Absolute paths are not allowed.',
    success: false,
  };
}

// SECURITY: Validate file path contains only safe characters
const safePathPattern = /^[a-zA-Z0-9\s._/\\-]+$/;
if (!safePathPattern.test(file_path)) {
  return {
    ok: false,
    error: 'File path contains invalid characters',
    success: false,
  };
}

// SECURITY: Limit file path length
if (file_path.length > 4096) {
  return {
    ok: false,
    error: 'File path exceeds maximum allowed length of 4096 characters',
    success: false,
  };
}
```

**After:**
```typescript
import { validateFilePath, ValidationError } from '../common/validators.js';
import { createErrorResponse } from '../common/error-handlers.js';

try {
  validateFilePath(file_path, false); // false = no absolute paths
} catch (error) {
  if (error instanceof ValidationError) {
    return createErrorResponse(error, this.correlationId);
  }
  throw error;
}
```

**Lines saved:** ~25 lines per file validation

### Pattern 5: Add JSDoc Comments

**Before:**
```typescript
public async testCredential(): Promise<boolean> {
  const response = await this.makeSlackApiCall('auth.test', {});
  if (response.ok) {
    return true;
  }
  return false;
}
```

**After:**
```typescript
/**
 * Test the validity of the Slack credential
 * @returns Promise that resolves to true if credential is valid, false otherwise
 * @throws ExternalServiceError if Slack API call fails
 */
public async testCredential(): Promise<boolean> {
  const response = await this.makeSlackApiCall('auth.test', {});
  return response.ok;
}
```

## Refactoring Checklist

For each bubble file:

- [ ] Import common utilities at the top
- [ ] Replace inline validation with common validators
- [ ] Replace inline error handling with common error classes
- [ ] Replace inline retry logic with retry utilities
- [ ] Replace file path validation with common validator
- [ ] Add JSDoc comments to all public methods
- [ ] Add JSDoc comments to private helper methods
- [ ] Improve variable names for clarity
- [ ] Ensure all error paths use common error types
- [ ] Test the refactored bubble

## Priority Order

### Phase 1: Critical Service Bubbles (High Usage)
1. slack.ts (~2100 lines) - HIGH PRIORITY
2. http.ts (~800 lines) - HIGH PRIORITY
3. postgresql.ts (~400 lines) - HIGH PRIORITY
4. ai-agent.ts (~600 lines) - HIGH PRIORITY
5. airtable.ts (~500 lines) - MEDIUM PRIORITY

### Phase 2: Apify Actors (~30 files)
6. google-maps-scraper.ts
7. instagram-scraper.ts
8. instagram-hashtag-scraper.ts
9. linkedin-jobs-scraper.ts
10. linkedin-posts-search.ts
11. linkedin-profile-posts.ts
12. tiktok-scraper.ts
13. twitter-scraper.ts
14. youtube-scraper.ts
15. youtube-transcript-scraper.ts
... (and ~15 more)

### Phase 3: Other Service Bubbles (~40 files)
16. elasticsearch-bubble.ts
17. gmail-bubble.ts
18. google-calendar-bubble.ts
19. notion-bubble.ts
20. stripe-bubble.ts
21. redis-bubble.ts
22. mongodb-bubble.ts
23. s3-bubble.ts
... (and ~30 more)

### Phase 4: Tool Bubbles (~30 files)
24. chart-js-tool.ts
25. code-edit-tool.ts
26. google-maps-tool.ts
27. instagram-tool.ts
28. linkedin-tool.ts
29. research-agent-tool.ts
30. sql-query-tool.ts
31. twitter-tool.ts
32. youtube-tool.ts
... (and ~20 more)

### Phase 5: Workflow Templates (~21 files)
33. templates/*.ts files

## Code Reduction Estimates

By refactoring pattern:
- File path validation: ~25 lines per file × 120 files = ~3,000 lines saved
- Inline validation: ~50 lines per file × 120 files = ~6,000 lines saved
- Error handling: ~20 lines per file × 120 files = ~2,400 lines saved
- Retry logic: ~10 lines per file × 30 files = ~300 lines saved
- JSDoc improves quality but doesn't reduce lines

**Total estimated reduction: ~11,700 lines**

With additional optimizations (removing dead code, consolidating schemas):
**Total estimated reduction: ~14,200 lines (11% of codebase)**

## Testing Strategy

After each bubble refactoring:
1. Run existing tests: `npm test -- <bubble-file>.test.ts`
2. Run integration tests if available
3. Manual smoke test for critical bubbles
4. Check for TypeScript errors: `npm run type-check`

## Execution Plan

Given 120 files and 12-13 hour estimate:
- **Hour 1-2:** Setup and Phase 1 (5 critical bubbles)
- **Hour 3-6:** Phase 2 (30 Apify actors)
- **Hour 7-9:** Phase 3 (40 service bubbles)
- **Hour 10-11:** Phase 4 (30 tool bubbles)
- **Hour 12:** Phase 5 (21 workflow templates)
- **Hour 13:** Final verification, testing, and report generation

## Automated Refactoring Script

See `scripts/refactor-bubbles.ts` for automated refactoring assistance.

The script will:
1. Detect common patterns that can be refactored
2. Generate refactored code as suggestions
3. Track line count reduction
4. Generate a report

**Note:** Manual review and testing is still required after automated refactoring.
