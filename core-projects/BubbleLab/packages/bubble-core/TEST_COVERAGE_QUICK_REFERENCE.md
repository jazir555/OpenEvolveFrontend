# Test Coverage Quick Reference Guide

## Current Test Statistics

### Overall Status
- **Test Files:** 61 total (24 passing, 37 failing)
- **Tests:** 1,197 total (807 passing, 330 failing, 60 skipped)
- **Pass Rate:** 67%
- **Coverage Goal:** 100% (currently blocked by test failures)

### Critical Issues
1. **12 test suites blocked** by missing dependencies
2. **330+ tests failing** across multiple categories
3. **Tool bubbles without tests:** 6 identified
4. **Workflow templates:** 43 templates need testing

## Quick Commands

### Run All Tests with Coverage
```bash
cd BubbleLab/packages/bubble-core
pnpm test:coverage
```

### Run Specific Test File
```bash
pnpm test src/bubbles/common/cache.test.ts
```

### Run All Tests in Watch Mode
```bash
pnpm test:watch
```

### Run Only Unit Tests
```bash
pnpm test:unit
```

### Run Only Integration Tests
```bash
pnpm test:integration
```

### View Coverage Report
```bash
pnpm test:coverage
npx vite preview --outDir coverage
# Open http://localhost:4173
```

## Blocked Test Suites (12 files)

### Missing `resilience.js` Adapter
Create: `src/integrations/openevolve/adapters/resilience.ts`

Files affected:
- `airtable-wrapper.test.ts`
- `apify-bubble.test.ts`
- `stripe-bubble.test.ts`
- `google-sheets-bubble.test.ts`
- `webhook-bubble.test.ts`

### Missing `test-utils.js`
Create: `src/tests/test-utils.ts`

Files affected:
- `http.comprehensive.test.ts`
- `postgresql.comprehensive.test.ts`
- `comprehensive-security.test.ts`

### Bun Test Incompatibility
Convert: `src/bubbles/service-bubble/deepseek.test.ts`

Change from:
```typescript
import { describe, it, expect } from 'bun:test';
```

To:
```typescript
// Vitest globals are already available
```

## Test Failures by Category

### 1. Cache Tests (3 failures)
**File:** `cache.test.ts`

Fix needed:
- `MultiTierCache.getStats()` not aggregating correctly
- L2 cache size tracking issue

### 2. Connection Pool Tests (2 failures)
**File:** `connection-pool.test.ts`

Fix needed:
- Timeout mechanism not rejecting promises
- Registry statistics not tracking HTTP pools

### 3. Retry Tests (21+ failures)
**File:** `retry.test.ts`

Fix needed:
- Sleep function overhead (15ms instead of <10ms)
- Test assertion format (use exact match, not StringContaining)
- CircuitBreakerState import/export issue

### 4. Validator Tests (1 failure)
**File:** `validators.test.ts`

Fix needed:
- `sanitizeString()` should remove `()` characters

### 5. Security Tests (5 failures)
**File:** `security-fixes.test.ts`

Fix needed:
- Expression length validation
- Empty expression validation
- Division by zero handling
- SSRF redirect chain prevention

## Tool Bubbles Needing Tests

### High Priority (No tests)
1. **chart-js-tool.ts** - Chart generation
2. **code-edit-tool.ts** - Code transformation
3. **google-maps-tool.ts** - Geocoding & maps
4. **instagram-tool.ts** - Instagram API
5. **linkedin-tool.ts** - LinkedIn API
6. **youtube-tool.ts** - YouTube API

### Medium Priority (Tests exist, may need expansion)
1. **sql-query-tool.ts** - Add injection prevention tests
2. **twitter-tool.ts** - Add error scenarios
3. **research-agent-tool.ts** - Add workflow tests

## Test File Template

```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ToolBubbleClass } from './tool-file';

describe('ToolBubbleClass', () => {
  let tool: ToolBubbleClass;

  beforeEach(() => {
    // Setup
    tool = new ToolBubbleClass({});
  });

  afterEach(() => {
    // Cleanup
  });

  describe('happy path', () => {
    it('should perform basic operation', async () => {
      const result = await tool.performAction();
      expect(result.success).toBe(true);
    });
  });

  describe('error cases', () => {
    it('should handle invalid input', async () => {
      tool = new ToolBubbleClass({ invalid: 'param' });
      const result = await tool.performAction();
      expect(result.success).toBe(false);
    });
  });

  describe('edge cases', () => {
    it('should handle empty input', async () => {
      tool = new ToolBubbleClass({});
      const result = await tool.performAction();
      expect(result).toBeDefined();
    });

    it('should handle large datasets', async () => {
      const largeData = Array(10000).fill('data');
      tool = new ToolBubbleClass({ data: largeData });
      const result = await tool.performAction();
      expect(result.success).toBe(true);
    });
  });
});
```

## Coverage Targets

### Minimum Viable (Phase 1-2)
- [ ] 0 test failures
- [ ] 80% line coverage
- [ ] 75% branch coverage
- [ ] 80% function coverage

### Ideal (Phase 3-4)
- [ ] 0 test failures
- [ ] 100% line coverage
- [ ] 95% branch coverage
- [ ] 100% function coverage

## Workflow Templates Testing

### Template Test Template
```typescript
import { describe, it, expect } from 'vitest';
import { runTemplate } from '../test-utils';

describe('Template Name', () => {
  it('should execute with valid parameters', async () => {
    const result = await runTemplate('template-name', {
      param1: 'value1',
      param2: 'value2',
    });
    expect(result.success).toBe(true);
  });

  it('should handle missing parameters', async () => {
    const result = await runTemplate('template-name', {});
    expect(result.success).toBe(false);
    expect(result.error).toContain('required');
  });
});
```

## Priority Order

### Phase 1: Critical Blockers (4-6 hours)
1. Create `resilience.ts` adapter
2. Create `test-utils.ts`
3. Convert `deepseek.test.ts` to vitest
4. Fix CircuitBreakerState export

### Phase 2: Failing Tests (6-8 hours)
1. Fix cache statistics
2. Fix connection pool timeout
3. Fix sleep function
4. Fix sanitization
5. Fix security tests

### Phase 3: Missing Coverage (15-20 hours)
1. Create tool bubble tests (6 files)
2. Create workflow template tests (43 files)
3. Enhance service bubble tests

### Phase 4: Optimization (8-10 hours)
1. Branch coverage
2. Line coverage
3. Edge cases
4. Error paths

## Common Patterns

### Testing Async Operations
```typescript
it('should handle async operation', async () => {
  const promise = tool.performAction();
  await expect(promise).resolves.toEqual({ success: true });
});

it('should handle async errors', async () => {
  const promise = tool.performAction();
  await expect(promise).rejects.toThrow('error message');
});
```

### Testing Timeouts
```typescript
it('should timeout after specified duration', async () => {
  const start = Date.now();
  try {
    await tool.performAction();
  } catch (e) {
    // Expected
  }
  const elapsed = Date.now() - start;
  expect(elapsed).toBeGreaterThanOrEqual(1000); // 1s timeout
  expect(elapsed).toBeLessThan(1500); // + buffer
});
```

### Testing Retry Logic
```typescript
it('should retry on failure', async () => {
  let attempts = 0;
  const mockFn = () => {
    attempts++;
    if (attempts < 3) throw new Error('fail');
    return { success: true };
  };
  const result = await retryWithBackoff(mockFn, { maxAttempts: 3 });
  expect(attempts).toBe(3);
  expect(result.success).toBe(true);
});
```

## Resources

- **Vitest Docs:** https://vitest.dev/
- **Test Coverage:** https://vitest.dev/guide/coverage.html
- **Best Practices:** https://vitest.dev/guide/why.html

## Status Tracking

- [x] Analysis complete
- [ ] Phase 1: Fix critical blockers
- [ ] Phase 2: Fix failing tests
- [ ] Phase 3: Add missing coverage
- [ ] Phase 4: Optimize to 100%
- [ ] Generate final coverage report

## Next Steps

1. Start with Phase 1 (Critical Blockers)
2. Run tests after each fix to verify
3. Track progress in this document
4. Update coverage percentages regularly
5. Generate HTML coverage report weekly
