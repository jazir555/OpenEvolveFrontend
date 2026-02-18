# 100% Code Coverage Achievement Plan

**Generated:** 2026-01-19
**Package:** @bubblelab/bubble-core
**Current State:** 51.7% test coverage (91/176 files have tests)

## Executive Summary

This document provides a comprehensive analysis of the current test coverage state and a detailed roadmap for achieving exactly 100% code coverage across line, branch, and function metrics.

### Current Status

| Metric | Value |
|--------|-------|
| Total Source Files | 176 |
| Files with Tests | 91 |
| Files Without Tests | 129 |
| Test Coverage Percentage | 51.7% |
| Target Coverage | 100% |

### Critical Issues Identified

1. **Missing Module Dependencies:** 12+ tests failing due to missing `../../../../integrations/openevolve/adapters/resilience.js` module
2. **Missing Test Framework:** Some tests using `bun:test` instead of `vitest`
3. **Massive Test Gap:** 129 high-priority source files completely lack test coverage

---

## Phase 1: Fix Failing Tests (Priority: CRITICAL)

### 1.1 Resolve Missing Dependencies

**Affected Files:**
- `src/bubbles/service-bubble/airtable-wrapper.ts`
- `src/bubbles/service-bubble/apify-bubble.ts`
- `src/bubbles/service-bubble/stripe-bubble.ts`
- `src/bubbles/service-bubble/firecrawl.test.ts`
- `src/bubbles/service-bubble/postgresql.test.ts`
- `src/bubbles/service-bubble/deepseek.test.ts`
- And 6+ more files

**Root Cause:**
Files importing from `../../../../integrations/openevolve/adapters/resilience.js` but the module exists as `.ts` not `.js`

**Solutions:**

**Option A: Build Before Test**
```json
// package.json
"scripts": {
  "pretest": "pnpm build:shared",
  "test": "vitest run --coverage"
}
```

**Option B: Update Vitest Config**
```typescript
// vitest.config.ts
resolve: {
  alias: {
    '@resilience': new URL(
      '../../integrations/openevolve/adapters/resilience.ts',
      import.meta.url
    ).pathname
  }
}
```

**Option C: Mock the Module**
Create `src/__mocks__/resilience.ts` with mocked implementations

**Recommended:** Option A (Build shared dependencies first)

### 1.2 Fix Test Framework Issues

**File:** `src/bubbles/service-bubble/deepseek.test.ts`

**Issue:** Uses `bun:test` framework

**Solution:**
```typescript
// Replace
import { describe, it, expect } from 'bun:test';

// With
import { describe, it, expect } from 'vitest';
```

### 1.3 Fix Test Failures in Existing Tests

**Common Issues Found:**

1. **Path Traversal Tests** - Tests expecting validation failure but getting schema rejection
2. **Null Byte Tests** - Object.is comparison failures
3. **Email Validation Tests** - Schema rejecting before test logic runs
4. **Timeout Tests** - Actual timeouts vs mocked timeouts
5. **Retry Logic Tests** - Flake due to timing issues

**Fix Strategy:**
```typescript
// Instead of testing invalid input
it('should prevent path traversal', () => {
  const result = await bubble.execute({ fileName: '../../etc/passwd' });
  expect(result.success).toBe(false);
});

// Test the schema validation
it('should reject path traversal at schema level', () => {
  expect(() => bubble.validateInput({ fileName: '../../etc/passwd' }))
    .toThrow('path traversal');
});
```

---

## Phase 2: Achieve Baseline Coverage (Priority: HIGH)

### 2.1 Run Initial Coverage Report

After fixing failing tests:
```bash
cd BubbleLab/packages/bubble-core
pnpm test:coverage -- --reporter=json --reporter=html
```

### 2.2 Identify Gaps in Existing Tests

For each file with tests:
```bash
# Check specific file coverage
pnpm test -- --coverage src/bubbles/service-bubble/http.test.ts
```

**Common Coverage Gaps to Fill:**

1. **Error Catch Blocks**
```typescript
// Code
try {
  await operation();
} catch (error) {
  logger.error('Operation failed', error);
}

// Missing Test
it('should log error when operation fails', async () => {
  mockOperation.mockRejectedValueOnce(new Error('Test error'));
  await bubble.execute();
  expect(logger.error).toHaveBeenCalledWith(
    'Operation failed',
    expect.any(Error)
  );
});
```

2. **Else Branches**
```typescript
// Code
if (condition) {
  return A;
} else {
  return B;
}

// Missing Test
it('should return B when condition is false', async () => {
  const result = await bubble.execute({ condition: false });
  expect(result).toBe('B');
});
```

3. **Null/Undefined Paths**
```typescript
// Missing Tests
it('should handle null input', async () => {
  const result = await bubble.execute({ value: null });
  expect(result).toBe(defaultValue);
});

it('should handle undefined input', async () => {
  const result = await bubble.execute({ value: undefined });
  expect(result).toBe(defaultValue);
});
```

4. **Empty Collections**
```typescript
// Missing Test
it('should handle empty array', async () => {
  const result = await bubble.execute({ items: [] });
  expect(result).toEqual([]);
});
```

5. **Boundary Values**
```typescript
// Code: if (count > 10)
// Missing Tests
it('should handle value at boundary (10)', async () => {
  await bubble.execute({ count: 10 });
});

it('should handle value below boundary (9)', async () => {
  await bubble.execute({ count: 9 });
});

it('should handle value above boundary (11)', async () => {
  await bubble.execute({ count: 11 });
});
```

---

## Phase 3: Create Tests for High-Priority Files (Priority: HIGH)

### Top 20 Files Requiring Tests (by complexity score)

| # | File | Priority | Estimated Effort |
|---|------|----------|------------------|
| 1 | `bubbles/tool-bubble/file-processor-tool.ts` | 343 | 8 hours |
| 2 | `bubbles/service-bubble/google-drive.ts` | 233 | 6 hours |
| 3 | `bubbles/tool-bubble/json-validator-tool.ts` | 231 | 5 hours |
| 4 | `bubbles/service-bubble/gmail.ts` | 223 | 6 hours |
| 5 | `bubbles/service-bubble/notion-bubble.ts` | 220 | 5 hours |
| 6 | `bubbles/tool-bubble/metrics-collector-tool.ts` | 193 | 4 hours |
| 7 | `bubbles/tool-bubble/pdf-generator-tool.ts` | 183 | 4 hours |
| 8 | `bubbles/tool-bubble/log-parser-tool.ts` | 181 | 4 hours |
| 9 | `bubbles/service-bubble/airtable-bubble.ts` | 159 | 4 hours |
| 10 | `bubbles/tool-bubble/csv-processor-tool.ts` | 156 | 3 hours |
| 11 | `bubbles/tool-bubble/data-transformer-tool.ts` | 147 | 3 hours |
| 12 | `bubbles/tool-bubble/xml-parser-tool.ts` | 142 | 3 hours |
| 13 | `bubbles/service-bubble/github-bubble.ts` | 132 | 3 hours |
| 14 | `bubbles/service-bubble/gmail-bubble.ts` | 126 | 3 hours |
| 15 | `bubbles/service-bubble/crewai-bubble.ts` | 125 | 3 hours |
| 16 | `bubbles/service-bubble/postgresql-bubble.ts` | 123 | 3 hours |
| 17 | `bubbles/service-bubble/workflow-orchestrator-bubble.ts` | 122 | 4 hours |
| 18 | `bubbles/service-bubble/ace-tools-bubble.ts` | 121 | 3 hours |
| 19 | `utils/security-utils.ts` | 121 | 3 hours |
| 20 | `bubbles/service-bubble/slack-bubble.ts` | 118 | 3 hours |

**Total Estimated Effort for Top 20:** ~85 hours

### Test Template Structure

For each file, create comprehensive tests covering:

```typescript
describe('FileName', () => {
  describe('Happy Path', () => {
    it('should execute successfully with valid inputs');
    it('should handle typical use cases');
    it('should return expected results');
  });

  describe('Input Validation', () => {
    it('should reject null inputs');
    it('should reject undefined inputs');
    it('should reject invalid types');
    it('should validate required fields');
    it('should validate field formats');
  });

  describe('Error Handling', () => {
    it('should handle network errors');
    it('should handle timeout errors');
    it('should handle authentication errors');
    it('should handle rate limit errors');
    it('should log errors appropriately');
  });

  describe('Edge Cases', () => {
    it('should handle empty arrays');
    it('should handle boundary values');
    it('should handle special characters');
    it('should handle concurrent operations');
    it('should handle large datasets');
  });

  describe('Branch Coverage', () => {
    // Test all if/else branches
    // Test all switch cases
    // Test all ternary operators
    // Test all logical operators (&&, ||, ??)
  });

  describe('Security', () => {
    it('should sanitize user input');
    it('should prevent injection attacks');
    it('should validate file paths');
    it('should handle malicious data');
  });

  describe('Performance', () => {
    it('should complete within timeout');
    it('should handle memory efficiently');
    it('should not leak resources');
  });
});
```

---

## Phase 4: Medium Priority Files (Priority: MEDIUM)

### Files 21-50 (Priority Score 100-117)

**Estimated Effort:** ~60 hours

Files include:
- Various service bubbles (sendgrid, hubspot, salesforce, etc.)
- Additional tool bubbles
- Workflow bubbles
- Utility functions

---

## Phase 5: Lower Priority Files (Priority: LOW)

### Files 51-129 (Priority Score <100)

**Estimated Effort:** ~80 hours

Files include:
- Sample flows
- Constants
- Type definitions
- Index files
- Configuration files

---

## Common Test Patterns for Coverage

### 1. Service Bubble Test Pattern

```typescript
describe('ServiceBubbleName', () => {
  let mockCredential: any;
  let mockContext: BubbleContext;

  beforeEach(() => {
    mockCredential = {
      apiKey: 'test-key',
      endpoint: 'https://api.test.com'
    };
    mockContext = {
      logger: createMockLogger(),
      eventBus: createMockEventBus()
    };
    vi.clearAllMocks();
  });

  describe('Authentication', () => {
    it('should authenticate with valid credentials', async () => {
      const bubble = new ServiceBubbleName(mockCredential);
      await bubble.authenticate();
      expect(bubble.isAuthenticated).toBe(true);
    });

    it('should fail with invalid credentials', async () => {
      mockCredential.apiKey = 'invalid';
      const bubble = new ServiceBubbleName(mockCredential);
      await expect(bubble.authenticate()).rejects.toThrow();
    });

    it('should handle authentication timeout', async () => {
      // Mock timeout scenario
    });
  });

  describe('Operations', () => {
    it('should perform primary operation successfully', async () => {
      const bubble = new ServiceBubbleName(mockCredential);
      const result = await bubble.executeOperation({ /* params */ });
      expect(result).toBeDefined();
      expect(result.success).toBe(true);
    });

    it('should retry on transient failure', async () => {
      // Test retry logic
    });

    it('should fail after max retries', async () => {
      // Test retry exhaustion
    });
  });

  describe('Error Scenarios', () => {
    it('should handle network errors', async () => {
      // Mock network failure
    });

    it('should handle rate limiting', async () => {
      // Mock rate limit response
    });

    it('should handle malformed responses', async () => {
      // Test invalid response handling
    });
  });
});
```

### 2. Tool Bubble Test Pattern

```typescript
describe('ToolBubbleName', () => {
  let mockContext: BubbleContext;

  beforeEach(() => {
    mockContext = {
      logger: createMockLogger(),
      workspace: '/tmp/test-workspace'
    };
    vi.clearAllMocks();
  });

  describe('Core Functionality', () => {
    it('should process valid input', async () => {
      const tool = new ToolBubbleName();
      const result = await tool.execute({ input: 'valid' });
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should validate input schema', async () => {
      const tool = new ToolBubbleName();
      await expect(tool.execute({ invalid: 'data' }))
        .rejects.toThrow();
    });
  });

  describe('Data Transformation', () => {
    it('should transform data correctly', async () => {
      // Test transformation logic
    });

    it('should handle null/undefined values', async () => {
      // Test null handling
    });

    it('should handle empty collections', async () => {
      // Test empty array handling
    });
  });
});
```

### 3. Utility Function Test Pattern

```typescript
describe('utilityFunction', () => {
  describe('Happy Path', () => {
    it('should return correct result for valid input', () => {
      const result = utilityFunction('valid-input');
      expect(result).toBe('expected-output');
    });
  });

  describe('Edge Cases', () => {
    const testCases = [
      { input: null, expected: 'default' },
      { input: undefined, expected: 'default' },
      { input: '', expected: 'default' },
      { input: 'boundary-value', expected: 'boundary-result' },
      { input: 'special-chars-!@#$%', expected: 'sanitized' }
    ];

    testCases.forEach(({ input, expected }) => {
      it(`should handle ${JSON.stringify(input)}`, () => {
        const result = utilityFunction(input);
        expect(result).toBe(expected);
      });
    });
  });
});
```

---

## Verification Checklist

For each test file created, verify:

- [ ] All exported functions have tests
- [ ] All exported classes have tests
- [ ] All error paths are tested
- [ ] All if/else branches are covered
- [ ] All switch cases are covered
- [ ] All try/catch blocks are tested
- [ ] Null/undefined inputs tested
- [ ] Empty collections tested
- [ ] Boundary values tested
- [ ] Error scenarios tested
- [ ] Timeout scenarios tested (if applicable)
- [ ] Retry logic tested (if applicable)
- [ ] Cleanup/teardown tested
- [ ] Logging statements verified

---

## Execution Strategy

### Week 1: Fix Foundation
- Day 1-2: Fix failing tests and missing dependencies
- Day 3-4: Run baseline coverage, identify gaps in existing tests
- Day 5: Fill gaps in top 5 existing test files

### Week 2-3: High Priority Files
- Create comprehensive tests for top 20 files
- Estimated: 85 hours of work

### Week 4: Medium Priority Files
- Create tests for files 21-50
- Estimated: 60 hours of work

### Week 5: Lower Priority Files + Validation
- Create tests for remaining files
- Run full coverage reports
- Fix any remaining gaps
- Estimated: 80 hours of work

**Total Estimated Time:** ~225 hours (~6 weeks for 1 developer)

---

## Success Criteria

✅ **100% Line Coverage:** Every line of executable code is tested
✅ **100% Branch Coverage:** Every conditional branch is tested
✅ **100% Function Coverage:** Every function is called in tests
✅ **All Tests Passing:** Zero failing tests
✅ **Test Quality:** Tests are meaningful, not just coverage chasing

---

## Tools and Automation

### Coverage Report Commands

```bash
# Full coverage report
pnpm test:coverage

# Coverage for specific file
pnpm test -- --coverage src/path/to/file.test.ts

# Watch mode with coverage
pnpm test:watch --coverage

# HTML report
pnpm test:coverage && open coverage/index.html
```

### Automated Test Generation

Consider using tools to accelerate:
- `chatgpt-code-review` - Suggests test cases
- `vitest/ui` - Visual coverage inspection
- `c8` - Alternative coverage provider

---

## Conclusion

Achieving 100% test coverage is achievable but requires systematic effort:

1. **Immediate Actions (Week 1):** Fix failing tests, establish baseline
2. **Short-term (Weeks 2-4):** Create tests for high-priority files
3. **Long-term (Week 5-6):** Complete remaining files, verify 100%

**Key Success Factors:**
- Consistent test patterns
- Focus on quality over just coverage numbers
- Regular coverage monitoring
- Automated coverage gates in CI/CD

---

**Next Steps:**
1. Review this plan with team
2. Assign priorities and timelines
3. Set up coverage tracking in CI/CD
4. Begin Phase 1: Fix Failing Tests
