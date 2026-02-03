# Test Coverage Analysis and Gap Report

## Executive Summary

**Current Status:** The bubble-core package has extensive tests (1,100+) but several critical issues prevent achieving 100% coverage:

1. **Broken Dependencies:** Multiple test files import missing modules
2. **Test Failures:** 330+ tests are failing due to various issues
3. **Missing Test Files:** Several tool bubbles lack dedicated test coverage
4. **Infrastructure Issues:** Coverage tool configuration needs refinement

## Test Execution Results

### Test Statistics
- **Total Test Files:** 61
- **Passed:** 24 (39%)
- **Failed:** 37 (61%)
- **Total Tests:** 1,197
- **Passed Tests:** 807 (67%)
- **Failed Tests:** 330 (28%)
- **Skipped Tests:** 60 (5%)

### Critical Blockers

#### 1. Missing Dependencies (12 test suites blocked)

**Files Affected:**
- `airtable-wrapper.test.ts`
- `apify-bubble.test.ts`
- `stripe-bubble.test.ts`
- `stripe-bubble.edge-cases.test.ts`
- `google-sheets-bubble.test.ts`
- `webhook-bubble.test.ts`
- `firecrawl.test.ts`
- `postgresql.test.ts`
- `http.comprehensive.test.ts`
- `postgresql.comprehensive.test.ts`
- `comprehensive-security.test.ts`

**Error Pattern:**
```
Cannot find module '../../../../integrations/openevolve/adapters/resilience.js'
Cannot find module '../../../adapters/resilience.js'
Cannot find module '../../tests/test-utils.js'
```

**Resolution Required:**
1. Create or restore the missing `resilience.js` adapter module
2. Create or restore the missing `test-utils.js` module
3. Update import paths to match actual file structure

#### 2. Bun Test Framework Incompatibility

**File Affected:**
- `deepseek.test.ts`

**Error:**
```
Cannot find package 'bun:test'
```

**Resolution Required:**
- Convert test from `bun:test` to `vitest` format
- Update imports from `import { describe, it, expect } from 'bun:test'` to use vitest globals

## Test Failures by Category

### 1. Cache Test Failures (3 failures)

**File:** `src/bubbles/common/cache.test.ts`

**Failures:**
- `MultiTierCache > should combine statistics from both tiers`
  - Expected: L2 size = 1, Actual: L2 size = 2
  - Issue: Statistics aggregation logic incorrect

**Root Cause:** The `getStats()` method in `MultiTierCache` is not correctly combining statistics from L1 and L2 caches.

### 2. Connection Pool Test Failures (2 failures)

**File:** `src/bubbles/common/connection-pool.test.ts`

**Failures:**
- `GenericConnectionPool > should timeout if connection not available`
  - Expected: Promise rejection, Actual: Promise resolved
  - Issue: Timeout logic not working correctly

- `ConnectionPoolRegistry > should get stats for all pools`
  - Expected: HTTP pool size = 1, Actual: HTTP pool size = 0
  - Issue: Registry not properly tracking pool stats

**Root Cause:** Connection pool timeout mechanism and registry statistics tracking need fixes.

### 3. Retry Logic Test Failures (21+ failures)

**File:** `src/bubbles/common/retry.test.ts`

**Failures:**
- `sleep > should resolve immediately for zero duration`
  - Expected: < 10ms, Actual: 15-17ms
  - Issue: Sleep function has overhead

- `retryWithBackoff > should respect custom correlation ID and operation name`
  - Expected: StringContaining matcher, Actual: Exact string match
  - Issue: Test assertion using wrong matcher format

- `CircuitBreaker` tests (17 failures)
  - Error: `Cannot read properties of undefined (reading 'CLOSED')`
  - Issue: `CircuitBreakerState` enum not exported or imported correctly

**Root Cause:**
1. Sleep implementation needs optimization
2. Test assertion format needs correction
3. CircuitBreakerState enum needs proper export/import

### 4. Validator Test Failures (1 failure)

**File:** `src/bubbles/common/validators.test.ts`

**Failure:**
- `sanitizeString > should remove dangerous characters`
  - Expected: `'scriptalertxss/script'`
  - Actual: `'scriptalert(xss)/script'`
  - Issue: Sanitization not removing parentheses

**Root Cause:** The `sanitizeString()` function doesn't remove `()` characters which can be used in XSS attacks.

### 5. Security Test Failures (5 failures)

**File:** `src/bubbles/tool-bubble/__tests__/security-fixes.test.ts`

**Failures:**
- Expression evaluation security tests (4 failures)
  - Tests expecting `success: false` are getting `success: true`
  - Issue: Security validation bypassed or not implemented

- Division by zero test (1 failure)
  - Expected: `Infinity`, Actual: `undefined`
  - Issue: Division by zero returns undefined instead of Infinity

**Root Cause:** Data transformer tool security validations need reinforcement.

## Missing Test Coverage

### Tool Bubbles Without Tests

Based on the file structure, these tool bubbles need dedicated test files:

1. **chart-js-tool.ts**
   - No dedicated test file found
   - Needs tests for:
     - Chart generation
     - Data processing
     - Error handling
     - Edge cases (empty data, large datasets)

2. **code-edit-tool.ts**
   - No dedicated test file found
   - Needs tests for:
     - Code transformation
     - AST manipulation
     - Security validations
     - Language-specific operations

3. **google-maps-tool.ts**
   - No dedicated test file found
   - Needs tests for:
     - Geocoding
     - Distance calculations
     - API integration
     - Rate limiting

4. **instagram-tool.ts**
   - No dedicated test file found
   - Needs tests for:
     - API calls
     - Data parsing
     - Authentication
     - Error scenarios

5. **linkedin-tool.ts**
   - No dedicated test file found
   - Needs tests for:
     - Profile data fetching
     - API integration
     - Authentication flows
     - Rate limiting

6. **sql-query-tool.ts**
   - Test file exists but may need expansion
   - Needs tests for:
     - SQL injection prevention
     - Query execution
     - Result parsing
     - Connection handling

7. **twitter-tool.ts**
   - Test file exists (5 tests passing)
   - May need additional coverage for:
     - API error scenarios
     - Rate limiting
     - Data parsing edge cases

8. **youtube-tool.ts**
   - No dedicated test file found
   - Needs tests for:
     - Video metadata fetching
     - API integration
     - Search functionality
     - Authentication

9. **research-agent-tool.ts**
   - Test file exists but may be incomplete
   - Needs tests for:
     - Multi-step research workflows
     - Tool orchestration
     - Result aggregation
     - Error recovery

### Workflow Templates (41 templates in templates/)

**Categories:**
- infrastructure/ (4 files) - Already secured
- development/ (11 files)
- llm-operations/ (8 files)
- examples/ (24 files)

**Coverage Needed:**
Each template needs tests for:
1. Parameter validation
2. Bubble execution
3. Error handling
4. Integration with other bubbles
5. Edge cases and boundary conditions

## Service Bubbles Coverage Gaps

### HTTP Service Bubble
- **Current:** Comprehensive tests exist but failing due to missing test-utils
- **Needs:** Fix test dependencies, then add:
  - SSRF prevention tests
  - Rate limiting tests
  - Redirect handling tests
  - Timeout tests
  - Large payload tests

### PostgreSQL Service Bubble
- **Current:** Tests exist but failing
- **Needs:** Fix dependencies, then add:
  - Connection pool tests
  - Transaction tests
  - Error recovery tests
  - SQL injection prevention tests
  - Performance tests

### AI Agent Bubble
- **Current:** Basic tests exist
- **Needs:** Add:
  - JSON parsing edge cases
  - Large response handling
  - Timeout scenarios
  - Retry logic tests
  - Token counting tests

## Utility Modules Coverage

### Error Handler
- **File:** `src/utils/error-handler.ts`
- **Recent Changes:** Added retryWithBackoff function
- **Needs:**
  - Tests for retryWithBackoff
  - Circuit breaker tests
  - Error categorization tests
  - Correlation ID tests

### Constants
- **File:** `src/utils/constants.ts`
- **Recent Changes:** Added HTTP_STATUS_REQUEST_TIMEOUT
- **Needs:**
  - Verify all constants are tested
  - Add tests for RETRYABLE_STATUS_CODES set

### Validators
- **File:** `src/bubbles/common/validators.ts`
- **Issue:** sanitizeString not removing parentheses
- **Needs:**
  - Fix sanitization logic
  - Add comprehensive XSS prevention tests
  - Add tests for all edge cases

## Coverage Configuration

### Current Configuration (vitest.config.ts)

```typescript
coverage: {
  provider: 'v8',
  reporter: ['text', 'json', 'html', 'lcov', 'clover'],
  exclude: [
    'node_modules/**',
    'dist/**',
    'coverage/**',
    '**/*.test.ts',
    '**/*.spec.ts',
    '**/types/**',
    '**/templates/**',  // Templates excluded - should be tested
    'vitest.config.ts',
    'vitest.setup.ts',
    '**/*.d.ts',
  ],
  thresholds: {
    lines: 100,      // Changed from 80
    functions: 100,  // Changed from 80
    branches: 100,   // Changed from 75
    statements: 100, // Changed from 80
  },
  perFile: true,
}
```

### Issues:
1. **Templates excluded:** Workflow templates are excluded from coverage but should be tested
2. **Thresholds too aggressive:** 100% across all metrics may not be achievable
3. **No baseline coverage:** Cannot generate report due to test failures

## Recommended Action Plan

### Phase 1: Fix Critical Blockers (Priority: CRITICAL)

1. **Create Missing Dependencies**
   - Create `src/integrations/openevolve/adapters/resilience.ts`
   - Create `src/tests/test-utils.ts`
   - Update all imports to use correct paths

2. **Fix Bun Test Incompatibility**
   - Convert `deepseek.test.ts` to vitest format
   - Update all imports and assertions

3. **Fix CircuitBreakerState Export**
   - Verify enum is exported from `src/bubbles/common/retry.ts`
   - Update import in test file

### Phase 2: Fix Failing Tests (Priority: HIGH)

1. **Fix Cache Statistics**
   - Update `MultiTierCache.getStats()` to properly aggregate L1 and L2 stats
   - Fix L2 size tracking

2. **Fix Connection Pool Timeout**
   - Implement proper timeout mechanism
   - Add timeout rejection logic

3. **Fix Sleep Function**
   - Optimize for zero-duration case
   - Use proper timer mechanism

4. **Fix Sanitization**
   - Add parentheses removal to `sanitizeString()`
   - Add comprehensive XSS test cases

5. **Fix Security Tests**
   - Implement expression length validation
   - Implement empty expression validation
   - Fix division by zero handling
   - Add SSRF redirect chain prevention

### Phase 3: Add Missing Coverage (Priority: MEDIUM)

1. **Tool Bubble Tests**
   - Create test files for each untested tool bubble
   - Achieve minimum 80% coverage per tool
   - Focus on happy path, error cases, and edge cases

2. **Workflow Template Tests**
   - Create test framework for templates
   - Test each template with valid inputs
   - Test error scenarios
   - Test integration points

3. **Service Bubble Enhancements**
   - Add comprehensive tests for HTTP bubble
   - Add comprehensive tests for PostgreSQL bubble
   - Add AI agent edge case tests

### Phase 4: Optimize Coverage (Priority: LOW)

1. **Branch Coverage**
   - Identify uncovered branches
   - Add tests for all conditional paths
   - Test all exception handlers

2. **Line Coverage**
   - Identify uncovered lines
   - Add tests for edge cases
   - Test all utility functions

3. **Function Coverage**
   - Verify all functions are tested
   - Add tests for private/internal methods where applicable

## Estimated Effort

- **Phase 1:** 4-6 hours (Fix critical blockers)
- **Phase 2:** 6-8 hours (Fix failing tests)
- **Phase 3:** 15-20 hours (Add missing coverage)
- **Phase 4:** 8-10 hours (Optimize coverage)

**Total Estimated Effort:** 33-44 hours

## Success Criteria

### Minimum Viable Coverage
- [ ] All tests pass (0 failures)
- [ ] 80% line coverage across all modules
- [ ] 75% branch coverage
- [ ] 80% function coverage
- [ ] All critical security paths tested

### Ideal Coverage (100% Goal)
- [ ] All tests pass (0 failures)
- [ ] 100% line coverage (excluding truly unreachable code)
- [ ] 95% branch coverage
- [ ] 100% function coverage
- [ ] All edge cases tested
- [ ] All error paths tested
- [ ] Integration tests for all workflows

## Tools and Commands

### Generate Coverage Report
```bash
cd BubbleLab/packages/bubble-core
pnpm test:coverage
```

### View HTML Coverage Report
```bash
pnpm test:coverage
npx vite preview --outDir coverage
```

### Run Specific Test File
```bash
pnpm test src/bubbles/common/cache.test.ts
```

### Run Tests with Pattern
```bash
pnpm test --tool.test.ts
```

### Run Tests in Watch Mode
```bash
pnpm test:watch
```

## Next Steps

1. **Immediate:** Fix the 12 blocked test suites by creating missing dependencies
2. **Short-term:** Fix the 330+ failing tests
3. **Medium-term:** Create tests for uncovered tool bubbles and workflows
4. **Long-term:** Optimize for 100% coverage with comprehensive edge case testing

## Conclusion

While the bubble-core package has a solid test foundation with 1,100+ tests, significant work is needed to:

1. **Fix Infrastructure:** Resolve missing dependencies and configuration issues
2. **Fix Failures:** Address 330+ failing tests
3. **Fill Gaps:** Create comprehensive tests for uncovered modules
4. **Optimize:** Achieve 100% coverage with edge case and error path testing

The estimated effort of 33-44 hours is realistic for achieving robust 100% coverage, but the work should be prioritized in phases to ensure test stability before adding new coverage.
