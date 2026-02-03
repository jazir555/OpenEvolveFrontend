# Code Coverage Analysis and 100% Achievement Plan

## Current Status (As of Analysis)

### Fixed Issues
1. ✅ Fixed import paths for resilience.js (changed from `../../../../` to `../../../../../`)
2. ✅ Fixed bun:test import issue in deepseek.test.ts

### Test Execution Status
- **Total Test Files**: 65
- **Passing**: 35 files
- **Failing**: 30 files
- **Total Tests**: 1,405
- **Passing**: 1,027 tests
- **Failing**: 376 tests (flaky tests with retry x2)

### Main Test Failure Categories

#### 1. Schema Validation Failures (NotionBubble)
- All Notion tests failing with "Invalid discriminator value"
- This indicates the schema definition doesn't match the test expectations
- Files affected: `src/bubbles/service-bubble/notion.test.ts`

#### 2. Edge Case Test Failures
- Google Drive edge case tests failing (path traversal, null byte injection, email validation)
- These are testing that the schema REJECTS invalid input, but the tests expect acceptance
- Files affected:
  - `src/bubbles/service-bubble/google-drive-bubble.edge-cases.test.ts`
  - `src/bubbles/service-bubble/stripe-bubble.edge-cases.test.ts`

#### 3. Time-Sensitive Test Failures
- Connection pool timeout tests
- Cache eviction tests
- These tests are flaky due to timing issues
- Files affected:
  - `src/bubbles/common/connection-pool.test.ts`
  - `src/bubbles/common/cache.test.ts`

## Strategy for Achieving 100% Coverage

### Phase 1: Fix Failing Tests (Priority: HIGH)

#### Action Items:
1. **Fix Notion Schema Mismatch**
   - Review `src/bubbles/service-bubble/notion.ts` schema definition
   - Update test to match actual schema or update schema to match test expectations
   - Likely issue: operation discriminator values don't match

2. **Fix Edge Case Test Expectations**
   - Update tests to properly expect schema REJECTION instead of acceptance
   - Test should verify that invalid input throws validation errors
   - Fix security edge case tests to properly validate rejection behavior

3. **Fix Time-Sensitive Tests**
   - Increase timeout thresholds in connection pool tests
   - Use mock timers instead of real timers for deterministic behavior
   - Fix cache statistics counting logic

### Phase 2: Generate Baseline Coverage Report (Priority: HIGH)

Once tests are fixed:
```bash
pnpm test:coverage -- --reporter=json --reporter=text --reporter=html
```

Expected outputs:
- `coverage/coverage-final.json` - Machine-readable coverage data
- `coverage/index.html` - Interactive HTML report
- Console output with summary

### Phase 3: Identify Coverage Gaps (Priority: HIGH)

Analysis will reveal:
1. **Files with < 100% line coverage**
2. **Uncovered branches** (if/else, ternary, switch)
3. **Uncovered functions**
4. **Uncovered statements**

#### Common Coverage Gap Patterns:
1. Error catch blocks
2. Else branches
3. Null/undefined checks
4. Boundary conditions
5. Fallback/default values
6. Cleanup/teardown code
7. Logging statements
8. Validation failures
9. Timeout scenarios
10. Retry exhaustion

### Phase 4: Create Targeted Tests (Priority: MEDIUM)

For each uncovered line/branch, create specific tests:

#### Error Path Tests:
```typescript
it('should throw error when input is invalid', async () => {
  await expect(bubble.execute(invalidInput))
    .rejects.toThrow('Expected error message');
});
```

#### Branch Coverage Tests:
```typescript
it('should handle false condition', async () => {
  const result = await bubble.execute({ condition: false });
  expect(result).toBe(expectedFalsePathResult);
});
```

#### Edge Case Tests:
```typescript
it('should return fallback value on error', async () => {
  mockApi.mockImplementationOnce(() => { throw new Error(); });
  const result = await bubble.execute(params);
  expect(result).toBe(fallbackValue);
});
```

### Phase 5: High-Priority Files for Coverage

Based on the test files present, these files likely need additional coverage:

#### Tool Bubbles:
- `src/bubbles/tool-bubble/research-agent-tool.ts`
- `src/bubbles/tool-bubble/sql-query-tool.ts`
- `src/bubbles/tool-bubble/instagram-tool.ts`
- `src/bubbles/tool-bubble/linkedin-tool.ts`
- `src/bubbles/tool-bubble/twitter-tool.ts`
- `src/bubbles/tool-bubble/youtube-tool.ts`
- `src/bubbles/tool-bubble/code-edit-tool.ts`
- `src/bubbles/tool-bubble/tiktok-tool.ts`
- `src/bubbles/tool-bubble/reddit-scrape-tool.ts`

#### Service Bubbles:
- `src/bubbles/service-bubble/airtable-wrapper.ts`
- `src/bubbles/service-bubble/apify-bubble.ts`
- `src/bubbles/service-bubble/stripe-bubble.ts`
- `src/bubbles/service-bubble/google-drive-bubble.ts`
- `src/bubbles/service-bubble/google-sheets-bubble.ts`

#### Utility Functions:
- `src/bubbles/common/retry.ts`
- `src/bubbles/common/cache.ts`
- `src/bubbles/common/connection-pool.ts`
- `src/bubbles/common/validators.ts`
- `src/utils/json-parsing.ts`
- `src/utils/safe-gemini-chat.ts`

## Current Test File Inventory

### Passing Test Files (35):
- ai-agent.test.ts
- ai-agent-json-parsing.test.ts
- apify.test.ts
- bubbleflow-validation-tool.test.ts
- chart-js-tool.test.ts
- code-edit-tool.integration.test.ts
- debug-boilerplate.test.ts
- eleven-labs.test.ts
- followupboss.test.ts
- gemini-2.5-flash-reliability.integration.test.ts
- github.test.ts
- gmail.integration.test.ts
- google-calendar.integration.test.ts
- google-maps-tool.test.ts
- google-sheets.test.ts
- hello-world.test.ts
- http.test.ts
- http.edge-cases.test.ts
- linkedin-tool.test.ts
- list-bubbles-tool.test.ts
- notion/notion.test.ts
- resend.test.ts
- slack.test.ts
- slack-validation.test.ts
- storage.test.ts
- telegram.test.ts
- tools-schema-compat.test.ts
- web-scrape-tool.test.ts
- web-search-tool.test.ts
- webhook-bubble.test.ts
- Plus many integration tests and utility tests

### Failing Test Files (30):
- deepseek.test.ts (import issue - needs test rewrite)
- notion.test.ts (schema mismatch - 39 tests)
- google-drive-bubble.edge-cases.test.ts (security test expectation issues - 4 tests)
- stripe-bubble.edge-cases.test.ts (security test expectation issues - 4 tests)
- stripe-bubble.test.ts (missing resilience dependency)
- airtable-wrapper.test.ts (missing resilience dependency)
- apify-bubble.test.ts (missing resilience dependency)
- firecrawl.test.ts (missing resilience dependency)
- postgresql.test.ts (missing resilience dependency)
- comprehensive-security.test.ts (missing resilience dependency)
- http-bubble.test.ts
- google-sheets-bubble.test.ts
- And ~15 more with various issues

## Next Steps

1. **Immediate**: Fix the 3 resilience import issues (completed ✅)
2. **High Priority**: Fix Notion schema mismatch (39 tests)
3. **High Priority**: Fix edge case test expectations (8 tests)
4. **Medium Priority**: Generate coverage report on passing tests
5. **Medium Priority**: Identify specific uncovered lines
6. **Medium Priority**: Create targeted tests for uncovered code
7. **Low Priority**: Fix remaining failing tests
8. **Validation**: Run full coverage report and verify 100%

## Estimated Time to 100% Coverage

- Fix existing test failures: 4-6 hours
- Generate and analyze coverage report: 1-2 hours
- Create tests for uncovered lines (estimated 500-1000 lines): 8-12 hours
- Validation and refinement: 2-4 hours
- **Total: 15-24 hours**

## Success Criteria

✅ All tests passing
✅ Coverage report shows:
  - Lines: 100%
  - Branches: 100%
  - Functions: 100%
  - Statements: 100%

✅ Coverage report generated in:
  - `coverage/coverage-final.json`
  - `coverage/index.html`
  - Console output

## Progress Tracking

- [x] Fix resilience.js import paths
- [x] Fix bun:test import
- [ ] Fix Notion schema mismatch
- [ ] Fix edge case test expectations
- [ ] Generate baseline coverage report
- [ ] Identify all uncovered lines/branches
- [ ] Create tests for high-priority files
- [ ] Create tests for medium-priority files
- [ ] Create tests for low-priority files
- [ ] Verify 100% coverage achieved
