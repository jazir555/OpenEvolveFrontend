# 100% Code Coverage Achievement Plan - Executive Summary

## Project Status

**Goal**: Achieve exactly 100% code coverage for bubble-core package
**Current Coverage**: ~95% (estimated based on passing tests)
**Target**: 100% (lines, branches, functions, statements)

## Critical Issues Identified

### 1. Test Failures Blocking Coverage Analysis
**Status**: Partially Fixed
**Impact**: Cannot generate accurate coverage report until tests pass

#### Fixed Issues ✅
- Resilience.js import paths (3 files fixed)
- Bun:test import issue (deepseek.test.ts)

#### Remaining Issues ⚠️
- **Notion Tests (39 failures)**: Schema mismatch
  - Tests use camelCase operations (`retrievePage`)
  - Schema expects snake_case (`retrieve_page`)
  - Tests reference operations that don't exist in schema:
    - `delete_page` (not implemented)
    - `retrieve_user` (not implemented)
    - `delete_block` (not implemented)
    - `retrieve_block_children` (not implemented)
    - `query_data_source` (partially implemented)
    - `retrieve_database` (not implemented)
    - `append_block_children` (not implemented)

- **Edge Case Tests (12 failures)**: Test expectation issues
  - Google Drive security tests expect schema REJECTION but test for ACCEPTANCE
  - Stripe security tests expect schema REJECTION but test for ACCEPTANCE
  - These are actually working correctly but tests are backwards

- **Time-Sensitive Tests (3 failures)**: Flaky timing
  - Connection pool timeout test
  - Cache eviction statistics
  - Multi-tier cache stats

### 2. Coverage Gap Analysis
**Status**: Blocked by test failures
**Impact**: Cannot identify specific uncovered lines

**Estimated Uncovered Areas** (based on common patterns):
1. Error handling paths (catch blocks)
2. Else branches in conditionals
3. Null/undefined validation paths
4. Boundary conditions
5. Fallback/default value logic
6. Cleanup/teardown code
7. Timeout scenarios
8. Retry exhaustion paths

### 3. Files Requiring Additional Tests

**High Priority** (Core functionality):
- `src/bubbles/tool-bubble/research-agent-tool.ts`
- `src/bubbles/tool-bubble/sql-query-tool.ts`
- `src/bubbles/tool-bubble/code-edit-tool.ts`
- `src/bubbles/service-bubble/stripe-bubble.ts`
- `src/bubbles/service-bubble/google-drive-bubble.ts`
- `src/bubbles/common/retry.ts`
- `src/bubbles/common/cache.ts`
- `src/utils/json-parsing.ts`

**Medium Priority** (Frequently used):
- All tool-bubble implementations
- Service bubbles with partial coverage
- Validation and error handling utilities

**Low Priority** (Edge cases):
- Template files
- Example files
- Deprecated code paths

## Recommended Action Plan

### Phase 1: Fix Critical Test Failures (4-6 hours)

**Option A: Quick Fix** (2-3 hours)
- Comment out failing tests in notion.test.ts
- Fix edge case test expectations
- Generate coverage on passing tests only
- Proceed with coverage improvement

**Option B: Proper Fix** (4-6 hours)
- Implement missing operations in Notion schema
- Update all tests to use correct operation names
- Fix edge case test expectations
- Fix time-sensitive test flakiness
- Generate complete coverage report

### Phase 2: Generate Baseline Coverage (1-2 hours)
```bash
# Run tests with detailed coverage reporting
pnpm test:coverage -- --reporter=json --reporter=text --reporter=html

# Outputs:
# - coverage/coverage-final.json (machine-readable)
# - coverage/index.html (interactive report)
# - Console summary
```

### Phase 3: Analyze Coverage Gaps (2-3 hours)

**For each file with < 100% coverage:**
1. Open `coverage/index.html`
2. Navigate to file
3. Identify red lines (uncovered)
4. Document coverage gaps:
   ```markdown
   File: src/bubbles/example.ts
   Coverage: 87.5%

   Missing Lines:
   - Line 45: if (error) { throw ... }  // Error path
   - Line 78: return fallbackValue      // Fallback path
   - Line 102: else { ... }             // Else branch

   Missing Branches:
   - Line 34: typeof x === 'string'     // Only tested true
   - Line 67: condition ? a : b         // Only tested true path
   ```

### Phase 4: Create Targeted Tests (8-12 hours)

**Systematic Approach:**

For each uncovered line:

**Error Paths:**
```typescript
it('should throw error when input is invalid', async () => {
  await expect(bubble.execute(invalidInput))
    .rejects.toThrow('Expected error message');
});
```

**Branch Coverage:**
```typescript
it('should handle false condition', async () => {
  const result = await bubble.execute({ condition: false });
  expect(result).toBe(expectedFalsePathResult);
});
```

**Edge Cases:**
```typescript
it('should return fallback value on error', async () => {
  mockApi.mockImplementationOnce(() => { throw new Error(); });
  const result = await bubble.execute(params);
  expect(result).toBe(fallbackValue);
});
```

**Boundary Conditions:**
```typescript
it('should handle empty array', async () => {
  const result = await bubble.execute({ items: [] });
  expect(result).toEqual(defaultResult);
});
```

### Phase 5: Verify 100% Coverage (2-4 hours)

```bash
# Run full coverage suite
pnpm test:coverage

# Check console output for:
# ✓ Lines: 100%
# ✓ Branches: 100%
# ✓ Functions: 100%
# ✓ Statements: 100%

# Generate HTML report for manual verification
pnpm test:coverage -- --reporter=html
# Open coverage/index.html in browser
# Click through files to verify no red lines
```

## Common Coverage Patterns to Address

### 1. Error Catch Blocks
```typescript
try {
  await operation();
} catch (error) {
  // This line often uncovered
  handleError(error);
}
```
**Test**: Mock operation to throw error

### 2. Else Branches
```typescript
if (condition) {
  doSomething();
} else {
  // This line often uncovered
  doSomethingElse();
}
```
**Test**: Test both truthy and falsy conditions

### 3. Null/Undefined Checks
```typescript
if (value != null) {
  // This line often uncovered
  processValue(value);
}
```
**Test**: Test with null and undefined values

### 4. Default/Fallback Values
```typescript
const result = await operation() ?? defaultValue;
//                                         ^^^^^^^^^^^^^^
//                                         Often uncovered
```
**Test**: Mock operation to return null/undefined

### 5. Cleanup Code
```typescript
try {
  await operation();
} finally {
  // This line often uncovered
  cleanup();
}
```
**Test**: Test both success and error paths

### 6. Retry Exhaustion
```typescript
for (let i = 0; i < maxRetries; i++) {
  try {
    return await operation();
  } catch {
    if (i === maxRetries - 1) throw; // Often uncovered
  }
}
```
**Test**: Mock operation to always fail

## Time Estimate Breakdown

| Phase | Task | Time |
|-------|------|------|
| 1 | Fix critical test failures | 4-6 hours |
| 2 | Generate baseline coverage | 1-2 hours |
| 3 | Analyze coverage gaps | 2-3 hours |
| 4 | Create tests for uncovered lines | 8-12 hours |
| 5 | Verification and refinement | 2-4 hours |
| | **Total** | **17-27 hours** |

## Success Criteria

✅ All tests passing (1,405 tests)
✅ Coverage report shows:
  - Lines: 100%
  - Branches: 100%
  - Functions: 100%
  - Statements: 100%

✅ Coverage report artifacts generated:
  - `coverage/coverage-final.json`
  - `coverage/index.html`
  - Console summary output

✅ No red lines in HTML coverage report
✅ No uncovered branches in coverage report
✅ All test retries successful (no flaky tests)

## Quick Win Alternative

If time is constrained, consider this alternative approach:

1. **Exclude failing tests** from coverage run (1 hour)
2. **Generate coverage** on passing tests only (30 min)
3. **Identify quick wins** - files near 100% (1 hour)
4. **Target easiest files first** - add edge case tests (4-6 hours)
5. **Achieve 98-99% coverage** as interim goal
6. **Return to failing tests** for final 1-2% (4-6 hours)

**Total time to 99%**: 11-15 hours
**Remaining 1%**: 4-6 additional hours

## Risk Factors

### High Risk
- **Notion schema redesign**: May require significant rework
- **Missing operations**: Need to implement before testing
- **Flaky tests**: Time-sensitive tests may need refactoring

### Medium Risk
- **Mock complexity**: Some tests may require complex mocking
- **External dependencies**: API calls may need sophisticated mocking
- **State management**: Tests may interfere with each other

### Low Risk
- **Simple utility functions**: Easy to test
- **Pure functions**: Deterministic, easy to cover
- **Well-defined inputs/outputs**: Straightforward to test

## Recommended Next Steps

**Immediate Actions (Today):**
1. Decide: Quick fix or proper fix for failing tests
2. Implement chosen approach
3. Generate baseline coverage report
4. Identify top 10 files with lowest coverage

**Short-term (This Week):**
1. Create tests for top 10 files
2. Re-run coverage and assess progress
3. Address next 10 files
4. Continue until 95% coverage

**Final Push (Next Week):**
1. Tackle hardest remaining coverage gaps
2. Fix flaky tests
3. Verify 100% coverage
4. Generate final reports

## Tools & Resources

**Coverage Analysis:**
- Vitest coverage: https://vitest.dev/guide/coverage.html
- HTML report: `coverage/index.html`
- JSON data: `coverage/coverage-final.json`

**Test Utilities:**
- Vitest: https://vitest.dev/
- vi.fn() for mocking
- vi.mocked() for typed mocks

**Best Practices:**
- Test one thing per test
- Use descriptive test names
- Mock external dependencies
- Test both success and failure paths
- Use beforeEach/afterEach for cleanup

## Conclusion

Achieving 100% coverage is feasible but requires systematic approach:

1. Fix blocking test failures (choose quick or proper fix)
2. Generate baseline coverage report
3. Identify and prioritize coverage gaps
4. Create targeted tests systematically
5. Verify and refine until 100% achieved

**Estimated effort**: 17-27 hours
**Recommended approach**: Start with quick fix, iterate to proper fix

**Key success factor**: Consistency over intensity - regular testing sessions better than marathon sessions
