# ROMA Test Fixes - Quick Reference Guide

## Summary

**User Selected:** Option A (Fix Tests to Match Implementation)
**Approach:** Update test expectations to match actual service APIs
**Estimated Time:** 1-2 hours
**Risk:** Low (only tests are modified)

---

## Already Fixed ✅

### 1. Validation Contract Tests (4 tests)

**Issue:** Tests expected `{isValid, errors}` object but implementation returns `boolean`

**Fix Applied:** Updated all validation tests to expect boolean returns

**File:** `src/tests/contract/roma-service.test.ts:244-300`

**Changes:**
```typescript
// Before:
const validation = service.validateExecutionResult(result);
expect(validation.isValid).toBe(true);
expect(validation.errors).toHaveLength(0);

// After:
const isValid = service.validateExecutionResult(result);
expect(isValid).toBe(true);
```

---

## Remaining Fixes Needed (13 tests)

### Category 1: Performance Analysis Tests (5 tests)

**Issue:** Tests pass `RomaExecutionResult` object but method expects `executionId: string`

**Method Signature:**
```typescript
public async analyzeExecutionPerformance(executionId: string): Promise<Record<string, any>>
```

**Tests to Fix (lines 298-410):**

1. `should calculate execution time` (line 299)
2. `should calculate completion rate` (line 326)
3. `should calculate efficiency score` (line 344)
4. `should track tool usage frequency` (line 363)
5. `should track module usage` (line 386)

**Fix Pattern:**
```typescript
// Before:
it('should calculate execution time', () => {
  const result: RomaExecutionResult = { ... };
  const analysis = service.analyzeExecutionPerformance(result);
  expect(analysis.totalExecutionTime).toBe(5000);
});

// After:
it('should calculate execution time', async () => {
  // Mock getExecution to return expected data
  (mockClient.getExecution as any).mockResolvedValue({
    executionId: 'roma-123',
    goal: 'Test goal',
    status: 'completed',
    statistics: {
      executionTime: 5000,
      subtasksCreated: 5,
      subtasksCompleted: 5,
      toolsUsed: ['search', 'calculator'],
      modulesUsed: ['atomizer', 'planner'],
    },
  });

  const analysis = await service.analyzeExecutionPerformance('roma-123');
  expect(analysis.totalTime).toBe(5000);  // Note: property name changed
  expect(analysis.averageSubtaskTime).toBe(1000);  // Note: property name changed
});
```

**Property Name Changes:**
- `totalExecutionTime` → `totalTime`
- `averageTimePerTask` → `averageSubtaskTime`
- `completionRate` → string format like "80.0%" (not 0.8)
- `efficiencyScore` → string format like "0.983" (not number)
- `toolsUsed` → count (number)
- `modulesUsed` → count (number)

---

### Category 2: Caching Contract Tests (3 tests)

**Issue A:** Timestamp format mismatch (number vs ISO string)

**Tests Affected:**
- `should cache execution results` (line ~160)
- `should be safe to call multiple times with same input` (line ~450)

**Fix:** Ensure mock data uses consistent timestamp format

```typescript
// Use this mock data consistently:
const mockExecutionResult = {
  executionId: 'roma-123',
  goal: 'Test goal',
  status: 'completed' as const,
  result: {
    summary: 'Test result',
    reasoning: 'Test reasoning',
  },
  statistics: {
    totalTasks: 1,
    completedTasks: 1,
    executionTime: 1000,
    averageTimePerTask: 1000,
    toolUsage: {},
    moduleUsage: {},
  },
  timestamp: '2026-02-22T00:00:00.000Z',  // Use ISO string consistently
};
```

**Issue B:** Cache TTL timing test

**Test:** `should respect cache TTL` (line ~170)

**Current Issue:** Cache doesn't expire in test timeframe

**Fix:** Increase test timeout or use jest.useFakeTimers()

```typescript
it('should respect cache TTL', async () => {
  // Use fake timers to control time
  vi.useFakeTimers();

  service.setCacheTTL(100);  // 100ms TTL
  (mockClient.executeTask as any).mockResolvedValue(mockExecutionResult);

  await service.executeTaskWithCache('Test', {});

  // Fast-forward past TTL
  vi.advanceTimersByTime(150);

  await service.executeTaskWithCache('Test', {});

  expect(mockClient.executeTask).toHaveBeenCalledTimes(2);

  vi.useRealTimers();
});
```

**Issue C:** Cache statistics structure

**Test:** `should return cache statistics` (line ~190)

**Expected:**
```typescript
{
  size: number;      // Number of cached items
  hitRate: number;   // Cache hit rate
}
```

**Check actual implementation:** `RomaService.ts:getCacheStatistics()`

---

### Category 3: Retry Logic Tests (2 tests)

**Issue A:** Exponential backoff not working

**Test:** `should use exponential backoff between retries` (line ~80)

**Current:** Delay is 2000ms (linear)
**Expected:** Delay should be 100ms (exponential)

**Fix:** Check retry implementation in `RomaService.ts:executeTaskWithRetry()`

```typescript
// Expected implementation:
const delay = Math.min(100 * Math.pow(2, attempt - 1), maxDelay);
// attempt=1: 100ms
// attempt=2: 200ms
// attempt=3: 400ms
```

**Issue B:** Error message validation

**Test:** `should not retry on validation errors` (line ~95)

**Current:** Error message is "Invalid goal"
**Expected:** Error message should include "Invalid error"

**Fix:** Update test expectation or fix error message in service

```typescript
// Option 1: Fix test
expect(async () => await service.executeTaskWithRetry('Invalid', {}))
  .toThrowError(/Invalid goal/);  // Match actual error

// Option 2: Fix service (if desired)
throw new Error(`Invalid error: ${message}`);
```

---

### Category 4: Idempotency Contract (1 test)

**Issue:** Timestamp format mismatch

**Test:** `should be safe to call multiple times with same input` (line ~450)

**Fix:** Same as Caching Contract - ensure consistent ISO string timestamps

---

### Category 5: Execution Plan Tests (2 tests)

**Issue:** Subtask structure mismatch

**Tests:**
- `should retrieve execution plan` (line 413)
- `should handle execution with no sub-tasks` (line 433)

**Problem:** Tests expect `subTasks` but mock data or implementation doesn't provide it

**Fix:** Check if `getExecutionPlan()` method exists and what it returns

```typescript
// Check implementation first:
// Does this method exist?
public async getExecutionPlan(executionId: string): Promise<ExecutionPlan>

// What does ExecutionPlan interface look like?
interface ExecutionPlan {
  executionId: string;
  goal: string;
  subTasks?: SubTask[];
  // ... other fields
}
```

---

### Category 6: Subtask Operations Tests (3 tests)

**Tests:**
- `should update subtask status` (line ~450)
- `should get subtasks by status` (line ~470)
- `should cancel subtask` (line ~490)

**Issue:** Methods may not exist in implementation

**Check if these methods exist:**
```typescript
updateSubtaskStatus(executionId: string, subtaskId: string, status: string): Promise<void>
getSubtasksByStatus(executionId: string, status: string): Promise<Subtask[]>
cancelSubtask(executionId: string, subtaskId: string): Promise<void>
```

**If methods don't exist:** Remove these tests (they're testing non-existent functionality)

**If methods exist:** Fix test expectations to match actual signatures

---

## Quick Fix Commands

### Fix 1: Apply Validation Contract Fixes (Already Done)

```bash
# Changes already applied to:
# src/tests/contract/roma-service.test.ts (lines 244-300)
```

### Fix 2: Run Tests to See Current State

```bash
cd glue/adapters/roma/roma-bubblelab-plugin
npm test -- --run
```

### Fix 3: Apply Remaining Fixes (Manual or Script)

For manual fixes:
1. Open `src/tests/contract/roma-service.test.ts`
2. Go to each section listed above
3. Apply the fix patterns
4. Re-run tests after each category

---

## Priority Order

### High Priority (Breaking Tests)

1. ✅ **Validation Contract** - FIXED
2. **Performance Analysis** - Fix method calls and property names
3. **Execution Plan** - Check if methods exist

### Medium Priority (Functional Issues)

4. **Retry Logic** - Fix backoff calculation
5. **Cache Statistics** - Fix return structure

### Low Priority (Cosmetic Issues)

6. **Timestamp Format** - Standardize on ISO strings
7. **Cache TTL** - Use fake timers

---

## Test Execution After Fixes

```bash
# Run all tests
npm test -- --run

# Run with coverage
npm test -- --run --coverage

# Run specific test file
npm test -- roma-service.test.ts --run

# Run with verbose output
npm test -- --run --reporter=verbose
```

---

## Expected Final Results

**Before Fixes:** 6/46 passed (13%)
**After All Fixes:** 46/46 passed (100%)

**Target:** All tests passing
**Coverage:** 85% target
**Duration:** < 3 seconds

---

## Need Help?

**For each test failure:**
1. Check method signature in `RomaService.ts`
2. Check return type in `RomaService.ts`
3. Update test to match actual implementation
4. Re-run test to verify fix

**Example Checklist:**
- [ ] Fix Performance Analysis tests (5 tests)
- [ ] Fix Caching tests (3 tests)
- [ ] Fix Retry Logic tests (2 tests)
- [ ] Fix Idempotency test (1 test)
- [ ] Fix Execution Plan tests (2 tests)
- [ ] Fix Subtask Operations tests (3 tests)

**Total:** 16 remaining tests to fix

---

**Last Updated:** 2026-02-22
**Status:** Validation Contract Fixed ✅
**Next:** Performance Analysis Contract
