# Quick Fix Guide for Test Suite

This guide provides step-by-step instructions to fix all test compilation errors.

## Prerequisites

```bash
# Ensure you're in the root directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Install dependencies (if not already done)
npm install --legacy-peer-deps
```

## Fix #1: Vitest Imports (5 minutes)

### Files to Fix:
1. `glue/adapters/ragbits/bubblelabs-ragbits-plugin/src/tests/contract/ragbits-api.test.ts`
2. `glue/adapters/bubblelab/src/tests/contract/openevolve-api.test.ts`
3. `glue/adapters/bubblelab/src/tests/contract/workflow-orchestrator.test.ts`

### Change:
```typescript
// BEFORE
import { describe, it, expect, beforeAll } from 'vitest';

// AFTER (Option 1: Use Jest globals - recommended)
// Just remove the import entirely - Jest provides these globally

// OR (Option 2: Explicit Jest import)
import { describe, it, expect, beforeAll } from '@jest/globals';
```

### Quick Fix Command:
```bash
# Remove vitest imports
sed -i "s/import { describe, it, expect, beforeAll } from 'vitest';//g" \
  glue/adapters/ragbits/bubblelabs-ragbits-plugin/src/tests/contract/ragbits-api.test.ts \
  glue/adapters/bubblelab/src/tests/contract/openevolve-api.test.ts \
  glue/adapters/bubblelab/src/tests/contract/workflow-orchestrator.test.ts
```

## Fix #2: LoongFlowSolution Mock Data (30 minutes)

### Files to Fix:
- `glue/schemas/__tests__/loongflow-schemas.test.ts`
- `glue/schemas/__tests__/hybrid-schemas.test.ts`
- `tests/test_hybrid_pes_evolution_e2e.test.ts`

### Add Missing Properties:
```typescript
// BEFORE
{
  solution: 'def solution1(x): return x * 2',
  solution_id: 'sol_1',
  generate_plan: 'Strategy 1',
  score: 0.8,
  evaluation: 'Good',
  summary: 'Test',
  parent_id: '',
  island_id: 0,
  iteration: 1,
  metadata: {}
}

// AFTER
{
  solution: 'def solution1(x): return x * 2',
  solution_id: 'sol_1',
  generate_plan: 'Strategy 1',
  score: 0.8,
  evaluation: 'Good',
  summary: 'Test',
  parent_id: '',
  island_id: 0,
  iteration: 1,
  timestamp: Date.now(),        // ← ADD THIS
  generation: 1,                 // ← ADD THIS
  sample_cnt: 1,                 // ← ADD THIS
  sample_weight: 1.0,            // ← ADD THIS
  metadata: {}
}
```

### Search Pattern:
```bash
# Find all occurrences
grep -rn "solution_id.*sol_1" glue/ tests/ --include="*.test.ts"
```

## Fix #3: LLMConfig Provider Property (10 minutes)

### Files to Fix:
- `glue/schemas/__tests__/loongflow-schemas.test.ts`
- `glue/schemas/__tests__/hybrid-schemas.test.ts`

### Change:
```typescript
// BEFORE
provider: 'openai',

// AFTER
model_provider: 'openai',  // OR just remove this line if not required
```

### Quick Fix Command:
```bash
# Replace provider with model_provider
sed -i 's/provider:.*openai/model_provider: "openai"/g' \
  glue/schemas/__tests__/loongflow-schemas.test.ts \
  glue/schemas/__tests__/hybrid-schemas.test.ts

sed -i 's/provider:.*anthropic/model_provider: "anthropic"/g' \
  glue/schemas/__tests__/loongflow-schemas.test.ts \
  glue/schemas/__tests__/hybrid-schemas.test.ts
```

## Fix #4: InMemoryEventBus Interface (45 minutes)

### Option A: Update InMemoryEventBus (Recommended)

File: `glue/orchestration/event-bus.ts` or wherever InMemoryEventBus is defined

```typescript
export class InMemoryEventBus implements EventBus {
  // Add missing properties
  config: EventBusConfig;
  subscriptions: Map<string, Set<EventHandler>>;
  stats: EventBusStats;
  startTime: number;

  // ... implement all EventBus methods
  constructor() {
    this.config = { /* default config */ };
    this.subscriptions = new Map();
    this.stats = { /* default stats */ };
    this.startTime = Date.now();
  }

  // Implement required methods
  async publish(event: Event): Promise<void> { /* ... */ }
  subscribe(eventType: string, handler: EventHandler): void { /* ... */ }
  unsubscribe(eventType: string, handler: EventHandler): void { /* ... */ }
  // ... etc
}
```

### Option B: Update Workflow Configs (Alternative)

Update workflow files to accept simplified event bus:

```typescript
// In workflow files
export interface PESEvolutionWorkflowConfig {
  loongFlowAdapter: LoongFlowAdapter;
  openEvolveAdapter: OpenEvolveAdapter;
  eventBus?: InMemoryEventBus;  // Use concrete type instead of EventBus
}
```

### Option C: Create Mock EventBus (Quick Fix)

File: `tests/mocks/event-bus-mock.ts`

```typescript
export class MockEventBus implements EventBus {
  config = { maxRetries: 3, timeout: 5000 };
  subscriptions = new Map();
  stats = { published: 0, failed: 0 };
  startTime = Date.now();

  async publish(_event: any) { this.stats.published++; }
  subscribe(_eventType: string, _handler: any) { /* no-op */ }
  unsubscribe(_eventType: string, _handler: any) { /* no-op */ }
  // ... implement other required methods as no-ops
}

// Use in tests:
import { MockEventBus } from './mocks/event-bus-mock';
const eventBus = new MockEventBus();
```

## Fix #5: DeadLetterQueue Constructor (10 minutes)

File: `tests/test_hybrid_pes_evolution_e2e.test.ts`

### Option A: Fix Constructor Call
```typescript
// Check DeadLetterQueue constructor signature
// If it's: constructor(config: RetryPolicy)
dlq = new DeadLetterQueue({
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 10000
});
```

### Option B: Remove DeadLetterQueue
```typescript
// If not actually used in tests, just remove it
// Remove these lines:
let dlq: DeadLetterQueue;
// ...
dlq = new DeadLetterQueue(eventBus);
```

## Fix #6: Remove Unused Imports (10 minutes)

### Auto-fix with ESLint:
```bash
npm run lint -- --fix
```

### Manual cleanup:
```typescript
// Remove these unused imports from tests/test_hybrid_pes_evolution_e2e.test.ts:
import { MultiStageReasoningWorkflow } from '../glue/orchestration/workflows/multi-stage-reasoning-workflow';
import { HybridTask, EvolutionConfig, AdaptiveTrigger } from '../glue/schemas/hybrid-pes-evolution-canonical';
// ... and others marked as unused
```

## Verify Fixes

After applying all fixes:

```bash
# Run tests
npm test

# Expected output:
# Test Suites: 28 passed, 28 total
# Tests:       XXX passed, XXX total
```

## If Tests Still Fail

### Check TypeScript Compilation:
```bash
npm run typecheck
```

### Check Individual Test Files:
```bash
# Run specific test
npm test -- glue/schemas/__tests__/pes-schemas.test.ts

# Run with verbose output
npm test -- --verbose
```

### Debug Mode:
```bash
# Run with node debugger
node --inspect-brk node_modules/.bin/jest --runInBand
```

## Order of Fixes

Recommended order (easiest to hardest):

1. ✅ Fix Vitest imports (5 min)
2. ✅ Remove unused imports (10 min)
3. ✅ Fix LLMConfig provider (10 min)
4. ⚠️ Fix DeadLetterQueue (10 min)
5. ⚠️ Fix LoongFlowSolution mock data (30 min)
6. ❌ Fix InMemoryEventBus interface (45 min)

**Total Time: ~2 hours**

## Common Pitfalls

### Don't Do This:
❌ Modify schema files to match tests
❌ Remove type checking
❌ Use `@ts-ignore` without understanding the issue
❌ Skip fixing the EventBus interface

### Do This Instead:
✅ Fix test data to match schemas
✅ Implement proper interfaces
✅ Fix underlying issues, not symptoms
✅ Run typecheck before committing

## Getting Help

If you're stuck:

1. Check error messages carefully
2. Look at the schema definitions
3. Check similar working tests
4. Review Federation Constitution for patterns

## Success Criteria

When all tests pass:
```bash
$ npm test

Test Suites: 28 passed, 28 total
Tests:       XXX passed, XXX total
Snapshots:   0 total
Time:        XX s
```

Coverage report should show:
```
--------------------|---------|----------|---------|---------|
File                | % Stmts | % Branch | % Funcs | % Lines |
--------------------|---------|----------|---------|---------|
All files           |    60+  |     60+  |    60+  |    60+  |
--------------------|---------|----------|---------|---------|
```

---

**Estimated Total Time:** 2 hours
**Difficulty:** Medium
**Impact:** High (enables all 28 test suites to run)
