# BubbleRunner Quick Reference Card

## Fixed: 5 Critical Blocking Gaps

### Property Initialization (Lines 50-59, 75-77)
```typescript
private bubbleFactory: BubbleFactory;
private currentStep: number;
private savedStates: Map<number, any>;

// Constructor
this.currentStep = 0;
this.savedStates = new Map();
this.bubbleFactory = bubbleFactory;
```

---

## New Public API Methods

### runStep(stepId: number)
**Purpose:** Execute a single step from the execution plan

**Usage:**
```typescript
const result = await runner.runStep('setup');
console.log(result.success); // true/false
console.log(result.data); // { stepId: 'setup', completed: true }
```

**Returns:** `Promise<ExecutionResult>`

**Throws:** Error if plan not initialized or step not found

**Side Effects:**
- Updates `currentStep`
- Saves state to `savedStates` Map
- Logs execution progress

---

### resumeFromStep(stepId: number)
**Purpose:** Resume execution from a previously saved state

**Usage:**
```typescript
// First execute a step
await runner.runStep('setup');

// Later resume from it
const result = await runner.resumeFromStep('setup');
console.log(result.data.resumedFrom); // 'setup'
```

**Returns:** `Promise<ExecutionResult>`

**Throws:** Error if no saved state exists for step

**Side Effects:**
- Restores `currentStep`
- Re-executes the step
- Updates saved state

---

### getSavedState(stepId: number)
**Purpose:** Retrieve saved state for a specific step

**Usage:**
```typescript
const state = runner.getSavedState('setup');
console.log(state);
// {
//   stepId: 'setup',
//   currentStep: 0,
//   variables: {...},
//   timestamp: '2026-01-10T...'
// }
```

**Returns:** `any | undefined`

---

### getAllSavedStates()
**Purpose:** Get copy of all saved states

**Usage:**
```typescript
const allStates = runner.getAllSavedStates();
console.log(allStates.size); // Number of saved states
```

**Returns:** `Map<number, any>`

---

### clearSavedStates()
**Purpose:** Reset all saved states and current step

**Usage:**
```typescript
runner.clearSavedStates();
console.log(runner['currentStep']); // 0
console.log(runner['savedStates'].size); // 0
```

**Returns:** `void`

**Side Effects:**
- Clears `savedStates` Map
- Resets `currentStep` to 0

---

## Private Helper Methods

### executeMiniStep(miniStep: MiniStep)
**Purpose:** Execute a single mini-step (bubble operation)

**Operations:**
- `new_bubble` - Bubble instantiation
- `await_action` - Bubble execution

**Usage:** Internal (called by runStep)

---

### saveState(stepId: number)
**Purpose:** Save current execution state

**State Includes:**
- stepId
- currentStep
- All user variables
- Timestamp

**Usage:** Internal (called by runStep)

---

## Execution Flow

### Basic Execution
```
1. Create BubbleRunner
2. Get execution plan: runner.getPlan()
3. Execute steps: await runner.runStep(stepId)
4. Access results: ExecutionResult
```

### Step-by-Step Execution
```
1. Execute step 1: await runner.runStep('setup')
2. Execute step 2: await runner.runStep('step_2')
3. Execute step 3: await runner.runStep('step_3')
```

### Resume Execution
```
1. Execute steps normally
2. Save state automatically with each runStep()
3. Resume: await runner.resumeFromStep(stepId)
4. Continue from that point
```

### State Inspection
```
1. Get specific state: runner.getSavedState(stepId)
2. Get all states: runner.getAllSavedStates()
3. Clear states: runner.clearSavedStates()
```

---

## ExecutionResult Structure

```typescript
{
  executionId: number,
  success: boolean,
  error: string,
  summary: ExecutionSummary | undefined,
  data: {
    stepId?: number,
    completed?: boolean,
    resumedFrom?: number
  }
}
```

---

## Error Handling

### Common Errors

**"Execution plan not initialized"**
- Plan is null - shouldn't happen in normal usage

**"Step {id} not found in execution plan"**
- Invalid step ID passed to runStep()

**"No saved state found for step {id}"**
- Trying to resume from non-executed step

### Best Practices

```typescript
// Always wrap in try-catch
try {
  const result = await runner.runStep(stepId);
  if (result.success) {
    console.log('Step completed:', result.data);
  } else {
    console.error('Step failed:', result.error);
  }
} catch (error) {
  console.error('Execution error:', error);
}

// Check for saved state before resume
const state = runner.getSavedState(stepId);
if (state) {
  await runner.resumeFromStep(stepId);
} else {
  console.log('No saved state - execute normally');
  await runner.runStep(stepId);
}
```

---

## Step Types

### setup
- Lines before first bubble
- Initialization code

### control_flow
- Contains bubbles within for/while/if blocks
- Has mini-steps array

### bubble_block
- Single bubble execution
- Has mini-steps array

### finalization
- Lines after last bubble
- Cleanup code

---

## Mini-Step Types

### new_bubble
```typescript
{
  type: 'new_bubble',
  bubbleName: string,
  variableName: string
}
```

### await_action
```typescript
{
  type: 'await_action',
  variableName: string
}
```

---

## Quick Examples

### Execute Single Step
```typescript
const runner = new BubbleRunner(script, factory, options);
const result = await runner.runStep('setup');
```

### Execute All Steps
```typescript
const plan = runner.getPlan();
for (const step of plan.steps) {
  const result = await runner.runStep(step.id);
  if (!result.success) break;
}
```

### Resume from Step
```typescript
const runner = new BubbleRunner(script, factory, options);

// First execution
await runner.runStep('setup');
await runner.runStep('step_2');

// Later resume
await runner.resumeFromStep('setup');
```

### Inspect States
```typescript
// Check specific step
const state = runner.getSavedState('setup');
console.log('Variables:', state.variables);

// Check all steps
const allStates = runner.getAllSavedStates();
allStates.forEach((state, stepId) => {
  console.log(`Step ${stepId}:`, state.timestamp);
});
```

### Reset and Restart
```typescript
// Clear all saved states
runner.clearSavedStates();

// Start fresh
await runner.runStep('setup');
```

---

## Type Signatures

```typescript
class BubbleRunner {
  // Properties
  private bubbleFactory: BubbleFactory;
  private currentStep: number;
  private savedStates: Map<number, any>;

  // Public methods
  async runStep(stepId: number): Promise<ExecutionResult>
  async resumeFromStep(stepId: number): Promise<ExecutionResult>
  getSavedState(stepId: number): any | undefined
  getAllSavedStates(): Map<number, any>
  clearSavedStates(): void
  getPlan(): ExecutionPlan

  // Private helpers
  private async executeMiniStep(miniStep: MiniStep): Promise<void>
  private saveState(stepId: number): void
}
```

---

## Memory Management

### State Accumulation
- Each runStep() adds to savedStates Map
- Can grow large for many steps

### Recommendations
```typescript
// Clear after full execution
await runAllSteps();
runner.clearSavedStates();

// Or periodically
if (runner['savedStates'].size > 100) {
  runner.clearSavedStates();
}
```

---

## Logging

All methods log to BubbleLogger:
- Info: Major operations
- Debug: Detailed progress
- Error: Failures with context

Example:
```typescript
runner.getLogger()?.info('Custom message');
```

---

## File Locations

**Main Implementation:**
`BubbleLab/packages/bubble-runtime/src/runtime/BubbleRunner.ts`

**Documentation:**
- `BUBBLERUNNER_FIXES_SUMMARY.md` - Detailed changes
- `BUBBLERUNNER_FIX_COMPLETE.md` - Comprehensive report
- `BUBBLERUNNER_QUICK_REFERENCE.md` - This file

**Verification:**
`verify_bubblerunner_fixes.ts` - Test script

---

*Last Updated: 2026-01-10*
