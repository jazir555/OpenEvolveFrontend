# MathSolver API Reference

Complete API documentation for the MathSolver module.

---

## Table of Contents

1. [Core API](#core-api)
2. [UI Components](#ui-components)
3. [Math Tools](#math-tools)
4. [Types](#types)
5. [Constants](#constants)
6. [Events](#events)

---

## Core API

### MathSolverCore

Main class for mathematical problem solving.

```typescript
import { MathSolverCore } from './MathSolver';

const core = new MathSolverCore();
```

#### Constructor

```typescript
constructor(config?: Partial<MathSolverConfig>)
```

**Parameters:**
- `config` (optional): Partial configuration object

**Example:**
```typescript
const core = new MathSolverCore({
  apiBaseUrl: 'http://localhost:8000',
  timeout: 30000,
  autoSaveToKnowledgeBase: true
});
```

#### Methods

##### solve()

Solve a mathematical problem.

```typescript
async solve(options: SolveOptions): Promise<SolveResult>
```

**Parameters:**
```typescript
interface SolveOptions {
  problem: string;           // The problem to solve
  solver?: SolverSystem;     // 'z3' | 'lean' | 'unified' | 'auto'
  domain?: MathDomain;       // Domain hint
  timeout?: number;          // Timeout in seconds
  knowledgeSearch?: boolean; // Search knowledge base first
}
```

**Returns:**
```typescript
interface SolveResult {
  success: boolean;
  solver: 'z3' | 'lean' | 'unified';
  result?: string;
  proof?: string;
  latex?: string;
  error?: string;
  executionTime: number;
  knowledgeUsed?: boolean;
}
```

**Example:**
```typescript
const result = await core.solve({
  problem: 'x + 5 = 10, solve for x',
  solver: 'z3',
  timeout: 30
});

if (result.success) {
  console.log('Solution:', result.result);
  console.log('LaTeX:', result.latex);
}
```

##### cancelSolve()

Cancel the current solve operation.

```typescript
cancelSolve(): void
```

**Example:**
```typescript
// Start solving
core.solve({ problem: '...' });

// Cancel after 5 seconds
setTimeout(() => core.cancelSolve(), 5000);
```

##### isSolving()

Check if a solve operation is in progress.

```typescript
isSolving(): boolean
```

**Returns:** `boolean`

##### getState() / setState()

Get or set the solver state.

```typescript
getState(): MathSolverState
setState(state: Partial<MathSolverState>): void
```

**State Structure:**
```typescript
interface MathSolverState {
  messages: MathSolverMessage[];
  isProcessing: boolean;
  currentSolver?: SolverSystem;
  backendStatus: 'unknown' | 'healthy' | 'unhealthy';
  apiVersion: string;
}
```

##### exportState() / importState()

Export or import state for persistence.

```typescript
exportState(): string
importState(serialized: string): void
```

**Example:**
```typescript
// Save state
const saved = core.exportState();
localStorage.setItem('mathSolver', saved);

// Restore state
const saved = localStorage.getItem('mathSolver');
if (saved) core.importState(saved);
```

##### on() / off()

Event subscription.

```typescript
on<K extends keyof MathSolverEventMap>(
  event: K,
  handler: (data: MathSolverEventMap[K]) => void
): void

off<K extends keyof MathSolverEventMap>(
  event: K,
  handler: (data: MathSolverEventMap[K]) => void
): void
```

**Example:**
```typescript
const onMessage = (msg: MathSolverMessage) => {
  console.log('New message:', msg.content);
};

core.on('messageAdded', onMessage);

// Later...
core.off('messageAdded', onMessage);
```

##### checkBackendHealth()

Check if the backend is available.

```typescript
async checkBackendHealth(): Promise<boolean>
```

##### getBackendStatus()

Get detailed backend status.

```typescript
async getBackendStatus(): Promise<BackendStatus>
```

**Returns:**
```typescript
interface BackendStatus {
  online: boolean;
  z3Available: boolean;
  leanAvailable: boolean;
  leanVersions: string[];
  apiVersion: string;
  knowledgeBaseSize: number;
}
```

---

## UI Components

### MathSolverUI

Main React component for the MathSolver interface.

```typescript
import { MathSolverUI } from './MathSolver';

function App() {
  return <MathSolverUI />;
}
```

#### Props

```typescript
interface MathSolverUIProps {
  initialProblem?: string;
  onProblemSubmit?: (problem: string) => void;
  onResult?: (result: SolveResult) => void;
  showKnowledgeBase?: boolean;
}
```

**Example:**
```typescript
<MathSolverUI
  initialProblem="x² + 5x + 6 = 0"
  onProblemSubmit={(p) => console.log('Solving:', p)}
  onResult={(r) => console.log('Result:', r)}
  showKnowledgeBase={true}
/>
```

### MathSolverErrorBoundary

Error boundary to prevent app crashes.

```typescript
import { MathSolverErrorBoundary } from './MathSolver';

<MathSolverErrorBoundary onReset={() => console.log('Reset')}>
  <MathSolverUI />
</MathSolverErrorBoundary>
```

---

## Math Tools

### Tool Functions

All tools are async functions that return `Promise<ToolResult>`.

#### solve_z3

Solve using Z3 SMT solver.

```typescript
const result = await executeMathToolCall({
  name: 'solve_z3',
  arguments: {
    problem: 'x + y = 10, x - y = 2',
    timeout: 30
  }
});
```

#### solve_lean

Solve using Lean theorem prover.

```typescript
const result = await executeMathToolCall({
  name: 'solve_lean',
  arguments: {
    problem: 'Prove that for all natural numbers n, n + 0 = n',
    version: '4'  // or '3'
  }
});
```

#### solve_unified

Solve using unified approach.

```typescript
const result = await executeMathToolCall({
  name: 'solve_unified',
  arguments: {
    problem: 'Find the maximum value of f(x) = -x² + 4x',
    strategy: 'default'
  }
});
```

#### search_math_knowledge

Search the knowledge base.

```typescript
const result = await executeMathToolCall({
  name: 'search_math_knowledge',
  arguments: {
    query: 'quadratic equation',
    limit: 5
  }
});
```

#### get_strategy

Get strategy recommendation.

```typescript
const result = await executeMathToolCall({
  name: 'get_strategy',
  arguments: {
    problemType: 'inequality',
    constraints: ['linear', 'integer']
  }
});
```

#### translate_math

Translate between formal systems.

```typescript
const result = await executeMathToolCall({
  name: 'translate_math',
  arguments: {
    problem: 'x > 0 → x² > 0',
    fromSystem: 'mathematical',
    toSystem: 'z3'
  }
});
```

#### formalize_problem

Convert natural language to formal.

```typescript
const result = await executeMathToolCall({
  name: 'formalize_problem',
  arguments: {
    problem: 'The sum of two numbers is 10 and their product is 21'
  }
});
```

#### explain_proof

Explain a proof step by step.

```typescript
const result = await executeMathToolCall({
  name: 'explain_proof',
  arguments: {
    proof: 'theorem add_zero (n : ℕ) : n + 0 = n := by induction n...'
  }
});
```

#### verify_proof

Verify a proof's correctness.

```typescript
const result = await executeMathToolCall({
  name: 'verify_proof',
  arguments: {
    proof: 'theorem example : 2 + 2 = 4 := rfl',
    system: 'lean4'
  }
});
```

### Tool Detection

Check if a string is a math tool call.

```typescript
import { isMathTool } from './MathSolver';

const isMath = isMathTool('solve_z3'); // true
const isNotMath = isMathTool('web_search'); // false
```

---

## Types

### SolverSystem

```typescript
type SolverSystem = 'z3' | 'lean' | 'unified' | 'auto';
```

### MathDomain

```typescript
type MathDomain = 
  | 'arithmetic' 
  | 'algebra' 
  | 'calculus' 
  | 'logic' 
  | 'linear_algebra' 
  | 'number_theory' 
  | 'geometry' 
  | 'statistics' 
  | 'discrete' 
  | 'general';
```

### MathSolverMessage

```typescript
interface MathSolverMessage {
  id: string;
  role: 'user' | 'solver' | 'system' | 'error';
  content: string;
  timestamp: Date;
  metadata?: {
    solver?: SolverSystem;
    domain?: MathDomain;
    executionTime?: number;
    latex?: string;
    proof?: string;
    knowledgeUsed?: boolean;
    strategy?: string;
  };
}
```

### MathProblem

```typescript
interface MathProblem {
  id: string;
  content: string;
  domain: MathDomain;
  timestamp: Date;
  complexity?: 'simple' | 'moderate' | 'complex';
  recommendedSolver?: SolverSystem;
}
```

### MathSolverConfig

```typescript
interface MathSolverConfig {
  apiBaseUrl: string;
  timeout: number;
  defaultSolver: SolverSystem;
  autoSaveToKnowledgeBase: boolean;
  maxHistorySize: number;
}
```

---

## Constants

### Version

```typescript
export const MATH_SOLVER_VERSION = '1.1.0';
export const MATH_SOLVER_BUILD_DATE = '2026-01-31';
```

### Default Config

```typescript
export const DEFAULT_MATH_SOLVER_CONFIG: MathSolverConfig = {
  apiBaseUrl: 'http://localhost:8000',
  timeout: 300000,  // 5 minutes
  defaultSolver: 'auto',
  autoSaveToKnowledgeBase: true,
  maxHistorySize: 1000
};
```

### System Prompt

```typescript
export const MATH_SOLVER_SYSTEM_PROMPT = `You are an expert mathematical assistant...`;
```

---

## Events

### Event Map

```typescript
type MathSolverEventMap = {
  'messageAdded': MathSolverMessage;
  'solvingStarted': { problem: MathProblem; solver?: SolverSystem };
  'solvingCompleted': SolveResult;
  'solvingError': SolveResult;
  'solvingCancelled': null;
  'backendStatusChanged': { online: boolean; message: string };
  'stateChanged': MathSolverState;
};
```

### Usage Examples

```typescript
// Listen for new messages
core.on('messageAdded', (message) => {
  console.log(`[${message.role}]: ${message.content}`);
});

// Track solving progress
core.on('solvingStarted', ({ problem, solver }) => {
  console.log(`Solving "${problem.content}" with ${solver}...`);
});

core.on('solvingCompleted', (result) => {
  if (result.success) {
    console.log(`✓ Solved in ${result.executionTime}ms`);
  } else {
    console.log(`✗ Failed: ${result.error}`);
  }
});

// Monitor backend status
core.on('backendStatusChanged', ({ online, message }) => {
  console.log(`Backend ${online ? 'online' : 'offline'}: ${message}`);
});
```

---

## Error Handling

### Error Types

```typescript
class MathSolverError extends Error {
  constructor(
    message: string,
    public code: string,
    public recoverable: boolean = false
  ) {
    super(message);
  }
}

// Common error codes:
// - 'BACKEND_UNAVAILABLE'
// - 'TIMEOUT'
// - 'INVALID_PROBLEM'
// - 'SOLVER_FAILED'
// - 'CANCELLED'
```

### Handling Example

```typescript
try {
  const result = await core.solve({ problem: '...' });
} catch (error) {
  if (error instanceof MathSolverError) {
    switch (error.code) {
      case 'BACKEND_UNAVAILABLE':
        // Show retry button
        break;
      case 'TIMEOUT':
        // Suggest increasing timeout
        break;
      case 'CANCELLED':
        // User cancelled, no action needed
        break;
    }
  }
}
```

---

## Best Practices

### 1. Always Check Backend Health

```typescript
const isHealthy = await core.checkBackendHealth();
if (!isHealthy) {
  // Show error message
  return;
}
```

### 2. Handle Cancellation Gracefully

```typescript
const controller = new AbortController();

// Start solving
const promise = core.solve({ 
  problem: '...',
  signal: controller.signal 
});

// Cancel button
<button onClick={() => controller.abort()}>Cancel</button>
```

### 3. Use Event Listeners for UI Updates

```typescript
useEffect(() => {
  const handleMessage = (msg: MathSolverMessage) => {
    setMessages(prev => [...prev, msg]);
  };
  
  core.on('messageAdded', handleMessage);
  return () => core.off('messageAdded', handleMessage);
}, []);
```

### 4. Export State Before Unmount

```typescript
useEffect(() => {
  return () => {
    // Save state on unmount
    const state = core.exportState();
    localStorage.setItem('mathSolverState', state);
  };
}, []);
```

---

*End of API Reference*
