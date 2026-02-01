# MathSolver Quick Start Guide

Get started with MathSolver in 5 minutes.

---

## Installation

The MathSolver is already integrated into Iterative Studio. No separate installation needed.

Ensure the Python backend is running:
```bash
cd /path/to/z3-leanaide-bubblelabs-ui
python z3_bubblelabs_advanced_ui.py
```

---

## Basic Usage

### 1. Using the UI

1. Open Iterative Studio
2. Click the mode selector (top-left)
3. Select "MathSolver" from the dropdown
4. Type your problem in the input area
5. Press **Ctrl+Enter** (or click Solve)

### 2. Using the API

```typescript
import { MathSolverCore } from './MathSolver';

// Create instance
const core = new MathSolverCore();

// Check backend
const healthy = await core.checkBackendHealth();
if (!healthy) {
  console.error('Backend not available');
  return;
}

// Solve a problem
const result = await core.solve({
  problem: 'x + 5 = 10, solve for x',
  solver: 'z3',
  timeout: 30
});

if (result.success) {
  console.log('Solution:', result.result);
  console.log('LaTeX:', result.latex);
} else {
  console.error('Error:', result.error);
}
```

---

## Common Patterns

### Pattern 1: Simple Solve

```typescript
const core = new MathSolverCore();

async function solveProblem(problem: string) {
  try {
    const result = await core.solve({ problem });
    return result.success ? result.result : null;
  } catch (e) {
    console.error('Solve failed:', e);
    return null;
  }
}
```

### Pattern 2: With Cancellation

```typescript
const core = new MathSolverCore();

// Start solving
const solvePromise = core.solve({
  problem: 'complex equation...',
  timeout: 300
});

// Cancel after 10 seconds
setTimeout(() => core.cancelSolve(), 10000);

// Wait for result
try {
  const result = await solvePromise;
} catch (e) {
  if (e.message === 'Cancelled') {
    console.log('Solve was cancelled');
  }
}
```

### Pattern 3: With Knowledge Base

```typescript
const result = await core.solve({
  problem: '...',
  knowledgeSearch: true  // Check KB first
});

if (result.knowledgeUsed) {
  console.log('Solution from knowledge base');
}
```

### Pattern 4: Event-Driven Updates

```typescript
const core = new MathSolverCore();

// Listen for messages
core.on('messageAdded', (msg) => {
  console.log(`[${msg.role}]: ${msg.content}`);
});

// Listen for completion
core.on('solvingCompleted', (result) => {
  if (result.success) {
    showSuccess(result.result);
  } else {
    showError(result.error);
  }
});

// Start solving
core.solve({ problem: '...' });
```

### Pattern 5: State Persistence

```typescript
const core = new MathSolverCore();

// Save state
function saveSession() {
  const state = core.exportState();
  localStorage.setItem('mathSession', state);
}

// Restore state
function restoreSession() {
  const saved = localStorage.getItem('mathSession');
  if (saved) {
    core.importState(saved);
  }
}

// Auto-save on page unload
window.addEventListener('beforeunload', saveSession);
```

---

## Tool Usage

### Using Math Tools in Agentic Mode

```typescript
import { executeMathToolCall, isMathTool } from './MathSolver';

// Execute a tool
const result = await executeMathToolCall({
  name: 'solve_z3',
  arguments: {
    problem: 'x + y = 10, x - y = 2'
  }
});

// Check if a name is a math tool
if (isMathTool('solve_z3')) {
  // It's a math tool
}
```

### Available Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `solve_z3` | SMT solving | Equations, constraints |
| `solve_lean` | Theorem proving | Proofs, logic |
| `solve_unified` | Auto-select | Unknown problems |
| `search_math_knowledge` | KB search | Find similar problems |
| `get_strategy` | Get approach | Optimization strategies |
| `translate_math` | Convert systems | Z3 ↔ Lean |
| `formalize_problem` | Natural → Formal | Convert text to math |
| `explain_proof` | Proof explanation | Step-by-step breakdown |
| `verify_proof` | Proof checking | Validate correctness |

---

## Solver Selection Guide

| Problem Type | Recommended Solver | Why |
|--------------|-------------------|-----|
| Linear equations | Z3 | Fast, efficient |
| Nonlinear constraints | Z3 | SMT handles well |
| Theorem proving | Lean | Built for proofs |
| Logic puzzles | Either | Depends on complexity |
| Unknown type | Unified | Auto-detection |
| Quick check | Z3 | Lower overhead |
| Formal verification | Lean | Rigorous checking |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Submit problem |
| `Escape` | Cancel solving |
| `Ctrl + K` | Focus input (when implemented) |
| `↑` / `↓` | Navigate history (when implemented) |

---

## Configuration

### Custom Configuration

```typescript
const core = new MathSolverCore({
  apiBaseUrl: 'http://localhost:8000',
  timeout: 300000,           // 5 minutes
  defaultSolver: 'auto',     // Auto-select
  autoSaveToKnowledgeBase: true,
  maxHistorySize: 1000
});
```

### Custom System Prompt

```typescript
import { updateMathSolverSystemPrompt } from './MathSolver';

updateMathSolverSystemPrompt(`
You are a mathematical assistant specialized in linear algebra.
Always provide solutions in matrix form when applicable.
`);
```

---

## Troubleshooting Quick Fixes

### "Backend is offline"
```bash
curl http://localhost:8000/health
# If fails, start backend:
python z3_bubblelabs_advanced_ui.py
```

### "A solve operation is already in progress"
```typescript
core.cancelSolve();  // Cancel existing
core.setState({ isProcessing: false });  // Force reset
```

### State not persisting
```typescript
// Check localStorage
console.log(localStorage.getItem('mathSolver'));

// Clear if corrupted
localStorage.removeItem('mathSolver');
location.reload();
```

---

## Performance Tips

### 1. Use Knowledge Base
```typescript
await core.solve({
  problem: '...',
  knowledgeSearch: true  // Faster for known problems
});
```

### 2. Set Appropriate Timeout
```typescript
// Don't wait 5 minutes for simple problems
await core.solve({
  problem: 'simple equation',
  timeout: 10  // 10 seconds
});
```

### 3. Clean Up Old Messages
```typescript
// Limit history to prevent memory issues
const messages = core.getState().messages;
if (messages.length > 100) {
  core.setState({
    messages: messages.slice(-100)
  });
}
```

---

## Integration Examples

### React Component

```typescript
import { MathSolverUI } from './MathSolver';

function App() {
  return (
    <div>
      <h1>Math Solver</h1>
      <MathSolverUI 
        initialProblem="x^2 + 5x + 6 = 0"
        onResult={(r) => console.log(r)}
      />
    </div>
  );
}
```

### Custom Wrapper

```typescript
import { MathSolverCore } from './MathSolver';

class MathHelper {
  private core = new MathSolverCore();
  
  async quickSolve(problem: string): Promise<string | null> {
    const result = await this.core.solve({
      problem,
      solver: 'auto',
      timeout: 30
    });
    return result.success ? result.result : null;
  }
  
  async solveWithSteps(problem: string) {
    const result = await this.core.solve({ problem });
    if (result.proof) {
      return this.formatSteps(result.proof);
    }
    return result.result;
  }
}
```

---

## Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Check API_REFERENCE.md** for detailed API docs
3. **See TROUBLESHOOTING.md** for problem solving
4. **Review DEVELOPMENT_HISTORY.md** for implementation details

---

## Support

- **Issues:** Check TROUBLESHOOTING.md
- **API Docs:** See API_REFERENCE.md
- **Examples:** Look in `__tests__/` directory

---

*Happy Solving! 🧮*
