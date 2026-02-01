# MathSolver Troubleshooting Guide

Common issues and their solutions.

---

## Quick Diagnostics

### Run Diagnostics Script

```typescript
import { MathSolverCore } from './MathSolver';

async function runDiagnostics() {
  const core = new MathSolverCore();
  
  console.log('=== MathSolver Diagnostics ===\n');
  
  // 1. Check backend
  console.log('1. Backend Health:');
  const healthy = await core.checkBackendHealth();
  console.log(`   Status: ${healthy ? '✓ Online' : '✗ Offline'}`);
  
  if (healthy) {
    const status = await core.getBackendStatus();
    console.log(`   Z3: ${status.z3Available ? '✓' : '✗'}`);
    console.log(`   Lean: ${status.leanAvailable ? '✓' : '✗'}`);
    console.log(`   Knowledge Base: ${status.knowledgeBaseSize} entries`);
    console.log(`   API Version: ${status.apiVersion}`);
  }
  
  // 2. Check knowledge engine
  console.log('\n2. Knowledge Engine:');
  const kbAvailable = await core.checkKnowledgeEngineAvailability();
  console.log(`   Status: ${kbAvailable ? '✓ Available' : '✗ Unavailable (graceful degradation)'}`);
  if (!kbAvailable) {
    const kbStatus = core.getKnowledgeEngineStatus();
    console.log(`   Note: ${kbStatus.error || 'Knowledge engine optional - direct solving works'}`);
  }
  
  // 2. Test solve
  console.log('\n2. Test Solve:');
  try {
    const result = await core.solve({
      problem: 'x + 2 = 5',
      solver: 'z3',
      timeout: 10
    });
    console.log(`   Result: ${result.success ? '✓ Success' : '✗ Failed'}`);
    if (result.success) {
      console.log(`   Solution: ${result.result}`);
    }
  } catch (e) {
    console.log(`   Error: ${e.message}`);
  }
  
  // 3. Test knowledge base
  console.log('\n3. Knowledge Base:');
  try {
    const result = await executeMathToolCall({
      name: 'search_math_knowledge',
      arguments: { query: 'test', limit: 1 }
    });
    console.log(`   Status: ${result.success ? '✓ Working' : '✗ Failed'}`);
  } catch (e) {
    console.log(`   Error: ${e.message}`);
  }
  
  console.log('\n=== Diagnostics Complete ===');
}

runDiagnostics();
```

---

## Backend Issues

### "Backend is offline" Error

**Symptoms:**
- Toast notification: "Backend is offline"
- Cannot solve any problems
- Health check fails

**Solutions:**

1. **Verify backend is running:**
```bash
curl http://localhost:8000/health
```

2. **Check port availability:**
```bash
# Windows
netstat -ano | findstr :8000

# macOS/Linux
lsof -i :8000
```

3. **Start the backend:**
```bash
cd /path/to/z3-leanaide-bubblelabs-ui
git pull  # Ensure latest code
python z3_bubblelabs_advanced_ui.py
```

4. **Check firewall:**
- Ensure port 8000 is not blocked
- Try accessing from browser: `http://localhost:8000`

---

### "API version mismatch" Warning

**Symptoms:**
- Console warning about version mismatch
- Some features may not work

**Solutions:**

1. **Check versions:**
```typescript
console.log('Frontend:', MATH_SOLVER_VERSION);
const status = await core.getBackendStatus();
console.log('Backend:', status.apiVersion);
```

2. **Update frontend:**
```bash
git pull origin main
```

3. **Update backend:**
```bash
git pull origin main
pip install -r requirements.txt
```

---

### "Z3 not available" or "Lean not available"

**Symptoms:**
- Solver-specific errors
- Backend healthy but solvers unavailable

**Solutions:**

1. **Check solver installation:**
```bash
# Check Z3
python -c "import z3; print(z3.get_version())"

# Check Lean
lean --version
```

2. **Reinstall dependencies:**
```bash
pip install z3-solver
# For Lean, follow official installation guide
```

---

## Knowledge Engine Issues

### "Knowledge base unavailable" Message

**Symptoms:**
- Toast notification: "Knowledge base unavailable - continuing with direct solving"
- Knowledge Base checkbox shows "(Unavailable)" in red
- KB status indicator in header shows ✗

**Impact:**
- **This is NOT a failure** - MathSolver continues to work normally
- Direct solving (Z3, Lean, Unified) works perfectly
- Only self-improving capabilities (learning, strategy recommendations) are disabled

**Explanation:**
The knowledge engine is optional. When unavailable:
- Problems are solved directly without searching the knowledge base
- Strategies are determined by local heuristics instead of learned patterns
- Solutions are not saved to the knowledge base for future use

**Solutions:**

1. **Verify knowledge endpoint:**
```bash
curl http://localhost:8000/knowledge/stats
```

2. **Check backend components:**
```bash
curl http://localhost:8000/health | grep knowledge
```

3. **Continue without knowledge base:**
The system is designed to function fully without the knowledge engine. You can:
- Continue solving problems normally
- Use all solver types (Z3, Lean, Unified)
- Export/import state manually

4. **To restore knowledge engine:**
```bash
# Restart the backend
cd /path/to/z3-leanaide-bubblelabs-ui
python z3_bubblelabs_advanced_ui.py
```

### Knowledge Search Returns Empty

**Symptoms:**
- `search_math_knowledge` tool returns "No similar problems found"
- Knowledge base appears empty

**Solutions:**

1. **Check if KB is populated:**
```bash
curl http://localhost:8000/knowledge/stats
```

2. **The KB learns over time** - Solve more problems to populate it

3. **Use broader search terms:**
```typescript
const result = await executeMathToolCall({
  name: 'search_math_knowledge',
  arguments: { query: 'algebra', top_k: 10 }  // Broader term
});
```

### Strategy Recommendations Failing

**Symptoms:**
- `get_strategy` tool fails
- Fallback message: "Using heuristic fallback"

**Explanation:**
When the knowledge engine is unavailable, MathSolver uses local heuristics based on problem content:
- Problems with "prove", "theorem", ∀, ∃ → Recommends Lean
- Problems with "solve", "=", ">", "<" → Recommends Z3
- Uncertain → Recommends Unified

**Solutions:**
1. Check knowledge engine availability
2. Manually select solver if you know the appropriate one
3. Use heuristic recommendations (they work well for common cases)

---

## UI Issues

### Component Not Rendering

**Symptoms:**
- Blank area where MathSolver should be
- No errors in console

**Solutions:**

1. **Check container exists:**
```typescript
const container = document.getElementById('mathsolver-container');
if (!container) {
  console.error('Container not found!');
}
```

2. **Verify React is loaded:**
```typescript
console.log('React:', React?.version);
console.log('ReactDOM:', ReactDOM?.version);
```

3. **Check for errors in ErrorBoundary:**
```typescript
// Wrap with error boundary
<MathSolverErrorBoundary onError={(e) => console.error(e)}>
  <MathSolverUI />
</MathSolverErrorBoundary>
```

---

### State Not Persisting

**Symptoms:**
- Lost work after refresh
- Export/import not working

**Solutions:**

1. **Check localStorage:**
```typescript
const state = core.exportState();
localStorage.setItem('mathSolver', state);

// Verify
console.log('Saved:', localStorage.getItem('mathSolver'));
```

2. **Check state size:**
```typescript
const state = core.exportState();
const size = new Blob([state]).size;
console.log(`State size: ${size} bytes`);
// If > 5MB, may exceed localStorage limit
```

3. **Clear corrupted state:**
```typescript
localStorage.removeItem('mathSolver');
location.reload();
```

---

### Keyboard Shortcuts Not Working

**Symptoms:**
- Ctrl+Enter doesn't submit
- Escape doesn't cancel

**Solutions:**

1. **Check event listeners:**
```typescript
document.addEventListener('keydown', (e) => {
  console.log('Key:', e.key, 'Ctrl:', e.ctrlKey);
});
```

2. **Verify focus:**
- Ensure MathSolverUI has focus
- Check if other elements are capturing events

3. **Check for preventDefault:**
```typescript
// Ensure no other handler is preventing
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    e.preventDefault(); // Must call this
    submit();
  }
}, { capture: true });
```

---

## Solve Issues

### "A solve operation is already in progress"

**Symptoms:**
- Error when trying to solve
- UI stuck in loading state

**Solutions:**

1. **Cancel existing solve:**
```typescript
core.cancelSolve();
```

2. **Reset state:**
```typescript
core.setState({
  isProcessing: false,
  currentSolver: undefined
});
```

3. **Check for zombie solves:**
```typescript
// Ensure proper cleanup
useEffect(() => {
  return () => {
    core.cancelSolve();
  };
}, []);
```

---

### Timeout Errors

**Symptoms:**
- "Request timed out" error
- Long-running problems fail

**Solutions:**

1. **Increase timeout:**
```typescript
const result = await core.solve({
  problem: '...',
  timeout: 600  // 10 minutes
});
```

2. **Simplify problem:**
- Break into smaller sub-problems
- Remove unnecessary constraints

3. **Check backend resources:**
- CPU usage
- Memory availability
- Disk space for temporary files

---

### "Invalid problem format"

**Symptoms:**
- Solver rejects input
- Format errors

**Solutions:**

1. **Validate input:**
```typescript
function validateProblem(problem: string): boolean {
  // Check length
  if (problem.length < 3) return false;
  
  // Check for valid characters
  const validPattern = /^[\w\s\+\-\*\/\=\<\>\(\)\[\]\{\}\^\,\.]+$/;
  return validPattern.test(problem);
}
```

2. **Use formalize tool:**
```typescript
const formalized = await executeMathToolCall({
  name: 'formalize_problem',
  arguments: { problem: naturalLanguageText }
});
```

---

## Memory Issues

### Memory Leaks

**Symptoms:**
- Browser slows down over time
- Memory usage keeps growing

**Solutions:**

1. **Clean up event listeners:**
```typescript
useEffect(() => {
  const handler = () => { /* ... */ };
  core.on('messageAdded', handler);
  
  return () => {
    core.off('messageAdded', handler);
  };
}, []);
```

2. **Limit message history:**
```typescript
// Auto-truncate old messages
if (messages.length > 100) {
  core.setState({
    messages: messages.slice(-100)
  });
}
```

3. **Clear toasts:**
```typescript
// After switching modes
clearAllToasts();
```

---

### "Out of memory" Errors

**Symptoms:**
- Browser crashes
- "Aw snap" error page

**Solutions:**

1. **Reduce history size:**
```typescript
const config: Partial<MathSolverConfig> = {
  maxHistorySize: 50  // Default is 1000
};
const core = new MathSolverCore(config);
```

2. **Clear state:**
```typescript
core.setState({
  messages: [],
  isProcessing: false
});
```

---

## TypeScript Issues

### Type Errors

**Symptoms:**
- TypeScript compilation errors
- Type mismatches

**Solutions:**

1. **Check imports:**
```typescript
// Correct
import { MathSolverCore } from './MathSolver';

// Incorrect - missing types
import { MathSolverCore } from './MathSolverCore';
```

2. **Update type definitions:**
```bash
npm update @types/react @types/react-dom
```

3. **Check strict mode:**
```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true
  }
}
```

---

### Circular Dependency Warnings

**Symptoms:**
- Build warnings
- Runtime errors

**Solutions:**

1. **Check import paths:**
```typescript
// Avoid this in MathSolverUI.tsx:
import { something } from './index';

// Use direct imports instead:
import { something } from './MathSolverCore';
```

2. **Use barrel exports carefully:**
```typescript
// index.ts - only re-export, no logic
export * from './MathSolverCore';
export * from './MathSolverUI';
```

---

## Performance Issues

### Slow UI Rendering

**Symptoms:**
- Lag when typing
- Slow message updates

**Solutions:**

1. **Virtualize long lists:**
```typescript
import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={400}
  itemCount={messages.length}
  itemSize={50}
>
  {({ index, style }) => (
    <div style={style}>
      <Message message={messages[index]} />
    </div>
  )}
</FixedSizeList>
```

2. **Debounce input:**
```typescript
const debouncedProblem = useDebounce(problem, 300);
```

3. **Use React.memo:**
```typescript
const Message = React.memo(({ message }) => {
  // Component
});
```

---

### Slow Solving

**Symptoms:**
- Problems take too long
- Timeout errors

**Solutions:**

1. **Use appropriate solver:**
```typescript
// Z3 for SMT
// Lean for proofs
// Auto for unknown

const result = await core.solve({
  problem: '...',
  solver: 'z3'  // Be explicit
});
```

2. **Enable knowledge base:**
```typescript
const result = await core.solve({
  problem: '...',
  knowledgeSearch: true  // Check KB first
});
```

3. **Optimize problem:**
- Remove redundant constraints
- Use simpler formulations
- Split complex problems

---

## Network Issues

### CORS Errors

**Symptoms:**
- "CORS policy" errors in console
- Requests blocked

**Solutions:**

1. **Check backend CORS:**
```python
# Python backend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Use proxy in development:**
```json
// package.json
{
  "proxy": "http://localhost:8000"
}
```

---

### Connection Refused

**Symptoms:**
- "ECONNREFUSED" errors
- Cannot connect to backend

**Solutions:**

1. **Check URL:**
```typescript
const core = new MathSolverCore({
  apiBaseUrl: 'http://localhost:8000'  // Correct port
});
```

2. **Test connection:**
```bash
curl -v http://localhost:8000/health
```

---

## Debug Mode

### Enable Debug Logging

```typescript
// Set before creating instance
window.MATH_SOLVER_DEBUG = true;

const core = new MathSolverCore();
// Now logs detailed information
```

### Performance Profiling

```typescript
// Chrome DevTools
performance.mark('solve-start');
await core.solve({ problem: '...' });
performance.mark('solve-end');
performance.measure('solve', 'solve-start', 'solve-end');

// View in DevTools > Performance
```

---

## Getting Help

If issues persist:

1. **Check logs:**
   - Browser console (F12)
   - Backend logs
   - Network tab for failed requests

2. **Run diagnostics:**
   ```typescript
   runDiagnostics();
   ```

3. **Create minimal reproduction:**
   ```typescript
   // Smallest code that shows the issue
   const core = new MathSolverCore();
   await core.solve({ problem: 'x = 1' });
   ```

4. **Report issue with:**
   - Browser version
   - Backend version
   - Error messages
   - Steps to reproduce

---

*Last updated: 2026-01-31*
