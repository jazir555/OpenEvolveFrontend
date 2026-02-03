# Graceful Degradation - Knowledge Engine

MathSolver is designed to function fully even when the knowledge engine (self-improving capabilities) is unavailable.

---

## Overview

The knowledge engine provides:
- **Pattern Recognition**: Finding similar solved problems
- **Strategy Recommendations**: Suggesting the best solver for a problem
- **Learning**: Caching solutions for future reuse
- **Statistics**: Tracking success rates and usage patterns

**When unavailable**, MathSolver **gracefully degrades** by:
1. Continuing with direct solving (Z3, Lean, Unified)
2. Using local heuristics for solver selection
3. Operating without caching/learning
4. Providing clear UI feedback

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MathSolver                                │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │  Knowledge      │    │         Core Solving             │   │
│  │  Engine         │◄───┤     (Always Available)           │   │
│  │  (Optional)     │    │                                  │   │
│  │                 │    │  • Z3 SMT Solver                 │   │
│  │  • Search KB    │    │  • Lean Theorem Prover           │   │
│  │  • Get Strategy │    │  • Unified Solver                │   │
│  │  • Learn        │    │  • State Management              │   │
│  │  • Stats        │    │  • Event System                  │   │
│  └────────┬────────┘    └──────────────────────────────────┘   │
│           │                                                      │
│           │ (Optional - failures handled gracefully)             │
│           ▼                                                      │
│  ┌─────────────────┐                                             │
│  │  Heuristic      │  Fallback when knowledge engine fails      │
│  │  Fallbacks      │                                             │
│  │                 │  • Problem content analysis                 │
│  │  • Solver rec   │  • Pattern matching                         │
│  │  • Strategy     │  • Default behaviors                        │
│  └─────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection & Status

### Checking Availability

```typescript
import { MathSolverCore } from './MathSolver';

const core = new MathSolverCore();

// Async check with network call
const available = await core.checkKnowledgeEngineAvailability();

// Sync check of cached status
const status = core.getKnowledgeEngineStatus();
console.log(status.available);  // boolean
console.log(status.lastChecked); // timestamp
console.log(status.error);       // error message if failed
```

### Status Object

```typescript
interface KnowledgeEngineStatus {
    available: boolean;      // Current availability
    lastChecked: number;     // Timestamp of last check
    error?: string;          // Error message if unavailable
}
```

---

## Graceful Degradation Behaviors

### 1. Knowledge Base Search

**Normal Operation:**
```typescript
// Searches KB for similar problems
const results = await searchKnowledge({ query: 'quadratic' });
// Returns: [problem1, problem2, ...]
```

**When Unavailable:**
```typescript
// Search fails gracefully
// Solving continues directly without KB lookup
// User sees: "Knowledge base unavailable - continuing with direct solving"
```

### 2. Strategy Recommendations

**Normal Operation:**
```typescript
const strategy = await getStrategy({ problem_statement: '...' });
// Returns: { strategy: 'z3', confidence: 0.95, ... }
```

**When Unavailable - Heuristic Fallback:**

| Problem Pattern | Recommended Solver |
|-----------------|-------------------|
| Contains "prove", "theorem", ∀, ∃ | Lean |
| Contains "solve", "=", ">", "<" | Z3 |
| Unclear/Complex | Unified |

```typescript
// Returns local heuristic recommendation
// "Using heuristic fallback"
// "Recommended Strategy: lean (60% confidence)"
```

### 3. Learning from Solutions

**Normal Operation:**
```typescript
// Successful solutions are saved to KB
await learnFromSolution({ problem, result, proof });
```

**When Unavailable:**
```typescript
// Learning is skipped silently
// Solving still works, just no caching
```

### 4. Tool Execution

**search_math_knowledge Tool:**
```typescript
// On failure, returns helpful fallback:
"⚠️ Knowledge engine currently unavailable

Suggestion: Continue with direct solving using:
- solve_z3 for constraint problems
- solve_lean for theorem proving  
- solve_unified for automatic selection"
```

**get_strategy Tool:**
```typescript
// On failure, returns heuristic recommendation with explanation:
"⚠️ Knowledge engine unavailable - using heuristic fallback

Recommended Strategy: z3
Confidence: 60% (heuristic-based)"
```

---

## UI Indicators

### Header Status

```
MathSolver
● Backend connected  ● KB ✓   [Refresh] [Close]  ← KB Available

MathSolver  
● Backend connected  ● KB ✗   [Refresh] [Close]  ← KB Unavailable
```

### Knowledge Base Checkbox

**Available:**
```
☑ Use Knowledge Base (Available)
```

**Unavailable:**
```
☐ Use Knowledge Base (Unavailable)  ← Grayed out, disabled
```

### Toast Notifications

**KB Unavailable:**
```
⚠️ Knowledge base unavailable - continuing with direct solving
```

**Solving Success (no KB):**
```
✓ Solution found!
```

---

## Code Examples

### Basic Usage (Handles KB Unavailability Automatically)

```typescript
const core = new MathSolverCore();

// Create and solve - KB availability handled automatically
const problem = core.createProblem('x + 5 = 10');
const result = await core.solve({
    problem,
    useKnowledgeBase: true  // Will be ignored if KB unavailable
});

// Result is valid regardless of KB status
console.log(result.success);
```

### Checking Before Using KB Features

```typescript
const core = new MathSolverCore();

// Check KB availability
const kbAvailable = await core.checkKnowledgeEngineAvailability();

if (kbAvailable) {
    // Use KB features
    const results = await searchKnowledge({ query: 'algebra' });
    const strategy = await getStrategy({ problem_statement: '...' });
} else {
    // Use direct solving with heuristics
    console.log('Using heuristic solver selection');
}
```

### Custom Fallback Logic

```typescript
async function solveWithFallback(problem: string) {
    const core = new MathSolverCore();
    const kbAvailable = core.isKnowledgeEngineAvailable();
    
    if (kbAvailable) {
        // Try KB-enhanced solving
        const search = await searchKnowledge({ query: problem });
        if (search.results.length > 0) {
            // Use similar problem approach
        }
    }
    
    // Always fall back to direct solving
    const result = await core.solve({
        problem: core.createProblem(problem),
        preferredSolver: 'auto'  // Heuristic selection
    });
    
    return result;
}
```

---

## Testing Graceful Degradation

### Unit Tests

```typescript
describe('Knowledge Engine Graceful Degradation', () => {
    test('should solve when knowledge engine is unavailable', async () => {
        // Mock KB failure
        mathSolverAPI.searchKnowledge = jest.fn().mockRejectedValue(
            new Error('Knowledge engine unavailable')
        );
        
        const core = new MathSolverCore();
        const problem = core.createProblem('x + 2 = 5');
        
        // Should not throw
        const result = await core.solve({
            problem,
            useKnowledgeBase: true
        });
        
        // Should succeed (or fail for non-KB reasons)
        expect(result.error).not.toContain('Knowledge engine');
    });
});
```

### Manual Testing

1. **Start backend without knowledge engine:**
   ```bash
   # If KB is a separate service, don't start it
   python z3_bubblelabs_advanced_ui.py  # Main backend only
   ```

2. **Verify graceful degradation:**
   ```typescript
   const core = new MathSolverCore();
   await core.checkKnowledgeEngineAvailability();  // Should return false
   
   // Solving still works
   const result = await core.solve({
       problem: core.createProblem('x + 1 = 2'),
       useKnowledgeBase: true  // Ignored
   });
   ```

---

## Troubleshooting

### "Knowledge base unavailable" Message

**This is NOT an error** - it's informational.

**What it means:**
- The knowledge engine endpoint is not accessible
- MathSolver will continue with direct solving
- All core functionality remains available

**What to do:**
1. **Nothing required** - system works without KB
2. **To restore KB:** Check backend status and restart if needed
3. **Verify:** `curl http://localhost:8000/knowledge/stats`

### Checking Component Health

```bash
# Check backend
curl http://localhost:8000/health

# Check knowledge engine specifically
curl http://localhost:8000/knowledge/stats

# Should return:
# {"total_patterns": 0, "total_strategies": 0, "learning_enabled": true}
# Or connection error (KB unavailable)
```

---

## Performance Impact

| Feature | With KB | Without KB | Impact |
|---------|---------|------------|--------|
| First solve | Slower (search) | Faster | +20-50ms |
| Repeat solve | Instant (cached) | Normal solve | No cache |
| Strategy selection | ML-based | Heuristic | Minimal |
| Learning | Automatic | Disabled | No learning |

**Conclusion:** Operating without KB is actually slightly faster for first-time problems, but loses caching benefits for repeats.

---

## Design Principles

1. **Fail Softly** - KB failures don't stop solving
2. **Clear Feedback** - Users know when KB is unavailable
3. **Functional Fallbacks** - Heuristics provide reasonable alternatives
4. **Automatic Recovery** - KB is rechecked periodically
5. **No Data Loss** - Solutions still work, just not cached

---

## Related Documentation

- [README.md](./README.md) - Main documentation
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Problem solving
- [API_REFERENCE.md](./API_REFERENCE.md) - API details

---

*The MathSolver works perfectly without the knowledge engine - the KB is an enhancement, not a requirement.*
