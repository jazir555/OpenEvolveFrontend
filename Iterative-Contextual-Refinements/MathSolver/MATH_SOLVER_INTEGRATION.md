# MathSolver Integration Guide

This document describes how the MathSolver mode integrates the Iterative Studio (TypeScript/React frontend) with the Z3-LeanAIDE mathematical knowledge system (Python backend).

## Overview

The MathSolver mode adds automated mathematical reasoning capabilities to Iterative Studio by connecting to a Python FastAPI backend that provides:

- **Z3 SMT Solver**: Constraint satisfaction, arithmetic, optimization
- **Lean Theorem Prover**: Formal proofs, logical deduction
- **Unified Solver**: Consensus between Z3 and Lean for high-confidence results
- **Knowledge Base**: ML-powered pattern matching for similar problems

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Iterative Studio (Frontend)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ MathSolver   │  │ Agentic Mode │  │ Deepthink Mode       │  │
│  │ UI Component │  │ + Math Tools │  │ + Math Strategy      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
│         └──────────────────┼──────────────────────┘              │
│                            │                                     │
│                    MathSolverCore.ts                             │
│         (API client, state management, event handling)           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP/JSON
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Z3-LeanAIDE Python Backend (FastAPI)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /solve/z3    │  │ /solve/lean  │  │ /solve/unified       │  │
│  │ /prove/lean  │  │ /knowledge/* │  │ /translate           │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
│         └──────────────────┼──────────────────────┘              │
│                            │                                     │
│              Unified Bridge + Knowledge Manager                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       ┌─────────────┐              ┌─────────────┐
       │ Z3 Solver   │              │ Lean Server │
       │ (SMT-LIB)   │              │ (Lean 4)    │
       └─────────────┘              └─────────────┘
```

## Installation

### 1. Start the Python Backend

```bash
# From the OpenEvolve Frontend directory
cd /path/to/OpenEvolve/Frontend

# Install dependencies (if not already installed)
pip install z3-solver fastapi uvicorn sqlalchemy pydantic redis

# Start the API server
python -c "from math_api_complete import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

Or using the provided startup script (if available):
```bash
python start_math_api.py
```

### 2. Configure Frontend Connection

The MathSolver frontend connects to the backend via the `MATH_SOLVER_API_URL` environment variable:

```bash
# Default: http://localhost:8000
export MATH_SOLVER_API_URL=http://localhost:8000
```

Or in a `.env` file:
```
MATH_SOLVER_API_URL=http://localhost:8000
```

### 3. Verify Integration

1. Open Iterative Studio in your browser
2. Navigate to the MathSolver mode (if added to the mode selector)
3. Enter a test problem: `x + 2 = 5`
4. Click "Solve"
5. Verify results are returned from the backend

## Usage Modes

### 1. Dedicated MathSolver Mode

The MathSolver can operate as a standalone mode with its own UI:

```typescript
import { MathSolverUI } from './MathSolver';

function App() {
    return (
        <MathSolverUI 
            onClose={() => console.log('Closed')}
            initialProblem="x² + 3x + 2 = 0"
        />
    );
}
```

### 2. Math Tools in Agentic Mode

Extend Agentic mode with mathematical reasoning tools:

```typescript
import { executeMathToolCall, isMathTool, MATH_TOOLS_PROMPT } from './MathSolver';

// In your agent configuration, add math tools to the system prompt
const systemPrompt = `${AGENTIC_SYSTEM_PROMPT}\n\n${MATH_TOOLS_PROMPT}`;

// In tool execution, handle math tools
async function executeTool(toolCall: ToolCall) {
    if (isMathTool(toolCall.type)) {
        return await executeMathToolCall(toolCall as MathToolCall);
    }
    // Handle other tools...
}
```

Available math tools:
- `solve_z3(problem_statement, constraints?)` - Z3 SMT solving
- `solve_lean(theorem_statement, timeout?)` - Lean theorem proving
- `solve_unified(problem_statement, ...)` - Consensus-based solving
- `search_math_knowledge(query, top_k?)` - Knowledge base search
- `translate_math(content, from, to)` - Z3 ↔ Lean translation
- `formalize_problem(problem_statement, target_format)` - Formalization help
- `explain_proof(proof_content, solver_type)` - Proof explanation
- `verify_proof(proof_content)` - Proof verification

### 3. Math Strategy in Deepthink Mode

For complex mathematical problems, integrate MathSolver as a Deepthink strategy:

```typescript
// In DeepthinkCore.ts, add a mathematical strategy
const strategies = [
    'StrategicSolver',
    'HypothesisExplorer', 
    'DissectedObservations',
    'MathematicalFormalization'  // New math-focused strategy
];
```

## API Reference

### Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/solve/z3` | POST | Solve using Z3 SMT solver |
| `/prove/lean` | POST | Prove using Lean theorem prover |
| `/solve/unified` | POST | Solve with Z3+Lean consensus |
| `/knowledge/search` | POST | Search knowledge base |
| `/knowledge/learn` | POST | Learn from solution |
| `/translate` | POST | Translate between formats |
| `/health` | GET | Backend health check |

### TypeScript Types

```typescript
// From MathSolverCore.ts
interface MathProblem {
    id: string;
    statement: string;
    constraints?: string[];
    domain?: 'algebra' | 'arithmetic' | 'geometry' | 'calculus' | 'logic' | 'number_theory' | 'other';
    difficulty?: 'easy' | 'medium' | 'hard' | 'expert';
}

interface Z3Result {
    status: 'sat' | 'unsat' | 'unknown' | 'timeout' | 'error';
    model?: Record<string, any>;
    proof?: string;
    solvingTimeMs: number;
}

interface LeanResult {
    status: 'proved' | 'failed' | 'partial' | 'timeout' | 'error';
    proof?: string;
    tactics?: string[];
    errors?: string[];
    provingTimeMs: number;
}

interface UnifiedResult {
    z3Result: Z3Result;
    leanResult: LeanResult;
    consensus: { agreement: boolean; confidence: number; discrepancies?: string[] };
    recommendedApproach: 'z3' | 'lean' | 'both';
}
```

## Configuration

### Solver Selection Guidelines

| Problem Type | Recommended Solver | Reason |
|--------------|-------------------|--------|
| Linear equations, constraints | Z3 | Efficient for arithmetic |
| Satisfiability checking | Z3 | SMT solver strength |
| Formal proofs, theorems | Lean | Proof generation |
| Inductive reasoning | Lean | Mathematical induction |
| Complex, critical systems | Unified | Cross-validation |
| Unknown domain | Auto | Automatic selection |

### Consensus Levels

- **Strict**: Both solvers must agree exactly (highest confidence)
- **Confidence**: Allow minor discrepancies with confidence scoring (default)
- **Permissive**: Use best available result even without agreement

## Troubleshooting

### Backend Not Available

```
Backend unavailable
```

**Solutions:**
1. Verify the Python backend is running: `curl http://localhost:8000/health`
2. Check firewall settings for port 8000
3. Verify `MATH_SOLVER_API_URL` environment variable

### Timeout Errors

```
Request timeout - proof may be too complex
```

**Solutions:**
1. Increase timeout: set timeout to 600 seconds
2. Simplify the problem
3. Use Z3 instead of Lean for constraint problems
4. Break problem into smaller parts

### Parse Errors

```
Syntax error in formalization
```

**Solutions:**
1. Check parentheses matching
2. Verify variable declarations
3. Use `formalize_problem` tool for guidance
4. Review SMT-LIB or Lean syntax documentation

## Examples

### Example 1: Simple Equation

```
Problem: x² + 3x + 2 = 0
Solver: Z3
Result: sat, x = -1 or x = -2
```

### Example 2: Theorem Proving

```
Problem: For all natural numbers n, n + 0 = n
Solver: Lean
Result: proved using induction tactic
Proof: by induction n; simp
```

### Example 3: Unified Consensus

```
Problem: Prove a² + b² ≥ 2ab for all reals a, b
Solver: Unified
Z3 Result: sat (verified with random testing)
Lean Result: proved (algebraic manipulation)
Consensus: Agreement with 95% confidence
```

## Performance Considerations

- **Z3**: Typically responds in milliseconds for simple problems
- **Lean**: May take seconds to minutes for complex proofs
- **Knowledge Base**: Searching is fast due to Redis caching
- **Translation**: May require AI calls for complex expressions

## Security

- All inputs are validated to prevent injection attacks
- SQL queries use parameterized statements
- Subprocess calls have strict timeouts and input validation
- No hardcoded secrets in the codebase

## Contributing

To extend the MathSolver integration:

1. **Add new tool**: Extend `MathToolCall` type and `executeMathToolCall` function
2. **Add new solver**: Extend backend API and `SolverSystem` type
3. **Improve UI**: Modify `MathSolverUI.tsx` with new features
4. **Add tests**: Create test files in `MathSolver/__tests__/`

## License

Apache-2.0 (same as Iterative Studio)
