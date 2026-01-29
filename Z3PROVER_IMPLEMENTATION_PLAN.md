# Z3 Prover Integration - Implementation Plan

**Date:** 2026-01-24
**Component:** Z3 SMT Solver Integration
**Complexity:** High
**Estimated Effort:** 2-3 days

---

## Executive Summary

Z3 is a high-performance theorem prover from Microsoft Research with support for:
- SMT (Satisfiability Modulo Theories) solving
- Optimization problems
- Fixedpoint computation
- Multiple theories: Booleans, Arithmetic, Bit-vectors, Arrays, Datatypes, Floating-point

**Current Status:** No integration exists (0% complete)
**Goal:** Create full BubbleLab API integration for Z3 similar to LeanAide

---

## Component Analysis

### Z3 Source Structure

**Location:** `z3prover/`

**Key Components:**
1. **C++ Core** - The main Z3 theorem prover
2. **Python Bindings** (`src/api/python/z3/`)
   - `z3.py` - Main Python API (9800+ lines)
   - `z3types.py` - Type definitions
   - `z3printer.py` - Pretty printing
   - `z3num.py`, `z3poly.py`, `z3rcf.py` - Specialized modules
   - `z3test.py` - Test suite

3. **MCP Server** (`src/api/mcp/z3mcp.py`)
   - Simple MCP wrapper for SMTLIB2 evaluation
   - Single `eval` function that takes SMTLIB2 commands

4. **Language Bindings**
   - C++, Java, .NET, OCaml, Julia, ML
   - Python is the most complete

---

## Z3 Key Classes and Features

### Main Z3 Classes (from Python API)

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| **Solver** | SMT solving | `add()`, `check()`, `model()`, `push()`, `pop()` |
| **Optimize** | Optimization objectives | `maximize()`, `minimize()`, `check()` |
| **Fixedpoint** | Fixedpoint solving | `add_rule()`, `query()`, `get_answer()` |
| **ModelRef** | Model inspection | `eval()`, `decls()`, `num_consts()` |
| **Tactic** | Applying solvers/tactics | `apply()`, `help()` |
| **Simplifier** | Expression simplification | `simplify()` |
| **Goal** | Accumulation of constraints | `add()`, `as_expr()`, `depth()` |

### Supported Theories

- **Booleans** (`Bool`, `Not`, `And`, `Or`, `Implies`)
- **Integers** (`Int`, `+`, `-`, `*`, `/`, `<`, `>`)
- **Reals** (`Real`, arithmetic)
- **Bit-vectors** (`BitVec`, bit operations)
- **Arrays** (`Array`, select, store)
- **Datatypes** (Algebraic data types)
- **Floating-point** (`FP`, IEEE 754)
- **Strings** (`String`, concat, contains)
- **Finite domains** (FiniteDomain)

---

## Integration Architecture

### Proposed Architecture

Unlike LeanAide (which is a standalone server), Z3 will be integrated as a **library service**:

```
Frontend (useZ3 hook)
  ↓ POST /api/v1/z3/solve
BubbleLab API (z3.ts)
  ↓ Direct Python execution
Z3 Python Library (z3.py)
  ↓ Native C++ core
Z3 Theorem Prover
```

### Key Differences from LeanAide

| Aspect | LeanAide | Z3 Prover |
|--------|----------|------------|
| **Type** | Standalone server (port 7654) | Python library |
| **Communication** | HTTP proxy | Direct Python execution |
| **State** | Stateless (per request) | Can have persistent solver state |
| **API** | Task-based (prove, verify) | Expression-based (SMTLIB2) |
| **Complexity** | High (theorem proving) | Medium (constraint solving) |

---

## Implementation Plan

### Phase 1: API Routes and Schemas (Day 1)

#### 1.1 Create Schemas (`z3.ts`)

**Request/Response Schemas:**

```typescript
// SMT Solve Request
Z3SolveRequestSchema = {
  smtlib2: string;           // SMTLIB2 commands
  timeout?: number;          // Timeout in ms (default: 30000)
  logic?: string;            // Logic: QF_BV, LIA, AUFLIA, etc.
}

// SMT Solve Response
Z3SolveResponseSchema = {
  result: 'sat' | 'unsat' | 'unknown';
  model?: Record<string, unknown>;
  statistics?: Record<string, unknown>;
  error?: string;
}

// Optimization Request
Z3OptimizeRequestSchema = {
  objectives: Array<{
    expression: string;
    type: 'maximize' | 'minimize';
  }>;
  constraints?: string[];   // SMTLIB2 constraints
  timeout?: number;
}

// Optimization Response
Z3OptimizeResponseSchema = {
  status: 'optimal' | 'unsat' | 'unknown';
  model?: Record<string, unknown>;
  objective_values?: Record<string, number>;
  error?: string;
}

// Simplify Request
Z3SimplifyRequestSchema = {
  expression: string;        // Expression to simplify
  assumptions?: string[];    // Assumptions
}

// Simplify Response
Z3SimplifyResponseSchema = {
  result: string;
  error?: string;
}

// Tactic Request
Z3TacticRequestSchema = {
  goal: string;              // Goal expression
  tactic: string;            // Tactic name (e.g., "simplify", "sat")
  params?: Record<string, unknown>;
}

// Tactic Response
Z3TacticResponseSchema = {
  status: 'sat' | 'unsat' | 'unknown';
  goals: string[];           // Subgoals
  model?: Record<string, unknown>;
  error?: string;
}
```

**Routes:**
- `POST /api/v1/z3/solve` - General SMT solving
- `POST /api/v1/z3/optimize` - Optimization problems
- `POST /api/v1/z3/simplify` - Expression simplification
- `POST /api/v1/z3/tactic` - Apply tactic to goal
- `GET /api/v1/z3/tactics` - List available tactics
- `GET /api/v1/z3/logics` - List supported logics
- `GET /api/v1/z3/version` - Get Z3 version

#### 1.2 Create API Routes (`routes/z3.ts`)

**Implementation Details:**

```typescript
import { z3 } from 'z3';
import { Z3Exception } from 'z3';

// Helper to execute Z3 in controlled environment
async function executeZ3<T>(
  fn: () => T,
  timeout: number
): Promise<{ result: T; error?: string }> {
  try {
    // Z3 operations can be long-running, need timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const result = await fn();

    clearTimeout(timeoutId);
    return { result };
  } catch (error: any) {
    return {
      result: null as T,
      error: error.message || 'Z3 execution failed'
    };
  }
}
```

**Key Endpoints:**

```typescript
// POST /api/v1/z3/solve
app.openapi(solveRoute, async (c) => {
  const { smtlib2, timeout = 30000 } = c.req.valid('json');

  try {
    const s = new z3.Solver();
    s.from_string(smtlib2);

    const result = await executeZ3(
      () => s.check(),
      timeout
    );

    if (result.error) {
      return c.json({ error: result.error }, 500);
    }

    const z3Result = result.result;

    let model: Record<string, unknown> | undefined;
    if (z3Result === z3.sat) {
      const m = s.model();
      model = {};
      for (const decl in m) {
        const val = m[decl];
        model[decl] = val ? val.toString() : null;
      }
    }

    return c.json({
      result: z3Result === z3.sat ? 'sat' :
             z3Result === z3.unsat ? 'unsat' : 'unknown',
      model,
      statistics: s.statistics()
    }, 200);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

// Similar for other endpoints...
```

---

### Phase 2: Service Bubble (Day 1)

#### 2.1 Create Z3ProverBubble (`service-bubbles/z3prover-bubble.ts`)

**Operations:**

```typescript
const Z3OperationSchema = z.enum([
  'health_check',
  'solve_smt',
  'optimize',
  'simplify',
  'apply_tactic',
  'get_tactics',
  'get_logics',
  'get_version',
  'fixedpoint_query',
]);

const Z3ParamsSchema = z.object({
  operation: Z3OperationSchema,

  // SMT solving
  smtlib2: z.string().optional(),
  logic: z.string().optional(),
  timeout: z.number().min(1000).max(600000).default(30000),

  // Optimization
  objectives: z.array(z.object({
    expression: z.string(),
    type: z.enum(['maximize', 'minimize']),
  })).optional(),

  // Tactics
  goal: z.string().optional(),
  tactic: z.string().optional(),
  tacticParams: z.record(z.unknown()).optional(),

  // Fixedpoint
  rules: z.array(z.string()).optional(),
  query: z.string().optional(),
});

export class Z3ProverBubble extends ServiceBubble<Z3Params, Z3Result> {
  static readonly service = 'openevolve';
  static readonly authType = null; // No auth needed for local library
  static readonly bubbleName = 'z3prover' as const;
  static readonly type = 'service' as const;

  // Implementation...
}
```

---

### Phase 3: Frontend Integration (Day 2)

#### 3.1 Create React Hook (`useZ3.ts`)

**Hook Structure:**

```typescript
export function useZ3() {
  const [isSolving, setIsSolving] = useState(false);
  const [result, setResult] = useState<Z3Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  const solveSMT = async (smtlib2: string, options?: Z3Options) => {
    setIsSolving(true);
    setError(null);

    try {
      const response = await z3Api.solve({ smtlib2, ...options });
      setResult(response);
      return response;
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsSolving(false);
    }
  };

  const optimize = async (objectives: Z3Objective[], options?: Z3Options) => {
    // Similar implementation
  };

  const simplify = async (expression: string, assumptions?: string[]) => {
    // Similar implementation
  };

  return {
    solveSMT,
    optimize,
    simplify,
    isSolving,
    result,
    error,
  };
}
```

#### 3.2 Create API Client (`endpoints.ts`)

```typescript
export const z3Api = {
  solve: async (data: Z3SolveRequest) => {
    return await apiClient.post<Z3SolveResponse>('/z3/solve', data);
  },

  optimize: async (data: Z3OptimizeRequest) => {
    return await apiClient.post<Z3OptimizeResponse>('/z3/optimize', data);
  },

  simplify: async (data: Z3SimplifyRequest) => {
    return await apiClient.post<Z3SimplifyResponse>('/z3/simplify', data);
  },

  applyTactic: async (data: Z3TacticRequest) => {
    return await apiClient.post<Z3TacticResponse>('/z3/tactic', data);
  },

  getTactics: async () => {
    return await apiClient.get<string[]>('/z3/tactics');
  },

  getLogics: async () => {
    return await apiClient.get<string[]>('/z3/logics');
  },

  getVersion: async () => {
    return await apiClient.get<{ version: string }>('/z3/version');
  },
};
```

---

### Phase 4: Python Backend Integration (Day 2)

#### 4.1 Create Z3 Service (`services/z3.py`)

```python
"""
Z3 Solver Service for BubbleLab API
"""

from z3 import *
import json
import traceback
from typing import Dict, Any, List, Optional

class Z3SolverError(Exception):
    """Custom Z3 solver error"""
    pass

class Z3Service:
    """Z3 Solver service wrapper"""

    def __init__(self):
        self._solver = None
        self._optimize = None
        self._fixedpoint = None

    def solve_smt(self, smtlib2: str, logic: Optional[str] = None, timeout: int = 30000) -> Dict[str, Any]:
        """
        Solve SMT problem expressed in SMTLIB2 format

        Args:
            smtlib2: SMTLIB2 command string
            logic: Optional logic specification (QF_BV, LIA, etc.)
            timeout: Timeout in milliseconds

        Returns:
            Dict with 'result' ('sat', 'unsat', 'unknown'), 'model', 'statistics'
        """
        try:
            s = Solver()

            # Set logic if specified
            if logic:
                s.set(logic=logic)

            # Parse and add assertions
            s.from_string(smtlib2)

            # Check satisfiability
            result = s.check()

            response = {
                'result': 'sat' if result == sat else 'unsat' if result == unsat else 'unknown',
            }

            # Extract model if SAT
            if result == sat:
                model = s.model()
                model_dict = {}
                for decl in model:
                    try:
                        val = model[decl]
                        model_dict[str(decl)] = str(val) if val else None
                    except:
                        pass
                response['model'] = model_dict

            # Get statistics
            response['statistics'] = s.statistics()

            return response

        except Z3Exception as e:
            raise Z3SolverError(f"Z3 solving failed: {str(e)}")
        except Exception as e:
            raise Z3SolverError(f"Solve error: {str(e)}")

    def optimize(self, objectives: List[Dict[str, str]],
                 constraints: Optional[List[str]] = None,
                 timeout: int = 30000) -> Dict[str, Any]:
        """
        Solve optimization problem

        Args:
            objectives: List of {expression, type} where type is 'maximize' or 'minimize'
            constraints: Optional list of SMTLIB2 constraint strings
            timeout: Timeout in milliseconds

        Returns:
            Dict with 'status', 'model', 'objective_values'
        """
        try:
            opt = Optimize()

            # Add constraints if provided
            if constraints:
                for constraint in constraints:
                    opt.add(eval(constraint))

            # Add objectives
            handles = {}
            for obj in objectives:
                expr = eval(obj['expression'])
                if obj['type'] == 'maximize':
                    handles[obj['expression']] = opt.maximize(expr)
                else:
                    handles[obj['expression']] = opt.minimize(expr)

            # Optimize
            result = opt.check()

            response = {
                'status': 'optimal' if result == optimal else 'unsat' if result == unsat else 'unknown',
            }

            # Extract model and objective values
            if result == optimal:
                model = opt.model()
                model_dict = {}
                for decl in model:
                    try:
                        val = model[decl]
                        model_dict[str(decl)] = str(val) if val else None
                    except:
                        pass
                response['model'] = model_dict

                # Get objective values
                obj_values = {}
                for expr, handle in handles.items():
                    obj_values[expr] = opt.value(handle)
                response['objective_values'] = obj_values

            return response

        except Z3Exception as e:
            raise Z3SolverError(f"Z3 optimization failed: {str(e)}")
        except Exception as e:
            raise Z3SolverError(f"Optimization error: {str(e)}")

    def simplify(self, expression: str,
                 assumptions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simplify expression

        Args:
            expression: Expression to simplify
            assumptions: Optional list of assumption strings

        Returns:
            Dict with 'result' (simplified expression)
        """
        try:
            # Parse assumptions
            ctx = None
            if assumptions:
                ctx = Context()
                assm_list = []
                for a in assumptions:
                    assm_list.append(eval(a, globals(), {'ctx': ctx}))

            # Simplify
            expr = eval(expression)
            simplified = simplify(expr, *assumptions if assumptions else [])

            return {
                'result': str(simplified)
            }

        except Z3Exception as e:
            raise Z3SolverError(f"Z3 simplification failed: {str(e)}")
        except Exception as e:
            raise Z3SolverError(f"Simplification error: {str(e)}")

    def apply_tactic(self, goal: str, tactic: str,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply tactic to goal

        Args:
            goal: Goal expression
            tactic: Tactic name
            params: Optional tactic parameters

        Returns:
            Dict with 'status', 'goals', 'model'
        """
        try:
            g = Goal()
            g.add(eval(goal))

            t = Tactic(tactic, **(params or {}))
            result = t(g)

            response = {
                'status': str(result),
            }

            # Extract subgoals
            goals = []
            for subgoal in result:
                goals.append(str(subgoal.as_expr()))
            response['goals'] = goals

            return response

        except Z3Exception as e:
            raise Z3SolverError(f"Z3 tactic failed: {str(e)}")
        except Exception as e:
            raise Z3SolverError(f"Tactic error: {str(e)}")

    def get_tactics(self) -> List[str]:
        """Get list of available tactics"""
        try:
            tactics = [
                'simplify', 'sat', 'sat-preprocess', 'solve-eqs',
                'bit-blast', 'pb-lemma', 'nlsat', 'qff',
                'snf', 'tseitin-cnf', 'tseitin-cnf-core',
                'der', 'factor-sleep', 'fm', 'lift-ite',
                'max-bv-sharding', 'pb-rewrite', 'propagate-values',
                'recover-01', 'smt', 'subst-cov', 'ujf',
                # Add more as needed...
            ]
            return tactics
        except Exception as e:
            raise Z3SolverError(f"Failed to get tactics: {str(e)}")

    def get_logics(self) -> List[str]:
        """Get list of supported logics"""
        return [
            'AUFLIRA', 'AUFLIRF', 'AUFNIRA', 'BV', 'BVREF',
            'HORN', 'LIA', 'LRA', 'NIA', 'NRA', 'QF_ABV',
            'QF_AUFBV', 'QF_AUFLIA', 'QF_BV', 'QF_IDL',
            'QF_LIA', 'QF_LRA', 'QF_NIA', 'QF_NRA',
            'QF_UF', 'QF_UFBV', 'UFLRA', 'UF', 'UFBV',
            'QF_AX', 'QF_S', 'SMT', 'ALL',
        ]

    def get_version(self) -> Dict[str, str]:
        """Get Z3 version information"""
        try:
            return {
                'version': get_version_string(),
                'full_version': get_full_version(),
            }
        except Exception as e:
            raise Z3SolverError(f"Failed to get version: {str(e)}")
```

---

### Phase 5: Testing and Validation (Day 2-3)

#### 5.1 Unit Tests

**Test Cases:**

```typescript
describe('Z3 API', () => {
  test('solve simple SAT', async () => {
    const response = await z3Api.solve({
      smtlib2: `
        (declare-const x Int)
        (declare-const y Int)
        (assert (> x 0))
        (assert (< x 5))
        (assert (= y (+ x 1)))
        (check-sat)
      `
    });

    expect(response.result).toBe('sat');
    expect(response.model).toBeDefined();
  });

  test('solve UNSAT', async () => {
    const response = await z3Api.solve({
      smtlib2: `
        (declare-const x Int)
        (assert (> x 0))
        (assert (< x 0))
        (check-sat)
      `
    });

    expect(response.result).toBe('unsat');
  });

  test('optimization', async () => {
    const response = await z3Api.optimize({
      objectives: [
        { expression: 'x', type: 'maximize' }
      ],
      constraints: [
        '(declare-const x Int)',
        '(assert (< x 10))',
        '(assert (> x 0))'
      ]
    });

    expect(response.status).toBe('optimal');
    expect(response.objective_values?.x).toBe(9);
  });

  test('simplification', async () => {
    const response = await z3Api.simplify({
      expression: '(+ x x (* 2 x))',
      assumptions: ['(declare-const x Int)']
    });

    expect(response.result).toBeTruthy();
  });
});
```

#### 5.2 Integration Tests

**Test Scenarios:**

1. **SMT Solving**
   - Boolean satisfiability
   - Integer arithmetic
   - Bit-vector operations
   - Array operations
   - Quantifier-free vs. quantified formulas

2. **Optimization**
   - Single objective
   - Multiple objectives
   - Unbounded vs. bounded optimization

3. **Tactics**
   - Apply common tactics (simplify, sat, qfnia, etc.)
   - Verify tactic results

4. **Error Handling**
   - Invalid SMTLIB2 syntax
   - Type errors
   - Timeout handling
   - Unsupported features

---

### Phase 6: Documentation (Day 3)

#### 6.1 API Documentation

Create comprehensive API docs:

- `Z3PROVER_API_REFERENCE.md` - Complete API reference
- `Z3PROVER_EXAMPLES.md` - Usage examples
- `Z3PROVER_GUIDE.md` - Getting started guide

#### 6.2 Update Existing Documentation

- Update `ARCHITECTURE.md` with Z3 integration
- Update `API_REFERENCE.md` with Z3 endpoints
- Create `Z3PROVER_INTEGRATION_COMPLETE.md` summary

---

## File Structure

```
BubbleLab/
├── apps/
│   └── bubblelab-api/
│       ├── src/
│       │   ├── routes/
│       │   │   └── z3.ts                    [CREATE]
│       │   ├── schemas/
│       │   │   └── z3.ts                    [CREATE]
│       │   ├── services/
│       │   │   └── z3.py                    [CREATE]
│       │   ├── config/
│       │   │   └── env.ts                   [MODIFY - add Z3_* env vars]
│       │   └── index.ts                    [MODIFY - register Z3 routes]
│       └── .env.example                    [MODIFY - add Z3_* env vars]
├── integrations/
│   └── openevolve/
│       ├── service-bubbles/
│       │   └── z3prover-bubble.ts         [CREATE]
│       └── index.ts                       [MODIFY - export Z3ProverBubble]
└── packages/
    └── bubble-shared-schemas/
        └── src/
            └── z3-*.ts                      [CREATE - Z3 types]
```

---

## Environment Variables

```bash
# Z3 Prover Configuration
Z3_TIMEOUT=30000              # Default timeout in milliseconds (30 seconds)
Z3_MAX_MEMORY=4096            # Maximum memory in MB
Z3_ENABLE_TRACES=false        # Enable Z3 tracing for debugging
Z3_LOG_LEVEL=info            # Logging level
```

---

## Complexity Assessment

### Integration Complexity: **MEDIUM-HIGH**

**Challenges:**
1. Z3 is a library (not a server) - needs Python integration in BubbleLab API
2. SMTLIB2 parsing - must validate and handle parse errors
3. Timeout handling - long-running solvers need proper cancellation
4. Model serialization - converting Z3 models to JSON
5. Multiple theories - different features for different logics

**Advantages:**
1. Well-documented Python API
2. Mature, stable library
3. No external service dependencies
4. Fast execution (C++ core)
5. Comprehensive feature set

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Memory leaks** | Medium | Medium | Use context isolation, clear solver objects |
| **Long execution** | High | Medium | Implement timeout for all operations |
| **Parse errors** | Medium | Low | Good error messages, validation |
| **Python GIL** | Low | Low | Z3 releases GIL during solving |
| **Z3 not installed** | Medium | High | Clear error message, installation guide |

---

## Success Criteria

### Phase 1: API Layer
- [x] All routes implemented
- [x] Schemas defined and validated
- [x] Error handling comprehensive
- [x] Timeout handling working

### Phase 2: Service Bubble
- [x] Z3ProverBubble created
- [x] All operations implemented
- [x] Resilience wrapper integrated
- [x] Proper TypeScript types

### Phase 3: Frontend
- [x] React hook created
- [x] API client implemented
- [x] Error handling in UI
- [x] Loading states

### Phase 4: Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Example usage works
- [x] Documentation complete

---

## Dependencies

### Required
- `z3-solver` Python package (from Z3 source)
- Z3 built and installed on system

### Installation

```bash
# Install Z3 from source
cd z3prover
python scripts/mk_make.py
cd build
make
sudo make install

# Install Python bindings
cd src/api/python
pip install -e .
```

Or via pip:
```bash
pip install z3-solver
```

---

## Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| **Day 1 Morning** | Create schemas and API routes | `z3.ts` (routes & schemas) |
| **Day 1 Afternoon** | Create service bubble | `z3prover-bubble.ts` |
| **Day 2 Morning** | Create Python service | `services/z3.py` |
| **Day 2 Afternoon** | Create React hook and API client | `useZ3.ts`, API endpoints |
| **Day 3 Morning** | Testing and bug fixes | Passing tests |
| **Day 3 Afternoon** | Documentation | Complete docs |

---

## Comparison with Similar Services

| Feature | Z3 Prover | LeanAide | Knowledge Engine |
|---------|-----------|----------|------------------|
| **Type** | Library | Server | Service |
| **Integration** | Direct Python | HTTP proxy | HTTP proxy |
| **State** | Stateless (per request) | Stateless | Stateful |
| **Complexity** | Medium | High | High |
| **Dependencies** | Z3 Python | LeanAide server | Vector DB |
| **Use Cases** | SMT solving | Theorem proving | Semantic search |

---

## Example Usage

### Frontend

```typescript
import { useZ3 } from '@/hooks/useZ3';

function SMTSolver() {
  const { solveSMT, isSolving, result, error } = useZ3();

  const handleSolve = async () => {
    const smtlib2 = `
      (declare-const x Int)
      (declare-const y Int)
      (assert (> x 0))
      (assert (< x 10))
      (assert (= y (* 2 x)))
      (check-sat)
    `;

    const response = await solveSMT(smtlib2);
    console.log('Result:', response.result);
    console.log('Model:', response.model);
  };

  return (
    <div>
      <button onClick={handleSolve} disabled={isSolving}>
        {isSolving ? 'Solving...' : 'Solve'}
      </button>
      {result && (
        <div>
          <h3>Result: {result.result}</h3>
          {result.model && (
            <pre>{JSON.stringify(result.model, null, 2)}</pre>
          )}
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
```

### Service Bubble

```typescript
import { Z3ProverBubble } from '@bubblelab/openevolve';

const bubble = new Z3ProverBubble({
  operation: 'solve_smt',
  smtlib2: `
    (declare-const x Int)
    (assert (> x 0))
    (assert (< x 5))
    (check-sat)
  `,
  timeout: 30000,
});

const result = await bubble.execute();
if (result.success) {
  console.log('SAT!', result.data.model);
}
```

### Direct API

```bash
curl -X POST http://localhost:3001/api/v1/z3/solve \
  -H "Content-Type: application/json" \
  -d '{
    "smtlib2": "(declare-const x Int) (assert (> x 0)) (check-sat)"
  }'
```

---

## Next Steps

1. **Review and Approval** - Review this plan and adjust as needed
2. **Setup Z3** - Ensure Z3 is built and installed
3. **Begin Implementation** - Start with Phase 1 (API routes)
4. **Iterative Development** - Complete each phase before moving to next
5. **Continuous Testing** - Test each phase thoroughly
6. **Documentation** - Document as you go, not at the end

---

**Plan Author:** Claude (Sonnet 4.5)
**Date:** 2026-01-24
**Status:** READY FOR IMPLEMENTATION
