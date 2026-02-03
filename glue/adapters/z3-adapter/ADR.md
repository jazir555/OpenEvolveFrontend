# Architecture Decision Record: Z3 Integration

## Status
**Accepted**

## Context

Z3 (Microsoft Z3 Theorem Prover) is a high-performance SMT (Satisfiability Modulo Theories) solver that provides critical capabilities for the OpenEvolve system:

- **Constraint Solving**: Solves complex constraint satisfaction problems across arithmetic, bit-vectors, arrays, and quantifiers
- **Theorem Proving**: Formal verification of mathematical theorems and properties
- **Optimization**: Multi-objective optimization with linear and non-linear objectives
- **Proof Generation**: Extractable proofs for unsatisfiability results
- **SMT-LIB2 Support**: Industry-standard interface for theorem proving

Z3 is used throughout OpenEvolve for:
- Quality gate enforcement (constraint-based verification)
- Workflow stage validation
- Knowledge graph constraint solving
- Cross-validation with LeanAIDE formal proofs

The integration must support:
1. **Dual Interface**: Both Python API (`z3-solver`) and CLI interface
2. **High Performance**: Sub-second solving for typical constraints
3. **Robustness**: Graceful degradation when Z3 unavailable
4. **Isolation**: Adapter pattern to prevent direct Z3 dependencies in core code

## Decision

### Architecture Pattern: Sidecar Adapter with Dual Interface

We chose a **Sidecar Adapter Pattern** with the following characteristics:

```
[Core OpenEvolve] --> [Z3 Adapter (Canonical Layer)] --> [Z3 Engine]
                                                      |
                                                      +-- [Z3 CLI (fallback)]
```

### Key Design Choices

1. **Adapter Location**: `/glue/adapters/z3-adapter/`
   - Isolated from core-projects (Law of Air Gap)
   - Rewritten Z3 utilities in adapter layer (no imports from core)
   - Canonical schema at `/glue/schemas/z3-canonical.json`

2. **Interface Strategy**: Python API primary, CLI fallback
   - Primary: `z3-solver` Python package (>=4.12.0)
   - Fallback: Subprocess to `z3` binary
   - Automatic detection at initialization

3. **Data Flow**:
   ```
   Input (Canonical Format)
       --> Z3SolverEngine.solve_constraints()
       --> Z3Result (Canonical Format)
       --> Output (Canonical Format)
   ```

4. **API Design**: MCP (Model Context Protocol) compatible
   - Standardized tool interface for AI systems
   - JSON request/response
   - 8 core tools covering major Z3 capabilities

## Consequences

### Positive Benefits

1. **Performance**: Solves typical constraints in <100ms via Python API
2. **Reliability**: Fallback to CLI ensures operation even if Python bindings fail
3. **Flexibility**: Supports multiple input formats (SMT-LIB, natural language, structured constraints)
4. **Integration**: Direct bridge to LeanAIDE for cross-validation
5. **Extensibility**: MCP interface enables easy tool composition
6. **Isolation**: Core system never directly depends on Z3

### Negative Tradeoffs

1. **Duplication**: Z3 utilities rewritten in adapter layer (law of air gap)
2. **Overhead**: Adapter layer adds ~10-20ms per call
3. **Complexity**: Dual interface (Python + CLI) increases maintenance burden
4. **Memory**: Z3 Python bindings can consume significant memory for large problems
5. **State**: No persistent state between calls (must be stateless for horizontal scaling)

### Known Limitations

1. **Timeout Handling**: Z3's Python API timeout is unreliable; use process-level timeouts
2. **Large Models**: Models with >10,000 variables may cause performance degradation
3. **Quantifiers**: Nested quantifiers can cause Z3 to hang (use timeout)
4. **Proof Extraction**: Only available for UNSAT results with proof generation enabled
5. **Parallel Solving**: Portfolio mode limited to 4 parallel strategies (Python GIL)

## Implementation Details

### Core Components

#### 1. Z3SolverEngine
```python
class Z3SolverEngine:
    def solve_constraints(
        variables: List[Z3Variable],
        constraints: List[Z3Constraint],
        timeout: float = 30.0
    ) -> Z3SolverResult
```

**Capabilities**:
- Constraint satisfaction (SAT/UNSAT/UNKNOWN)
- Model extraction (variable assignments)
- Multiple constraint types: BOOLEAN, INTEGER, REAL, BIT_VECTOR, ARRAY, STRING
- Configurable timeout (default: 30s)

**Example**:
```python
variables = [Z3Variable("x", Z3ConstraintType.INTEGER)]
constraints = [Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER)]
result = solver.solve_constraints(variables, constraints)
# result.status == Z3ResultStatus.SAT
# result.model.assignments == {"x": 1}
```

#### 2. Z3TheoremProver
```python
class Z3TheoremProver:
    def prove_theorem(
        theorem: str,
        assumptions: List[str] = [],
        extract_proof: bool = False
    ) -> Z3TheoremResult
```

**Capabilities**:
- Theorem proving via SMT-LIB
- Counterexample generation for false theorems
- Proof extraction (if enabled)
- Tactic selection (auto, smt, qflia, default)

**Example**:
```python
theorem = "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(check-sat)"
result = prover.prove_theorem(theorem)
# result.proven == True
# result.tactic_used == "smt"
```

#### 3. Z3AdvancedSolver (Portfolio Mode)
```python
class Z3AdvancedSolver:
    def solve_portfolio(
        smtlib: str,
        strategies: List[str] = ["default", "smt", "qflia"],
        timeout: float = 30.0
    ) -> PortfolioResult
```

**Capabilities**:
- Parallel solving with multiple strategies
- Automatic strategy selection
- Winner reporting with execution times

**Example**:
```python
result = solver.solve_portfolio(problem, strategies=["smt", "qflia"])
# result.winner_strategy == "qflia"
# result.parallel_speedup == 1.8
```

### API Endpoints (MCP Tools)

| Tool | Purpose | Timeout | Retry Strategy |
|------|---------|---------|----------------|
| `z3_solve_constraints` | Constraint satisfaction | 30s | 3 attempts, exponential backoff |
| `z3_optimize` | Optimization problems | 60s | 2 attempts, linear backoff |
| `z3_prove_theorem` | Theorem proving | 30s | 3 attempts, exponential backoff |
| `z3_translate_smt_to_lean` | SMT-LIB → Lean 4 | 10s | No retry (deterministic) |
| `z3_solve_incremental` | Incremental solving | N/A | No retry (stateful) |
| `z3_extract_proof` | Proof extraction | 30s | 2 attempts, exponential backoff |
| `z3_analyze_problem` | Problem analysis | 5s | No retry (fast) |
| `z3_solve_portfolio` | Portfolio solving | 60s | No retry (parallel) |

### Data Flow Diagrams

#### Constraint Solving Flow
```
[Client]
  --> {variables: [...], constraints: [...], timeout: 30}
[Z3 Adapter]
  --> Normalize to Canonical Schema
  --> Parse variables to Z3Variable objects
  --> Parse constraints to Z3Constraint objects
[Z3 Engine]
  --> Convert to Z3 Python objects
  --> solver.add() for each constraint
  --> solver.check()
  --> solver.model()
[Z3 Adapter]
  --> Convert Z3Result to Canonical Format
  --> Add execution_time, errors, warnings
[Client]
  <-- {status: "sat", model: {...}, execution_time: 0.05}
```

#### Theorem Proving Flow
```
[Client]
  --> {theorem: "(set-logic LIA)...", assumptions: [...]}
[Z3 Adapter]
  --> Detect if SMT-LIB or natural language
  --> If natural language, translate to SMT-LIB
[Z3 Engine]
  --> Parse SMT-LIB with parse_smt2_string()
  --> Apply tactic (auto/smt/qflia)
  --> Check satisfiability
  --> Extract proof if UNSAT and extract_proof=True
[Z3 Adapter]
  --> Map to Z3TheoremResult
  --> Add tactic_used, counterexample, proof
[Client]
  <-- {proven: true, proof: "...", tactic: "smt"}
```

### Configuration Requirements

#### Environment Variables
```bash
# Z3 Configuration
Z3_TIMEOUT=30              # Default timeout (seconds)
Z3_PROOF_GENERATION=true  # Enable proof extraction
Z3_MAX_MEMORY=4096         # Max memory (MB)
Z3_PARALLEL_STRATEGIES=4   # Number of parallel strategies

# Z3 Binary (fallback)
Z3_BINARY_PATH=/usr/bin/z3 # Path to z3 binary

# Adapter Configuration
Z3_ADAPTER_HOST=z3-adapter # Service name (Docker)
Z3_ADAPTER_PORT=8000       # HTTP port
Z3_LOG_LEVEL=INFO          # Logging level
```

#### Python Configuration
```python
Z3Config(
    timeout=30.0,
    proof_generation=True,
    max_memory=4096,
    parallel_strategies=4
)
```

### LeanAIDE Bridge Integration

Z3 integrates with LeanAIDE via `Z3LeanAideBridge`:

```python
bridge = Z3LeanAideBridge()

# Translate SMT-LIB to Lean 4
result = await bridge.translate_smt_to_lean(smtlib_content)

# Verify with both systems
result = await bridge.verify_with_both(
    problem,
    strategy=VerificationStrategy.PARALLEL
)
```

**Verification Strategies**:
- `Z3_FIRST`: Try Z3, fall back to Lean
- `LEAN_FIRST`: Try Lean, fall back to Z3
- `PARALLEL`: Run both, take first success
- `CONSENSUS`: Both must agree
- `ADAPTIVE`: Choose based on problem type

## Gotchas

### API Quirks Discovered

1. **Timeout Unreliability**:
   - Python API `set_option(timeout=...)` often ignored
   - **Solution**: Use process-level timeout via `asyncio.wait_for()`

2. **Proof Generation**:
   - Must enable before adding constraints: `solver.set(proof=True)`
   - Only works for UNSAT results
   - **Solution**: Check `result.status == UNSAT` before extracting proof

3. **Model Extraction**:
   - Calling `solver.model()` on UNSAT result raises exception
   - **Solution**: Always check `solver.check()` first

4. **Bit-Vector Widths**:
   - Must declare width in variable: `Z3Variable("x", BIT_VECTOR, bit_width=32)`
   - **Gotcha**: Default width varies by logic (QF_BV vs UFBV)

5. **Incremental Solving**:
   - `push()` and `pop()` must be balanced
   - **Gotcha**: Popping below base level raises exception
   - **Solution**: Track stack depth manually

6. **Parse SMT-LIB**:
   - `parse_smt2_string()` requires `decls` dict for function symbols
   - **Gotcha**: Empty decls dict causes parse errors for complex formulas
   - **Solution**: Extract declarations from solver before parsing

### Version Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|---------------------|-------|
| z3-solver (Python) | 4.12.0 | 4.13.0+ | 4.13 adds better Python API |
| Z3 binary | 4.12.0 | 4.13.0+ | Must match Python version |
| Python | 3.10 | 3.11+ | 3.11 improves performance |

### Non-Obvious Behaviors

1. **Memory Leaks**:
   - Z3 solver objects accumulate memory if reused
   - **Solution**: Create new solver per request, or call `solver.reset()` periodically

2. **GIL Contention**:
   - Z3 Python calls hold GIL, blocking other threads
   - **Solution**: Use `run_in_executor()` for concurrent solves

3. **Quantifier Alternations**:
   - `∀∃` (forall-exists) patterns can cause Z3 to hang
   - **Solution**: Always use timeout for quantified formulas

4. **Array Operations**:
   - Arrays are lazy by default; `store` doesn't copy
   - **Gotcha**: Multiple `store` operations create chain, not array
   - **Solution**: Use `ConstArray` for constant arrays

5. **SMT-LIB Logic Declarations**:
   - `(set-logic QF_LIA)` required before variable declarations
   - **Gotcha**: Wrong logic causes parse errors
   - **Solution**: Auto-detect logic or use `ALL`

6. **String Constraints**:
   - String operations require `QF_S` logic
   - **Gotcha**: String operations unsupported in `LIA` logic
   - **Solution**: Separate string constraints into different solver

## Circuit Breaker Configuration

### Timeout Values
```python
TIMEOUTS = {
    "solve_constraints": 30.0,    # seconds
    "optimize": 60.0,              # seconds
    "prove_theorem": 30.0,         # seconds
    "extract_proof": 30.0,         # seconds
    "solve_portfolio": 60.0,       # seconds
    "analyze_problem": 5.0,        # seconds
    "incremental": None,           # no timeout (stateful)
}
```

### Retry Strategies

#### Exponential Backoff (Default)
```python
@retry(
    attempts=3,
    base_delay=1.0,      # seconds
    max_delay=10.0,      # seconds
    exponential=2.0,     # backoff multiplier
    jitter=0.1           # add random jitter
)
async def solve_with_retry(...):
    ...
```

**Usage**: Constraint solving, theorem proving

#### Linear Backoff (Optimization)
```python
@retry(
    attempts=2,
    base_delay=2.0,
    max_delay=5.0,
    exponential=1.0      # linear
)
async def optimize_with_retry(...):
    ...
```

**Usage**: Optimization problems (long-running)

#### No Retry (Fast/Stateful)
```python
# No retry decorator
async def translate_smt_to_lean(...):
    ...
```

**Usage**: Translation, incremental solving, problem analysis

### Failure Thresholds

```python
CIRCUIT_BREAKER = {
    "failure_threshold": 5,        # open after 5 failures
    "success_threshold": 2,        # close after 2 successes
    "timeout": 60.0,               # open state duration (seconds)
    "half_open_max_calls": 1       # test call in half-open state
}
```

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit tripped, requests fail immediately
- **HALF_OPEN**: Test if service recovered, allow 1 call

**Triggers**:
- 5 consecutive failures (timeout or exception)
- 3 consecutive timeouts (>30s)
- Memory usage >4GB

**Recovery**:
- 2 consecutive successes → CLOSE
- 60s timeout → HALF_OPEN
- Manual reset via API

### Error Classification

| Error Type | Retryable | Circuit Breaker | Fallback |
|------------|-----------|-----------------|----------|
| `Z3Exception` (timeout) | Yes | Yes | LeanAIDE |
| `Z3Exception` (parse error) | No | No | Return error |
| `MemoryError` | No | Yes | Return error |
| `ImportError` (Z3 unavailable) | No | Yes | CLI fallback |
| `subprocess.TimeoutExpired` | Yes | Yes | Return error |

## Security Considerations

### Input Validation

#### SMT-LIB Sanitization
```python
def validate_smtlib(smtlib: str) -> bool:
    """Reject malicious SMT-LIB patterns."""
    # Block file access
    if any(pattern in smtlib.lower() for pattern in ["(get-info", "(get-"]):
        raise ValueError("Info commands not allowed")

    # Block set-option (could disable security checks)
    if "(set-option" in smtlib:
        raise ValueError("Set-option not allowed")

    # Limit recursion depth
    paren_depth = smtlib.count("(") - smtlib.count(")")
    if abs(paren_depth) > 1000:
        raise ValueError("Excessive nesting")

    # Check for declared logic
    if "(set-logic" not in smtlib:
        raise ValueError("Logic declaration required")

    return True
```

#### Natural Language Input
```python
def validate_nl_input(problem: str) -> bool:
    """Sanitize natural language problem statements."""
    # Max length
    if len(problem) > 10000:
        raise ValueError("Problem too long")

    # Block shell commands
    shell_indicators = ["; rm", "| rm", "$(", "`", "$(syscall"]
    if any(indicator in problem for indicator in shell_indicators):
        raise ValueError("Shell commands not allowed")

    return True
```

### Authentication Requirements

**Z3 has no authentication** (local tool). Security is enforced at adapter layer:

```python
# Adapter-level rate limiting
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/z3/solve")
@limiter.limit("100/minute")  # Max 100 requests per minute
async def solve_constraints(request: Request):
    ...
```

### Data Sensitivity

#### Sensitive Data in Constraints
```python
# WARNING: Constraints may contain sensitive information
# Example: (assert (= password "secret123"))

# Best practice: Hash sensitive values before sending to Z3
def sanitize_constraint(constraint: str) -> str:
    """Replace potential secrets with hashes."""
    import re

    # Detect patterns that look like secrets
    secret_patterns = [
        (r'password\s*=\s*"([^"]+)"', lambda m: f'password = "{hash_sha256(m.group(1))}"'),
        (r'api_key\s*=\s*"([^"]+)"', lambda m: f'api_key = "{hash_sha256(m.group(1))}"'),
    ]

    for pattern, replacer in secret_patterns:
        constraint = re.sub(pattern, replacer, constraint, flags=re.IGNORECASE)

    return constraint
```

#### Logging Security
```python
# NEVER log constraint contents
logger.info(f"Solving with {len(constraints)} constraints")  # OK
logger.info(f"Constraints: {constraints}")  # BAD - leaks data

# Log only metadata
logger.info({
    "msg": "Solving constraints",
    "num_constraints": len(constraints),
    "constraint_types": [c.type for c in constraints],  # types only
    "correlation_id": ctx.id
})
```

### Resource Limits

```python
# Prevent resource exhaustion
MAX_CONSTRAINTS = 10000
MAX_VARIABLES = 5000
MAX_EXECUTION_TIME = 300.0  # 5 minutes
MAX_MEMORY_MB = 8192        # 8GB

def enforce_limits(constraints, variables):
    if len(constraints) > MAX_CONSTRAINTS:
        raise ValueError(f"Too many constraints: {len(constraints)} > {MAX_CONSTRAINTS}")

    if len(variables) > MAX_VARIABLES:
        raise ValueError(f"Too many variables: {len(variables)} > {MAX_VARIABLES}")
```

---

## References

- **Z3 GitHub**: https://github.com/Z3Prover/z3
- **Z3 Python API**: https://z3prover.github.io/api/html/namespacez3py.html
- **SMT-LIB Standard**: http://smtlib.cs.uiowa.edu/
- **OpenEvolve Integration**: `/openevolve/openevolve/knowledge_engine/integrations/z3_integration.py`
- **MCP Tools**: `/z3_mcp_tools.py`
- **LeanAIDE Bridge**: `/z3_leanaide_bridge.py`

**Created**: 2026-02-03
**Author**: OpenEvolve Architecture Team
**Status**: Accepted, Implemented
**Last Updated**: 2026-02-03
