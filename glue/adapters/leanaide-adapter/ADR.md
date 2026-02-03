# Architecture Decision Record: LeanAide Integration

## Status
**Accepted**

## Context

LeanAide is an AI-powered formal verification assistant built on Lean 4 theorem prover, providing advanced mathematical reasoning capabilities for the OpenEvolve system:

- **Formal Verification**: Machine-checked proofs with Lean 4 kernel
- **Tactic Automation**: Aesop tactic for automated proof search
- **Mathlib Integration**: Access to 500,000+ mathematical theorems
- **Semantic Search**: Embedding-based theorem retrieval
- **Code Generation**: Lean 4 code synthesis from natural language

LeanAide is used throughout OpenEvolve for:
- Mathematical problem decomposition
- Formal verification of evolutionary code changes
- Knowledge graph theorem validation
- Cross-validation with Z3 constraint solving

The integration must support:
1. **Async/Await Pattern**: Lean 4 compilation is slow, requires async interface
2. **Long-Running Operations**: Proof search can take 60-300 seconds
3. **Robust Fallbacks**: Mock implementation when Lean unavailable
4. **Bridge Integration**: Bidirectional translation with Z3

## Decision

### Architecture Pattern: Async Sidecar with Mock Fallback

We chose an **Async Sidecar Pattern** with the following characteristics:

```
[Core OpenEvolve] --> [LeanAide Adapter (Async Canonical Layer)] --> [Lean 4 Server]
                                                                  |
                                                                  +-- [Mock Fallback]
```

### Key Design Choices

1. **Adapter Location**: `/glue/adapters/leanaide-adapter/`
   - Isolated from core-projects (Law of Air Gap)
   - Rewritten Lean utilities in adapter layer
   - Canonical schema at `/glue/schemas/leanaide-canonical.json`

2. **Interface Strategy**: Client-Server with subprocess communication
   - Primary: `lean-client-python` for Lean 4 interaction
   - Fallback: Mock implementation for graceful degradation
   - Async interface via `asyncio.run_in_executor()`

3. **Data Flow**:
   ```
   Input (Canonical Format)
       --> LeanAideIntegration.verify_theorem()
       --> Lean 4 compilation + proving
       --> LeanAideResult (Canonical Format)
       --> Output (Canonical Format)
   ```

4. **API Design**: Task-based async pattern with correlation IDs
   - Correlation ID tracking for distributed tracing
   - UTC timestamp enforcement (Law of UTC)
   - Structured JSON logging

## Consequences

### Positive Benefits

1. **Formal Correctness**: Lean 4 kernel provides mathematical guarantees
2. **Mathlib Access**: 500K+ theorems for proof automation
3. **Semantic Search**: Embedding-based retrieval speeds proof finding
4. **Z3 Bridge**: Bidirectional SMT-LIB ↔ Lean 4 translation
5. **Graceful Degradation**: Mock fallback ensures system availability
6. **Async Design**: Non-blocking interface for long-running proofs

### Negative Tradeoffs

1. **Slow Compilation**: Lean 4 compilation adds 5-30s overhead
2. **Memory Intensive**: Lean 4 server requires 2-4GB RAM
3. **Steep Learning Curve**: Lean 4 tactic system is complex
4. **Limited Parallelism**: Single Lean 4 server (process-level isolation required)
5. **Cold Starts**: First proof after server start is slow (~10s)

### Known Limitations

1. **Proof Search Timeout**: No reliable timeout for infinite proof loops
2. **Mathlib Dependency**: Some proofs require specific mathlib versions
3. **Tactic Failure**: Aesop can fail on certain quantifier structures
4. **Code Generation**: Generated Lean code may not compile
5. **Incremental Solving**: Limited support for incremental proof updates

## Implementation Details

### Core Components

#### 1. LeanAideIntegration
```python
class LeanAideIntegration:
    async def verify_theorem(
        theorem: str,
        proof: Optional[str] = None,
        auto_prove: bool = True,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult
```

**Capabilities**:
- Theorem verification with optional proof
- Automated proof generation (auto_prove)
- Batch verification (parallel processing)
- Formal verification pipeline (generate + verify)

**Example**:
```python
integration = LeanAideIntegration(config)
result = await integration.verify_theorem(
    theorem="theorem add_comm (a b : Nat) : a + b = b + a := by simp",
    auto_prove=True,
    correlation_id="verify_001"
)
# result.verified == True
# result.proof contains Lean 4 proof
```

#### 2. Proof Generation
```python
async def generate_proof(
    theorem: str,
    search_depth: int = 10,
    timeout: int = 30,
    correlation_id: Optional[str] = None
) -> LeanAideResult
```

**Capabilities**:
- Automated proof search using Aesop
- Configurable search depth (default: 10)
- Timeout-based termination (default: 30s)
- Tactic application results

**Example**:
```python
result = await integration.generate_proof(
    theorem="∀ n : Nat, n + 0 = n",
    search_depth=15,
    timeout=60
)
# result.success == True
# result.proof contains step-by-step proof
```

#### 3. Tactic Application
```python
async def apply_tactic(
    goal: str,
    tactic: str,
    correlation_id: Optional[str] = None
) -> LeanAideResult
```

**Capabilities**:
- Apply single tactic to proof goal
- Return updated goal state
- Support for all Lean 4 tactics (simp, rw, cases, etc.)

**Example**:
```python
result = await integration.apply_tactic(
    goal="⊢ a + b = b + a",
    tactic="simp [add_comm]"
)
# result.proof contains new goal state
```

#### 4. Similar Theorem Search
```python
async def search_similar_theorems(
    query: str,
    num_results: int = 5,
    correlation_id: Optional[str] = None
) -> List[Dict[str, Any]]
```

**Capabilities**:
- Embedding-based semantic search
- Returns similar theorems from mathlib
- Similarity scores for ranking

**Example**:
```python
similar = await integration.search_similar_theorems(
    query="addition is commutative",
    num_results=5
)
# Returns theorems like add_comm, add_left_comm, etc.
```

### API Endpoints

| Endpoint | Purpose | Timeout | Async |
|----------|---------|---------|-------|
| `verify_theorem` | Verify theorem with proof | 60s | Yes |
| `generate_proof` | Auto-generate proof | 30s | Yes |
| `apply_tactic` | Apply tactic to goal | 10s | Yes |
| `search_similar_theorems` | Semantic search | 5s | Yes |
| `formal_verification_pipeline` | Full pipeline (gen + verify) | 90s | Yes |
| `batch_verify` | Verify multiple theorems | 60s each | Yes (parallel) |

### Data Flow Diagrams

#### Theorem Verification Flow
```
[Client]
  --> {theorem: "∀ n, n + 0 = n", auto_prove: true, correlation_id: "..." }
[LeanAide Adapter]
  --> Validate input (max length, no shell commands)
  --> Check if Lean 4 server available
  --> If unavailable, use mock implementation
[Lean 4 Server]
  --> Compile theorem to Lean 4
  --> Run tactic auto (Aesop)
  --> Check kernel verification
  --> Extract proof term
[LeanAide Adapter]
  --> Convert to LeanAideResult
  --> Add processing_time_ms, correlation_id
  --> Structured log (JSON lines)
[Client]
  <-- {success: true, verified: true, proof: "...", processing_time_ms: 1234}
```

#### Z3-LeanAide Bridge Flow
```
[Client]
  --> {problem: SMT-LIB constraint, strategy: PARALLEL}
[LeanAide Adapter]
  --> Detect problem type (SMT vs natural language)
[Z3-LeanAide Bridge]
  --> If SMT: translate to Lean 4 via SMTtoLeanTranslator
  --> If natural language: use Lean directly
  --> Verify with both Z3 and LeanAIDE
  --> Compare results (consensus checking)
[LeanAide Adapter]
  --> Combine results into CombinedVerificationResult
  --> Add confidence_score, recommendation
[Client]
  <-- {
       success: true,
       z3_result: {...},
       lean_result: {...},
       agreement: true,
       confidence_score: 0.95,
       recommendation: "Both verified"
     }
```

### Configuration Requirements

#### Environment Variables
```bash
# LeanAide Configuration
LEANAIDE_HOST=localhost          # Lean 4 server host
LEANAIDE_PORT=7654              # Lean 4 server port
LEANAIDE_TIMEOUT=300            # Default timeout (seconds)
LEAN_VERSION=4.0.0             # Lean 4 version

# Proof Search Configuration
AUTO_TACTIC_TIMEOUT=30         # Aesop timeout (seconds)
PROOF_SEARCH_DEPTH=10          # Max proof search depth
MAX_PROOF_STEPS=100            # Max steps in proof
ENABLE_AESOP=true              # Enable Aesop tactic
ENABLE_MATHLIB=true            # Enable mathlib integration

# Cache Configuration
CACHE_PROOFS=true              # Cache generated proofs
PROOF_CACHE_TTL=3600           # Cache TTL (seconds)

# Adapter Configuration
LEANAIDE_ADAPTER_HOST=leanaide-adapter  # Service name (Docker)
LEANAIDE_ADAPTER_PORT=8001              # HTTP port
LEANAIDE_LOG_LEVEL=INFO                 # Logging level
```

#### Python Configuration
```python
config = {
    "lean_version": "4.0.0",
    "auto_tactic_timeout": 30,
    "proof_search_depth": 10,
    "max_proof_steps": 100,
    "enable_auto_search": True,
    "enable_aesop": True,
    "enable_mathlib": True,
    "cache_proofs": True,
    "proof_cache_ttl": 3600,
    "embedding_search": {
        "enabled": True,
        "num_results": 5,
        "similarity_threshold": 0.8
    },
    "verification": {
        "check_termination": True,
        "check_type_correctness": True,
        "check_axioms": True
    }
}
```

### Z3 Bridge Integration

LeanAide integrates with Z3 via `Z3LeanAideBridge`:

```python
bridge = Z3LeanAideBridge()

# Verify with both systems
result = await bridge.verify_with_both(
    problem="(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(check-sat)",
    strategy=VerificationStrategy.PARALLEL
)

# Cross-validate (translate SMT to Lean, verify both ways)
result = await bridge.cross_validate(smtlib_problem)
```

**Verification Strategies**:
- `Z3_FIRST`: Try Z3, fall back to Lean
- `LEAN_FIRST`: Try Lean, fall back to Z3
- `PARALLEL`: Run both, take first success
- `CONSENSUS`: Both must agree
- `ADAPTIVE`: Choose based on problem type

## Gotchas

### API Quirks Discovered

1. **Lean 4 Compilation Overhead**:
   - First proof after server start takes 5-10s (server warmup)
   - **Solution**: Pre-warm server with dummy proof on startup

2. **Aesop Timeout Unreliability**:
   - Aesop `timeout` parameter is "soft" (may exceed)
   - **Solution**: Use process-level timeout via `asyncio.wait_for()`

3. **Proof Term Extraction**:
   - `print` output from Lean doesn't include proof term by default
   - **Solution**: Use `set_option pp.proofs true` before proving

4. **Mathlib Import Latency**:
   - Importing full mathlib adds 10-20s startup time
   - **Solution**: Lazy import only required theorems

5. **Incremental Compilation**:
   - Lean 4 doesn't support incremental compilation in server mode
   - **Gotcha**: Changing imported files requires server restart
   - **Solution**: Use `.olean` cache and restart server daily

6. **Unicode in Lean Code**:
   - Lean 4 uses Unicode symbols (∀, ∃, →, etc.)
   - **Gotcha**: Must encode as UTF-8 when sending over HTTP
   - **Solution**: Always set `Content-Type: application/json; charset=utf-8`

### Version Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|---------------------|-------|
| Lean 4 | 4.0.0 | 4.8.0+ | 4.8 adds better Aesop support |
| lean-client-python | 0.4.0 | 0.6.0+ | 0.6 fixes async bugs |
| Mathlib | v4.8.0 | latest | Use same version as Lean |
| Python | 3.10 | 3.11+ | 3.11 improves asyncio performance |

### Non-Obvious Behaviors

1. **Tactic State Mutation**:
   - Lean 4 tactics modify proof state destructively
   - **Gotcha**: Can't revert failed tactic application
   - **Solution**: Use `tactic <;> try` for safe application

2. **Type Class Resolution**:
   - Lean 4 uses type classes for overloading (e.g., `+` works on Nat, Int, Real)
   - **Gotcha**: Ambiguous type classes cause "don't know which to use" errors
   - **Solution**: Add explicit type annotations: `(x : Nat)`

3. **Universe Levels**:
   - Lean 4 has universe polymorphism (`Type`, `Type 1`, `Type 2`, etc.)
   - **Gotcha**: Universe inconsistency errors can be cryptic
   - **Solution**: Use `.ulift` to adjust universe levels manually

4. **Proof Irrelevance**:
   - Lean 4 treats proof terms as irrelevant (can discard them)
   - **Gotcha**: Extracting proof term from `by` tactic requires special handling
   - **Solution**: Use `have` instead of `by` for proof extraction

5. **Partial Evaluation**:
   - Lean 4 doesn't support partial evaluation of tactics
   - **Gotcha**: Can't pause and resume proof search
   - **Solution**: Use `try` tactic for best-effort search

6. **Quotient Types**:
   - Lean 4 uses quotients for sets, multisets, etc.
   - **Gotcha**: Quotient types can't be projected directly
   - **Solution**: Use `quotient.out` to extract representative

## Circuit Breaker Configuration

### Timeout Values
```python
TIMEOUTS = {
    "verify_theorem": 60.0,            # seconds
    "generate_proof": 30.0,            # seconds
    "apply_tactic": 10.0,              # seconds
    "search_similar_theorems": 5.0,    # seconds
    "formal_verification_pipeline": 90.0,  # seconds
    "batch_verify": 60.0,              # per theorem
}
```

### Retry Strategies

#### Exponential Backoff (Default)
```python
@retry(
    attempts=3,
    base_delay=2.0,      # seconds (Lean 4 is slower than Z3)
    max_delay=30.0,      # seconds
    exponential=2.0,
    jitter=0.1
)
async def verify_with_retry(...):
    ...
```

**Usage**: Theorem verification, proof generation

#### Linear Backoff (Long-Running)
```python
@retry(
    attempts=2,
    base_delay=5.0,
    max_delay=10.0,
    exponential=1.0      # linear
)
async def long_running_proof(...):
    ...
```

**Usage**: Complex proofs (>100 steps)

#### No Retry (Fast/Stateful)
```python
# No retry decorator
async def apply_tactic(...):
    ...
```

**Usage**: Tactic application, theorem search (fast operations)

### Failure Thresholds

```python
CIRCUIT_BREAKER = {
    "failure_threshold": 3,        # open after 3 failures (Lean is slower)
    "success_threshold": 2,        # close after 2 successes
    "timeout": 120.0,              # open state duration (seconds)
    "half_open_max_calls": 1       # test call in half-open state
}
```

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit tripped, use mock fallback
- **HALF_OPEN**: Test if Lean 4 server recovered

**Triggers**:
- 3 consecutive failures (compilation error, proving timeout)
- 2 consecutive timeouts (>60s)
- Lean 4 server unresponsive (health check fails)

**Recovery**:
- 2 consecutive successes → CLOSE
- 120s timeout → HALF_OPEN
- Manual restart of Lean 4 server

### Error Classification

| Error Type | Retryable | Circuit Breaker | Fallback |
|------------|-----------|-----------------|----------|
| `Lean4Error` (compilation) | No | No | Return error |
| `Lean4Error` (proving timeout) | Yes | Yes | Mock implementation |
| `ImportError` (Lean unavailable) | No | Yes | Mock implementation |
| `TimeoutError` (asyncio) | Yes | Yes | Mock implementation |
| `ConnectionError` (Lean server) | Yes | Yes | Mock implementation |

## Security Considerations

### Input Validation

#### Lean 4 Code Sanitization
```python
def validate_lean_code(code: str) -> bool:
    """Reject malicious Lean 4 patterns."""
    # Block meta-programming (could cause arbitrary code execution)
    dangerous_macros = ["run_cmd", "eval", "infer_type"]
    if any(macro in code for macro in dangerous_macros):
        raise ValueError("Meta-programming not allowed")

    # Block system calls
    if "IO." in code or "System.IO." in code:
        raise ValueError("IO operations not allowed")

    # Limit recursion depth
    paren_depth = code.count("(") - code.count(")")
    if abs(paren_depth) > 500:
        raise ValueError("Excessive nesting")

    return True
```

#### Natural Language Input
```python
def validate_nl_input(problem: str) -> bool:
    """Sanitize natural language problem statements."""
    # Max length
    if len(problem) > 10000:
        raise ValueError("Problem too long")

    # Block shell commands (same as Z3)
    shell_indicators = ["; rm", "| rm", "$(", "`", "$(syscall"]
    if any(indicator in problem for indicator in shell_indicators):
        raise ValueError("Shell commands not allowed")

    # Block Lean 4 tactic injection
    tactic_keywords = ["by", " := by", "simp", "rw"]
    if keyword in problem.lower():
        raise ValueError("Tactic keywords not allowed in natural language")

    return True
```

### Authentication Requirements

**Lean 4 Server has no authentication** (local tool). Security enforced at adapter layer:

```python
# Adapter-level rate limiting
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/leanaide/verify")
@limiter.limit("20/minute")  # Stricter than Z3 (Lean is slower)
async def verify_theorem(request: Request):
    ...
```

### Data Sensitivity

#### Sensitive Data in Theorems
```python
# WARNING: Theorems may contain sensitive information
# Example: theorem password_correct : password = "secret123" → True

# Best practice: Hash sensitive constants before proving
def sanitize_theorem(theorem: str) -> str:
    """Replace potential secrets with hashes."""
    import re

    # Detect string literals that look like secrets
    secret_patterns = [
        (r'"([^"]{20,})"', lambda m: f'"{hash_sha256(m.group(1))}"'),  # Long strings
        (r'password\s*=\s*"([^"]+)"', lambda m: f'password = "{hash_sha256(m.group(1))}"'),
    ]

    for pattern, replacer in secret_patterns:
        theorem = re.sub(pattern, replacer, theorem, flags=re.IGNORECASE)

    return theorem
```

#### Logging Security
```python
# NEVER log theorem contents
logger.info(f"Verifying theorem of length {len(theorem)}")  # OK
logger.info(f"Theorem: {theorem}")  # BAD - leaks data

# Log only metadata
logger.info({
    "msg": "Verifying theorem",
    "theorem_length": len(theorem),
    "auto_prove": auto_prove,
    "correlation_id": correlation_id,
    "timestamp": datetime.now(timezone.utc).isoformat()
})
```

### Resource Limits

```python
# Prevent resource exhaustion
MAX_THEOREM_LENGTH = 10000      # characters
MAX_PROOF_STEPS = 100           # proof steps
MAX_EXECUTION_TIME = 300.0      # 5 minutes
MAX_CONCURRENT_PROOFS = 5       # parallel proofs

def enforce_limits(theorem: str, proof: Optional[str]):
    if len(theorem) > MAX_THEOREM_LENGTH:
        raise ValueError(f"Theorem too long: {len(theorem)} > {MAX_THEOREM_LENGTH}")

    if proof and proof.count("by ") > MAX_PROOF_STEPS:
        raise ValueError(f"Proof too long: >{MAX_PROOF_STEPS} steps")
```

### UTC Enforcement (Law of UTC)

All timestamps in UTC:

```python
from datetime import datetime, timezone

# BAD - local time
timestamp = datetime.now()

# GOOD - UTC
timestamp = datetime.now(timezone.utc)

# Log with UTC
logger.info({
    "msg": "Theorem verification completed",
    "timestamp": datetime.now(timezone.utc).isoformat(),  # UTC
    "correlation_id": correlation_id
})
```

## Integration Patterns

### Correlation ID Tracking

Every request must have correlation ID:

```python
correlation_id = correlation_id or f"lean_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

logger.info({
    "msg": "Starting verification",
    "correlation_id": correlation_id,
    "timestamp": datetime.now(timezone.utc).isoformat()
})
```

### Structured Logging (JSON Lines)

All logs must be structured JSON:

```python
# BAD - unstructured
logger.info("Verification failed")

# GOOD - structured JSON
logger.error({
    "msg": "Verification failed",
    "correlation_id": correlation_id,
    "error": str(e),
    "theorem_length": len(theorem),
    "timestamp": datetime.now(timezone.utc).isoformat()
})
```

### Async/Await Pattern

All Lean 4 operations must be async:

```python
# BAD - blocking
result = integration.verify_theorem(theorem)

# GOOD - async
result = await integration.verify_theorem(theorem)
```

### Error Handling

Always handle errors gracefully:

```python
try:
    result = await integration.verify_theorem(theorem)
except Lean4CompilationError as e:
    # Compilation error - don't retry
    return LeanAideResult(success=False, error=str(e))
except Lean4TimeoutError as e:
    # Timeout - retry with backoff
    logger.warning({"msg": "Timeout, retrying", "correlation_id": correlation_id})
    return await retry_with_backoff(lambda: integration.verify_theorem(theorem))
except Exception as e:
    # Unexpected error - log and return
    logger.error({"msg": "Unexpected error", "error": str(e), "correlation_id": correlation_id})
    return LeanAideResult(success=False, error=str(e))
```

---

## References

- **Lean 4 Documentation**: https://leanprover.github.io/lean4/doc/
- **Mathlib**: https://leanprover-community.github.io/mathlib4/
- **Aesop Tactic**: https://github.com/JLimperg/aesop
- **OpenEvolve Integration**: `/openevolve/openevolve/knowledge_engine/integrations/leanaide_integration.py`
- **Z3 Bridge**: `/z3_leanaide_bridge.py`
- **Lean Client Python**: https://github.com/leanprover/lean-client-python

**Created**: 2026-02-03
**Author**: OpenEvolve Architecture Team
**Status**: Accepted, Implemented
**Last Updated**: 2026-02-03
