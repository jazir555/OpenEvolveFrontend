# LeanAide Integration Guide

Complete guide for LeanAide integration in the RESE-Z3 Bridge Adapter.

**Author:** RESE Team
**Created:** 2026-02-04
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [RESE Phase Integration](#rese-phase-integration)
7. [Advanced Patterns](#advanced-patterns)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)

---

## Overview

The RESE-Z3 Bridge now includes full LeanAide integration, enabling:

- **Autoformalization**: Convert natural language to Lean 4 theorems
- **AI-Powered Proving**: Generate proofs using LeanAide AI
- **Z3-Lean Translation**: Bridge Z3 constraints to Lean 4
- **Tactic Suggestions**: Get AI-recommended proof tactics

### Key Features

- **Zero Trust**: Runtime verification via probe scripts
- **Circuit Breaker**: Automatic failure detection and recovery
- **Idempotent Operations**: Safe to run multiple times
- **Structured Logging**: JSON logs with correlation IDs
- **Performance Monitoring**: Built-in metrics and caching

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RESE-Z3 Bridge                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Z3 Client      │         │  LeanAide Client │         │
│  │  (Port 8000)     │         │  (Port 7654)     │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                            │                     │
│           └───────────┬────────────────┘                     │
│                       │                                      │
│              ┌────────▼────────┐                           │
│              │ Z3-LeanAide     │                           │
│              │ Bridge          │                           │
│              │ (Existing)      │                           │
│              └─────────────────┘                           │
│                       │                                      │
│              ┌────────▼────────┐                           │
│              │ Canonical       │                           │
│              │ Schema Layer    │                           │
│              └─────────────────┘                           │
│                       │                                      │
│              ┌────────▼────────┐                           │
│              │ RESE API        │                           │
│              │ Methods         │                           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Natural Language Theorem
         │
         ▼
┌─────────────────┐
│ Autoformalize   │ ──> Lean 4 Code
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Prove with AI   │ ──> Proof
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Z3 Verify       │ ──> Verification
└─────────────────┘
```

---

## Configuration

### Environment Variables

```bash
# LeanAide Server Configuration
LEANAIDE_BASE_URL=http://localhost:7654
LEANAIDE_TIMEOUT_MS=60000
LEANAIDE_ENABLE=true

# Z3 Server Configuration
Z3_BASE_URL=http://localhost:8000
Z3_TIMEOUT_MS=30000

# Circuit Breaker Configuration
Z3_CIRCUIT_BREAKER_THRESHOLD=5
Z3_CIRCUIT_BREAKER_TIMEOUT_MS=60000

# Retry Configuration
Z3_MAX_RETRIES=3
Z3_RETRY_BACKOFF_MS=1000

# Cache Configuration
Z3_ENABLE_CACHE=true
Z3_CACHE_TTL_MS=300000

# Monitoring Configuration
Z3_ENABLE_MONITORING=true
```

### Python Configuration

```python
from rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig

# Load from environment
config = RESEZ3BridgeConfig.from_env()

# Or configure manually
config = RESEZ3BridgeConfig(
    leanaide_base_url="http://localhost:7654",
    leanaide_timeout_ms=60000,
    leanaide_enable=True,
    z3_base_url="http://localhost:8000",
    z3_timeout_ms=30000,
    enable_cache=True,
    enable_monitoring=True,
)

# Create bridge
bridge = RESEZ3Bridge(config=config)
```

---

## API Reference

### 1. Autoformalization

Convert natural language to Lean 4 theorems.

```python
def autoformalize(
    natural_language: str,
    theorem_name: Optional[str] = None,
    correlation_id: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> LeanAideAutoformalizeResponse
```

**Parameters:**
- `natural_language`: Natural language theorem statement
- `theorem_name`: Optional name for the theorem
- `correlation_id`: Tracing ID (auto-generated if not provided)
- `timeout_ms`: Timeout override (default: 60000ms)

**Returns:**
- `LeanAideAutoformalizeResponse` with:
  - `success`: True if successful
  - `lean_code`: Generated Lean 4 code
  - `theorem_name`: Theorem name
  - `theorem_type`: Elaborated type
  - `proof_sketch`: Optional proof sketch
  - `execution_time_ms`: Execution time

**Example:**
```python
response = bridge.autoformalize(
    natural_language="There are infinitely many prime numbers",
    theorem_name="infinitely_many_primes",
)

if response.success:
    print(f"Lean 4 code:\n{response.lean_code}")
    print(f"Theorem type: {response.theorem_type}")
```

---

### 2. AI-Powered Proving

Generate proofs using LeanAide AI.

```python
def prove_with_ai(
    theorem_text: str,
    theorem_code: Optional[str] = None,
    theorem_statement: Optional[str] = None,
    correlation_id: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> LeanAideProveResponse
```

**Parameters:**
- `theorem_text`: Natural language theorem
- `theorem_code`: Optional pre-formalized Lean 4 code
- `theorem_statement`: Optional elaborated theorem type
- `correlation_id`: Tracing ID
- `timeout_ms`: Timeout override (default: 60000ms)

**Returns:**
- `LeanAideProveResponse` with:
  - `success`: True if proof generated
  - `proof`: Generated proof
  - `tactics_used`: List of tactics used
  - `proof_script`: Complete proof script
  - `execution_time_ms`: Execution time

**Example:**
```python
response = bridge.prove_with_ai(
    theorem_text="For all natural numbers n, n + 0 = n",
    theorem_code="theorem add_zero (n : Nat) : n + 0 = n",
)

if response.success:
    print(f"Proof:\n{response.proof}")
    print(f"Tactics: {response.tactics_used}")
```

---

### 3. Z3 to Lean Translation

Translate Z3 SMT-LIB to Lean 4.

```python
def translate_z3_to_lean(
    smtlib_content: str,
    constraint_type: ConstraintType = ConstraintType.BOOLEAN,
    correlation_id: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> Z3ToLeanTranslationResponse
```

**Parameters:**
- `smtlib_content`: SMT-LIB2 content
- `constraint_type`: Type of constraints (BOOLEAN, INTEGER, REAL, etc.)
- `correlation_id`: Tracing ID
- `timeout_ms`: Timeout override (default: 30000ms)

**Returns:**
- `Z3ToLeanTranslationResponse` with:
  - `success`: True if translation successful
  - `lean_code`: Generated Lean 4 code
  - `theorem_statement`: Theorem statement
  - `variables`: List of variables
  - `translated_constraints`: Translated constraints

**Example:**
```python
smtlib = """
(declare-fun x () Real)
(declare-fun y () Real)
(assert (> x 0.0))
(assert (> y 0.0))
(assert (> (+ x y) 0.0))
"""

response = bridge.translate_z3_to_lean(
    smtlib_content=smtlib,
    constraint_type=ConstraintType.REAL,
)

if response.success:
    print(f"Lean 4 translation:\n{response.lean_code}")
    print(f"Variables: {response.variables}")
```

---

### 4. Tactic Suggestions

Get AI-recommended proof tactics.

```python
def suggest_tactics(
    goal_state: str,
    context: Optional[str] = None,
    num_suggestions: int = 3,
    correlation_id: Optional[str] = None,
    timeout_ms: Optional[int] = None,
) -> LeanAideTacticSuggestionResponse
```

**Parameters:**
- `goal_state`: Current goal state in Lean 4
- `context`: Additional context
- `num_suggestions`: Number of suggestions (1-10, default: 3)
- `correlation_id`: Tracing ID
- `timeout_ms`: Timeout override (default: 15000ms)

**Returns:**
- `LeanAideTacticSuggestionResponse` with:
  - `success`: True if suggestions generated
  - `suggestions`: List of `LeanAideTacticSuggestion`
    - `tactic`: Tactic name
    - `description`: Description
    - `confidence`: Confidence score (0-1)
    - `reasoning`: Explanation

**Example:**
```python
response = bridge.suggest_tactics(
    goal_state="⊢ x + y = y + x",
    context="Working with real numbers",
    num_suggestions=3,
)

if response.success:
    for suggestion in response.suggestions:
        print(f"{suggestion.tactic}: {suggestion.description}")
        print(f"Confidence: {suggestion.confidence}")
```

---

## Usage Examples

### Example 1: Complete Workflow

Autoformalize and prove a theorem:

```python
from rese_z3_bridge import RESEZ3Bridge
import uuid

# Create bridge
bridge = RESEZ3Bridge()
correlation_id = str(uuid.uuid4())

# Step 1: Autoformalize
formalize_response = bridge.autoformalize(
    natural_language="The square root of 2 is irrational",
    theorem_name="sqrt2_irrational",
    correlation_id=correlation_id,
)

if not formalize_response.success:
    print(f"Autoformalization failed: {formalize_response.error}")
    exit(1)

print(f"Generated Lean code:\n{formalize_response.lean_code}\n")

# Step 2: Generate proof
prove_response = bridge.prove_with_ai(
    theorem_text="The square root of 2 is irrational",
    theorem_code=formalize_response.lean_code,
    theorem_statement=formalize_response.theorem_type,
    correlation_id=correlation_id,
)

if prove_response.success:
    print(f"Proof generated:\n{prove_response.proof}")
    print(f"\nTactics used: {prove_response.tactics_used}")
else:
    print(f"Proof generation failed: {prove_response.error}")

# Cleanup
bridge.close()
```

---

### Example 2: Z3 to Lean to Proof

Translate Z3 constraint and prove:

```python
from rese_z3_bridge import RESEZ3Bridge, ConstraintType

bridge = RESEZ3Bridge()

# Z3 constraint
smtlib = """
(declare-fun p () Bool)
(declare-fun q () Bool)
(assert (implies p q))
(assert (not q))
(check-sat)
"""

# Step 1: Translate to Lean
translate_response = bridge.translate_z3_to_lean(
    smtlib_content=smtlib,
    constraint_type=ConstraintType.BOOLEAN,
)

if translate_response.success:
    print(f"Lean translation:\n{translate_response.lean_code}")

    # Step 2: Try to prove (if we have a complete formalization)
    if translate_response.theorem_statement:
        prove_response = bridge.prove_with_ai(
            theorem_text="Logical implication",
            theorem_code=translate_response.lean_code,
            theorem_statement=translate_response.theorem_statement,
        )

        if prove_response.success:
            print(f"Proof: {prove_response.proof}")

bridge.close()
```

---

### Example 3: Interactive Proof Assistance

Get tactic suggestions during proof development:

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

# Current proof state
goal_state = "⊢ ∀ (n : Nat), n + 0 = n"

# Get suggestions
response = bridge.suggest_tactics(
    goal_state=goal_state,
    context="Natural number addition",
    num_suggestions=5,
)

if response.success:
    print(f"Top {len(response.suggestions)} tactic suggestions:")
    for i, suggestion in enumerate(response.suggestions, 1):
        print(f"\n{i}. {suggestion.tactic}")
        print(f"   {suggestion.description}")
        print(f"   Confidence: {suggestion.confidence:.2f}")
        if suggestion.reasoning:
            print(f"   Reasoning: {suggestion.reasoning}")

bridge.close()
```

---

### Example 4: Batch Autoformalization

Autoformalize multiple theorems:

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

theorems = [
    "There are infinitely many primes",
    "The square root of 2 is irrational",
    "Every natural number has a unique prime factorization",
    "The sum of two even numbers is even",
]

results = []
for theorem in theorems:
    response = bridge.autoformalize(
        natural_language=theorem,
        theorem_name=theorem.lower().replace(" ", "_")[:30],
    )

    results.append({
        "theorem": theorem,
        "success": response.success,
        "lean_code": response.lean_code if response.success else None,
        "error": response.error if not response.success else None,
    })

# Summary
successful = sum(1 for r in results if r["success"])
print(f"Successfully formalized: {successful}/{len(theorems)}")

for result in results:
    if result["success"]:
        print(f"\n✓ {result['theorem']}")
        print(f"  {result['lean_code'][:100]}...")
    else:
        print(f"\n✗ {result['theorem']}")
        print(f"  Error: {result['error']}")

bridge.close()
```

---

## RESE Phase Integration

### SCE (Symbolic Constraint Engine)

Use LeanAide for constraint formalization:

```python
from rese_z3_bridge import RESEZ3Bridge, CanonicalVariable, CanonicalConstraint

bridge = RESEZ3Bridge()

# Define constraints
constraints = [
    CanonicalConstraint(
        expression="forall x. x > 0",
        constraint_type=ConstraintType.BOOLEAN,
        description="Positive numbers",
    ),
]

# Autoformalize constraint
for constraint in constraints:
    response = bridge.autoformalize(
        natural_language=constraint.description or constraint.expression,
    )

    if response.success:
        print(f"Formalized: {response.lean_code}")

bridge.close()
```

---

### DITO (Dynamic Inference Trace Optimizer)

Use LeanAide for ATP and proof generation:

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

# Detect contradiction and generate proof
contradiction_response = bridge.prove_with_ai(
    theorem_text="If P and Q, then not (not P or not Q)",
)

if contradiction_response.success:
    print(f"Contradiction proof: {contradiction_response.proof}")

bridge.close()
```

---

### ACI (Anomaly Characterization Index)

Use LeanAide for anomaly verification:

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

# Verify anomaly constraints
anomaly_response = bridge.prove_with_ai(
    theorem_text="System constraints are violated",
    theorem_code="""
theorem anomaly_detected :
  ∃ (x : ℝ), x > 1000.0 := by
    sorry
""",
)

if anomaly_response.success:
    print(f"Anomaly verified: {anomaly_response.proof}")

bridge.close()
```

---

## Advanced Patterns

### Pattern 1: Hybrid Verification

Combine Z3 and LeanAide for maximum confidence:

```python
from rese_z3_bridge import RESEZ3Bridge, ConstraintType

bridge = RESEZ3Bridge()

theorem = "For all real numbers x and y, x + y = y + x"

# Step 1: Verify with Z3 (quick check)
smtlib = f"""
(declare-fun x () Real)
(declare-fun y () Real)
(assert (not (= (+ x y) (+ y x))))
(check-sat)
"""

z3_response = bridge.solve_constraints(
    variables=[],
    constraints=[
        CanonicalConstraint(
            expression="not (x + y = y + x)",
            constraint_type=ConstraintType.REAL,
        )
    ],
)

# Step 2: Formalize with LeanAide
leanaide_response = bridge.autoformalize(
    natural_language=theorem,
)

# Step 3: Generate formal proof
if leanaide_response.success:
    proof_response = bridge.prove_with_ai(
        theorem_text=theorem,
        theorem_code=leanaide_response.lean_code,
    )

    print(f"Z3 result: {z3_response.result}")
    print(f"Lean proof: {proof_response.proof if proof_response.success else 'Failed'}")

bridge.close()
```

---

### Pattern 2: Incremental Proof Development

Build proofs incrementally with tactic suggestions:

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

goal = "⊢ ∀ (n : Nat), n + 0 = n"

# Start with intro
print("Current goal:", goal)
response = bridge.suggest_tactics(
    goal_state=goal,
    num_suggestions=1,
)

if response.success and response.suggestions:
    tactic = response.suggestions[0].tactic
    print(f"Suggested tactic: {tactic}")
    print(f"Reasoning: {response.suggestions[0].reasoning}")

    # Apply tactic (simulated)
    new_goal = "⊢ n + 0 = n"  # After intro
    print(f"\nNew goal: {new_goal}")

    # Get next suggestion
    response = bridge.suggest_tactics(
        goal_state=new_goal,
        num_suggestions=1,
    )

    if response.success and response.suggestions:
        print(f"Next tactic: {response.suggestions[0].tactic}")

bridge.close()
```

---

### Pattern 3: Counterexample Search

Use Z3 to find counterexamples before formal proving:

```python
from rese_z3_bridge import RESEZ3Bridge, ConstraintType

bridge = RESEZ3Bridge()

# Theorem to test
theorem = "All prime numbers are odd"

# Step 1: Try to find counterexample with Z3
smtlib = """
(declare-fun p () Int)
(assert (Prime p))
(assert (not (odd p)))
(check-sat)
"""

response = bridge.solve_constraints(
    variables=[],
    constraints=[
        CanonicalConstraint(
            expression="Prime p and not (odd p)",
            constraint_type=ConstraintType.INTEGER,
        )
    ],
)

if response.result.value == "sat":
    print("Found counterexample!")
    if response.model:
        print(f"Model: {response.model.assignments}")

    # Step 2: Understand why with LeanAide
    explanation = bridge.prove_with_ai(
        theorem_text="2 is a prime number but is even",
    )

    if explanation.success:
        print(f"\nExplanation: {explanation.proof}")
else:
    print("No counterexample found - theorem may be true")

bridge.close()
```

---

## Troubleshooting

### Issue 1: LeanAide Connection Refused

**Symptom:**
```
Connection error to LeanAide server
```

**Solution:**
1. Verify LeanAide server is running:
   ```bash
   curl http://localhost:7654/
   ```

2. Check configuration:
   ```python
   health = bridge.get_health()
   print(health)
   ```

3. Verify environment variable:
   ```bash
   echo $LEANAIDE_BASE_URL
   ```

---

### Issue 2: Autoformalization Timeout

**Symptom:**
```
Request timed out after 60000ms
```

**Solution:**
1. Increase timeout:
   ```python
   response = bridge.autoformalize(
       natural_language=complex_theorem,
       timeout_ms=120000,  # 2 minutes
   )
   ```

2. Simplify theorem statement:
   ```python
   # Instead of complex statement
   simple = "For all x, P(x) implies Q(x)"
   ```

3. Check if LeanAide server is overloaded

---

### Issue 3: Translation Fails

**Symptom:**
```
Translation failed: Could not parse SMT-LIB
```

**Solution:**
1. Validate SMT-LIB syntax:
   ```python
   # Ensure valid SMT-LIB
   smtlib = """
   (set-logic ALL)
   (declare-fun x () Int)
   (assert (> x 0))
   (check-sat)
   """
   ```

2. Check constraint type matches:
   ```python
   response = bridge.translate_z3_to_lean(
       smtlib_content=smtlib,
       constraint_type=ConstraintType.INTEGER,  # Match SMT-LIB type
   )
   ```

3. Use simpler constraints for testing

---

### Issue 4: Circuit Breaker Open

**Symptom:**
```
Circuit breaker is OPEN, rejecting request
```

**Solution:**
1. Check circuit breaker status:
   ```python
   stats = bridge.get_stats()
   print(stats["client_stats"]["circuit_breaker"])
   ```

2. Wait for timeout (default: 60 seconds)

3. Manually reset by restarting bridge

4. Increase failure threshold:
   ```python
   config = RESEZ3BridgeConfig(
       circuit_breaker_threshold=10,  # Increase from 5
   )
   bridge = RESEZ3Bridge(config=config)
   ```

---

## Performance Tuning

### Caching

Enable caching for repeated queries:

```python
config = RESEZ3BridgeConfig(
    enable_cache=True,
    cache_ttl_ms=300000,  # 5 minutes
)

bridge = RESEZ3Bridge(config=config)

# First call - cache miss
response1 = bridge.autoformalize(
    natural_language="Test theorem",
)

# Second call - cache hit (much faster)
response2 = bridge.autoformalize(
    natural_language="Test theorem",
)
```

---

### Connection Pooling

Adjust connection pool size:

```python
config = RESEZ3BridgeConfig(
    # LeanAide client will use connection pooling
    leanaide_timeout_ms=60000,
)

bridge = RESEZ3Bridge(config=config)
```

---

### Concurrent Requests

Use threading for concurrent requests:

```python
from concurrent.futures import ThreadPoolExecutor

bridge = RESEZ3Bridge()

theorems = ["Theorem 1", "Theorem 2", "Theorem 3"]

def formalize(theorem):
    return bridge.autoformalize(natural_language=theorem)

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(formalize, theorems))

for result in results:
    print(f"Success: {result.success}")

bridge.close()
```

---

### Monitoring

Check performance metrics:

```python
bridge = RESEZ3Bridge()

# ... perform operations ...

# Get performance summary
stats = bridge.get_stats()
performance = stats["performance_summary"]

print(f"Total operations: {performance['total_operations']}")
print(f"Success rate: {performance['success_rate']:.2%}")
print(f"Average duration: {performance['average_duration_ms']:.2f}ms")
print(f"Cached operations: {performance['cached_operations']}")
```

---

## Best Practices

### 1. Always Use Correlation IDs

```python
import uuid

correlation_id = str(uuid.uuid4())

response = bridge.autoformalize(
    natural_language=theorem,
    correlation_id=correlation_id,  # For tracing
)
```

### 2. Handle Errors Gracefully

```python
response = bridge.autoformalize(natural_language=theorem)

if not response.success:
    print(f"Autoformalization failed: {response.error}")
    # Handle error
else:
    # Process result
    pass
```

### 3. Set Appropriate Timeouts

```python
# Simple theorems: 30s
response = bridge.autoformalize(
    natural_language=simple_theorem,
    timeout_ms=30000,
)

# Complex theorems: 120s
response = bridge.autoformalize(
    natural_language=complex_theorem,
    timeout_ms=120000,
)
```

### 4. Cleanup Resources

```python
bridge = RESEZ3Bridge()

try:
    # Use bridge
    response = bridge.autoformalize(natural_language=theorem)
finally:
    # Always cleanup
    bridge.close()
```

### 5. Use Structured Logging

```python
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_app")

response = bridge.autoformalize(natural_language=theorem)

logger.info(json.dumps({
    "event": "autoformalization",
    "correlation_id": response.correlation_id,
    "success": response.success,
    "execution_time_ms": response.execution_time_ms,
}))
```

---

## Appendix

### A. Constraint Types

```python
from rese_z3_schema import ConstraintType

ConstraintType.BOOLEAN    # Boolean logic
ConstraintType.INTEGER    # Integer arithmetic
ConstraintType.REAL       # Real arithmetic
ConstraintType.BIT_VECTOR # Bit vectors
ConstraintType.ARRAY      # Arrays
ConstraintType.STRING     # Strings
```

### B. Error Types

```python
# Client errors
rese_z3_client.Z3ClientError
rese_z3_client.Z3ClientConnectionError
rese_z3_client.Z3ClientTimeoutError
rese_z3_client.Z3ClientCircuitBreakerOpenError

# Schema errors (ValueError)
validate_autoformalize_request()
validate_prove_request()
validate_translation_request()
validate_tactic_suggestion_request()
```

### C. Response Structures

All responses include:
- `success`: bool
- `correlation_id`: str
- `timestamp`: str (ISO-8601 UTC)
- `execution_time_ms`: float
- `error`: Optional[str]
- `metadata`: Dict[str, Any]

---

## Support

For issues, questions, or contributions:
- Check probe scripts: `glue/adapters/rese-z3-bridge/probes/`
- Run tests: `pytest tests/test_leanaide_integration.py`
- Check logs: Set `Z3_ENABLE_MONITORING=true`

**Law of Runtime Truth:** Always verify via probe scripts before integration.
