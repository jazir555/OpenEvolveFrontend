# RESE-Z3 Bridge Usage Guide

**Complete guide for using the RESE-Z3 Bridge in all RESE phases**

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Phase-Specific Usage](#phase-specific-usage)
   - [Phase 1: SCE Constraint Solving](#phase-1-sce-constraint-solving)
   - [Phase 2: DITO Contradiction Detection](#phase-2-dito-contradiction-detection)
   - [Phase 3: ACI Anomaly Verification](#phase-3-aci-anomaly-verification)
   - [Phase 4: Output Generation](#phase-4-output-generation)
4. [Advanced Usage](#advanced-usage)
5. [Error Handling](#error-handling)
6. [Performance Tuning](#performance-tuning)
7. [Best Practices](#best-practices)

## Introduction

The RESE-Z3 Bridge provides a unified API for all RESE phases to access Z3 capabilities. It abstracts away the complexity of Z3 integration, providing:

- **Simple API**: One method call per operation
- **Resilience**: Circuit breaker and retry logic built-in
- **Monitoring**: Performance metrics automatically tracked
- **Idempotency**: Same input always produces same output

## Quick Start

### Installation

```python
# Add to your path
import sys
sys.path.insert(0, 'glue/adapters/rese-z3-bridge/src')

# Import bridge
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import (
    CanonicalVariable,
    CanonicalConstraint,
    ConstraintType,
    ProblemType,
)

# Initialize
bridge = RESEZ3Bridge()
```

### Basic Example

```python
# Define a simple constraint problem
variables = [CanonicalVariable("x", ConstraintType.INTEGER)]
constraints = [
    CanonicalConstraint("(> x 0)", ConstraintType.INTEGER, "x > 0"),
    CanonicalConstraint("(< x 100)", ConstraintType.INTEGER, "x < 100"),
]

# Solve
response = bridge.solve_constraints(variables, constraints)

# Check result
if response.result.value == "sat":
    print(f"✓ Satisfiable")
    print(f"  Solution: {response.model.assignments}")
elif response.result.value == "unsat":
    print(f"✗ Unsatisfiable")
else:
    print(f"? Unknown")
```

## Phase-Specific Usage

### Phase 1: SCE Constraint Solving

**Purpose**: Find satisfying assignments for symbolic constraints

**Use Case**: When SCE phase needs to verify if assumptions can be satisfied

#### Example 1: Simple Constraint Solving

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalVariable, CanonicalConstraint, ConstraintType

class SCEAdapter:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def verify_assumptions(self, assumptions):
        """
        Verify if assumptions are mutually satisfiable

        Args:
            assumptions: List of assumption strings

        Returns:
            Tuple of (is_satisfiable, model)
        """
        # Convert assumptions to canonical constraints
        constraints = []
        for i, assumption in enumerate(assumptions):
            constraints.append(
                CanonicalConstraint(
                    expression=assumption,
                    constraint_type=ConstraintType.BOOLEAN,
                    constraint_id=f"assumption_{i}",
                )
            )

        # Solve via bridge
        response = self.bridge.solve_constraints(
            variables=[],
            constraints=constraints,
            correlation_id=f"sce-{uuid.uuid4()}",
        )

        # Return result
        is_sat = response.result.value == "sat"
        model = response.model.assignments if response.model else None

        return is_sat, model

# Usage
sce = SCEAdapter()
assumptions = [
    "(> x 0)",
    "(< x 100)",
    "(= y (+ x 1))",
]

is_sat, model = sce.verify_assumptions(assumptions)
print(f"Satisfiable: {is_sat}")
if model:
    print(f"Model: {model}")
```

#### Example 2: Constraint Hardening

```python
def harden_constraints(base_constraints, safety_invariants):
    """
    Harden constraints with safety invariants

    Args:
        base_constraints: Base problem constraints
        safety_invariants: Safety constraints to add

    Returns:
        Tuple of (is_safe, violating_constraints)
    """
    bridge = RESEZ3Bridge()

    # Combine constraints
    all_constraints = base_constraints + safety_invariants

    # Check if satisfiable
    response = bridge.solve_constraints(
        variables=[],
        constraints=all_constraints,
    )

    if response.result.value == "sat":
        return True, []
    else:
        # Find violating constraint via binary search
        violating = []
        for constraint in safety_invariants:
            test_constraints = base_constraints + [constraint]
            response = bridge.solve_constraints(
                variables=[],
                constraints=test_constraints,
            )
            if response.result.value == "unsat":
                violating.append(constraint)

        return False, violating
```

### Phase 2: DITO Contradiction Detection

**Purpose**: Detect contradictions in constraint sets efficiently

**Use Case**: When DITO optimizer needs to check for contradictions in activated subgraph

#### Example 1: Targeted Contradiction Detection

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

class DITOOptimizer:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def check_contradiction_targeted(
        self,
        node_constraint: CanonicalConstraint,
        active_constraints: List[CanonicalConstraint],
        correlation_id: str,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if node constraint contradicts any active constraint

        Args:
            node_constraint: Constraint to check
            active_constraints: Currently active constraints
            correlation_id: Correlation ID for tracing

        Returns:
            Tuple of (has_contradiction, counterexample)
        """
        # Combine constraints
        all_constraints = [node_constraint] + active_constraints

        # Detect contradiction via bridge
        has_contradiction, counterexample = self.bridge.detect_contradictions(
            constraints=all_constraints,
            correlation_id=correlation_id,
        )

        return has_contradiction, counterexample

# Usage
dito = DITOOptimizer()

node_constraint = CanonicalConstraint(
    "(> temperature 1000)",
    ConstraintType.REAL,
    "Temperature > 1000",
)

active_constraints = [
    CanonicalConstraint("(< temperature 500)", ConstraintType.REAL),
]

has_contradiction, counterexample = dito.check_contradiction_targeted(
    node_constraint,
    active_constraints,
    correlation_id="dito-check-123",
)

if has_contradiction:
    print("⚠️  Contradiction detected!")
    print(f"   Node constraint contradicts active subgraph")
else:
    print("✓ No contradiction")
```

#### Example 2: Batch Contradiction Detection

```python
def detect_contradictions_batch(constraints, batch_size=10):
    """
    Detect contradictions in batches for efficiency

    Args:
        constraints: List of constraints to check
        batch_size: Number of constraints per batch

    Returns:
        List of (constraint1, constraint2) pairs that contradict
    """
    bridge = RESEZ3Bridge()
    contradictions = []

    # Check pairs
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            pair = [constraints[i], constraints[j]]

            has_contradiction, _ = bridge.detect_contradictions(
                constraints=pair,
            )

            if has_contradiction:
                contradictions.append((constraints[i], constraints[j]))

    return contradictions
```

### Phase 3: ACI Anomaly Verification

**Purpose**: Verify if detected anomalies violate safety constraints

**Use Case**: When ACI calculator needs to verify anomalies against constraints

#### Example 1: Single Anomaly Verification

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

class ACICalculator:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def verify_anomaly(
        self,
        anomaly_constraints: List[CanonicalConstraint],
        safety_constraints: List[CanonicalConstraint],
        correlation_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify if anomaly violates safety constraints

        Args:
            anomaly_constraints: Constraints representing anomaly
            safety_constraints: Safety constraints to verify against
            correlation_id: Correlation ID for tracing

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Combine anomaly with safety constraints
        all_constraints = anomaly_constraints + safety_constraints

        # Verify via bridge
        is_valid, error = self.bridge.verify_anomaly(
            constraints=all_constraints,
            correlation_id=correlation_id,
        )

        return is_valid, error

# Usage
aci = ACICalculator()

# Anomaly: temperature exceeds safety limit
anomaly = [
    CanonicalConstraint("(> temperature 900)", ConstraintType.REAL),
]

# Safety constraint
safety = [
    CanonicalConstraint("(< temperature 800)", ConstraintType.REAL),
]

is_valid, error = aci.verify_anomaly(
    anomaly,
    safety,
    correlation_id="aci-verify-456",
)

if not is_valid:
    print(f"⚠️  Anomaly detected: {error}")
else:
    print("✓ No anomaly")
```

#### Example 2: Batch Anomaly Verification

```python
def verify_anomalies_batch(anomalies, safety_constraints):
    """
    Verify multiple anomalies efficiently

    Args:
        anomalies: List of anomaly constraint lists
        safety_constraints: Safety constraints to verify against

    Returns:
        List of (anomaly_idx, is_valid, error) tuples
    """
    bridge = RESEZ3Bridge()
    results = []

    for idx, anomaly in enumerate(anomalies):
        is_valid, error = bridge.verify_anomaly(
            constraints=anomaly + safety_constraints,
            correlation_id=f"aci-batch-{idx}",
        )
        results.append((idx, is_valid, error))

    return results
```

### Phase 4: Output Generation

**Purpose**: Generate formal proofs and verified output

**Use Case**: When generating final RESE output with formal verification

#### Example 1: Theorem Proving for Output

```python
from rese_z3_bridge import RESEZ3Bridge

class OutputGenerator:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def prove_output_correctness(
        self,
        theorem_statement: str,
        assumptions: List[str],
        correlation_id: str,
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Prove output correctness via theorem proving

        Args:
            theorem_statement: Theorem to prove
            assumptions: Assumptions for the proof
            correlation_id: Correlation ID for tracing

        Returns:
            Tuple of (proven, proof, counterexample)
        """
        response = self.bridge.prove_theorem(
            theorem_statement=theorem_statement,
            assumptions=assumptions,
            correlation_id=correlation_id,
        )

        if response.proven:
            return True, response.proof, None
        else:
            return False, None, response.counterexample

# Usage
generator = OutputGenerator()

# Theorem: If x > 0 and y > 0, then x + y > 0
theorem = "(implies (and (> x 0) (> y 0)) (> (+ x y) 0))"
assumptions = []

proven, proof, counterexample = generator.prove_output_correctness(
    theorem,
    assumptions,
    correlation_id="output-proof-789",
)

if proven:
    print("✓ Output theorem proven")
    print(f"Proof: {proof}")
else:
    print("✗ Theorem disproven")
    print(f"Counterexample: {counterexample}")
```

#### Example 2: Lean 4 Translation

```python
def generate_lean4_output(smtlib_problem: str) -> str:
    """
    Generate Lean 4 formalization from SMT-LIB

    Args:
        smtlib_problem: SMT-LIB2 problem

    Returns:
        Lean 4 code
    """
    bridge = RESEZ3Bridge()

    lean4_code = bridge.translate_to_lean4(
        smtlib_content=smtlib_problem,
        correlation_id="lean4-gen-101",
    )

    return lean4_code
```

## Advanced Usage

### Custom Timeout

```python
# Use custom timeout for complex problems
response = bridge.solve_constraints(
    variables=variables,
    constraints=constraints,
    timeout_ms=120000,  # 2 minutes
)
```

### Performance Monitoring

```python
# Get performance statistics
bridge = RESEZ3Bridge()

# Do some work...
response = bridge.solve_constraints(variables, constraints)

# Get stats
stats = bridge.get_stats()
performance = stats["performance_summary"]

print(f"Total operations: {performance['total_operations']}")
print(f"Success rate: {performance['success_rate']:.1%}")
print(f"Average duration: {performance['average_duration_ms']:.2f}ms")
print(f"Cached operations: {performance['cached_operations']}")
```

### Circuit Breaker Monitoring

```python
# Check circuit breaker state
health = bridge.get_health()
cb_state = health["circuit_breaker"]["state"]

if cb_state == "open":
    print("⚠️  Circuit breaker is OPEN")
    print("   Z3 server may be down")
elif cb_state == "half_open":
    print("⚠️  Circuit breaker is HALF_OPEN")
    print("   Testing if Z3 server recovered")
else:
    print("✓ Circuit breaker is CLOSED")
```

### Disable Caching

```python
from rese_z3_bridge import RESEZ3BridgeConfig

# Disable cache for fresh results
config = RESEZ3BridgeConfig(
    enable_cache=False,
)
bridge = RESEZ3Bridge(config)
```

## Error Handling

### Circuit Breaker Open

```python
from rese_z3_client import Z3ClientCircuitBreakerOpenError

try:
    response = bridge.solve_constraints(variables, constraints)
except Z3ClientCircuitBreakerOpenError:
    print("⚠️  Circuit breaker is OPEN")
    print("   Too many failures detected")
    print("   Please wait or fix Z3 server")
    # Handle gracefully - return cached result or fallback
```

### Timeout Error

```python
from rese_z3_client import Z3ClientTimeoutError

try:
    response = bridge.solve_constraints(variables, constraints)
except Z3ClientTimeoutError as e:
    print(f"⚠️  Request timed out: {e}")
    print("   Try simplifying constraints or increasing timeout")
    # Handle gracefully - return partial result or retry
```

### Connection Error

```python
from rese_z3_client import Z3ClientConnectionError

try:
    response = bridge.solve_constraints(variables, constraints)
except Z3ClientConnectionError as e:
    print(f"⚠️  Cannot connect to Z3 server: {e}")
    print("   Check if Z3 server is running")
    # Handle gracefully - use fallback or retry later
```

### Generic Error

```python
try:
    response = bridge.solve_constraints(variables, constraints)
except Exception as e:
    print(f"⚠️  Unexpected error: {e}")
    # Log and handle gracefully
    bridge.logger.error(f"Solver error: {e}")
```

## Performance Tuning

### Increase Timeout for Complex Problems

```bash
export Z3_TIMEOUT_MS=120000  # 2 minutes
```

### Adjust Circuit Breaker Threshold

```bash
export Z3_CIRCUIT_BREAKER_THRESHOLD=10  # More tolerant
```

### Enable Caching for Idempotent Operations

```bash
export Z3_ENABLE_CACHE=true
export Z3_CACHE_TTL_MS=600000  # 10 minutes
```

### Tune Retry Logic

```bash
export Z3_MAX_RETRIES=5
export Z3_RETRY_BACKOFF_MS=2000  # 2 seconds
```

## Best Practices

### 1. Always Use Correlation IDs

```python
# Good
response = bridge.solve_constraints(
    variables,
    constraints,
    correlation_id=f"sce-{uuid.uuid4()}",
)

# Bad
response = bridge.solve_constraints(variables, constraints)
```

### 2. Handle All Result Types

```python
response = bridge.solve_constraints(variables, constraints)

if response.result.value == "sat":
    # Handle satisfiable
    pass
elif response.result.value == "unsat":
    # Handle unsatisfiable
    pass
else:
    # Handle unknown
    pass
```

### 3. Check Health Before Critical Operations

```python
health = bridge.get_health()
if health["status"] != "healthy":
    print("⚠️  Bridge not healthy, aborting")
    return

# Proceed with operation
response = bridge.solve_constraints(variables, constraints)
```

### 4. Use Descriptions for Constraints

```python
# Good
constraint = CanonicalConstraint(
    "(> temperature 0)",
    ConstraintType.REAL,
    description="Temperature must be positive",
)

# Bad
constraint = CanonicalConstraint(
    "(> temperature 0)",
    ConstraintType.REAL,
)
```

### 5. Log Operations for Debugging

```python
import logging

logger = logging.getLogger("rese.phase1")

response = bridge.solve_constraints(variables, constraints)
logger.info(f"Solve result: {response.result.value}")
if response.errors:
    logger.warning(f"Errors: {response.errors}")
```

### 6. Clean Up Resources

```python
bridge = RESEZ3Bridge()

try:
    # Use bridge
    response = bridge.solve_constraints(variables, constraints)
finally:
    # Always close
    bridge.close()
```

### 7. Use Context Managers (when available)

```python
# Future Python 3.12+ syntax
with RESEZ3Bridge() as bridge:
    response = bridge.solve_constraints(variables, constraints)
```

## Summary

The RESE-Z3 Bridge provides a simple, unified API for all RESE phases:

- **Phase 1 (SCE)**: Use `solve_constraints()` for constraint satisfaction
- **Phase 2 (DITO)**: Use `detect_contradictions()` for contradiction detection
- **Phase 3 (ACI)**: Use `verify_anomaly()` for anomaly verification
- **Phase 4 (Output)**: Use `prove_theorem()` and `translate_to_lean4()` for formal proofs

All methods include:
- Circuit breaker resilience
- Automatic retry with backoff
- Performance monitoring
- Structured logging with correlation IDs

For more details, see:
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Architecture documentation
- [README.md](../README.md) - Main README
