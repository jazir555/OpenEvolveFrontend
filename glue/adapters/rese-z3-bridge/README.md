# RESE-Z3 Bridge Adapter

**Unified interface for all RESE phases to access Z3 capabilities.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/test_rese_z3_bridge.py)
[![Documentation](https://img.shields.io/badge/docs-complete-blue)](ARCHITECTURE.md)
[![Version](https://img.shields.io/badge/version-1.0.0-orange)](src/__init__.py)

## Overview

The RESE-Z3 Bridge Adapter provides a centralized, unified API for all RESE phases (SCE, DITO, ACI, Output) to access Z3 constraint solving and theorem proving capabilities. It implements the Anti-Corruption Layer pattern, maintaining strict isolation between RESE components and the Z3 core system.

## Features

- **Unified API**: Single interface for all Z3 operations
- **LeanAide Integration**: AI-powered autoformalization and theorem proving
- **Circuit Breaker**: Prevents cascading failures
- **Exponential Backoff Retry**: Handles transient failures
- **Canonical Schema**: Anti-corruption layer for data transformation
- **Performance Monitoring**: Built-in metrics and tracking
- **Caching**: Idempotent operations with result caching
- **Structured Logging**: JSON logs with correlation IDs
- **Zero Dependencies on Core**: Law of the "Air Gap" compliance

## Quick Start

### Installation

```bash
cd glue/adapters/rese-z3-bridge
pip install -r requirements.txt
```

### Configuration

Set environment variables:

```bash
# Z3 Server Configuration
export Z3_BASE_URL=http://localhost:8000
export Z3_TIMEOUT_MS=30000
export Z3_CIRCUIT_BREAKER_THRESHOLD=5

# LeanAide Server Configuration (NEW!)
export LEANAIDE_BASE_URL=http://localhost:7654
export LEANAIDE_TIMEOUT_MS=60000
export LEANAIDE_ENABLE=true
```

### Basic Usage

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalVariable, CanonicalConstraint, ConstraintType

# Initialize bridge
bridge = RESEZ3Bridge()

# Solve constraints
variables = [CanonicalVariable("x", ConstraintType.INTEGER)]
constraints = [CanonicalConstraint("(> x 0)", ConstraintType.INTEGER)]

response = bridge.solve_constraints(variables, constraints)
print(f"Result: {response.result.value}")
if response.model:
    print(f"Solution: {response.model.assignments}")

# Check health
health = bridge.get_health()
print(f"Bridge status: {health['status']}")
```

## API Methods

### 1. `solve_constraints()` - SCE (Phase 1)

Find satisfying assignment for constraints.

```python
response = bridge.solve_constraints(
    variables=[
        CanonicalVariable("temperature", ConstraintType.REAL),
        CanonicalVariable("pressure", ConstraintType.REAL),
    ],
    constraints=[
        CanonicalConstraint("(> temperature 0)", ConstraintType.REAL),
        CanonicalConstraint("(< temperature 1000)", ConstraintType.REAL),
    ],
    correlation_id="sce-123",
    timeout_ms=30000,
)

# Check result
if response.result.value == "sat":
    print(f"Solution found: {response.model.assignments}")
elif response.result.value == "unsat":
    print("No solution exists")
else:
    print("Unknown result")
```

### 2. `detect_contradictions()` - DITO (Phase 2)

Detect contradictions in constraint sets.

```python
has_contradiction, counterexample = bridge.detect_contradictions(
    constraints=[
        CanonicalConstraint("(> x 100)", ConstraintType.INTEGER),
        CanonicalConstraint("(< x 0)", ConstraintType.INTEGER),
    ],
    correlation_id="dito-456",
)

if has_contradiction:
    print("Contradiction detected!")
else:
    print(f"No contradiction, counterexample: {counterexample}")
```

### 3. `verify_anomaly()` - ACI (Phase 3)

Verify if anomaly violates constraints.

```python
is_valid, error = bridge.verify_anomaly(
    constraints=[
        CanonicalConstraint("(> temperature 500)", ConstraintType.REAL),
    ],
    correlation_id="aci-789",
)

if not is_valid:
    print(f"Anomaly detected: {error}")
```

### 4. `prove_theorem()` - Formal Verification

Prove mathematical theorems.

```python
response = bridge.prove_theorem(
    theorem_statement="(implies (> x 0) (> (+ x 1) 0))",
    variables={"x": "Int"},
    assumptions=[],
    correlation_id="theorem-101",
)

if response.proven:
    print("Theorem proven!")
    print(f"Proof: {response.proof}")
else:
    print("Theorem disproven")
    print(f"Counterexample: {response.counterexample}")
```

### 5. `translate_to_lean4()` - Lean 4 Integration

Translate SMT-LIB to Lean 4 format.

```python
lean4_code = bridge.translate_to_lean4(
    smtlib_content="(declare-const x Int) (assert (> x 0))",
    correlation_id="lean4-202",
)
print(lean4_code)
```

### 6. `autoformalize()` - LeanAide Autoformalization (NEW!)

Convert natural language to Lean 4 theorems.

```python
response = bridge.autoformalize(
    natural_language="There are infinitely many prime numbers",
    theorem_name="infinitely_many_primes",
    correlation_id="auto-123",
)

if response.success:
    print(f"Lean 4 code: {response.lean_code}")
    print(f"Theorem type: {response.theorem_type}")
```

### 7. `prove_with_ai()` - AI-Powered Proving (NEW!)

Generate proofs using LeanAide AI.

```python
response = bridge.prove_with_ai(
    theorem_text="For all natural numbers n, n + 0 = n",
    theorem_code="theorem add_zero (n : Nat) : n + 0 = n",
    correlation_id="prove-456",
)

if response.success:
    print(f"Proof: {response.proof}")
    print(f"Tactics used: {response.tactics_used}")
```

### 8. `translate_z3_to_lean()` - Z3 to Lean Translation (NEW!)

Bridge Z3 constraints to Lean 4.

```python
response = bridge.translate_z3_to_lean(
    smtlib_content="(declare-fun x () Int)(assert (> x 0))",
    constraint_type=ConstraintType.INTEGER,
    correlation_id="trans-789",
)

if response.success:
    print(f"Lean code: {response.lean_code}")
    print(f"Variables: {response.variables}")
```

### 9. `suggest_tactics()` - AI Tactic Suggestions (NEW!)

Get AI-recommended proof tactics.

```python
response = bridge.suggest_tactics(
    goal_state="⊢ x + y = y + x",
    context="Working with real numbers",
    num_suggestions=3,
    correlation_id="tactics-101",
)

if response.success:
    for suggestion in response.suggestions:
        print(f"{suggestion.tactic}: {suggestion.description}")
        print(f"Confidence: {suggestion.confidence}")
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `Z3_BASE_URL` | Z3 server URL | `http://localhost:8000` | No |
| `Z3_TIMEOUT_MS` | Request timeout (ms) | `30000` | No |
| `LEANAIDE_BASE_URL` | LeanAide server URL | `http://localhost:7654` | No |
| `LEANAIDE_TIMEOUT_MS` | LeanAide timeout (ms) | `60000` | No |
| `LEANAIDE_ENABLE` | Enable LeanAide features | `true` | No |
| `Z3_CIRCUIT_BREAKER_THRESHOLD` | Failures before opening | `5` | No |
| `Z3_CIRCUIT_BREAKER_TIMEOUT_MS` | Time to stay open (ms) | `60000` | No |
| `Z3_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `Z3_RETRY_BACKOFF_MS` | Retry backoff (ms) | `1000` | No |
| `Z3_ENABLE_CACHE` | Enable result caching | `true` | No |
| `Z3_CACHE_TTL_MS` | Cache TTL (ms) | `300000` | No |
| `Z3_ENABLE_MONITORING` | Enable monitoring | `true` | No |

**Law of Configuration Explicitness**: All configurable values must be set via environment. No magic defaults.

## Testing

### Run All Tests

```bash
# Run unit tests
python -m pytest tests/test_rese_z3_bridge.py -v

# Run with coverage
python -m pytest tests/test_rese_z3_bridge.py --cov=src --cov-report=html
```

### Run Specific Test

```bash
# Test schema transformations
python -m pytest tests/test_rese_z3_bridge.py::TestCanonicalSchema -v

# Test circuit breaker
python -m pytest tests/test_rese_z3_bridge.py::TestCircuitBreaker -v

# Test bridge API
python -m pytest tests/test_rese_z3_bridge.py::TestRESEZ3Bridge -v
```

### Runtime Verification Probe

```bash
# Verify Z3 bridge is working end-to-end
bash probes/check_z3_bridge.sh

# Verify LeanAide integration (NEW!)
bash probes/check_leanaide.sh
```

**Law of Runtime Truth**: Probes execute actual calls to verify functionality.

## Docker Deployment

### Build Image

```bash
docker build -t rese-z3-bridge:latest .
```

### Run Container

```bash
docker run -d \
  -e Z3_BASE_URL=http://z3-core:8000 \
  -e Z3_TIMEOUT_MS=30000 \
  --name rese-z3-bridge \
  rese-z3-bridge:latest
```

### Check Health

```bash
docker exec rese-z3-bridge python -c "
from rese_z3_bridge import RESEZ3Bridge
bridge = RESEZ3Bridge()
health = bridge.get_health()
print(health['status'])
"
```

## Monitoring

### Get Bridge Stats

```python
bridge = RESEZ3Bridge()
stats = bridge.get_stats()

print(json.dumps(stats, indent=2))
```

**Output:**
```json
{
  "config": {
    "z3_base_url": "http://localhost:8000",
    "z3_timeout_ms": 30000,
    "circuit_breaker_threshold": 5
  },
  "client_stats": {
    "circuit_breaker": {
      "state": "closed",
      "failure_count": 0,
      "total_calls": 10,
      "total_successes": 10
    }
  },
  "performance_summary": {
    "total_operations": 10,
    "successful_operations": 10,
    "success_rate": 1.0,
    "average_duration_ms": 45.2
  }
}
```

## Integration Examples

### Phase 1 (SCE) Integration

```python
# In glue/adapters/rese-phase1/src/phase1_adapter.py

from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

class SCEAdapter:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def solve_constraints(self, assumptions):
        # Convert assumptions to canonical constraints
        constraints = [
            CanonicalConstraint(
                assumption.to_smtlib(),
                ConstraintType.BOOLEAN,
                assumption.description,
            )
            for assumption in assumptions
        ]

        # Solve via bridge
        response = self.bridge.solve_constraints(
            variables=[],
            constraints=constraints,
        )

        return response.result.value == "sat"
```

### Phase 2 (DITO) Integration

```python
# In glue/adapters/rese-sce/src/dito_optimizer.py

from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

class DITOOptimizer:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def check_contradiction_targeted(self, constraint, active_constraints):
        # Convert to canonical format
        constraints = [
            CanonicalConstraint(c.expression, c.constraint_type)
            for c in [constraint] + active_constraints
        ]

        # Check via bridge
        has_contradiction, _ = self.bridge.detect_contradictions(
            constraints=constraints,
        )

        return has_contradiction
```

### Phase 3 (ACI) Integration

```python
# In glue/adapters/rese-phase3/src/aci_calculator.py

from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

class ACICalculator:
    def __init__(self):
        self.bridge = RESEZ3Bridge()

    def verify_anomaly_constraints(self, anomaly_constraints):
        # Convert to canonical format
        constraints = [
            CanonicalConstraint(c.expression, c.constraint_type)
            for c in anomaly_constraints
        ]

        # Verify via bridge
        is_valid, error = self.bridge.verify_anomaly(
            constraints=constraints,
        )

        return is_valid, error
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.
See [docs/LEANAIDE_INTEGRATION.md](docs/LEANAIDE_INTEGRATION.md) for LeanAide-specific documentation.

**Key Principles:**
- Anti-Corruption Layer pattern
- Circuit breaker for resilience
- Canonical schema for data isolation
- Structured logging with correlation IDs
- Performance monitoring built-in
- LeanAide integration for AI-powered formalization

## Troubleshooting

### Circuit Breaker Open

**Problem:** Requests rejected with "Circuit breaker is OPEN"

**Solution:**
```bash
# Check Z3 server health
curl http://localhost:8000/health

# Wait for circuit breaker timeout (default 60s)
# Or fix Z3 server and restart bridge
```

### Connection Timeout

**Problem:** `Z3ClientTimeoutError`

**Solution:**
```bash
# Increase timeout
export Z3_TIMEOUT_MS=60000

# Or simplify constraints to reduce solve time
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'rese_z3_bridge'`

**Solution:**
```bash
# Add src to path
import sys
sys.path.insert(0, 'glue/adapters/rese-z3-bridge/src')

from rese_z3_bridge import RESEZ3Bridge
```

## Contributing

When adding features to the bridge:

1. **Follow CLAUDE.md laws**:
   - No imports from `core-projects/`
   - Runtime verification via probes
   - Configuration via environment
   - UTC timestamps in ISO-8601

2. **Add tests**:
   - Unit tests for new functionality
   - Contract tests for API changes
   - Idempotency tests

3. **Update documentation**:
   - Update README.md
   - Update ARCHITECTURE.md
   - Add usage examples

4. **Run probe**:
   ```bash
   bash probes/check_z3_bridge.sh
   ```

## License

Copyright (c) 2026 OpenEvolve RESE Team

## Authors

- RESE Team
- Created: 2026-02-04

## Version

**1.0.0** - Initial release with unified API for all RESE phases
