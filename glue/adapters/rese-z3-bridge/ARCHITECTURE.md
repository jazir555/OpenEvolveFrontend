# RESE-Z3 Bridge Architecture

## Overview

The RESE-Z3 Bridge Adapter provides a unified interface for all RESE phases to access Z3 capabilities. It implements the Anti-Corruption Layer pattern from CLAUDE.md, maintaining strict isolation between RESE components and the Z3 core system.

## Architecture Principles

Following CLAUDE.md laws:

1. **Law of the "Air Gap"**: No imports from `core-projects/`. All Z3 interactions via HTTP API.
2. **Law of Runtime Truth**: Probe scripts verify actual functionality before claiming it works.
3. **Law of the "Untouchable DB"**: Read-only access to constraint data.
4. **Law of Idempotency**: All operations safe to run 100 times.
5. **Law of Configuration Explicitness**: All config via environment variables (no magic defaults).
6. **Law of UTC**: All timestamps in UTC ISO-8601 format.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RESE Phases                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Phase 1  │  │ Phase 2  │  │ Phase 3  │  │ Phase 4  │   │
│  │   SCE    │  │   DITO   │  │   ACI    │  │  Output  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼────────────┼────────────┼────────────┼────────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  RESE-Z3 Bridge   │
                    │   (This Module)   │
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │      Anti-Corruption Layer    │
              │   (Canonical Schema Mapping)  │
              └───────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Z3 HTTP Client  │
                    │  (Circuit Breaker)│
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Z3 API Server   │
                    │ (z3prover_integ.) │
                    └───────────────────┘
```

## Components

### 1. Canonical Schema (`rese_z3_schema.py`)

Defines the canonical data models for Z3 interactions:

**Key Classes:**
- `CanonicalSolverRequest`: Standardized solver request
- `CanonicalSolverResponse`: Standardized solver response
- `CanonicalTheoremRequest`: Theorem proving request
- `CanonicalTheoremResponse`: Theorem proving response
- `CanonicalVariable`: Variable definition
- `CanonicalConstraint`: Constraint definition

**Transformation Functions:**
- `canonical_to_z3_request()`: Transform canonical to Z3 format
- `z3_to_canonical_response()`: Transform Z3 to canonical format
- `canonical_to_smtlib()`: Generate SMT-LIB2 format

### 2. Z3 Client (`rese_z3_client.py`)

HTTP client for communicating with Z3 API server:

**Features:**
- **Circuit Breaker Pattern**: Detects failures and prevents cascading
- **Exponential Backoff Retry**: Handles transient failures
- **Connection Pooling**: Efficient HTTP connections
- **Timeout Enforcement**: All requests bounded

**Key Classes:**
- `Z3Client`: Main HTTP client
- `CircuitBreaker`: Implements circuit breaker pattern
- `Z3ClientError`: Exception hierarchy

### 3. Bridge Adapter (`rese_z3_bridge.py`)

Main bridge providing unified API for all RESE phases:

**API Methods:**

1. **`solve_constraints()`** - For SCE (Phase 1)
   - Finds satisfying assignment for constraints
   - Returns model with variable assignments

2. **`detect_contradictions()`** - For DITO (Phase 2)
   - Detects unsatisfiable constraint sets
   - Returns counterexample if satisfiable

3. **`verify_anomaly()`** - For ACI (Phase 3)
   - Checks if anomaly violates constraints
   - Returns validation result

4. **`prove_theorem()`** - For formal verification
   - Proves mathematical theorems
   - Returns proof or counterexample

5. **`translate_to_lean4()`** - For Lean 4 integration
   - Translates SMT-LIB to Lean 4 format
   - Enables cross-system formal verification

**Additional Features:**
- **Performance Monitoring**: Tracks operation metrics
- **Caching**: Caches results for idempotency
- **Structured Logging**: JSON logs with correlation IDs

## Data Flow

### Example: SCE Constraint Solving

```
1. Phase 1 (SCE) creates CanonicalVariable and CanonicalConstraint objects
2. Calls bridge.solve_constraints(variables, constraints)
3. Bridge converts to CanonicalSolverRequest
4. Transform to Z3 request format via canonical_to_z3_request()
5. Generate SMT-LIB2 via canonical_to_smtlib()
6. Z3Client sends HTTP POST to Z3 server
7. Circuit breaker monitors for failures
8. Z3 server returns raw response
9. Transform to CanonicalSolverResponse via z3_to_canonical_response()
10. Cache result for idempotency
11. Return canonical response to Phase 1
```

## Circuit Breaker Logic

The circuit breaker prevents cascading failures:

**States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failing, requests rejected immediately
- **HALF_OPEN**: Testing if service recovered

**Transitions:**
```
CLOSED ──[failures >= threshold]──> OPEN
OPEN ──[timeout elapsed]──> HALF_OPEN
HALF_OPEN ──[successes >= threshold]──> CLOSED
HALF_OPEN ──[failures >= threshold]──> OPEN
```

**Configuration:**
```bash
Z3_CIRCUIT_BREAKER_THRESHOLD=5      # Failures before opening
Z3_CIRCUIT_BREAKER_TIMEOUT_MS=60000 # Time to stay open
```

## Performance Monitoring

The bridge monitors all operations:

**Metrics Tracked:**
- Operation name
- Duration (ms)
- Success/failure status
- Cache hit/miss
- Error messages

**Access Metrics:**
```python
bridge = RESEZ3Bridge()
stats = bridge.get_stats()
performance = stats["performance_summary"]
print(f"Average duration: {performance['average_duration_ms']:.2f}ms")
print(f"Success rate: {performance['success_rate']:.1%}")
```

## Error Handling

**Error Hierarchy:**
```
Z3ClientError
├── Z3ClientConnectionError  # Network/connection failures
├── Z3ClientTimeoutError      # Request timeouts
└── Z3ClientCircuitBreakerOpenError  # Circuit breaker open
```

**Error Handling Strategy:**
- **Transient Failures**: Retry with exponential backoff
- **Logic Failures**: Return error response (don't crash)
- **System Failures**: Circuit breaker opens, stop requests

## Configuration

All configuration via environment variables:

```bash
# Z3 Server Configuration
Z3_BASE_URL=http://localhost:8000
Z3_TIMEOUT_MS=30000

# Circuit Breaker Configuration
Z3_CIRCUIT_BREAKER_THRESHOLD=5
Z3_CIRCUIT_BREAKER_TIMEOUT_MS=60000

# Retry Configuration
Z3_MAX_RETRIES=3
Z3_RETRY_BACKOFF_MS=1000

# Caching Configuration
Z3_ENABLE_CACHE=true
Z3_CACHE_TTL_MS=300000

# Monitoring Configuration
Z3_ENABLE_MONITORING=true
```

## Testing Strategy

### 1. Unit Tests (`tests/test_rese_z3_bridge.py`)

Tests individual components:
- Schema transformations
- Circuit breaker logic
- API methods
- Error handling

### 2. Contract Tests

Prevents API breakage:
- Validate request/response formats
- Ensure canonical schema consistency
- Check transformation functions

### 3. Integration Tests

End-to-end testing:
- Full request/response cycle
- Circuit breaker state transitions
- Cache functionality

### 4. Idempotency Tests

Verify idempotency:
- Same input → same output
- Cached results identical
- Multiple calls safe

## Runtime Verification

Probe script (`probes/check_z3_bridge.sh`) verifies:
1. Bridge can connect to Z3 server
2. All API methods work
3. Circuit breaker functional
4. Schema transformations correct

**Run probe:**
```bash
cd glue/adapters/rese-z3-bridge
bash probes/check_z3_bridge.sh
```

## Usage Examples

### Phase 1 (SCE) - Constraint Solving

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalVariable, CanonicalConstraint, ConstraintType

bridge = RESEZ3Bridge()

# Define variables
variables = [
    CanonicalVariable("temperature", ConstraintType.REAL),
    CanonicalVariable("pressure", ConstraintType.REAL),
]

# Define constraints
constraints = [
    CanonicalConstraint("(> temperature 0)", ConstraintType.REAL, "T > 0"),
    CanonicalConstraint("(< temperature 1000)", ConstraintType.REAL, "T < 1000"),
    CanonicalConstraint("(> pressure 0)", ConstraintType.REAL, "P > 0"),
]

# Solve
response = bridge.solve_constraints(variables, constraints)
if response.result.value == "sat":
    print(f"Solution: {response.model.assignments}")
```

### Phase 2 (DITO) - Contradiction Detection

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

bridge = RESEZ3Bridge()

# Check for contradictions
constraints = [
    CanonicalConstraint("(> x 100)", ConstraintType.INTEGER),
    CanonicalConstraint("(< x 0)", ConstraintType.INTEGER),
]

has_contradiction, counterexample = bridge.detect_contradictions(constraints)
if has_contradiction:
    print("Contradiction detected!")
else:
    print(f"No contradiction, counterexample: {counterexample}")
```

### Phase 3 (ACI) - Anomaly Verification

```python
from rese_z3_bridge import RESEZ3Bridge
from rese_z3_schema import CanonicalConstraint, ConstraintType

bridge = RESEZ3Bridge()

# Verify anomaly constraints
constraints = [
    CanonicalConstraint("(> temperature 500)", ConstraintType.REAL),
]

is_valid, error = bridge.verify_anomaly(constraints)
if not is_valid:
    print(f"Anomaly detected: {error}")
```

### Theorem Proving

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

# Prove theorem: x > 0 implies x + 1 > 0
theorem = "(implies (> x 0) (> (+ x 1) 0))"
response = bridge.prove_theorem(
    theorem_statement=theorem,
    variables={"x": "Int"},
)

if response.proven:
    print("Theorem proven!")
    print(f"Proof: {response.proof}")
else:
    print("Theorem disproven")
    print(f"Counterexample: {response.counterexample}")
```

## Deployment

### Docker

```bash
# Build image
docker build -t rese-z3-bridge:latest .

# Run container
docker run -d \
  -e Z3_BASE_URL=http://z3-core:8000 \
  -e Z3_TIMEOUT_MS=30000 \
  --name rese-z3-bridge \
  rese-z3-bridge:latest
```

### Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rese-z3-bridge-config
data:
  Z3_BASE_URL: "http://z3-core:8000"
  Z3_TIMEOUT_MS: "30000"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rese-z3-bridge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rese-z3-bridge
  template:
    metadata:
      labels:
        app: rese-z3-bridge
    spec:
      containers:
      - name: bridge
        image: rese-z3-bridge:latest
        envFrom:
        - configMapRef:
            name: rese-z3-bridge-config
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 10
          periodSeconds: 30
```

## Monitoring

The bridge exposes metrics for monitoring:

**Health Check:**
```bash
curl http://localhost:9090/health
```

**Metrics:**
```python
bridge = RESEZ3Bridge()
stats = bridge.get_stats()
print(json.dumps(stats, indent=2))
```

**Metrics Include:**
- Circuit breaker state
- Operation success rate
- Average duration
- Cache hit rate
- Error counts

## Troubleshooting

### Circuit Breaker Open

**Symptom:** Requests rejected with "Circuit breaker is OPEN"

**Solution:**
1. Check Z3 server health
2. Wait for timeout (default 60s)
3. Fix underlying issue
4. Circuit will auto-recover

### Timeout Errors

**Symptom:** `Z3ClientTimeoutError`

**Solution:**
1. Increase `Z3_TIMEOUT_MS`
2. Simplify constraints
3. Check Z3 server performance

### Connection Errors

**Symptom:** `Z3ClientConnectionError`

**Solution:**
1. Verify `Z3_BASE_URL` is correct
2. Check Z3 server is running
3. Check network connectivity
4. Verify firewall rules

## Future Enhancements

1. **Async API**: Async/await support for concurrent operations
2. **Batch Operations**: Process multiple requests efficiently
3. **Advanced Caching**: Redis-based distributed caching
4. **Metrics Export**: Prometheus metrics endpoint
5. **GraphQL API**: Alternative to REST
6. **WebSocket Support**: Real-time constraint updates
7. **Advanced Translation**: Better SMT-LIB to Lean 4 translation
