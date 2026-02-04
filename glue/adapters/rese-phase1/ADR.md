# ADR: RESE Phase I - Epistemic Audit Implementation

**Status:** Accepted
**Date:** 2025-02-04
**Context:** RESE Integration Task #7

## Context

Phase I of the Recursive Epistemic Solvability Engine (RESE) performs an Epistemic Audit and Falsification using the Red Team Protocol. This phase is critical for identifying and testing both explicit and tacit assumptions in incumbent theoretical frameworks.

**Technical Manual References:**
- Section 3.0: Phase I - Epistemic Audit and Falsification
- Section 3.1: Initial Hypothesis Cluster Definition (Φ₁)
- Section 3.1.5: Tacit Assumption Mining (Φ₁.₅)
- Section 3.3: Formal Logic Audit and Contradiction Detection (Φ₃)

## Decision

Implemented Phase I as a Python-based adapter with the following architecture:

### 1. Core Components

**EpistemicAuditExecutor** - Main orchestrator
- Integrates all Phase I subroutines (Φ₁, Φ₁.₅, Φ₃, Φ₄)
- Follows CLAUDE.md laws (idempotency, timeouts, circuit breakers)
- Outputs canonical schema format

**ConstraintHardener** (Φ₁)
- Extracts and hardens constraints from problem descriptions
- Implements logical inversion (ℂ → ¬ℂ)
- Creates Category A (Hard Parameter Inequality) constraints

**AssumptionMiner** (Φ₁.₅)
- Mines tacit assumptions from failure patterns
- Implements inverse inference analysis
- Creates Category C (Tacit Assumption) constraints

**RedTeamProtocator** (Φ₄)
- Adversarial testing of assumptions
- Calculates Hypothesis Robustness Score (HRS)
- Tests against cross-domain adversarial data

**SCEAdapter**
- Integration with TypeScript Symbolic Constraint Engine
- Executes contradiction detection via Node.js subprocess
- Maintains air gap between Python and TypeScript

### 2. Technology Choices

**Python 3.11** as primary language
- Rationale: Better dataclass support, strong typing
- Alignment with RESE core (Python-based)
- Easier integration with scientific Python ecosystem

**TypeScript SCE** via subprocess
- Rationale: SCE already implemented in TypeScript (Task #2)
- Maintains separation of concerns
- Follows Law of the "Air Gap"

**Structured Logging** (JSON Lines)
- Rationale: CLAUDE.md requirement for observability
- Enables distributed tracing via correlation_id
- Machine-parseable for monitoring

**Circuit Breaker Pattern**
- Rationale: Prevents cascading failures
- CLAUDE.md requirement for failure management
- Protects both Phase I and SCE from overload

### 3. Data Flow

```
Problem Description + Failure Patterns
    ↓
ConstraintHardener (Φ₁)
    ↓
AssumptionMiner (Φ₁.₅)
    ↓
SCEAdapter → SCE (Φ₃)
    ↓
RedTeamProtocator (Φ₄)
    ↓
Canonical EpistemicAuditResult
```

## Alternatives Considered

### Alternative 1: Pure TypeScript Implementation

**Pros:**
- Direct integration with SCE
- Single language stack

**Cons:**
- Less alignment with RESE core (Python)
- Harder to integrate with scientific Python libraries
- RESE Technical Manual examples use Python-like syntax

**Decision:** Rejected - Python better aligns with RESE core

### Alternative 2: Direct SCE Import (via Python bridge)

**Pros:**
- Faster execution (no subprocess overhead)

**Cons:**
- Violates Law of the "Air Gap"
- Tight coupling between Python and TypeScript
- Harder to maintain isolation

**Decision:** Rejected - Subprocess approach maintains isolation

### Alternative 3: No Circuit Breaker

**Pros:**
- Simpler implementation

**Cons:**
- Violates CLAUDE.md failure management strategy
- Risk of cascading failures
- No protection against SCE overload

**Decision:** Rejected - Circuit breaker is mandatory per CLAUDE.md

## Consequences

### Positive

1. **Idempotency**: All operations safe to run 100x
2. **Observability**: Structured logging with correlation_id
3. **Resilience**: Circuit breaker prevents cascading failures
4. **Canonical Format**: Seamless integration with other phases
5. **Runtime Verification**: Probe scripts validate functionality

### Negative

1. **Subprocess Overhead**: SCE integration via Node.js adds latency
2. **Complexity**: Multiple components increase maintenance burden
3. **Python-TypeScript Gap**: Two languages require expertise in both

### Mitigations

1. **Subprocess Overhead**: Added timeouts and caching
2. **Complexity**: Comprehensive documentation and tests
3. **Language Gap**: Clear interfaces and canonical schema

## Implementation Details

### Configuration Management

All configuration via environment variables (Law of Configuration Explicitness):

```python
@dataclass
class Phase1Config:
    TIMEOUT_MS: int
    MAX_ASSUMPTIONS: int
    CIRCUIT_BREAKER_THRESHOLD: int
    # ... etc

    @classmethod
    def from_env(cls) -> 'Phase1Config':
        # Load from environment, validate, crash if invalid
```

### Circuit Breaker Logic

```python
class CircuitBreaker:
    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.state = CircuitBreakerState.OPEN

    def can_execute(self) -> bool:
        if self.state == OPEN:
            elapsed = now() - self.opened_at
            if elapsed >= self.timeout_ms:
                self.state = HALF_OPEN
                return True
            return False
        return True
```

### Dead Letter Queue

```python
class DeadLetterQueue:
    def enqueue(self, item: Dict[str, Any]) -> bool:
        if len(self._queue) >= self.max_size:
            self._queue.pop(0)  # Drop oldest
        self._queue.append(item)
```

## Testing Strategy

### 1. Probe Scripts

`probes/check_phase1.sh` validates:
- Module imports
- Configuration loading
- Component instantiation
- Dataclass serialization
- End-to-end audit execution

### 2. Unit Tests (Future)

- ConstraintHardener tests
- AssumptionMiner tests
- RedTeamProtocator tests
- CircuitBreaker tests
- DLQ tests

### 3. Integration Tests (Future)

- SCEAdapter integration
- Full Phase I audit
- Error scenarios

## Gotchas and Edge Cases

### 1. UTC Timestamps

**Issue:** Inconsistent timezone handling

**Solution:**
```python
timestamp = datetime.now(timezone.utc).isoformat()  # Law of UTC
```

### 2. Circuit Breaker State

**Issue:** Circuit breaker stays open indefinitely

**Solution:** Automatic timeout to HALF_OPEN state after configured timeout

### 3. Idempotency Violations

**Issue:** Mining same assumption twice

**Solution:** Check before create pattern (implemented in AssumptionMiner)

### 4. Timeout Enforcement

**Issue:** Operations hang indefinitely

**Solution:** All operations have timeouts via subprocess and explicit checks

### 5. SCE Integration

**Issue:** TypeScript SCE not available

**Solution:** Fallback to basic contradiction detection (implemented in `_detect_contradictions`)

## Future Work

### 1. Lean 4 Integration

Currently stubbed (`ENABLE_LEAN4_INTEGRATION = false`).
Future: Integrate Lean 4 theorem prover for formal verification.

### 2. Enhanced Assumption Mining

Current: Simple heuristic pattern matching.
Future: NLP-based inverse inference with statistical correlation.

### 3. Cross-Domain Red Team

Current: Simulated adversarial testing.
Future: Real cross-domain data integration.

### 4. Performance Optimization

Current: O(n²) contradiction detection.
Future: Implement DITO (Dynamic Inference Trace Optimizer).

## References

- CLAUDE.md: Federation Constitution
- RESE Technical Manual: Section 3.0
- Task #2: Symbolic Constraint Engine (TypeScript)
- Task #5: Canonical Schemas (TypeScript)
- Task #6: Probe Scripts

## Appendix: Code Structure

```
glue/adapters/rese-phase1/
├── src/
│   ├── phase1_executor.py      # Main executor + components
│   └── phase1_adapter.py       # SCE adapter + CLI
├── probes/
│   └── check_phase1.sh         # Runtime verification
├── tests/
│   └── (future unit tests)
├── Dockerfile
├── README.md
└── ADR.md                      # This document
```

---

**Author:** RESE Integration Team
**Reviewers:** TBD
**Approved:** TBD
