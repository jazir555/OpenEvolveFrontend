# Architecture Decision Record: RESE Deep Exploration Engine Implementation

**Date:** 2026-02-04
**Status:** Accepted
**Component:** RESE Deep Exploration Engine (DEE)
**Task:** #3 Implement RESE Deep Exploration Engine

---

## Context

The RESE (Recursive Epistemic Solvability Engine) system exists only in bytecode format with no source code recoverable (see SOURCE_RECOVERY_REPORT.md). Task #3 requires implementing the Deep Exploration Engine (DEE), a critical component of Phase III (MCTS Search) that provides:

1. Hypothesis generation from problem statements
2. Cross-domain pattern recognition
3. MCTS-based exploration and refinement
4. Integration with the broader RESE pipeline

The implementation must follow CLAUDE.md principles:
- Law of Configuration Explicitness (all config via env vars)
- Law of Idempotency (UPSERT logic, deduplication)
- Circuit Breaker (failure detection)
- Structured Logging (JSON with correlation_id)
- Timeout (all operations bounded)

## Decision

### 1. Core Library Structure

**Decision:** Implement DEE as a pure Python library in `glue/lib/rese_dee.py` with no external dependencies.

**Rationale:**
- Python standard library provides all required functionality (dataclasses, typing, datetime, uuid, etc.)
- Zero external dependencies reduces deployment complexity
- Easier to maintain and test
- Consistent with CLAUDE.md isolation principles

**Components:**
- `DeepExplorationEngine`: Main orchestrator
- `HypothesisGenerator`: Creates testable hypotheses
- `PatternRecognizer`: Cross-domain pattern matching
- `MCTSExplainer`: Monte Carlo Tree Search implementation

### 2. Canonical Schemas

**Decision:** Define all canonical schemas in `glue/schemas/rese_schemas.py` before implementing logic.

**Rationale:**
- Schema-first approach ensures data contracts are defined early
- Enables contract-based testing (CLAUDE.md Phase 2)
- Provides clear API boundaries
- Facilitates serialization/deserialization

**Key Schemas:**
- `Hypothesis`: Testable hypothesis with evidence tracking
- `SearchTreeNode`: MCTS tree node with UCB calculation
- `Pattern`: Recognized cross-domain pattern
- `MCTSSearchResult`: Complete search result with statistics
- `ExplorationConfig`: Configuration from environment variables

### 3. Idempotency Implementation

**Decision:** Implement UPSERT logic with deduplication by unique IDs for all stateful operations.

**Rationale:**
- CLAUDE.md Law of Idempotency requires safe replay of operations
- Prevents duplicate hypotheses, patterns, and evidence
- Enables safe retry of failed requests

**Implementation:**
```python
# Hypothesis deduplication by hypothesis_id
def _deduplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> Dict[str, Hypothesis]:
    unique = {}
    for hypothesis in hypotheses:
        existing = unique.get(hypothesis.hypothesis_id)
        if existing is None or hypothesis.confidence > existing.confidence:
            unique[hypothesis.hypothesis_id] = hypothesis
    return unique

# Evidence deduplication by evidence_id
def update_evidence(self, new_evidence: Dict[str, Any], is_supporting: bool = True):
    evidence_id = new_evidence.get("evidence_id") or hash(...)
    if evidence_id not in existing_ids:
        self.evidence.append({**new_evidence, "evidence_id": evidence_id})
```

### 4. Circuit Breaker Pattern

**Decision:** Implement circuit breaker for pattern recognition failures.

**Rationale:**
- Pattern recognition is computationally expensive
- Failures could cascade and crash the system
- CLAUDE.md requires graceful degradation

**States:**
- `CLOSED`: Normal operation
- `OPEN`: Failures detected, stop attempting
- `HALF_OPEN`: Testing if service recovered

**Configuration:**
- `failure_threshold`: Number of failures before opening (default: 5)
- `recovery_timeout_ms`: Time before attempting recovery (default: 60000ms)
- `half_open_max_calls`: Number of calls to test recovery (default: 3)

### 5. Exponential Backoff with Jitter

**Decision:** Implement retry with exponential backoff and jitter for transient failures.

**Rationale:**
- Prevents thundering herd problem
- CLAUDE.md failure management strategy specifies: "Transient Failure → Exponential Backoff Retry (Jittered)"

**Implementation:**
```python
def retry_with_backoff(func, max_retries=3, base_delay_ms=100, jitter_factor=0.1):
    delay_ms = min(base_delay_ms * (2 ** attempt), max_delay_ms)
    jitter = delay_ms * jitter_factor * (random.random() * 2 - 1)
    final_delay_ms = max(0, delay_ms + jitter)
    time.sleep(final_delay_ms / 1000.0)
```

### 6. Dead Letter Queue (DLQ)

**Decision:** Implement DLQ for logic and system failures (not transient).

**Rationale:**
- CLAUDE.md: "Logic Failure → Dead Letter Queue. Do not block the pipeline."
- Enables post-mortem analysis
- Allows manual retry of failed requests

**Error Classification:**
- `transient`: Network, timeout (retry, no DLQ)
- `logic`: Validation, bad data (DLQ)
- `system`: Circuit breaker, pattern recognition (DLQ)

### 7. Configuration Management

**Decision:** All configuration via environment variables with validation at startup.

**Rationale:**
- CLAUDE.md Law of Configuration Explicitness: "Every configurable value must be injected via Environment Variables"
- "If TARGET_API_URL is missing, the service crashes immediately with a loud error"
- No magic defaults

**Required Variables:**
```bash
EXPLORATION_DEPTH=10
MCTS_ITERATIONS=1000
MCTS_EXPLORATION_CONSTANT=1.414
CONVERGENCE_THRESHOLD=0.001
EXPLORATION_TIMEOUT_MS=10000
MAX_HYPOTHESES=100
PATTERN_RECOGNITION_THRESHOLD=0.7
```

### 8. Structured Logging

**Decision:** JSON Lines format with correlation_id, source_service, timestamp.

**Rationale:**
- CLAUDE.md: "Format: JSON Lines (jsonl)"
- "Context: correlation_id, source_service, target_service"
- Enables log aggregation and analysis

**Implementation:**
```python
{
    "msg": "Exploration started",
    "level": "info",
    "correlation_id": "uuid",
    "source_service": "rese_dee",
    "timestamp": "2026-02-04T12:00:00Z",
    "domain": "performance"
}
```

### 9. Timeout Enforcement

**Decision:** All exploration operations bounded by EXPLORATION_TIMEOUT_MS (default: 10000ms).

**Rationale:**
- CLAUDE.md: "Timeout: All exploration operations have timeout"
- "MANDATORY. Every HTTP request must have a timeout"
- Prevents infinite hangs

**Implementation:**
```python
start_time = time.time()
for iteration in range(mcts_iterations):
    elapsed_ms = (time.time() - start_time) * 1000
    if elapsed_ms > self.config.timeout_ms:
        logger.warning("MCTS timeout reached", iteration=iteration)
        break
```

### 10. Adapter Architecture

**Decision:** Separate adapter layer in `glue/adapters/rese-dee/src/dee_adapter.py`.

**Rationale:**
- CLAUDE.md: "Keep the 'Glue' distinct from the 'Core'"
- Adapter wraps library for API interface
- Provides request validation and transformation
- Handles DLQ and error classification

**Adapter Responsibilities:**
- Request validation (checks required fields, types)
- Response transformation to canonical format
- DLQ management
- Health checks
- Circuit breaker integration

## Alternatives Considered

### Alternative 1: Use NumPy for numerical operations
**Rejected:** Would add external dependency. Python standard library is sufficient for current requirements. Can optimize later if needed.

### Alternative 2: Implement full DITO optimizer with R-tree
**Rejected:** Too complex for initial implementation. Started with simpler O(n²) approach. Can optimize later following SOURCE_RECOVERY_REPORT.md recommendations.

### Alternative 3: Use existing MCTS library
**Rejected:** No suitable pure-Python MCTS library available. Custom implementation allows better integration with RESE schemas.

### Alternative 4: Implement Lean 4 integration immediately
**Rejected:** SOURCE_RECOVERY_REPORT.md marks this as "Tier 6 (Optional/Advanced)". Defer to later phase.

## Consequences

### Positive
- Zero external dependencies simplifies deployment
- Schema-first approach enables contract testing
- Idempotency ensures safe retry operations
- Circuit breaker prevents cascading failures
- Structured logging enables observability
- Configuration validation catches errors early

### Negative
- Pure Python implementation may be slower than optimized C/C++ libraries
- Initial implementation uses simpler algorithms (O(n²) instead of O(n log n))
- Additional complexity from circuit breaker and DLQ management

### Risks
- MCTS convergence may be slow for complex problems
- Pattern recognition may produce false positives
- Hypothesis quality depends on problem statement clarity

### Mitigations
- Configurable timeouts and iteration limits
- Confidence thresholds filter low-quality patterns
- Circuit breaker prevents runaway failures
- DLQ enables analysis of failed requests

## Implementation Status

**Completed:**
- [x] Canonical schemas in `glue/schemas/rese_schemas.py`
- [x] Core library in `glue/lib/rese_dee.py`
  - [x] HypothesisGenerator
  - [x] PatternRecognizer
  - [x] MCTSExplainer
  - [x] DeepExplorationEngine (orchestrator)
- [x] Adapter in `glue/adapters/rese-dee/src/dee_adapter.py`
- [x] Circuit breaker implementation
- [x] DLQ implementation
- [x] Structured logging (JSON Lines)
- [x] Timeout enforcement
- [x] Configuration from environment
- [x] Probe scripts in `probes/check_dee.sh`
- [x] Unit tests in `tests/test_dee.py`
- [x] Integration tests in `tests/test_integration.py`
- [x] Dockerfile
- [x] Requirements.txt
- [x] README.md

**Testing:**
- Probe script validates all 10 core functionalities
- Unit tests cover schemas, generators, recognizers, MCTS
- Integration tests cover full API, error handling, DLQ

**Next Steps:**
1. Run probe scripts to verify installation
2. Integrate with RESE pipeline phases
3. Performance testing and optimization
4. Add Lean 4 integration (optional, Tier 6)

## References

- SOURCE_RECOVERY_REPORT.md: Bytecode analysis and reimplementation strategy
- CLAUDE.md: Project constitution and immutable laws
- The Recursive Epistemic Solvability Engine (RESE)_ A Technical Manual for Overcoming Intractable Problem Spaces.txt

---

**Signed:** Claude (AI Assistant)
**Date:** 2026-02-04
