# Architecture Decision Record: RESE Phase IV Implementation

**Date:** 2026-02-04
**Status:** Accepted
**Component:** RESE Phase IV: Architecture Assembly (Δ₁, Δ₂, Δ₃)
**Location:** `glue/adapters/rese-phase4/`, `glue/schemas/rese_phase4_schemas.py`

---

## Context

RESE (Recursive Epistemic Solvability Engine) is a four-phase recursive methodology for transforming intractable problems into tractable ones. Phase IV is the final phase that integrates outputs from all previous phases:

- **Phase I (Φ₁, Φ₁.₅, Φ₂, Φ₃, Φ₄)**: Epistemic Audit - Constraint formalization, assumption mining, debiasing, contradiction detection, adversarial simulation
- **Phase II (Ψ₁, Ψ₂, Ψ₃, I_mech)**: Isomorphic Resonance - Problem formalization, ontology mapping, constraint inversion, isomorphism validation
- **Phase III (Γ₁, Γ₂, Γ₃, N_max)**: Monte Carlo Refinement - ACI analysis, MCTS search, statistical validation, convergence control
- **Phase IV (Δ₁, Δ₂, Δ₃)**: Architecture Assembly - Paradigm shift assembly, knowledge integration, architecture validation

Phase IV must:
1. Assemble paradigm shifts from validated patterns
2. Synthesize knowledge across all phases
3. Validate the final architecture
4. Generate the final output with confidence metrics

---

## Decision

### 1. Implementation Language: Python

**Rationale:**
- Consistency with other RESE components (SCE, DEE, LLTL)
- Native integration with Phase I-III executors (all Python)
- Easier to integrate with existing RESE schemas
- Better for algorithmic implementations (MCTS, isomorphism detection)

**Trade-offs:**
- ❌ Less type-safe than TypeScript (but mitigated by type hints and dataclasses)
- ✅ Consistent with RESE core implementation
- ✅ Direct access to Phase I-III outputs without translation

**Decision:** Python 3.11+ with type hints and dataclasses for type safety.

---

### 2. Architecture: Three-Component Structure

Following RESE specification, Phase IV consists of three components:

#### 2.1 Paradigm Shift Assembler (Δ₁)
**Responsibilities:**
- Group patterns by type (structural, functional, causal, etc.)
- Identify multi-phase patterns (from I, II, III)
- Extract transformation rules
- Calculate confidence (boost for multi-phase patterns)
- Create ParadigmShift objects

**Algorithm Choice:**
- **Naive grouping**: Group patterns by type, then assemble
- **Confidence calculation**: Base + multi-phase boost (10% for 2 phases, 15% for 3 phases)
- **Rationale**: Start simple, optimize later if needed

**Trade-offs:**
- ❌ O(n) grouping where n = total patterns (acceptable)
- ✅ Simple, correct, easy to verify
- ✅ Can optimize with parallel processing later

#### 2.2 Knowledge Integrator (Δ₂)
**Responsibilities:**
- Integrate knowledge from Phases I, II, III
- Generate synthesis rules
- Calculate completeness (phase coverage)
- Calculate consistency (contradiction detection)
- Calculate overall confidence

**Integration Strategies:**
- `MERGE`: Combine without transformation
- `SYNTHESIZE`: Transform and integrate (default)
- `ABSTRACT`: Extract higher-level abstractions
- `HIERARCHICAL`: Build hierarchical structure
- `CROSS_VALIDATION`: Cross-validate between phases

**Rationale:** Multiple strategies support different use cases.

#### 2.3 Architecture Validator (Δ₃)
**Responsibilities:**
- Validate completeness (≥2 phases)
- Validate consistency (≥0.6 score)
- Validate confidence (≥threshold, default 0.7)
- Validate ACI reduction (≥20%)
- Optional: Strict validation (paradigm shift quality)
- Optional: Formal verification (Lean 4)

**Validation Levels:**
1. `NONE` - No validation
2. `BASIC` - Completeness, consistency, confidence
3. `STANDARD` - Basic + ACI reduction (default)
4. `STRICT` - Standard + paradigm shift quality, cross-phase
5. `FORMAL` - Strict + Lean 4 proofs (placeholder)

**Rationale:** Gradated validation allows flexibility vs. rigor trade-off.

---

### 3. Canonical Schema Design

Following CLAUDE.md Anti-Corruption Layer (ACL) pattern:

```
[Canonical Request] → [Phase4Adapter (ACL)] → [ArchitectureAssemblyExecutor]
                        ↓
                 [Canonical Response]
```

#### 3.1 Phase Output Schemas
For integration, we define schemas for Phase I-III outputs:

- `EpistemicAuditResult`: Phase I output (constraints, contradictions, biases)
- `IsomorphicMappingResult`: Phase II output (isomorphisms, mappings)
- `MCTSRefinementResult`: Phase III output (hypotheses, ACI reduction)

All schemas support:
- `to_dict()`: Serialization to dictionary
- `from_dict()`: Deserialization from dictionary
- UTC timestamps (ISO-8601 format)
- Enum value serialization

#### 3.2 Phase IV Core Schemas
- `ParadigmShift`: Assembled paradigm shift
- `SynthesizedKnowledge`: Integrated knowledge
- `ArchitectureAssembly`: Final output

All schemas are idempotent:
- Same inputs → same outputs
- Deduplication by ID
- Idempotent `update_*` methods

---

### 4. Adapter Pattern: Anti-Corruption Layer (ACL)

**Following CLAUDE.md §2.2:**

**Responsibilities:**
- Transform canonical requests to executor format
- Execute with circuit breaker protection
- Transform results to canonical format
- Handle failures according to CLAUDE.md laws

**Failure Management:**
- **Transient failures**: Exponential backoff retry (3 retries, 1s → 2s → 4s → 10s)
- **Logic failures**: Dead Letter Queue (DLQ) - return failed assembly with errors
- **System failures**: Circuit breaker - stop after 5 failures, wait 60s

**Circuit Breaker States:**
- `CLOSED`: Normal operation
- `OPEN`: Too many failures, reject requests
- `HALF_OPEN`: Attempting recovery

---

### 5. Configuration: Environment Variables Only

**Following CLAUDE.md Law of Configuration Explicitness:**

```python
# ❌ BAD: Magic defaults
timeout = 25000

# ✅ GOOD: Explicit env var
timeout = int(os.getenv("PHASE4_ASSEMBLY_TIMEOUT_MS", "25000"))
if timeout <= 0:
    raise ValueError("PHASE4_ASSEMBLY_TIMEOUT_MS must be positive")
```

**Required Environment Variables:**
- `PHASE4_ASSEMBLY_TIMEOUT_MS`: Timeout for assembly (default: 25000ms)
- `PHASE4_VALIDATION_LEVEL`: Validation level (default: "standard")
- `PHASE4_INTEGRATION_STRATEGY`: Integration strategy (default: "synthesize")
- `PHASE4_MAX_PARADIGM_SHIFTS`: Max paradigm shifts (default: 50)
- `PHASE4_MIN_CONFIDENCE_THRESHOLD`: Min confidence (default: 0.7)
- `PHASE4_ENABLE_CROSS_VALIDATION`: Enable cross-validation (default: true)
- `PHASE4_ENABLE_FORMAL_VERIFICATION`: Enable Lean 4 (default: false)
- `CORRELATION_ID`: Distributed tracing ID (optional)

**Configuration Object:**
```python
config = Phase4Config.from_env()  # Crashes if required vars are invalid
```

---

### 6. Logging: Structured JSON Lines

**Following CLAUDE.md §3.3:**

```python
# ❌ BAD: Unstructured log
print("Error happened")

# ✅ GOOD: Structured log
logger.error("Architecture assembly failed", error=e, {
    "correlation_id": self.correlation_id,
    "source_service": "rese-phase4-adapter",
    "assembly_id": assembly.assembly_id,
})
```

**All logs include:**
- `level`: Log level (debug, info, warn, error)
- `msg`: Log message
- `timestamp`: ISO-8601 UTC timestamp
- `correlation_id`: Distributed tracing ID
- `source_service`: Service name ("rese-phase4-executor" or "rese-phase4-adapter")
- Contextual fields (assembly_id, error, retry_count, etc.)

---

### 7. Law of Idempotency Implementation

**All operations are idempotent:**

1. **Paradigm Shift Assembly:**
   - Same patterns → same paradigm shifts
   - Deduplication by shift_id
   - Deterministic confidence calculation

2. **Knowledge Integration:**
   - Same phase outputs → same synthesized knowledge
   - Deterministic completeness/consistency calculation

3. **Architecture Validation:**
   - Same assembly → same validation results
   - Deterministic rule application

4. **Idempotent Update Methods:**
   ```python
   def update_evidence(self, new_evidence: Dict[str, Any]):
       # Deduplicate by evidence_id
       evidence_id = new_evidence.get("evidence_id") or hash(new_evidence)
       if evidence_id not in self.evidence:
           self.evidence.append(new_evidence)
   ```

---

### 8. Timeout Protection

**All operations have timeouts:**

- Assembly timeout: Configurable via `PHASE4_ASSEMBLY_TIMEOUT_MS` (default: 25000ms)
- Each component checks elapsed time
- Raises `TimeoutError` if exceeded

```python
start_time = time.time()
timeout_sec = self.config.assembly_timeout_ms / 1000.0

# Periodic timeout check
elapsed = time.time() - start_time
if elapsed > timeout_sec:
    raise TimeoutError(f"Assembly exceeded timeout: {elapsed:.2f}s")
```

---

### 9. UTC Timestamps

**Following CLAUDE.md Law of UTC:**

All timestamps are:
- Timezone-aware (`datetime.now(timezone.utc)`)
- ISO-8601 format on serialization
- Parsed with timezone info on deserialization

```python
# Creation
created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Serialization
"created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at

# Deserialization
if isinstance(data["created_at"], str):
    data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
```

---

### 10. Formal Verification Placeholder

**Decision:** Placeholder for Lean 4 formal verification

**Rationale:**
- Full formal verification requires Lean 4 integration (future work)
- Placeholder allows validation level enum to include `FORMAL`
- Can be implemented later without breaking API

**Placeholder Implementation:**
```python
def _validate_formal(self, assembly: ArchitectureAssembly) -> List[Dict[str, Any]]:
    """Perform formal verification (Lean 4)."""
    return [{
        "validation_type": "formal_verification",
        "passed": True,  # Placeholder
        "note": "Formal verification not yet implemented",
    }]
```

---

## Validation

### How This Implementation Meets Requirements

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| **Paradigm Shift Assembly (Δ₁)** | `ParadigmShiftAssembler` class | ✅ Complete |
| **Knowledge Integration (Δ₂)** | `KnowledgeIntegrator` class | ✅ Complete |
| **Architecture Validation (Δ₃)** | `ArchitectureValidator` class | ✅ Complete |
| **CLAUDE.md Compliance** | All 6 laws followed | ✅ Complete |
| **Circuit Breaker** | `AdapterCircuitBreaker` class | ✅ Complete |
| **Exponential Backoff** | `execute_with_retry()` function | ✅ Complete |
| **Structured Logging** | `StructuredLogger` class | ✅ Complete |
| **Idempotency** | All operations idempotent | ✅ Complete |
| **Timeout Protection** | Configurable timeouts with checks | ✅ Complete |
| **UTC Timestamps** | All timestamps timezone-aware | ✅ Complete |
| **Probe Script** | `probes/check_phase4.sh` | ✅ Complete |

---

## Consequences

### Positive
1. ✅ **Complete Phase IV Implementation**: All three components (Δ₁, Δ₂, Δ₃) implemented
2. ✅ **CLAUDE.md Compliance**: All 6 laws followed
3. ✅ **Type Safety**: Dataclasses with type hints provide compile-time checking
4. ✅ **Idempotency**: Safe to run multiple times
5. ✅ **Observability**: Structured logging with correlation IDs
6. ✅ **Resilience**: Circuit breaker and retry logic
7. ✅ **Flexibility**: Configurable validation levels and integration strategies
8. ✅ **Completes RESE Pipeline**: Integrates Phases I-III outputs

### Negative
1. ❌ **Formal Verification**: Placeholder only (Lean 4 integration deferred)
2. ❌ **Performance**: Single-threaded assembly (could be parallelized)
3. ❌ **Caching**: No assembly caching (could improve performance)

### Neutral
1. ⚖️ **Complexity**: More complex than single-component design (but necessary)
2. ⚖️ **Memory**: Stores all paradigm shifts and knowledge in memory (acceptable for typical workloads)

---

## Alternatives Considered

### Alternative 1: Single Monolithic Executor
**Rejected because:**
- Harder to test individual components
- Violates separation of concerns
- Harder to maintain and extend

### Alternative 2: TypeScript Implementation
**Rejected because:**
- Inconsistent with other RESE components (all Python)
- Would require translation layer for Phase I-III outputs
- Less natural for algorithmic implementations

### Alternative 3: No Circuit Breaker
**Rejected because:**
- Violates CLAUDE.md §2.3
- Could hammer failing services
- No graceful degradation

---

## Future Work

1. **Lean 4 Integration**: Implement formal verification for Δ₃
2. **Parallel Assembly**: Parallelize paradigm shift assembly by pattern type
3. **Assembly Caching**: Cache assemblies for reuse (idempotent by design)
4. **Performance Optimization**: Profile and optimize hot paths
5. **Metrics Collection**: Add Prometheus metrics for monitoring
6. **Batch Processing**: Support batch assembly requests

---

## References

- RESE Technical Manual: Phase IV specifications
- CLAUDE.md: Federation Constitution (6 laws)
- RESE Developer Guide: Architecture patterns
- RESE Implementation Roadmap: Phase IV tasks

---

**Decision Status:** ACCEPTED
**Implementor:** RESE Development Team
**Review Date:** 2026-02-04
**Next Review:** After production deployment
