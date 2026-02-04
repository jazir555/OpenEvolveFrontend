# Architecture Decision Record: RESE Symbolic Constraint Engine (SCE) Implementation

**Date:** 2026-02-04
**Status:** Accepted
**Component:** RESE Symbolic Constraint Engine (SCE)
**Location:** `glue/lib/rese-sce.ts`, `glue/adapters/rese-sce/`

---

## Context

The RESE (Recursive Epistemic Solvability Engine) exists in bytecode-only format with all `.py` source files missing. After extensive decompilation attempts (uncompyle6, decompyle3), we concluded that complete source code recovery from Python 3.11 bytecode is **NOT feasible**.

According to the **SOURCE_RECOVERY_REPORT.md**, the `rese/core/symbolic_constraint_engine.py` module had:
- **3 classes**: ConstraintType, Constraint, SymbolicConstraintEngine
- **2 functions**
- **Priority**: CRITICAL (foundation for all RESE phases)
- **Complexity**: HIGH

This ADR documents the reimplementation decisions for the SCE.

---

## Decision

### 1. Implementation Language: TypeScript instead of Python

**Rationale:**
- The glue layer is primarily TypeScript/Node.js
- Better integration with existing adapters
- Type safety with TypeScript interfaces
- Easier to maintain consistency with canonical schemas (Zod)

**Trade-offs:**
- ❌ Lose direct Python bytecode compatibility
- ✅ Gain type safety and better tooling
- ✅ Consistent with rest of glue layer

### 2. Architecture: Three-Class Structure

Following bytecode analysis, we implemented three core classes:

#### 2.1 SymbolicConstraintEngine (Main Engine)
**Responsibilities:**
- Constraint management (add, remove, query)
- Contradiction detection orchestration
- Consistency checking
- Tacit assumption mining (Φ₁.₅)
- Full Epistemic Audit (Phase I)

**Key Design Decisions:**
- **In-memory constraint storage**: Using `Map<string, Constraint>` for O(1) lookups
- **Idempotent operations**: All `add*` operations are UPSERTs (Law of Idempotency)
- **UTC timestamps**: All timestamps in ISO-8601 UTC format (Law of UTC)

#### 2.2 ContradictionDetector
**Responsibilities:**
- Pairwise contradiction detection
- Circular dependency detection
- Hard/Soft constraint mismatch detection

**Algorithm Choice:**
- **Naive O(n²) implementation**: Baseline for correctness
- **DITO optimization**: Deferred to future implementation (O(n log n))
- **Rationale**: Start simple, optimize later (SOURCE_RECOVERY_REPORT.md §9.2)

**Trade-offs:**
- ❌ O(n²) complexity for large constraint sets
- ✅ Simpler implementation, easier to verify correctness
- ✅ Can optimize with DITO later without changing API

#### 2.3 ConsistencyChecker
**Responsibilities:**
- Duplicate constraint detection
- Orphaned dependency detection
- Dependency cycle detection (DFS-based)

**Algorithm Choice:**
- **Depth-First Search (DFS)**: Standard cycle detection algorithm
- **Adjacency list representation**: Efficient for sparse graphs

### 3. Adapter Pattern: Anti-Corruption Layer (ACL)

**Following CLAUDE.md §2.2:**

```
[Canonical Request] → [SCEAdapter (ACL)] → [SymbolicConstraintEngine]
                        ↓
                 [Canonical Response]
```

**Responsibilities:**
- Transform canonical requests to SCE internal format
- Execute SCE operations with circuit breaker protection
- Transform SCE results to canonical format
- Handle failures according to CLAUDE.md laws

**Failure Management:**
- **Transient failures**: Exponential backoff retry (3 retries, 1s → 2s → 4s → 10s)
- **Logic failures**: Dead Letter Queue (DLQ) - doesn't block pipeline
- **System failures**: Circuit breaker - stops hammering dead service

### 4. Configuration: Environment Variables Only

**Following CLAUDE.md Law of Configuration Explicitness:**

```typescript
// ❌ BAD: Magic defaults
const timeout = 5000;

// ✅ GOOD: Explicit env var
const timeout = parseInt(process.env.SCE_TIMEOUT_MS || '5000', 10);
if (timeout <= 0) {
    throw new Error('SCE_TIMEOUT_MS must be positive');
}
```

**Required Environment Variables:**
- `SCE_TIMEOUT_MS`: Default operation timeout
- `SCE_MAX_ITERATIONS`: Max iterations for algorithms
- `SCE_MAX_CONSTRAINTS`: Maximum constraint limit
- `SCE_CIRCUIT_BREAKER_THRESHOLD`: Failures before tripping
- `SCE_ENABLE_LEAN4`: Enable Lean 4 integration
- `SCE_ENABLE_TACIT_MINING`: Enable tacit assumption mining

### 5. Logging: Structured JSON Lines

**Following CLAUDE.md §3.3:**

```typescript
// ❌ BAD: Unstructured log
console.log("Error happened");

// ✅ GOOD: Structured log
logger.error('User Sync Failed', error, {
    correlation_id: ctx.id,
    source_service: 'sce-adapter',
    target_service: 'rese-core',
    retry_count: 2,
});
```

**All logs include:**
- `level`: Log level (debug, info, warn, error)
- `msg`: Log message
- `timestamp`: ISO-8601 UTC timestamp
- `correlation_id`: Distributed tracing ID (auto-generated)
- `source_service`: Service name

### 6. Canonical Schema Integration

**Following CLAUDE.md Law of the "Air Gap":**

All SCE results conform to the canonical schema defined in `glue/schemas/rese-canonical.ts`:

```typescript
// EpistemicAuditResult from canonical schema
const result: EpistemicAuditResult = {
    phase: 'phase1_epistemic_audit',
    audit_id: uuidv4(),
    problem_description: '...',
    tacit_assumptions: [...],
    contradictions: [...],
    falsification_results: [...],
    correlation_id: correlationId,
    timestamp: new Date().toISOString(), // UTC
};
```

**Validation:**
- All results validated against Zod schema before returning
- Invalid results trigger errors (sent to DLQ in adapter)

---

## Technical Implementation Details

### 1. Constraint Data Structure

```typescript
export interface Constraint {
    constraint_id: string;
    type: ConstraintType;           // HARD | SOFT
    category: ConstraintCategoryInternal;
    description: string;
    expression?: any;               // Logical expression (Lean 4)
    dependencies: string[];         // IDs of dependent constraints
    formalized_in_lean4?: boolean;
    lean4_theorem?: string;
    created_at: Date;               // UTC timestamp
}
```

**Design Decisions:**
- **UUID v4 for IDs**: Globally unique, no collisions
- **Dependencies as ID array**: Explicit dependency tracking
- **Optional Lean 4 fields**: Formal verification is optional

### 2. Contradiction Detection Algorithm

**Naive O(n²) Pairwise Comparison:**

```typescript
for (let i = 0; i < constraints.length; i++) {
    for (let j = i + 1; j < constraints.length; j++) {
        const contradiction = this.checkPairwiseContradiction(c1, c2);
        if (contradiction) {
            contradictions.push(contradiction);
        }
    }
}
```

**Contradiction Types Detected:**
1. **Direct negation**: "X is true" vs "X is not true"
2. **Circular dependency**: A depends on B, B depends on A
3. **Hard/Soft mismatch**: Different constraint types on same premise

### 3. Tacit Assumption Mining (Φ₁.₅)

**From RESE Manual §3.1.5:**

"Inverse Inference Analysis: The DEE analyzes high-entropy data (e.g., 50% null results) via statistical correlation. This performs inverse inference, inferring the unstated rule set (C_tacit) by correlating patterns of failure with known, unmeasured variables."

**Implementation:**
```typescript
async mineTacitAssumptions(failurePatterns: FailurePattern[], correlationId: string) {
    const assumptions: TacitAssumption[] = [];

    for (const pattern of failurePatterns) {
        // High failure rate (>30%) suggests tacit assumption
        if (pattern.failure_rate > 0.3) {
            assumptions.push({
                id: uuidv4(),
                description: this.inferAssumptionFromPattern(pattern.pattern_description),
                confidence_score: Math.min(pattern.failure_rate, 1.0),
                supporting_evidence_count: pattern.data_points,
                formalized_in_lean4: false,
            });
        }
    }

    return assumptions;
}
```

**Heuristic:**
- Failure rate > 30% → likely tacit assumption
- Confidence score = min(failure_rate, 1.0)
- Supporting evidence = data point count

### 4. Circuit Breaker Pattern

**Following CLAUDE.md §3.3:**

**States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit tripped, requests fail immediately
- **HALF_OPEN**: Testing if service has recovered

**Configuration:**
```typescript
CIRCUIT_BREAKER_THRESHOLD: 5      // Trip after 5 failures
CIRCUIT_BREAKER_TIMEOUT_MS: 60000 // Stay open for 1 minute
```

**Usage:**
```typescript
const result = await this.circuitBreaker.execute(async () => {
    return await this.sce.performEpistemicAudit(...);
});
```

### 5. Dead Letter Queue (DLQ)

**From CLAUDE.md: Logic failures → Dead Letter Queue**

**Purpose:**
- Capture logic failures (bad data, validation errors)
- Don't block the pipeline for invalid data
- Allow reprocessing of failed operations

**Implementation:**
```typescript
class DeadLetterQueue {
    private queue: DLQEntry[] = [];
    private maxSize: number;

    add(operation, payload, error, correlationId, retryCount) {
        // Remove oldest if full
        if (this.queue.length >= this.maxSize) {
            this.queue.shift();
        }

        this.queue.push({
            id: uuidv4(),
            timestamp: new Date(),
            operation,
            payload,
            error,
            correlation_id: correlationId,
            retry_count: retryCount,
        });
    }
}
```

---

## Consequences

### Positive

1. **Type Safety**: TypeScript provides compile-time type checking
2. **CLAUDE.md Compliance**: All laws followed (idempotency, explicit config, circuit breaker, structured logging, UTC)
3. **Canonical Schema**: Anti-Corruption Layer prevents data corruption
4. **Testability**: Comprehensive test coverage
5. **Documentation**: Extensive inline comments and README

### Negative

1. **Performance**: O(n²) contradiction detection (can optimize later with DITO)
2. **Language Mismatch**: TypeScript instead of Python (bytecode compatibility lost)
3. **Complexity**: Three-class architecture adds complexity
4. **Memory**: In-memory constraint storage (no persistence)

### Risks

1. **DITO Not Implemented**: O(n²) algorithm may not scale to large constraint sets
   - **Mitigation**: Document DITO as future optimization, keep API stable

2. **No Lean 4 Integration Yet**: Formal verification not implemented
   - **Mitigation**: Designed with Lean 4 hooks, can add later

3. **In-Memory Storage**: Constraints lost on restart
   - **Mitigation**: Can add persistence layer later without changing API

---

## Alternatives Considered

### Alternative 1: Python Reimplementation

**Approach:** Rewrite SCE in Python to match bytecode

**Pros:**
- ✅ Bytecode compatibility
- ✅ Direct mapping to original implementation

**Cons:**
- ❌ Integration complexity with TypeScript glue layer
- ❌ Less type safety
- ❌ Inconsistent with rest of glue layer

**Decision:** Rejected - TypeScript provides better integration

### Alternative 2: Direct Python Bytecode Execution

**Approach:** Use `__pycache__` bytecode directly via Python import

**Pros:**
- ✅ Original bytecode execution
- ✅ No reimplementation needed

**Cons:**
- ❌ Law of the "Air Gap" violated (direct import from core-projects)
- ❌ Can't modify or extend functionality
- ❌ Python 3.11 bytecode may change

**Decision:** Rejected - Violates CLAUDE.md Law of the "Air Gap"

### Alternative 3: Immediate DITO Implementation

**Approach:** Implement DITO (O(n log n)) from the start

**Pros:**
- ✅ Optimal performance
- ✅ Scales to large constraint sets

**Cons:**
- ❌ Higher complexity
- ❌ Harder to verify correctness
- ❌ Takes longer to implement

**Decision:** Rejected - Start simple, optimize later (SOURCE_RECOVERY_REPORT.md §9.2)

---

## Future Work

### Phase 1: Complete SCE Foundation (Current)
- ✅ Basic SCE implementation
- ✅ Adapter with ACL
- ✅ Circuit breaker and DLQ
- ✅ Comprehensive tests

### Phase 2: DITO Optimization (Future)
- ⏳ Implement R-tree spatial indexing
- ⏳ Implement LSH (Locality-Sensitive Hashing)
- ⏳ Implement Hierarchical Abstraction Graph (HAG)
- ⏳ Achieve O(n log n) complexity

### Phase 3: Lean 4 Integration (Future)
- ⏳ Lean 4 proposition formalization
- ⏳ ATP (Automated Theorem Proving) integration
- ⏳ Formal verification of constraints

### Phase 4: Persistence Layer (Future)
- ⏳ Database-backed constraint storage
- ⏳ Constraint history/audit log
- ⏳ Distributed constraint management

---

## References

1. **SOURCE_RECOVERY_REPORT.md**: Complete bytecode analysis and reimplementation plan
2. **RESE Technical Manual**: "The Recursive Epistemic Solvability Engine (RESE): A Technical Manual for Overcoming Intractable Problem Spaces"
3. **CLAUDE.md**: Federation Constitution
4. **rese-canonical.ts**: Canonical schema definitions
5. **rese/core/symbolic_constraint_engine.py** (bytecode): Original implementation structure

---

**Authors:** Claude (AI Assistant)
**Reviewers:** OpenEvolve Frontend Team
**Status:** Accepted
**Implementation Date:** 2026-02-04
