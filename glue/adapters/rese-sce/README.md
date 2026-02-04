# RESE Symbolic Constraint Engine (SCE) Adapter

## Overview

This adapter implements the RESE (Recursive Epistemic Solvability Engine) Symbolic Constraint Engine, which serves as the foundation for all RESE phases. It enforces logical consistency using formal logic and contradiction detection.

**From RESE Technical Manual §2.1:**
> The Symbolic Constraint Engine (SCE) enforces internal consistency and adherence to explicit epistemic norms through formal logic and contradiction detection (Φ₃).

## Architecture

### Core Components

1. **SymbolicConstraintEngine** (`glue/lib/rese-sce.ts`)
   - Main engine class for RESE Phase I: Epistemic Audit
   - Constraint management (add, remove, query)
   - Contradiction detection
   - Consistency checking
   - Tacit assumption mining (Φ₁.₅)

2. **ContradictionDetector**
   - Identifies logical contradictions in constraint sets
   - Naive O(n²) pairwise comparison (baseline)
   - DITO optimization support (O(n log n)) - to be implemented

3. **ConsistencyChecker**
   - Verifies internal consistency of constraint sets
   - Dependency cycle detection
   - Duplicate constraint detection

4. **SCEAdapter** (`glue/adapters/rese-sce/src/sce-adapter.ts`)
   - Anti-Corruption Layer implementation
   - Circuit breaker protection
   - Dead Letter Queue for logic failures
   - Exponential backoff retry for transient failures

## Features

### CLAUDE.md Compliance

✅ **Law of Idempotency**: All operations safe to run 100x
✅ **Law of Configuration Explicitness**: All config via env vars
✅ **Circuit Breaker Pattern**: Detect system failures
✅ **Structured Logging**: JSON with correlation_id
✅ **Timeout Enforcement**: Every operation has timeout

### Phase I: Epistemic Audit

The SCE implements RESE Phase I subroutines:

- **Φ₁**: Initial Hypothesis Cluster Definition (Constraint Hardening)
- **Φ₁.₅**: Tacit Assumption Mining (Inverse Inference Analysis)
- **Φ₃**: Formal Logic Audit and Contradiction Detection

## Configuration

### Environment Variables

```bash
# SCE Core Configuration
SCE_TIMEOUT_MS=5000                              # Default operation timeout
SCE_CONSTRAINT_TIMEOUT_MS=3000                   # Constraint operation timeout
SCE_CONTRADICTION_TIMEOUT_MS=10000               # Contradiction detection timeout
SCE_MAX_ITERATIONS=1000                          # Max iterations for algorithms
SCE_MAX_CONSTRAINTS=10000                        # Maximum constraint limit
SCE_MAX_CONTRADICTION_SET_SIZE=100               # Max contradiction set size
SCE_CIRCUIT_BREAKER_THRESHOLD=5                  # Failures before tripping
SCE_CIRCUIT_BREAKER_TIMEOUT_MS=60000             # Circuit breaker timeout
SCE_ENABLE_LEAN4=true                            # Enable Lean 4 integration
SCE_ENABLE_TACIT_MINING=true                     # Enable tacit assumption mining

# SCE Adapter Configuration
RESE_SCE_URL=http://localhost:8000               # SCE service URL
RESE_SCE_TIMEOUT_MS=30000                        # Adapter timeout
SCE_ADAPTER_MAX_RETRIES=3                        # Max retry attempts
SCE_ADAPTER_INITIAL_DELAY_MS=1000                # Initial retry delay
SCE_ADAPTER_MAX_DELAY_MS=10000                   # Max retry delay
SCE_ADAPTER_CB_THRESHOLD=5                       # Circuit breaker threshold
SCE_ADAPTER_CB_TIMEOUT_MS=60000                  # Circuit breaker timeout
SCE_DLQ_ENABLED=true                             # Enable Dead Letter Queue
SCE_DLQ_MAX_SIZE=1000                            # DLQ max size
```

## Usage

### Basic Constraint Management

```typescript
import SCEAdapter from './glue/adapters/rese-sce/src/sce-adapter';

const adapter = new SCEAdapter();

// Add a constraint
await adapter.addConstraint({
    type: 'hard',
    category: 'hard_parameter_inequality',
    description: 'Energy cannot be created or destroyed',
    dependencies: [],
}, 'correlation-id-123');

// Detect contradictions
const result = await adapter.detectContradictions('correlation-id-123');
console.log(`Found ${result.contradictions.length} contradictions`);
```

### Epistemic Audit (Phase I)

```typescript
import SCEAdapter from './glue/adapters/rese-sce/src/sce-adapter';

const adapter = new SCEAdapter();

// Perform full Phase I audit
const auditResult = await adapter.performEpistemicAudit({
    problem_description: 'LENR thermal coefficient inconsistency',
    failure_patterns: [
        {
            pattern_description: 'Lattice defects non-uniform distribution',
            failure_rate: 0.5,
            data_points: 100,
        },
    ],
    correlation_id: 'audit-456',
});

console.log(`Audit ID: ${auditResult.audit_id}`);
console.log(`Tacit assumptions: ${auditResult.tacit_assumptions.length}`);
console.log(`Contradictions: ${auditResult.contradictions.length}`);
```

## API Reference

### SCEAdapter

#### Methods

- `performEpistemicAudit(request)` - Perform Phase I Epistemic Audit
- `addConstraint(constraint, correlationId)` - Add constraint to SCE
- `removeConstraint(constraintId, correlationId)` - Remove constraint from SCE
- `getConstraint(constraintId)` - Get constraint by ID
- `getAllConstraints()` - Get all constraints
- `detectContradictions(correlationId)` - Detect contradictions
- `getStats()` - Get adapter statistics
- `getDLQEntries()` - Get Dead Letter Queue entries
- `clearDLQ()` - Clear Dead Letter Queue
- `resetCircuitBreakers()` - Reset circuit breakers
- `healthCheck()` - Health check

### Canonical Schema

All results conform to the canonical schema defined in `glue/schemas/rese-canonical.ts`:

- `EpistemicAuditResult` - Result from Phase I audit
- `TacitAssumption` - Mined tacit assumption
- `ContradictionDetection` - Detected contradiction

## Failure Management

### Transient Failures

Network blips, temporary timeouts → **Exponential Backoff Retry**

```typescript
// Automatic retry with exponential backoff
const result = await adapter.performEpistemicAudit(request);
// Retries up to 3 times with delays: 1s, 2s, 4s
```

### Logic Failures

Bad data, validation errors → **Dead Letter Queue**

```typescript
// Invalid data sent to DLQ, doesn't block pipeline
const dlqEntries = adapter.getDLQEntries();
// Review and reprocess DLQ entries separately
```

### System Failures

Service down, persistent failures → **Circuit Breaker**

```typescript
// Circuit opens after 5 failures
const health = await adapter.healthCheck();
if (health.circuit_state === 'open') {
    // Service is down, use fallback or cached data
}
```

## Testing

### Run Probe Script

```bash
cd glue/adapters/rese-sce
./probes/check-sce.sh
```

This verifies:
- TypeScript compilation
- File structure
- Environment variable configuration
- CLAUDE.md compliance
- Canonical schema integration
- Key classes and methods

## Development

### Directory Structure

```
glue/
├── adapters/
│   └── rese-sce/
│       ├── src/
│       │   └── sce-adapter.ts        # Main adapter
│       ├── probes/
│       │   └── check-sce.sh          # Probe script
│       ├── Dockerfile                # Container definition
│       ├── README.md                 # This file
│       └── package.json              # Dependencies
└── lib/
    └── rese-sce.ts                   # Core SCE implementation
```

### Adding New Features

1. Update `glue/lib/rese-sce.ts` with core logic
2. Update `glue/adapters/rese-sce/src/sce-adapter.ts` with adapter layer
3. Update `glue/schemas/rese-canonical.ts` if new result types
4. Add probe verification in `probes/check-sce.sh`
5. Update README with new configuration and usage

## Integration with RESE Pipeline

The SCE is used by:

1. **Phase I** (Epistemic Audit) - Main functionality
2. **Phase II** (Isomorphic Mapping) - Constraint inversion
3. **Phase III** (MCTS Search) - Constraint enforcement during search
4. **Phase IV** (Architecture Assembly) - Final consistency validation

## Lean 4 Integration

When `SCE_ENABLE_LEAN4=true`, the SCE can:

- Formalize constraints in Lean 4 propositions
- Verify contradictions using Lean 4 ATP (Automated Theorem Proving)
- Generate formal proofs for constraint satisfaction

```typescript
const constraint = {
    type: 'hard',
    category: 'hard_parameter_inequality',
    description: 'Energy conservation',
    formalized_in_lean4: true,
    lean4_theorem: 'theorem energy_conservation : ∀ (E : Energy), ...',
};
```

## Troubleshooting

### Circuit Breaker Tripped

```bash
# Check circuit state
curl http://localhost:3000/health

# Reset circuit breaker
curl -X POST http://localhost:3000/reset
```

### Dead Letter Queue Full

```bash
# View DLQ entries
curl http://localhost:3000/dlq

# Clear DLQ
curl -X DELETE http://localhost:3000/dlq
```

### Timeout Issues

```bash
# Increase timeouts
export SCE_TIMEOUT_MS=10000
export RESE_SCE_TIMEOUT_MS=60000
```

## References

- RESE Technical Manual: `rese/The Recursive Epistemic Solvability Engine (RESE)_ A Technical Manual for Overcoming Intractable Problem Spaces.txt`
- Canonical Schema: `glue/schemas/rese-canonical.ts`
- CLAUDE.md: Federation Constitution
- Source Recovery Report: `glue/adapters/rese-integration/SOURCE_RECOVERY_REPORT.md`

## License

Part of the OpenEvolve Frontend project.
