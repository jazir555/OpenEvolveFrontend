# RESE Phase I: Epistemic Audit Adapter

**Following CLAUDE.md Principles:**
- Law of the "Air Gap": No imports from core-projects
- Law of Runtime Truth: Verify via probes, not documentation
- Law of Idempotency: Safe to run 100x
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker Pattern: Detect system failures
- Structured Logging: JSON with correlation_id
- Timeout Enforcement: All operations have timeouts

## Overview

Phase I of the Recursive Epistemic Solvability Engine (RESE) performs an Epistemic Audit and Falsification using the Red Team Protocol.

**Technical Manual Reference:**
- Section 3.0: Phase I - Epistemic Audit and Falsification
- Section 3.1: Initial Hypothesis Cluster Definition (Φ₁)
- Section 3.1.5: Tacit Assumption Mining (Φ₁.₅)
- Section 3.3: Formal Logic Audit and Contradiction Detection (Φ₃)

## Architecture

### Components

1. **EpistemicAuditExecutor** - Main orchestrator for Phase I
   - Φ₁: Constraint Hardening
   - Φ₁.₅: Tacit Assumption Mining
   - Φ₃: Contradiction Detection (via SCE)
   - Φ₄: Red Team Protocol

2. **ConstraintHardener** - Strengthens constraints from patterns
3. **AssumptionMiner** - Extracts tacit assumptions from failure patterns
4. **RedTeamProtocator** - Adversarial testing of assumptions
5. **SCEAdapter** - Integration with Symbolic Constraint Engine

### Integration Points

- **SCE Adapter**: Connects to `glue/lib/rese-sce.ts` for contradiction detection
- **Canonical Schema**: Uses `glue/schemas/rese-canonical.ts` for data format
- **Probe Scripts**: Runtime verification via `probes/check_phase1.sh`

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ (for SCE integration)
- RESE probe scripts validated

### Local Development

```bash
# Set environment variables
export PHASE1_TIMEOUT_MS=15000
export PHASE1_MAX_ASSUMPTIONS=100
export PHASE1_CIRCUIT_BREAKER_THRESHOLD=5

# Run probe to verify installation
bash glue/adapters/rese-phase1/probes/check_phase1.sh
```

### Docker Deployment

```bash
# Build image
docker build -t rese-phase1:latest -f glue/adapters/rese-phase1/Dockerfile .

# Run container
docker run -d \
  -e PHASE1_TIMEOUT_MS=20000 \
  -e PHASE1_MAX_ASSUMPTIONS=200 \
  -v $(pwd)/data:/app/data \
  --name rese-phase1 \
  rese-phase1:latest
```

## Usage

### Command Line Interface

```bash
# Perform audit
python3 glue/adapters/rese-phase1/src/phase1_executor.py \
  --problem "LENR thermal coefficient inconsistency" \
  --patterns '[{
    "pattern_description": "Lattice defects cause irregular heat",
    "failure_rate": 0.5,
    "data_points": 100
  }]'

# Health check
python3 glue/adapters/rese-phase1/src/phase1_adapter.py health
```

### Python API

```python
from phase1_executor import EpistemicAuditExecutor, Phase1Config

# Load configuration from environment
config = Phase1Config.from_env()

# Create executor
executor = EpistemicAuditExecutor(config=config)

# Perform audit
result = executor.perform_audit(
    problem_description="Problem description here",
    failure_patterns=[
        {
            "pattern_description": "Pattern causing failure",
            "failure_rate": 0.6,
            "data_points": 50,
        }
    ],
    correlation_id="unique-trace-id",
)

# Access canonical result
print(f"Audit ID: {result.audit_id}")
print(f"Assumptions found: {len(result.tacit_assumptions)}")
print(f"Contradictions: {len(result.contradictions)}")
print(f"Hypotheses falsified: {result.metrics['hypotheses_falsified']}")
```

## Configuration

### Environment Variables

**Timeout Settings:**
- `PHASE1_TIMEOUT_MS` - Overall timeout (default: 15000)
- `PHASE1_CONSTRAINT_TIMEOUT_MS` - Constraint hardening timeout (default: 5000)
- `PHASE1_ASSUMPTION_TIMEOUT_MS` - Assumption mining timeout (default: 5000)
- `PHASE1_CONTRADICTION_TIMEOUT_MS` - Contradiction detection timeout (default: 10000)
- `PHASE1_FALSIFICATION_TIMEOUT_MS` - Red team timeout (default: 5000)

**Iteration Limits:**
- `PHASE1_MAX_ASSUMPTIONS` - Maximum assumptions to mine (default: 100)
- `PHASE1_MAX_CONSTRAINTS` - Maximum constraints (default: 1000)
- `PHASE1_MAX_CONTRADICTIONS` - Maximum contradictions (default: 100)
- `PHASE1_MAX_FALSIFICATION_ATTEMPTS` - Max red team attempts (default: 50)

**Circuit Breaker:**
- `PHASE1_CIRCUIT_BREAKER_THRESHOLD` - Failures before opening (default: 5)
- `PHASE1_CIRCUIT_BREAKER_TIMEOUT_MS` - Time to stay open (default: 60000)

**Confidence Thresholds:**
- `PHASE1_MIN_ASSUMPTION_CONFIDENCE` - Minimum confidence (default: 0.3)
- `PHASE1_MIN_ROBUSTNESS_SCORE` - Minimum robustness (default: 0.5)

**Feature Flags:**
- `PHASE1_ENABLE_TACIT_MINING` - Enable assumption mining (default: true)
- `PHASE1_ENABLE_LEAN4` - Enable Lean 4 integration (default: false)
- `PHASE1_ENABLE_RED_TEAM` - Enable red team protocol (default: true)

## Canonical Schema

Phase I outputs follow the canonical schema from `glue/schemas/rese-canonical.ts`:

```typescript
{
  phase: "phase1_epistemic_audit",
  audit_id: string (UUID),
  problem_description: string,
  tacit_assumptions: TacitAssumption[],
  contradictions: ContradictionDetection[],
  falsification_results: FalsificationResult[],
  hardened_constraints: Constraint[],
  metrics: {
    total_assumptions_analyzed: number,
    confirmed_contradictions: number,
    hypotheses_falsified: number,
    reduction_in_failure_rate?: number,
  },
  metadata: {
    execution_time_ms: number,
    lean4_version?: string,
    epoch_number: number,
  },
  correlation_id: string (UUID),
  timestamp: string (ISO-8601 UTC),
}
```

## Failure Management

### Circuit Breaker

The circuit breaker prevents cascading failures:

- **CLOSED**: Normal operation
- **OPEN**: Too many failures, rejecting requests
- **HALF_OPEN**: Testing if service recovered

### Dead Letter Queue

Failed audits are sent to DLQ for retry:

```python
# Check DLQ size
stats = executor.get_stats()
print(f"DLQ size: {stats['dlq_size']}")

# Peek at DLQ items
items = executor.dlq.peek()
```

### Error Handling

- **Transient Failure** → Exponential backoff retry
- **Logic Failure** → Dead Letter Queue (bad assumptions)
- **System Failure** → Circuit breaker opens

## Testing

### Probe Scripts

```bash
# Run all Phase I checks
bash glue/adapters/rese-phase1/probes/check_phase1.sh

# Expected output: JSON with all checks PASS
{
  "probe_name": "check_phase1",
  "overall_status": "PASS",
  "checks": {
    "directory_exists": { "status": "PASS" },
    "executor_module_exists": { "status": "PASS" },
    ...
  }
}
```

### Unit Tests

```bash
# Run tests (to be implemented)
python3 -m pytest glue/adapters/rese-phase1/tests/
```

## Monitoring

### Structured Logging

All logs are JSON Lines format:

```json
{
  "level": "info",
  "component": "EpistemicAuditExecutor",
  "timestamp": "2025-02-04T12:34:56.789Z",
  "message": "Starting Phase I: Epistemic Audit",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_description": "LENR thermal coefficient",
  "failure_patterns_count": 1
}
```

### Health Checks

```bash
# Via adapter
python3 glue/adapters/rese-phase1/src/phase1_adapter.py health

# Via Docker
docker exec rese-phase1 python3 /app/src/phase1_adapter.py health

# HTTP endpoint (if exposed)
curl http://localhost:8080/health
```

## Troubleshooting

### Circuit Breaker Open

**Symptom:** RuntimeError: "Circuit breaker is OPEN"

**Solution:**
1. Check logs for root cause
2. Wait for timeout (default: 60s)
3. Fix underlying issue
4. Reset circuit breaker

### High Failure Rate

**Symptom:** Many assumptions falsified

**Solution:**
1. Check `MIN_ASSUMPTION_CONFIDENCE` threshold
2. Review failure pattern quality
3. Validate problem description clarity

### Timeout Errors

**Symptom:** TimeoutError during audit

**Solution:**
1. Increase `PHASE1_TIMEOUT_MS`
2. Reduce `MAX_ASSUMPTIONS` or `MAX_CONSTRAINTS`
3. Check system resources

## References

- **RESE Technical Manual**: `rese/docs/RESE_TECHNICAL_MANUAL.md`
- **SCE Implementation**: `glue/lib/rese-sce.ts`
- **Canonical Schema**: `glue/schemas/rese-canonical.ts`
- **Probe Scripts**: `glue/adapters/rese-integration/probes/`

## License

Part of the RESE integration following CLAUDE.md principles.
