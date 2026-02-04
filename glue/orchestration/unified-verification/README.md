# Unified Verification Orchestrator

**Glue Layer Component** - Cross-validation orchestration between Z3 SMT Solver and LeanAide Theorem Prover

## Overview

The Unified Verification Orchestrator provides intelligent coordination between multiple formal verification systems (Z3 and LeanAide), enabling cross-validation, confidence aggregation, and adaptive strategy selection.

### Mission

Following the Federation Constitution, this orchestrator:
- **Air Gaps**: No direct imports from core-projects
- **Runtime Truth**: All capabilities verified via probe scripts before use
- **Idempotency**: Safe to retry verification requests
- **Configuration Explicitness**: All endpoints via environment variables
- **UTC**: All timestamps in UTC ISO-8601 format

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Unified Verification Orchestrator                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Strategy        │    │     Confidence Aggregator        │ │
│  │ Selector        │───▶│  - Normalizes scores             │ │
│  │ - Problem type  │    │  - Dynamic weights               │ │
│  │   analysis      │    │  - Evidence generation           │ │
│  │ - System choice │    └─────────────────────────────────┘ │
│  └─────────────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Cross Validator                           │   │
│  │  - Parallel execution                                │   │
│  │  - Sequential fallback                               │   │
│  │  - Result comparison                                 │   │
│  │  - Conflict detection                                │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                  │
│           ├──────────────┬─────────────────┐               │
│           ▼              ▼                 ▼               │
│    ┌──────────┐   ┌──────────────┐   ┌──────────┐        │
│    │   Z3     │   │  LeanAide    │   │ Learning │        │
│    │  SMT     │   │  Theorem     │   │  Loop    │        │
│    │  Solver  │   │  Prover      │   │          │        │
│    └──────────┘   └──────────────┘   └──────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
glue/orchestration/unified-verification/
├── src/
│   ├── canonical.ts              # Canonical data models (Zod schemas)
│   ├── orchestrator.ts           # Main orchestrator
│   ├── strategy-selector.ts      # Strategy selection logic
│   ├── cross-validator.ts        # Cross-validation execution
│   ├── confidence-aggregator.ts  # Confidence score aggregation
│   └── index.ts                  # Public API exports
├── probes/
│   ├── check_z3.sh              # Z3 API verification
│   ├── check_leanaide.sh        # LeanAide API verification
│   └── check_cross_validation.sh # Integration verification
├── tests/
│   ├── contract.test.ts         # API contract tests
│   └── jest.config.js           # Jest configuration
├── package.json
├── tsconfig.json
└── README.md
```

## Installation

```bash
cd glue/orchestration/unified-verification
npm install
```

## Configuration

Required environment variables:

```bash
# Z3 SMT Solver
export Z3_URL="http://localhost:8080"
export Z3_TIMEOUT="30000"
export Z3_HEALTH_CHECK="/health"
export Z3_VERIFY_PATH="/verify"

# LeanAide Theorem Prover
export LEANAIDE_URL="http://localhost:8081"
export LEANAIDE_TIMEOUT="45000"
export LEANAIDE_HEALTH_CHECK="/health"
export LEANAIDE_VERIFY_PATH="/verify"

# Optional
export DEBUG="true"  # Enable debug logging
```

## Probes: Law of Runtime Truth

Before using this orchestrator, verify the systems are accessible:

```bash
# Run all probes
npm run probes

# Individual probes
npm run probe:z3
npm run probe:leanaide
npm run probe:cross
```

**The probes must pass before the orchestrator will start successfully.**

## Usage

### Basic Verification

```typescript
import { UnifiedVerificationOrchestrator } from '@glue/unified-verification';

const orchestrator = new UnifiedVerificationOrchestrator(
  process.env.Z3_URL!,
  process.env.LEANAIDE_URL!
);

// Simple verification (auto-selects strategy)
const result = await orchestrator.verify(
  {
    id: uuidv4(),
    type: 'SMT_CONSTRAINTS',
    description: 'Verify constraint: x > 0',
    statement: '(declare-const x Int) (assert (> x 0)) (check-sat)'
  },
  {
    timeout: 5000,
    precision: 'high',
    allowedSystems: ['both'],
    requiredConfidence: 0.95
  },
  {
    confidenceRequired: 0.95,
    crossValidate: true,
    storeResults: true
  }
);

console.log('Verified:', result.verified);
console.log('Confidence:', result.confidence);
```

### Cross-Validation

```typescript
const crossValidationResult = await orchestrator.verifyWithCrossValidation(
  {
    id: uuidv4(),
    type: 'THEOREM_PROVING',
    description: 'Prove additive identity',
    statement: 'theorem add_zero (n : Nat) : n + 0 = n := by simp'
  },
  {
    confidenceRequired: 0.95,
    strategy: 'parallel',  // Run both systems
    storeResults: true
  }
);

console.log('Agreement:', crossValidationResult.agreement);
console.log('Resolution:', crossValidationResult.resolution);
console.log('Confidence:', crossValidationResult.confidence);
```

### Batch Verification

```typescript
const problems = [
  { id: uuidv4(), type: 'SMT_CONSTRAINTS', statement: '...', description: '...' },
  { id: uuidv4(), type: 'THEOREM_PROVING', statement: '...', description: '...' }
];

const results = await orchestrator.verifyBatch(
  problems,
  { timeout: 10000, precision: 'medium', allowedSystems: ['both'] },
  { storeResults: true }
);

results.forEach((result, problemId) => {
  console.log(`${problemId}: ${result.verified}`);
});
```

## Verification Strategies

The orchestrator automatically selects the optimal strategy based on problem type:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `z3_only` | Z3 only | SMT constraints, SAT solving |
| `leanaide_only` | LeanAide only | Theorem proving, formal verification |
| `parallel` | Both simultaneously | High confidence required, cross-validation |
| `sequential` | Z3 first, then LeanAide | Time-critical, early termination |
| `hybrid` | Adaptive approach | Complex problems, proof refinement |

### Problem Type Mapping

| Problem Type | Best Strategy |
|--------------|---------------|
| `SMT_CONSTRAINTS` | Z3 (95% success rate) |
| `THEOREM_PROVING` | LeanAide (92% success rate) |
| `FORMAL_VERIFICATION` | Parallel (cross-validation) |
| `CODE_CORRECTNESS` | Hybrid (Z3 → LeanAide) |
| `MODEL_CHECKING` | Z3 (90% success rate) |
| `SAT_SOLVING` | Z3 (98% success rate) |

## Confidence Aggregation

The orchestrator combines results from multiple systems:

1. **Normalization**: Adjust scores based on:
   - Historical accuracy
   - Problem type match
   - Execution quality
   - Confidence consistency

2. **Dynamic Weighting**: Systems weighted by:
   - Base strategy weights
   - Success/failure of verification
   - Confidence level achieved
   - Error handling

3. **Aggregation**: Combined confidence calculated using weighted average

4. **Evidence Generation**: Full audit trail including:
   - Individual system contributions
   - Normalization factors
   - Cross-validation agreement
   - Confidence variance analysis

## Cross-Validation

### Agreement Types

- **Full Agreement**: Both systems agree (verified/not verified) + high confidence alignment
- **Partial Agreement**: Systems agree on outcome but have confidence variance
- **Disagreement**: Systems disagree on verification outcome
- **Inconclusive**: Significant discrepancies in results

### Conflict Resolution

| Conflict Type | Resolution Strategy |
|---------------|---------------------|
| `verification_outcome` | Trust higher confidence |
| `confidence_level` | Trust higher confidence |
| `proof_structure` | Requires manual review |
| `timeout_mismatch` | Escalate for review |

### Resolution Outcomes

- `verified`: Problem verified with high confidence
- `not_verified`: Verification failed
- `inconclusive`: Results inconclusive
- `requires_review`: Manual review needed
- `escalated`: Escalated for expert review

## Learning Feedback Loop

The orchestrator learns from verification outcomes:

### Tracked Metrics

- Strategy effectiveness by problem type
- System accuracy rates
- Average execution times
- Confidence calibration

### Adaptive Improvements

- Strategy selection refined based on success rates
- Confidence weights adjusted based on accuracy
- Historical accuracy updated with exponential moving average

### Storage

Results stored for learning (TODO: Integrate with Vector DB + Graphiti):

```typescript
// Statistics
const stats = await orchestrator.getStatistics();

console.log('Total verifications:', stats.totalVerifications);
console.log('Success rate:', stats.successRate);
console.log('Average confidence:', stats.averageConfidence);
console.log('Z3 stats:', stats.systemBreakdown.z3);
console.log('LeanAide stats:', stats.systemBreakdown.leanaide);
```

## API Reference

### UnifiedVerificationOrchestrator

#### Constructor

```typescript
constructor(
  z3Url: string,
  leanaideUrl: string,
  logger?: Logger
)
```

#### Methods

- `verify(problem, constraints, options)`: Simple verification
- `verifyWithCrossValidation(problem, options)`: Cross-validation
- `verifyBatch(problems, constraints, options)`: Batch processing
- `getStatistics()`: Performance statistics
- `healthCheck()`: Component health status

### Canonical Schemas

All data validated using Zod schemas:

- `Problem`: Problem representation
- `Constraints`: Verification constraints
- `VerificationRequest`: Verification request
- `VerificationResult`: Single system result
- `CrossValidationResult`: Cross-validation result
- `ConfidenceScore`: Confidence breakdown
- `VerificationOptions`: User options

## Testing

### Contract Tests

Verify API contracts before deployment:

```bash
npm run test:contract
```

The contract tests validate:
- Required response fields from Z3
- Required response fields from LeanAide
- Canonical schema enforcement
- Cross-validation integration
- Error handling

**If contract tests fail, the adapter refuses to start.**

### Unit Tests

```bash
npm test
```

### Coverage

```bash
npm test -- --coverage
```

## Error Handling

The orchestrator follows Federation Constitution failure management:

- **Transient Failure**: Exponential backoff retry (network blips)
- **Logic Failure**: Dead letter queue (bad data)
- **System Failure**: Circuit breaker (target down)

### Graceful Degradation

```typescript
// If Z3 fails, LeanAide can still verify
// If one system fails, return results from the other
// If both fail, return error with details
```

## Logging

Structured JSON Lines logging:

```json
{
  "level": "info",
  "msg": "Starting cross-validation",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_service": "UnifiedVerificationOrchestrator",
  "problemId": "problem-123",
  "problemType": "SMT_CONSTRAINTS"
}
```

## Design Decisions (ADR)

### 1. Why Cross-Validation?

**Decision**: Implement cross-validation between Z3 and LeanAide

**Rationale**:
- Different systems have different strengths
- Agreement increases confidence
- Disagreement indicates need for review
- Prevents single-point-of-failure

**Trade-offs**:
- Slower than single system (parallel execution mitigates)
- More complex (orchestrator abstraction manages complexity)

### 2. Why Dynamic Strategy Selection?

**Decision**: Automatically select strategy based on problem type

**Rationale**:
- Users shouldn't need to be formal methods experts
- Historical data informs selection
- Adaptive learning improves over time

**Trade-offs**:
- Learning curve at start (defaults are reasonable)
- Overhead to track metrics (minimal impact)

### 3. Why Confidence Aggregation?

**Decision**: Combine confidence scores rather than simple majority

**Rationale**:
- Different systems have different accuracy profiles
- Weighted combination accounts for this
- Evidence trail provides explainability

**Trade-offs**:
- More complex than boolean logic (worth it for nuanced results)

## Troubleshooting

### Probe Failures

```bash
# Z3 not responding
curl ${Z3_URL}/health

# LeanAide not responding
curl ${LEANAIDE_URL}/health

# Check logs
docker logs z3-container
docker logs leanaide-container
```

### Contract Test Failures

```bash
# Run tests with verbose output
npm run test:contract -- --verbose

# Check API responses manually
curl -X POST ${Z3_URL}/verify -d '{...}'
```

### Low Confidence Results

- Check if problem type matches system capabilities
- Increase timeout constraints
- Review error messages in results
- Consider manual review for complex problems

## Federation Constitution Compliance

✅ **Law of Air Gap**: No imports from core-projects
✅ **Law of Runtime Truth**: All APIs verified via probes
✅ **Law of Untouchable DB**: Read-only access (when integrated)
✅ **Law of Idempotency**: Safe to retry verification requests
✅ **Law of Configuration Explicitness**: All env vars required
✅ **Law of UTC**: All timestamps in UTC ISO-8601

## Future Enhancements

- [ ] Integration with Vector DB for semantic search of proofs
- [ ] Integration with Graphiti for proof lineage tracking
- [ ] Support for additional verification systems
- [ ] Advanced learning algorithms
- [ ] Real-time strategy adjustment
- [ ] Distributed verification (multiple instances)

## Contributing

When contributing to this component:

1. **Run probes first**: `npm run probes`
2. **Write contract tests**: Verify API contracts
3. **Follow canonical schemas**: Use defined Zod schemas
4. **Add logging**: All actions logged with correlation_id
5. **Update README**: Document changes
6. **Test idempotency**: Ensure safe retries

## License

MIT

## Contact

For questions or issues, contact the Glue Layer team.
