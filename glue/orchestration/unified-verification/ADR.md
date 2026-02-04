# Architecture Decision Record: Unified Verification Orchestrator

**Status**: Accepted
**Date**: 2025-01-15
**Component**: Unified Verification Orchestrator
**Deciders**: Glue Layer Team

## Context

We need to integrate Z3 SMT Solver and LeanAide Theorem Prover into a cohesive verification system. These systems have:
- Different strengths (SMT solving vs theorem proving)
- Different APIs and data formats
- Different performance characteristics
- Different confidence metrics

## Problem

How do we provide a unified interface for formal verification while:
1. Leveraging the strengths of both systems?
2. Providing high confidence in verification results?
3. Handling failures gracefully?
4. Maintaining separation from core projects (Air Gap Law)?
5. Ensuring runtime correctness (Runtime Truth Law)?

## Decision

### 1. Cross-Validation Architecture

**Decision**: Implement cross-validation between Z3 and LeanAide

**Rationale**:
- **Complementary Strengths**: Z3 excels at SMT constraints; LeanAide excels at theorem proving
- **Confidence Boost**: Agreement between systems increases confidence
- **Error Detection**: Disagreement indicates need for review
- **Resilience**: If one system fails, the other may still succeed

**Implementation**:
- Parallel execution when both systems applicable
- Sequential execution for time-critical verification
- Hybrid approach for complex problems
- Result comparison and conflict detection

**Trade-offs**:
- ✅ Higher confidence through cross-validation
- ✅ Better error detection
- ❌ Slower than single system (mitigated by parallel execution)
- ❌ More complex (managed by orchestrator abstraction)

### 2. Canonical Data Models (Anti-Corruption Layer)

**Decision**: Define canonical schemas using Zod

**Rationale**:
- **Data Normalization**: Different systems use different formats
- **Type Safety**: Zod provides runtime validation
- **Documentation**: Schemas serve as documentation
- **Air Gap Compliance**: No imports from core-projects

**Implementation**:
- `Problem`: Unified problem representation
- `VerificationResult`: Unified result format
- `CrossValidationResult`: Cross-validation output
- `ConfidenceScore`: Confidence breakdown

**Trade-offs**:
- ✅ Clean separation from external systems
- ✅ Runtime validation prevents data corruption
- ✅ Clear API contracts
- ❌ Additional mapping layer (worth it for safety)

### 3. Strategy Selection Pattern

**Decision**: Automatically select verification strategy based on problem type

**Rationale**:
- **User Simplicity**: Users shouldn't need formal methods expertise
- **Optimal Performance**: Right tool for the job
- **Adaptive Learning**: Improve based on historical outcomes

**Implementation**:
- Problem type analysis (SMT_CONSTRAINTS, THEOREM_PROVING, etc.)
- System capability mapping (Z3: 95% for SMT, LeanAide: 92% for theorems)
- Strategy selection (z3_only, leanaide_only, parallel, sequential, hybrid)
- Learning feedback loop for continuous improvement

**Strategy Mapping**:
| Problem Type | Strategy | Success Rate |
|--------------|----------|--------------|
| SMT_CONSTRAINTS | z3_only | 95% |
| THEOREM_PROVING | leanaide_only | 92% |
| FORMAL_VERIFICATION | parallel | 85% |
| CODE_CORRECTNESS | hybrid | 80% |

**Trade-offs**:
- ✅ Optimal system selection
- ✅ Adaptive improvement over time
- ❌ Learning curve at start (defaults are reasonable)
- ❌ Overhead to track metrics (minimal impact)

### 4. Confidence Aggregation

**Decision**: Combine confidence scores using weighted aggregation

**Rationale**:
- **Nuanced Results**: Binary yes/no is insufficient
- **Weighted Combination**: Different systems have different accuracy
- **Explainability**: Evidence trail shows how confidence was calculated
- **Threshold-Based**: Users specify required confidence

**Implementation**:
- Normalize individual scores (historical accuracy, execution quality, etc.)
- Calculate dynamic weights (strategy, success, confidence level)
- Combine using weighted average
- Generate evidence trail (system contributions, cross-validation, etc.)

**Confidence Levels**:
- `very_high`: ≥95%
- `high`: ≥85%
- `medium`: ≥70%
- `low`: <70%

**Trade-offs**:
- ✅ More informative than boolean
- ✅ Accounts for system differences
- ✅ Explainable via evidence trail
- ❌ More complex (abstraction manages complexity)

### 5. Probe-Based Verification (Runtime Truth Law)

**Decision**: Verify API capabilities before using them

**Rationale**:
- **Documentation Lies**: APIs may not match docs
- **Version Changes**: Updates can break contracts
- **Early Detection**: Catch issues before runtime

**Implementation**:
- `check_z3.sh`: Verify Z3 health check and verify endpoint
- `check_leanaide.sh`: Verify LeanAide health check and verify endpoint
- `check_cross_validation.sh`: Verify integration end-to-end
- Contract tests: Run on container startup

**Probe Tests**:
1. Health check endpoint (200 OK)
2. Simple verification request (valid response)
3. Required fields present (verified, confidence, output)
4. Confidence score in valid range [0, 1]
5. Execution time within limits

**Trade-offs**:
- ✅ Catches integration issues early
- ✅ Enforces runtime correctness
- ✅ Documents actual API behavior
- ❌ Additional setup step (necessary for safety)

### 6. Graceful Degradation

**Decision**: Handle failures gracefully rather than fail fast

**Rationale**:
- **Partial Success**: One system failing doesn't mean both fail
- **User Experience**: Return best available result
- **Debugging**: Error messages help diagnose issues

**Implementation**:
- Transient failures: Exponential backoff retry
- Logic failures: Dead letter queue (log but don't block)
- System failures: Circuit breaker (stop hammering dead service)
- Return partial results if one system succeeds

**Failure Modes**:
| Failure Type | Handling |
|--------------|----------|
| Network blip | Retry with backoff |
| Bad data | Log to DLQ, continue |
| System down | Circuit breaker, use other system |
| Both down | Return error with details |

**Trade-offs**:
- ✅ More resilient than fail-fast
- ✅ Better user experience
- ✅ Easier debugging
- ❌ More complex error handling (necessary for production)

### 7. Learning Feedback Loop

**Decision**: Track outcomes and improve strategy selection

**Rationale**:
- **Adaptive System**: Improve over time
- **Data-Driven**: Decisions based on actual outcomes
- **Continuous Improvement**: No manual tuning needed

**Implementation**:
- Track strategy effectiveness by problem type
- Update success rates with exponential moving average
- Adjust confidence weights based on accuracy
- Store results for analysis (TODO: Vector DB + Graphiti)

**Metrics Tracked**:
- Success rate by (system, problem_type)
- Average execution time
- Confidence calibration (predicted vs actual)
- Strategy effectiveness

**Trade-offs**:
- ✅ Self-improving system
- ✅ Data-driven decisions
- ❌ Requires storage (minimal overhead)
- ❌ Cold start problem (reasonable defaults)

## Consequences

### Positive

1. **High Confidence**: Cross-validation provides strong assurance
2. **Optimal Performance**: Right tool for each problem type
3. **Resilient**: Graceful degradation handles failures
4. **Adaptive**: Learning improves over time
5. **Air Gap Compliant**: No imports from core-projects
6. **Runtime Correct**: Probes verify actual API behavior

### Negative

1. **Complexity**: More complex than single system (managed by abstraction)
2. **Latency**: Parallel execution adds overhead (acceptable for confidence gain)
3. **Storage**: Learning requires result storage (minimal impact)

### Risks

1. **Cold Start**: Learning system starts with defaults
   - **Mitigation**: Defaults based on known system strengths

2. **API Changes**: Core projects may change APIs
   - **Mitigation**: Contract tests catch changes early

3. **Confidence Calibration**: May be off initially
   - **Mitigation**: Exponential moving average adjusts slowly

## Alternatives Considered

### Alternative 1: Single System (Z3 Only)

**Rejected Because**:
- LeanAide better for theorem proving
- No cross-validation
- Single point of failure

### Alternative 2: User-Selected Strategy

**Rejected Because**:
- Requires formal methods expertise
- Suboptimal choices likely
- No learning/improvement

### Alternative 3: Simple Majority Voting

**Rejected Because**:
- Doesn't account for confidence levels
- Doesn't weight by system strength
- Less informative than aggregation

### Alternative 4: Fail-Fast Error Handling

**Rejected Because**:
- One system failure breaks entire flow
- Poor user experience
- No partial results

## Implementation Status

- ✅ Canonical schemas defined
- ✅ Strategy selector implemented
- ✅ Cross validator implemented
- ✅ Confidence aggregator implemented
- ✅ Main orchestrator implemented
- ✅ Probe scripts created
- ✅ Contract tests written
- ✅ Documentation completed
- ⏳ Vector DB integration (TODO)
- ⏳ Graphiti integration (TODO)

## Related Decisions

- [ADR: Canonical Data Models](./canonical.ts) - Why we use Zod schemas
- [ADR: Strategy Selection](./strategy-selector.ts) - How strategies are chosen
- [ADR: Confidence Aggregation](./confidence-aggregator.ts) - How scores combine

## References

- [Federation Constitution](../../../CLAUDE.md) - Operating principles
- [Anti-Corruption Layer Pattern](https://martinfowler.com/bliki/AnticorruptionLayer.html) - ACL pattern
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html) - Failure handling
- [Exponential Moving Average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) - Learning algorithm
