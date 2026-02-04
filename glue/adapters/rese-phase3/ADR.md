# Architecture Decision Record: RESE Phase III - MCTS Search

**Status**: Accepted
**Date**: 2026-02-04
**Phase**: III - Monte Carlo Refinement
**Component**: MC-NEST Executor

## Context

RESE Phase III requires implementing **Monte Carlo Refinement** with the following requirements:

1. **MC-NEST Algorithm**: Monte Carlo Nash Equilibrium Self-Refine Tree for adaptive search
2. **Statistical Validation**: Hypothesis testing with confidence intervals
3. **ACI Convergence Detection**: Algorithmic Convergence Indicator for termination
4. **Idempotent Operations**: Deduplication and replayability (Law of Idempotency)
5. **Circuit Breaker**: Failure detection and graceful degradation
6. **CLAUDE.md Compliance**: All 6 laws must be followed

### Key Challenges

1. **Search Space Explosion**: MCTS can explore exponential number of nodes
2. **Convergence Detection**: Need reliable method to detect when search is complete
3. **Statistical Validity**: Hypotheses must be statistically validated, not just heuristically
4. **Failure Handling**: Search can fail in various ways (timeout, contradictions, etc.)
5. **Performance**: Must complete within reasonable time (default 30s timeout)

## Decision

### Core Architecture

We implement **MC-NEST** as a four-phase MCTS algorithm:

```
Selection (UCB1) → Expansion → Simulation → Backpropagation → Validation → Convergence Check
```

### Component Structure

```
MCTSSearchExecutor (Main Orchestrator)
├── SearchTreeBuilder (Tree Management)
│   ├── Idempotent node updates
│   ├── Deduplication by hypothesis_id
│   └── Max depth enforcement
├── UCB1SelectionStrategy (Node Selection)
│   ├── Exploitation: mean_value
│   └── Exploration: C * sqrt(ln(parent) / visits)
├── HypothesisValidator (Statistical Validation)
│   ├── T-tests for significance
│   ├── Confidence intervals
│   └── Sample size validation
├── ConvergenceDetector (ACI)
│   ├── Stability metric
│   └── Window-based detection
└── HypothesisDLQ (Failure Handling)
    └── Capture failed hypotheses
```

### Key Algorithms

#### 1. UCB1 Selection

**Formula**:
```
UCB1 = mean_value + C * sqrt(ln(parent_visits) / visits)
```

**Rationale**:
- Balances exploration (high C) and exploitation (high mean)
- Proven convergence guarantees
- Default C = sqrt(2) ≈ 1.414

#### 2. Statistical Validation

**Method**: One-sample t-test

- **Null Hypothesis**: mean_reward <= 0.5
- **Alternative**: mean_reward > 0.5
- **Significance Level**: α = 0.05 (configurable)
- **Confidence Interval**: 95% (configurable)

**Rationale**:
- Provides statistical guarantees
- Prevents false positives
- Quantifies uncertainty

#### 3. ACI Convergence Detection

**Formula**:
```
ACI = variance(confidence_history) / (max(confidence) - min(confidence))
```

**Convergence Condition**:
```
ACI < threshold AND window_size >= min_window
```

**Rationale**:
- Measures stability of best hypothesis
- Normalized for scale-independence
- Window-based for robustness

#### 4. Idempotent Tree Updates

**Mechanism**:
- Hypothesis cache: `Set[hypothesis_id]`
- Check before insertion: `if hypothesis_id in cache: skip`
- UPSERT semantics: Update if exists, insert if new

**Rationale**:
- Law of Idempotency (CLAUDE.md)
- Safe replay on network failure
- Deduplicates duplicate hypotheses

## Alternatives Considered

### Alternative 1: Pure Greedy Search

**Description**: Always select highest-value node.

**Pros**:
- Simpler implementation
- Faster execution

**Cons**:
- Gets stuck in local optima
- No exploration guarantees
- Rejected: Violates MC-NEST requirements

### Alternative 2: Thompson Sampling

**Description**: Bayesian approach with beta distributions.

**Pros**:
- Theoretically optimal
- Good for bandit problems

**Cons**:
- More complex implementation
- Requires prior specification
- Rejected: Overkill for this use case

### Alternative 3: Genetic Algorithm

**Description**: Population-based evolution.

**Pros**:
- Parallelizable
- Good for global search

**Cons**:
- No convergence guarantees
- Requires population management
- Rejected: MCTS is better suited for tree search

### Alternative 4: Heuristic Validation

**Description**: Use simple threshold for validation.

**Pros**:
- Faster than statistical tests
- Simpler implementation

**Cons**:
- No statistical guarantees
- Higher false positive rate
- Rejected: Violates RESE rigor requirements

## Consequences

### Positive

1. **Proven Convergence**: UCB1 has theoretical convergence guarantees
2. **Statistical Rigor**: T-tests provide quantified uncertainty
3. **Idempotent**: Safe to retry failed requests
4. **Observable**: DLQ captures all failures for analysis
5. **Configurable**: All parameters via environment variables
6. **CLAUDE.md Compliant**: All 6 laws followed

### Negative

1. **Complexity**: More complex than simple greedy search
2. **Performance**: Statistical tests add overhead
3. **Tuning Required**: Many parameters to tune (UCB1 C, ACI threshold, etc.)
4. **Sample Size**: Requires minimum samples for validation

### Risks

1. **Risk: Slow Convergence**
   - **Mitigation**: Configurable timeout, early stopping
   - **Status**: Accepted

2. **Risk: Circuit Breaker False Positives**
   - **Mitigation**: Configurable threshold, timeout for recovery
   - **Status**: Accepted

3. **Risk: ACI Not Detecting Convergence**
   - **Mitigation**: Fallback to max iterations, configurable window
   - **Status**: Accepted

4. **Risk: Statistical Validation Too Strict**
   - **Mitigation**: Configurable significance level, sample size
   - **Status**: Accepted

## CLAUDE.md Compliance

### ✅ Law 1: Air Gap (Source Code Isolation)

**Implementation**: No imports from `core-projects/`

```python
# All imports from glue/lib or glue/schemas
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))

from rese_schemas import ...  # Canonical schemas
from rese_dee import ...  # Glue library
```

**Verification**: Code review confirms no core-projects imports

### ✅ Law 2: Runtime Truth (Anti-Hallucination)

**Implementation**: Probe script executes actual search

```bash
# probes/check_phase3.sh
# - Executes actual search with 10 iterations
# - Validates all components
# - Must pass before Phase III is considered functional
```

**Verification**: Probe script passes all checks

### ✅ Law 3: Untouchable DB (Read-Only State)

**Implementation**: No database writes

- Search results are in-memory
- DLQ is in-memory (can be persisted to file if needed)
- No direct database operations

**Verification**: Code review confirms no DB writes

### ✅ Law 4: Idempotency (Replayability Pact)

**Implementation**: Deduplication by hypothesis_id

```python
# SearchTreeBuilder.expand_node()
if hypothesis.hypothesis_id in self.hypothesis_cache:
    self.logger.debug("Skipping duplicate hypothesis")
    continue  # Skip duplicate

# Add to cache
self.hypothesis_cache.add(hypothesis.hypothesis_id)
```

**Verification**: Test cases confirm deduplication works

### ✅ Law 5: Configuration Explicitness

**Implementation**: All config via environment variables

```python
@dataclass
class Phase3Config:
    iterations: int  # From PHASE3_ITERATIONS
    ucb1_c: float  # From PHASE3_UCB1_C
    # ... all fields from env vars

    @classmethod
    def from_env(cls) -> "Phase3Config":
        # Load from os.getenv()
        # Crash immediately if required vars missing
```

**Verification**: Adapter validates config at startup

### ✅ Law 6: UTC (All timestamps in UTC)

**Implementation**: All timestamps use timezone-aware UTC

```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)  # UTC timestamp
```

**Verification**: Code review confirms all timestamps use UTC

## Integration Points

### DEE (Hypothesis Generation)

```python
# DEE provides diverse hypotheses
def hypothesis_generator():
    explore_result = dee_adapter.explore({...})
    return extract_hypotheses(explore_result)
```

### LLTL (Constraint-Based Reward)

```python
# LLTL evaluates constraint satisfaction
def reward_function(hypothesis):
    result, error = lltl_adapter.translate_constraints([hypothesis])
    return 1.0 - result["total_loss"]  # Inverse relationship
```

### Canonical Schemas

```python
# All data uses canonical schemas
from rese_schemas import Hypothesis, SearchTreeNode, MCTSSearchResult
```

## Testing Strategy

### Unit Tests

- **Configuration**: Test env var loading and validation
- **UCB1 Selection**: Test child selection logic
- **Tree Builder**: Test node expansion and deduplication
- **Validator**: Test statistical validation with various inputs
- **Convergence**: Test ACI calculation and detection
- **DLQ**: Test failure capture and max size

### Integration Tests

- **End-to-End Search**: Test full MC-NEST pipeline
- **DEE Integration**: Test with DEE hypothesis generator
- **LLTL Integration**: Test with LLTL reward function
- **Validation**: Test search followed by validation

### Probe Script

```bash
./probes/check_phase3.sh
```

- Verifies all components initialize
- Executes actual search
- Validates output
- **Must pass before deployment**

## Performance Characteristics

### Time Complexity

- **Selection**: O(log n) per level (tree traversal)
- **Expansion**: O(k) where k = num_children
- **Simulation**: O(n) where n = num_simulations
- **Backpropagation**: O(depth) per update
- **Overall per iteration**: O(depth + k + n)

### Space Complexity

- **Tree Storage**: O(total_nodes)
- **Hypothesis Cache**: O(total_nodes)
- **DLQ**: O(dlq_max_size)
- **Overall**: O(total_nodes)

### Scalability

- **Iterations**: Linear scaling (configurable)
- **Max Depth**: Limits branching factor
- **Timeout**: Prevents infinite searches
- **Circuit Breaker**: Prevents cascading failures

## Future Enhancements

1. **Parallel MCTS**: Run multiple searches in parallel
2. **GPU Acceleration**: Accelerate reward calculations
3. **Adaptive Parameters**: Adjust UCB1 C based on progress
4. **Transfer Learning**: Reuse trees from similar searches
5. **Distributed Search**: Coordinate across multiple machines

## References

- [RESE Implementation Roadmap](../../../docs/guides/RESE_IMPLEMENTATION_ROADMAP.md) - Phase III specification
- [CLAUDE.md](../../../CLAUDE.md) - Federation Constitution
- [Kocsis & Szepesvári (2006)]: "Bandit-based Monte-Carlo Planning"
- [Coulom (2007)]: "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"

## Appendix: Parameter Tuning Guide

### UCB1 Exploration Constant (C)

- **Range**: 0.5 to 3.0
- **Default**: sqrt(2) ≈ 1.414
- **Higher**: More exploration (slower convergence, better global optimum)
- **Lower**: More exploitation (faster convergence, may get stuck)

### Convergence Threshold

- **Range**: 0.0001 to 0.01
- **Default**: 0.001
- **Lower**: Stricter convergence (slower, more accurate)
- **Higher**: Looser convergence (faster, less accurate)

### ACI Window Size

- **Range**: 20 to 500
- **Default**: 100
- **Smaller**: Faster detection (more volatile)
- **Larger**: More stable detection (slower)

### Statistical Significance

- **Range**: 0.01 to 0.10
- **Default**: 0.05
- **Lower**: Stricter validation (fewer hypotheses pass)
- **Higher**: Looser validation (more hypotheses pass)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-04
**Author**: RESE Team
**Status**: Accepted and Implemented
