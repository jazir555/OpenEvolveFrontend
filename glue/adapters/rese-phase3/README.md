# RESE Phase III: MCTS Search Adapter

## Overview

This adapter implements **RESE Phase III: Monte Carlo Refinement** with the **MC-NEST** (Monte Carlo Nash Equilibrium Self-Refine Tree) algorithm.

### Key Components

1. **MCTSSearchExecutor**: Main orchestrator for MC-NEST search
2. **SearchTreeBuilder**: Constructs and manages MCTS search tree with idempotent updates
3. **HypothesisValidator**: Statistical validation using t-tests and confidence intervals
4. **ConvergenceDetector**: ACI (Algorithmic Convergence Indicator) for convergence detection
5. **UCB1SelectionStrategy**: UCB1 selection for balanced exploration/exploitation
6. **HypothesisDLQ**: Dead Letter Queue for failed hypotheses

## Features

### Core Capabilities

- **MC-NEST Algorithm**: Full MCTS implementation with Selection, Expansion, Simulation, Backpropagation
- **Statistical Validation**: T-tests, confidence intervals, sample size validation
- **ACI Convergence Detection**: Stability-based convergence with configurable windows
- **Idempotent Operations**: Deduplication by hypothesis_id (Law of Idempotency)
- **Circuit Breaker**: Failure detection and graceful degradation
- **Dead Letter Queue**: Capture and analyze failed hypotheses
- **Timeout Enforcement**: All operations bounded by configurable timeouts

### CLAUDE.md Compliance

✅ **Law of Configuration Explicitness**: All config via environment variables
✅ **Law of Idempotency**: UPSERT logic, deduplicate by hypothesis_id
✅ **Law of Runtime Truth**: Probe script verifies functionality
✅ **Circuit Breaker**: Detects and handles search failures
✅ **Structured Logging**: JSON logs with correlation_id
✅ **Timeout**: All operations have enforced timeouts

## Installation

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Required dependencies
pip install numpy scipy
```

### Environment Variables

Required environment variables (with defaults):

```bash
# MCTS Parameters
export PHASE3_ITERATIONS=1000
export PHASE3_UCB1_C=1.414
export PHASE3_CONVERGENCE_THRESHOLD=0.001
export PHASE3_TIMEOUT_MS=30000

# Search Tree
export PHASE3_MAX_DEPTH=20
export PHASE3_MAX_CHILDREN=10
export PHASE3_MIN_VISITS=5

# Validation
export PHASE3_SIG_THRESHOLD=0.05
export PHASE3_CONFIDENCE_INTERVAL=0.95
export PHASE3_MIN_SAMPLE_SIZE=30

# ACI Convergence
export PHASE3_ACI_WINDOW=100
export PHASE3_ACI_STABILITY=0.01

# Deduplication
export PHASE3_DEDUP_ENABLED=true
export PHASE3_CACHE_SIZE=10000

# Circuit Breaker
export PHASE3_CB_THRESHOLD=5
export PHASE3_CB_TIMEOUT=60000
```

## Usage

### Basic Usage

```python
from glue.adapters.rese_phase3.src.phase3_adapter import Phase3Adapter

# Initialize adapter
adapter = Phase3Adapter()

# Execute search
request = {
    "root_hypothesis": {
        "statement": "Test hypothesis",
        "type": "causal",
        "domain": "physics",
        "confidence": 0.5,
    },
    "num_children": 5,
}

result = adapter.search(request)
print(f"Best hypothesis: {result['best_hypothesis']['statement']}")
print(f"Confidence: {result['best_confidence']:.3f}")
print(f"Iterations: {result['tree_statistics']['iterations']}")
```

### Advanced Usage

```python
from glue.adapters.rese_phase3.src.phase3_executor import (
    MCTSSearchExecutor,
    Phase3Config,
)
from glue.schemas.rese_schemas import Hypothesis

# Custom configuration
config = Phase3Config(
    iterations=5000,
    ucb1_c=1.5,
    convergence_threshold=0.0001,
    timeout_ms=60000,
    max_depth=30,
    correlation_id="my-search-123",
)

# Initialize executor
executor = MCTSSearchExecutor(config)

# Create root hypothesis
root_hypothesis = Hypothesis(
    statement="Root hypothesis",
    type="causal",
    domain="physics",
    confidence=0.5,
)

# Define hypothesis generator
def hypothesis_generator():
    # Use DEE's HypothesisGenerator in production
    children = []
    for i in range(5):
        child = Hypothesis(
            statement=f"Child hypothesis {i}",
            type="causal",
            domain="physics",
            confidence=0.6,
            source_hypotheses=[root_hypothesis.hypothesis_id],
        )
        children.append(child)
    return children

# Define reward function (use LLTL for constraint-based evaluation)
def reward_function(hypothesis):
    # Higher confidence = higher reward
    return hypothesis.confidence

# Execute search
search_result, error = executor.execute_search(
    root_hypothesis=root_hypothesis,
    hypothesis_generator=hypothesis_generator,
    reward_function=reward_function,
)

if error:
    print(f"Search failed: {error}")
else:
    print(f"Search ID: {search_result.search_id}")
    print(f"Best hypothesis: {search_result.best_hypothesis.statement}")
    print(f"Best confidence: {search_result.best_hypothesis.confidence:.3f}")
    print(f"Iterations: {search_result.iterations}")
    print(f"Converged: {search_result.convergence_reached}")
    print(f"Execution time: {search_result.execution_time_ms:.1f}ms")
```

### Hypothesis Validation

```python
# Validate hypothesis with statistical tests
request = {
    "hypothesis": {
        "statement": "Test hypothesis",
        "type": "causal",
        "domain": "physics",
        "confidence": 0.7,
    },
    "rewards": [0.65, 0.72, 0.68, 0.71, 0.69, ...],  # From simulations
}

result = adapter.validate_hypothesis(request)

if result["success"]:
    validation = result["validation_result"]
    print(f"Valid: {validation['is_valid']}")
    print(f"Confidence: {validation['confidence']:.3f}")
    print(f"P-value: {validation['p_value']:.4f}")
    print(f"95% CI: [{validation['confidence_interval'][0]:.3f}, {validation['confidence_interval'][1]:.3f}]")
```

### Convergence Checking

```python
# Check convergence during search
request = {
    "iteration": 100,
    "best_confidence": 0.85,
    "best_reward": 0.82,
}

result = adapter.check_convergence(request)

if result["success"]:
    print(f"Converged: {result['is_converged']}")
    print(f"ACI Value: {result['aci_value']}")
```

## Testing

### Run Probe Script

```bash
cd glue/adapters/rese-phase3
chmod +x probes/check_phase3.sh
./probes/check_phase3.sh
```

Expected output:
```
==================================
RESE Phase III Probe
Testing MCTS Search Executor
==================================

Check 1: Python availability... PASSED
Check 2: Setting environment variables... PASSED
Check 3: Testing imports... PASSED
Check 4: Configuration validation... PASSED
Check 5: Executor initialization... PASSED
Check 6: Search execution (10 iterations)... PASSED
  SEARCH_SUCCESS: iterations=10, best_confidence=0.620
Check 7: Hypothesis validation... PASSED
  VALIDATION_SUCCESS: is_valid=True, confidence=0.630
Check 8: Convergence detection... PASSED
  CONVERGENCE_CHECK_SUCCESS: is_converged=True, aci_value=0.0008

==================================
ALL CHECKS PASSED
==================================
```

### Run Unit Tests

```bash
cd glue/adapters/rese-phase3
python tests/test_phase3.py
```

## Architecture

### MC-NEST Algorithm Flow

```
┌─────────────────────────────────────┐
│  1. Selection (UCB1)                │
│     Select promising node           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Expansion                       │
│     Generate child hypotheses       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Simulation                     │
│     Evaluate rewards (with CB)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Backpropagation                │
│     Update values up tree           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Validation                     │
│     Statistical hypothesis tests    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Convergence Check (ACI)        │
│     Check stability of best         │
└──────────────┬──────────────────────┘
               │
               ▼
         Repeat until converged or max iterations
```

### Component Interactions

```
┌──────────────────┐
│ Phase3Adapter    │
│  (REST API)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  MCTSSearchExecutor                  │
│  ┌────────────────────────────────┐ │
│  │ SearchTreeBuilder              │ │
│  │  - Build tree                  │ │
│  │  - Expand nodes (idempotent)   │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │ UCB1SelectionStrategy          │ │
│  │  - Select best child           │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │ HypothesisValidator            │ │
│  │  - T-tests                     │ │
│  │  - Confidence intervals        │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │ ConvergenceDetector            │ │
│  │  - ACI calculation             │ │
│  │  - Stability detection         │ │
│  └────────────────────────────────┘ │
│  ┌────────────────────────────────┐ │
│  │ HypothesisDLQ                  │ │
│  │  - Failed hypotheses           │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

## Integration with RESE Components

### DEE Integration (Hypothesis Generation)

```python
from glue.adapters.rese_dee.src.dee_adapter import DEEAdapter

# Initialize DEE
dee_adapter = DEEAdapter()

# Use DEE for hypothesis generation
def hypothesis_generator():
    # DEE generates hypotheses using various strategies
    explore_result = dee_adapter.explore({
        "problem_statement": "Your problem",
        "domain": "physics",
    })

    # Extract hypotheses from DEE result
    hypotheses = []
    for pattern in explore_result.get("patterns", []):
        hypothesis = Hypothesis(
            statement=pattern["description"],
            type="pattern_based",
            domain="physics",
            confidence=pattern["confidence"],
        )
        hypotheses.append(hypothesis)

    return hypotheses
```

### LLTL Integration (Constraint-Based Reward)

```python
from glue.adapters.rese_lltl.src.lltl_adapter import LLTLAdapter

# Initialize LLTL
lltl_adapter = LLTLAdapter()

# Use LLTL for constraint-based reward calculation
def reward_function(hypothesis):
    # Translate hypothesis to constraints
    constraints = [hypothesis.statement]  # Simplified

    # Translate to loss function
    result, error = lltl_adapter.translate_constraints(constraints)

    if error:
        return 0.0  # Penalize failures

    # Calculate reward from loss (inverse relationship)
    loss = result.get("total_loss", 1.0)
    reward = max(0.0, 1.0 - loss)

    return reward
```

## Monitoring and Debugging

### Health Check

```python
health = adapter.get_health()

print(f"Status: {health['status']}")
print(f"Circuit Breaker: {health['circuit_breaker_state']}")
print(f"DLQ Size: {health['dlq_size']}")
```

### Dead Letter Queue

```python
# Get failed hypotheses
dlq_contents = adapter.get_dlq_contents()

for entry in dlq_contents:
    print(f"Hypothesis: {entry['statement']}")
    print(f"Error: {entry['error']}")
    print(f"Type: {entry['error_type']}")

# Clear DLQ
adapter.clear_dlq()
```

### Structured Logs

All logs are JSON Lines format with correlation_id:

```json
{"msg": "Starting MC-NEST search", "level": "info", "correlation_id": "abc-123", "search_id": "xyz-789", "timestamp": "2026-02-04T12:00:00Z"}
{"msg": "Convergence reached", "level": "info", "correlation_id": "abc-123", "iteration": 450, "aci_value": 0.0008, "timestamp": "2026-02-04T12:01:30Z"}
```

## Performance Tuning

### Convergence Speed

- **Lower ACI stability threshold**: Faster convergence (may be less accurate)
- **Smaller ACI window**: Faster detection (more volatile)
- **Higher UCB1 C**: More exploration (slower convergence)

### Search Quality

- **More iterations**: Better solutions (slower)
- **Deeper max_depth**: More thorough search (slower)
- **Higher min_visits**: More reliable statistics (slower)

### Resource Limits

- **Timeout**: Prevents infinite searches
- **Max children per node**: Limits branching factor
- **DLQ size**: Limits memory for failed hypotheses

## Troubleshooting

### Search Not Converging

- Increase `PHASE3_ITERATIONS`
- Decrease `PHASE3_ACI_STABILITY`
- Decrease `PHASE3_ACI_WINDOW`
- Check hypothesis generator diversity

### Circuit Breaker Opening

- Increase `PHASE3_CB_THRESHOLD`
- Increase `PHASE3_CB_TIMEOUT`
- Check reward function for errors
- Review DLQ for error patterns

### Slow Performance

- Decrease `PHASE3_ITERATIONS`
- Decrease `PHASE3_MAX_DEPTH`
- Decrease `PHASE3_MAX_CHILDREN`
- Decrease `PHASE3_MIN_VISITS`

## Contributing

See `ADR.md` for architectural decisions and rationale.

## License

RESE Project License

## References

- [RESE Technical Manual](../../../docs/guides/RESE_IMPLEMENTATION_ROADMAP.md)
- [CLAUDE.md](../../../CLAUDE.md)
- [Phase I: Epistemic Audit](../rese-phase1/README.md) - (Future)
- [Phase II: Isomorphic Mapping](../rese-phase2/README.md) - (Future)
- [Phase IV: Architectural Synthesis](../rese-phase4/README.md) - (Future)
