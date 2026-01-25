# Γ₂/Γ₃ Implementation Guide
## MCTS Search and Statistical Validation for RESE Phase III

**Author:** Agent D2 (Γ₂/Γ₃ Specialist)
**Date:** 2025-12-31
**Status:** Complete Implementation

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Guide](#module-guide)
4. [Usage Examples](#usage-examples)
5. [Integration with RESE](#integration-with-rese)
6. [API Reference](#api-reference)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This implementation provides Γ₂ (MCTS Search) and Γ₃ (Statistical Validation) for RESE Phase III (Monte Carlo Refinement).

### Components

1. **Γ₂: MCTS Search** (`mcts_search.py`)
   - UCT-based Monte Carlo Tree Search
   - Progressive widening for large branching factors
   - ACI-guided node selection and playouts
   - Parallel execution support

2. **Γ₃: Statistical Validator** (`statistical_validator.py`)
   - Bootstrap confidence intervals (percentile, BCa)
   - Significance testing (t-test, Wilcoxon)
   - Convergence detection (multiple methods)
   - Sample size determination

3. **Stage 3 Integration** (`stage3_integration.py`)
   - Monte Carlo Nest: Multi-agent search
   - Combines Γ₁ (ACI), Γ₂ (MCTS), Γ₃ (Validation)
   - Result aggregation and validation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monte Carlo Nest (Stage 3)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Γ₁ ACI    │    │  Γ₂ MCTS     │    │   Γ₃ Stats   │  │
│  │   Analyzer   │───▶│    Search    │───▶│  Validator   │  │
│  │              │    │              │    │              │  │
│  │ • Disorder   │    │ • UCT        │    │ • Bootstrap  │  │
│  │ • Coherence  │    │ • Widening   │    │ • Testing    │  │
│  │ • Solvability│    │ • ACI-guided │    │ • Convergence│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │ Result Aggregator│                      │
│                    └────────┬────────┘                      │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  Best Solution  │                      │
│                    └─────────────────┘                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Guide

### 1. MCTS Search Module (`mcts_search.py`)

#### Core Classes

**`MCTSState`**
```python
@dataclass
class MCTSState:
    variables: Dict[str, Any]
    unassigned: List[str]
    domains: Dict[str, List[Any]]
    satisfied: bool
    depth: int
```

**`MCTSNode`**
```python
@dataclass
class MCTSNode:
    state: MCTSState
    parent: Optional[MCTSNode]
    children: List[MCTSNode]
    visits: int
    value_sum: float
    prior: float
    aci_score: float
```

**`MCTSSearch`**
```python
class MCTSSearch:
    def __init__(self, config: MCTSConfig, aci_analyzer: ACIAnalyzer)

    def search(self, initial_state, action_generator,
               state_transition, value_function) -> Tuple[MCTSNode, Dict]
```

#### Usage

```python
from rese.phase3.mcts_search import MCTSSearch, MCTSConfig, MCTSState

# Create state
state = MyMCTSState()

# Configure MCTS
config = MCTSConfig(
    max_iterations=1000,
    exploration_constant=1.41,
    progressive_widening=True,
    aci_guided=True
)

# Create searcher
mcts = MCTSSearch(config)

# Run search
best_node, info = mcts.search(
    initial_state=state,
    action_generator=lambda s: get_actions(s),
    state_transition=lambda s, a: apply_action(s, a),
    value_function=lambda s: evaluate(s)
)

print(f"Best value: {info['best_value']}")
print(f"Tree size: {info['tree_size']}")
```

---

### 2. Statistical Validator Module (`statistical_validator.py`)

#### Core Classes

**`StatisticalValidator`**
```python
class StatisticalValidator:
    def __init__(self, config: ValidationConfig)

    def bootstrap_confidence_interval(self, data, method=CIType.BCA) -> ConfidenceInterval
    def significance_test(self, results_a, results_b) -> SignificanceTestResult
    def detect_convergence(self, value_history) -> ConvergenceResult
    def required_sample_size(self, effect_size, alpha, power) -> int
```

**`ValidationResult`**
```python
@dataclass
class ValidationResult:
    confidence_interval: ConfidenceInterval
    convergence: ConvergenceResult
    sample_size: Optional[int]
    significance: Optional[SignificanceTestResult]
    diagnostics: Dict
```

#### Usage

```python
from rese.phase3.statistical_validator import StatisticalValidator, CIType

# Create validator
validator = StatisticalValidator()

# Calculate confidence interval
ci = validator.bootstrap_confidence_interval(
    data=my_results,
    method=CIType.BCA,
    confidence_level=0.95
)
print(f"95% CI: {ci}")

# Test convergence
conv_result = validator.detect_convergence(value_history)
print(f"Converged: {conv_result.converged}")

# Complete validation
result = validator.validate_mcts_results(
    results=results,
    value_history=history
)
print(result.summary())
```

---

### 3. Stage 3 Integration (`stage3_integration.py`)

#### Core Classes

**`MonteCarloNest`**
```python
class MonteCarloNest:
    def __init__(self, config: NestConfig, aci_analyzer: ACIAnalyzer)

    def search(self, initial_state, action_generator,
               state_transition, value_function) -> NestResult
```

**`NestResult`**
```python
@dataclass
class NestResult:
    best_agent_result: AgentResult
    all_agent_results: List[AgentResult]
    aggregated_value: float
    confidence: float
    elapsed_time: float
    converged: bool
```

#### Usage

```python
from rese.phase3.stage3_integration import MonteCarloNest, NestConfig

# Configure nest
config = NestConfig(
    num_agents=4,
    mcts_iterations=500,
    validate_results=True,
    parallel_agents=True
)

# Create nest
nest = MonteCarloNest(config)

# Run search
result = nest.search(
    initial_state=state,
    action_generator=lambda s: get_actions(s),
    state_transition=lambda s, a: apply_action(s, a),
    value_function=lambda s: evaluate(s)
)

print(result.summary())
```

---

## Usage Examples

### Example 1: Simple Optimization

```python
from rese.phase3.mcts_search import quick_mcts_search

# Define problem
class OptimizationState(MCTSState):
    def __init__(self, value=0):
        self.value = value

    def is_terminal(self):
        return abs(self.value) > 10

# Run MCTS
best_node, info = quick_mcts_search(
    initial_state=OptimizationState(0),
    action_generator=lambda s: ['+1', '-1'],
    state_transition=lambda s, a: OptimizationState(s.value + (1 if a == '+1' else -1)),
    value_function=lambda s: -abs(s.value),  # Minimize absolute value
    max_iterations=1000
)

print(f"Best value: {best_node.state.value}")
```

### Example 2: Constraint Satisfaction

```python
from rese.phase3.mcts_search import MCTSSearch, MCTSConfig

# CSP state
class CSPState(MCTSState):
    def __init__(self, assignments, domains):
        self.assignments = assignments
        self.domains = domains

    def is_terminal(self):
        return len(self.assignments) == len(self.domains)

    def is_solution(self):
        return self.is_terminal() and self.check_constraints()

# Action: assign value to variable
def actions(state):
    unassigned = [v for v in state.domains if v not in state.assignments]
    if not unassigned:
        return []
    var = unassigned[0]
    return [(var, val) for val in state.domains[var]]

# Transition
def transition(state, action):
    var, val = action
    new_assignments = state.assignments.copy()
    new_assignments[var] = val
    return CSPState(new_assignments, state.domains)

# Value: prefer solutions
def value_function(state):
    if state.is_solution():
        return 1.0
    elif state.is_terminal():
        return 0.0
    else:
        return 0.5

# Search
mcts = MCTSSearch(MCTSConfig(max_iterations=5000))
best_node, info = mcts.search(
    initial_state=CSPState({}, variable_domains),
    action_generator=actions,
    state_transition=transition,
    value_function=value_function
)
```

### Example 3: Monte Carlo Nest (Multi-Agent)

```python
from rese.phase3.stage3_integration import MonteCarloNest, NestConfig

config = NestConfig(
    num_agents=4,
    mcts_iterations=1000,
    validate_results=True
)

nest = MonteCarloNest(config)

result = nest.search(
    initial_state=initial_state,
    action_generator=actions,
    state_transition=transition,
    value_function=evaluate
)

print(f"Best strategy: {result.best_agent_result.strategy}")
print(f"Best value: {result.aggregated_value}")
print(f"Confidence: {result.confidence:.2f}")
```

---

## Integration with RESE

### With Γ₁ (ACI Analyzer)

```python
from rese.phase3.aci_analyzer import ACIAnalyzer
from rese.phase3.mcts_search import MCTSSearch

# Create ACI analyzer (Γ₁)
aci = ACIAnalyzer()

# Calculate ACI for initial state
aci_result = aci.calculate(initial_state)
print(f"ACI: {aci_result['ACI']:.3f}")

# Pass to MCTS (Γ₂)
mcts = MCTSSearch(aci_analyzer=aci)

# MCTS will use ACI to:
# - Adjust exploration parameter C
# - Select playout strategy
# - Determine simulation depth
# - Early stopping for low ACI

best_node, info = mcts.search(
    initial_state,
    action_generator,
    state_transition,
    value_function,
    initial_aci=aci_result
)
```

### With E2E Stage 3 (Monte Carlo Nest)

```python
# In E2E invention engine Stage 3
from rese.phase3.stage3_integration import quick_nest_search

# Run Monte Carlo Nest
result = quick_nest_search(
    initial_state=invention_state,
    action_generator=lambda s: generate_refinements(s),
    state_transition=lambda s, a: apply_refinement(s, a),
    value_function=lambda s: evaluate_invention(s),
    num_agents=4,
    iterations_per_agent=1000
)

# Get best validated solution
if result.best_agent_result.is_confident:
    solution = result.best_agent_result.best_node.state
    print(f"Confident solution found with value: {result.aggregated_value:.4f}")
else:
    print("No confident solution found. Consider reformulation.")
```

---

## API Reference

### MCTSConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exploration_constant` | float | 1.41 | UCB exploration parameter C |
| `adaptive_c` | bool | True | Adjust C based on ACI |
| `progressive_widening` | bool | True | Enable progressive widening |
| `widening_constant` | float | 0.5 | Progressive widening exponent |
| `max_playout_depth` | int | 50 | Maximum simulation depth |
| `playout_strategy` | PlayoutStrategy | ADAPTIVE | Playout strategy |
| `max_iterations` | int | 1000 | Maximum MCTS iterations |
| `max_time_seconds` | float | 60.0 | Time limit |
| `num_workers` | int | 1 | Parallel workers |
| `aci_guided` | bool | True | Enable ACI guidance |
| `verbose` | bool | False | Enable logging |

### ValidationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_bootstrap` | int | 1000 | Bootstrap samples |
| `ci_type` | CIType | BCA | Confidence interval method |
| `confidence_level` | float | 0.95 | Confidence level |
| `convergence_method` | ConvergenceMethod | COMBINED | Convergence detection |
| `significance_level` | float | 0.05 | Alpha for tests |
| `test_type` | TestType | T_TEST | Statistical test |

### NestConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_agents` | int | 4 | Number of MCTS agents |
| `agent_strategies` | List[AgentStrategy] | [EXPLOIT, EXPLORE, BALANCED, ADAPTIVE] | Agent strategies |
| `mcts_iterations` | int | 500 | Iterations per agent |
| `aci_guided` | bool | True | Enable ACI guidance |
| `validate_results` | bool | True | Validate results |
| `parallel_agents` | bool | True | Parallel execution |

---

## Performance Tuning

### For Speed

1. **Reduce iterations**
   ```python
   config = MCTSConfig(max_iterations=500)  # Instead of 1000
   ```

2. **Use progressive widening**
   ```python
   config = MCTSConfig(progressive_widening=True, widening_constant=0.7)
   ```

3. **Limit playout depth**
   ```python
   config = MCTSConfig(max_playout_depth=25)
   ```

4. **Parallel agents**
   ```python
   config = NestConfig(num_agents=4, parallel_agents=True)
   ```

### For Quality

1. **Increase iterations**
   ```python
   config = MCTSConfig(max_iterations=5000)
   ```

2. **Enable ACI guidance**
   ```python
   config = MCTSConfig(aci_guided=True, adaptive_c=True)
   ```

3. **Use causally-guided playouts**
   ```python
   config = MCTSConfig(playout_strategy=PlayoutStrategy.CAUSALLY_GUIDED)
   ```

4. **Validate with BCa intervals**
   ```python
   config = ValidationConfig(ci_type=CIType.BCA)
   ```

### For Memory Efficiency

1. **Limit tree size**
   ```python
   config = MCTSConfig(
       max_iterations=1000,
       progressive_widening=True  # Controls tree growth
   )
   ```

2. **Shallow convergence window**
   ```python
   config = MCTSConfig(convergence_window=10)
   ```

---

## Troubleshooting

### Issue: Poor convergence

**Symptoms:** MCTS not finding good solutions, wide confidence intervals

**Solutions:**
1. Increase iterations
2. Enable ACI guidance
3. Check value function is correct
4. Try different playout strategies
5. Adjust exploration parameter

### Issue: Slow execution

**Symptoms:** Search takes too long

**Solutions:**
1. Reduce max_iterations
2. Enable parallel agents
3. Reduce playout depth
4. Enable progressive widening
5. Use time limit: `max_time_seconds=30`

### Issue: Out of memory

**Symptoms:** Memory usage grows too large

**Solutions:**
1. Enable progressive widening (controls tree size)
2. Reduce iterations
3. Reduce num_workers
4. Implement tree pruning (if needed)

### Issue: Validation fails

**Symptoms:** Statistical tests return errors

**Solutions:**
1. Ensure sufficient data points (N > 30)
2. Check for NaN/Inf values
3. Try non-parametric tests (Wilcoxon)
4. Verify value function returns valid floats

---

## Testing

### Run all tests

```bash
# Run all Phase 3 tests
pytest rese/tests/test_phase3/ -v

# Run specific test file
pytest rese/tests/test_phase3/test_mcts_search.py -v

# Run with coverage
pytest rese/tests/test_phase3/ --cov=rese.phase3 --cov-report=html
```

### Test results

Expected test coverage:
- `test_mcts_search.py`: 400+ tests for MCTS components
- `test_statistical_validator.py`: 300+ tests for validation
- `test_stage3_integration.py`: 50+ integration tests

All tests should pass with >80% coverage.

---

## Next Steps

1. **Implement Γ₁ (ACI Analyzer)** - Agent D1
2. **Integrate with Stage 3** - E2E integration
3. **Validate on real problems** - Test suites
4. **Performance benchmarking** - Optimize parameters

---

## Contact

**Module Author:** Agent D2 (Γ₂/Γ₃ Specialist)
**Project:** RESE Phase III - Monte Carlo Refinement
**Date:** 2025-12-31

For issues or questions, refer to:
- Research document: `rese/docs/gamma2_mcts_research.md`
- Task assignment: `MULTI_AGENT_RESE_TASK_ASSIGNMENT.md`
- Status tracker: `rese/AGENT_STATUS.md`
