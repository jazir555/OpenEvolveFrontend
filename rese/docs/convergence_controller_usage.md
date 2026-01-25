# Convergence Controller Usage Guide

**Agent:** D3 (N_max Specialist)
**Date:** 2025-12-31
**Status:** Complete
**Module:** `rese.phase3.convergence_controller`

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Guide](#configuration-guide)
4. [Integration Examples](#integration-examples)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Convergence Controller provides intelligent, adaptive stopping for Monte Carlo Tree Search (MCTS) in the RESE framework. It integrates with Γ₁ (ACI calculation) and Stage 9 (E2E validation) to optimize computational resources while ensuring solution quality.

### Key Features

- **Multiple Convergence Detectors**: ACI stability, solution stability, variance, gradient, Gelman-Rubin
- **Dynamic N_max Adjustment**: Adapts to search progress in real-time
- **Early Stopping**: Identifies intractable problems early
- **Γ₁ Integration**: Uses ACI to predict convergence
- **Stage 9 Reporting**: Provides validation feedback
- **Composite Stopping Rules**: ANY, ALL, MAJORITY, WEIGHTED strategies

### Architecture

```
ConvergenceController
├── Detectors
│   ├── ACIStabilityDetector
│   ├── SolutionStabilityDetector
│   ├── VarianceDetector
│   ├── GradientDetector
│   └── GelmanRubinDetector
├── NMaxEstimator
│   ├── ACI-based estimation
│   ├── Structural estimation
│   └── Dynamic adjustment
├── EarlyStoppingRule
└── Stage9Reporter
```

---

## Quick Start

### Basic Usage

```python
from rese.phase3.convergence_controller import (
    create_convergence_controller,
    SearchState
)
import time

# 1. Create controller
controller = create_convergence_controller(
    use_aci=True,
    use_dynamic_adjustment=True,
    verbose=True
)

# 2. Get initial N_max
n_max = controller.get_n_max(problem_size=100)
print(f"Estimated N_max: {n_max}")

# 3. Initialize search state
search_state = SearchState(
    start_time=time.time(),
    n_max=n_max
)

# 4. Run MCTS with convergence control
for iteration in range(1, n_max + 1):
    # ... Run MCTS iteration ...
    value = run_mcts_iteration()  # Your MCTS code here
    aci = compute_aci()  # Optional: Compute ACI

    # Update search state
    search_state.update(iteration, value, aci)

    # Check if should stop
    should_stop, reason = controller.should_stop(search_state)

    # Adjust N_max if needed
    new_n_max, adj_reason = controller.adjust_n_max(search_state)
    if new_n_max != search_state.n_max:
        search_state.n_max = new_n_max
        print(f"Adjusted N_max: {new_n_max} ({adj_reason})")

    # Stop if converged
    if should_stop:
        print(f"Converged: {reason}")
        break

print(f"Final value: {search_state.current_value}")
print(f"Iterations: {search_state.iteration}")
```

### Minimal Example

```python
from rese.phase3.convergence_controller import create_convergence_controller

# Create with defaults
controller = create_convergence_controller()

# Get N_max for problem
n_max = controller.get_n_max(problem_size=50)

# Use in MCTS loop
# ... (see full example above)
```

---

## Configuration Guide

### Default Configuration

```python
from rese.phase3.convergence_controller import ConvergenceConfig

config = ConvergenceConfig()
```

Default values are suitable for most use cases:
- Variance threshold: 0.001
- Convergence window: 20
- Stopping strategy: MAJORITY
- Dynamic adjustment: enabled
- Early stopping: enabled

### Fast Configuration (Development)

```python
config = ConvergenceConfig(
    # Aggressive thresholds for faster convergence
    variance_threshold=0.01,
    convergence_window=10,
    min_n_max=50,
    max_n_max=500,

    # Fast stopping
    stopping_strategy='ANY',
    enable_early_stopping=True,
    min_iterations_before_stop=10,

    # Minimal computation
    use_gelman_rubin=False,
    aci_computation_interval=50
)
```

### Thorough Configuration (Production)

```python
config = ConvergenceConfig(
    # Strict thresholds
    variance_threshold=0.0001,
    convergence_window=50,
    min_n_max=500,
    max_n_max=10000,

    # Require all detectors
    stopping_strategy='ALL',
    enable_early_stopping=False,
    min_iterations_before_stop=200,

    # Full computation
    use_gelman_rubin=True,
    aci_computation_interval=10
)
```

### Configuration Parameters

#### Detection Methods

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_aci_stability` | bool | True | Enable ACI stability detector |
| `use_solution_stability` | bool | True | Enable solution stability detector |
| `use_variance` | bool | True | Enable variance-based detector |
| `use_gradient` | bool | True | Enable gradient-based detector |
| `use_gelman_rubin` | bool | False | Enable Gelman-Rubin detector (expensive) |

#### Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variance_threshold` | float | 0.001 | Variance convergence threshold |
| `gradient_threshold` | float | 0.001 | Gradient convergence threshold |
| `aci_variance_threshold` | float | 0.01 | ACI variance threshold |
| `r_hat_threshold` | float | 1.1 | Gelman-Rubin R-hat threshold |

#### Window Sizes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `convergence_window` | int | 20 | Window for convergence detectors |
| `stability_window` | int | 50 | Window for stability detectors |
| `aci_window` | int | 30 | Window for ACI trajectory |

#### N_max Estimation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_n_max` | int | 1000 | Base number of iterations |
| `min_n_max` | int | 100 | Minimum iterations |
| `max_n_max` | int | 10000 | Maximum iterations |
| `aci_weight_n_max` | float | 0.7 | Weight for ACI vs structure |
| `use_dynamic_adjustment` | bool | True | Enable dynamic adjustment |

#### Early Stopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_early_stopping` | bool | True | Enable early stopping |
| `low_aci_threshold` | float | 0.3 | ACI threshold for early stop |
| `no_improvement_iterations` | int | 100 | Iterations without improvement |
| `diminishing_returns_threshold` | float | 0.001 | Improvement rate threshold |

#### Stopping Strategy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stopping_strategy` | str | 'MAJORITY' | ANY, ALL, MAJORITY, or WEIGHTED |
| `min_iterations_before_stop` | int | 50 | Minimum iterations before stop |

---

## Integration Examples

### Integration with Γ₁ (ACI Calculator)

```python
from rese.gamma1.core.aci_calculator import ACICalculator
from rese.gamma1.core.csp_models import CSPInstance
from rese.phase3.convergence_controller import ConvergenceController

# Create ACI calculator
aci_calculator = ACICalculator(alpha=0.35, beta=0.35, gamma=0.30)

# Create CSP instance
csp = CSPInstance(...)  # Your CSP here

# Calculate initial ACI
aci_result = aci_calculator.calculate(csp)

# Create controller with ACI calculator
controller = ConvergenceController(
    config=ConvergenceConfig(),
    aci_calculator=aci_calculator
)

# Get N_max based on ACI
n_max = controller.get_n_max(csp=csp, aci_result=aci_result)

# During search, compute ACI periodically
for iteration in range(1, n_max + 1):
    # ... MCTS iteration ...

    # Compute ACI every N iterations
    if iteration % config.aci_computation_interval == 0:
        current_csp = get_partial_assignment()  # Get current state
        current_aci = aci_calculator.calculate(current_csp)
    else:
        current_aci = None

    # Update search state
    search_state.update(iteration, value, current_aci)

    # Check convergence (uses ACI stability detector)
    should_stop, reason = controller.should_stop(search_state)
```

### Integration with Stage 9 (E2E Validation)

```python
from rese.phase3.convergence_controller import Stage9Reporter

# Create Stage 9 reporter
stage9_reporter = Stage9Reporter(stage9_validator=your_stage9_validator)

# Create controller with Stage 9 reporter
controller = ConvergenceController(
    config=ConvergenceConfig(report_to_stage9=True),
    stage9_reporter=stage9_reporter
)

# During search, reports are sent automatically
# Get all reports after search
reports = stage9_reporter.get_reports()

# Analyze reports
for report in reports:
    if report['type'] == 'convergence_check':
        print(f"Iteration {report['iteration']}: {report['decision']}")
    elif report['type'] == 'n_max_adjustment':
        print(f"Iteration {report['iteration']}: N_max {report['old_n_max']} -> {report['new_n_max']}")
```

### Integration with MCTS

```python
from rese.phase3.mcts_search import MCTSSearch, MCTSConfig, MCTSState
from rese.phase3.convergence_controller import ConvergenceController

# Create MCTS config
mcts_config = MCTSConfig(
    max_iterations=10000,  # Will be overridden by convergence control
    convergence_window=20,
    verbose=False
)

# Create convergence controller
controller = create_convergence_controller(verbose=True)

# Get initial N_max
n_max = controller.get_n_max(problem_size=100)

# Update MCTS config
mcts_config.max_iterations = n_max

# Create MCTS
mcts = MCTSSearch(config=mcts_config)

# Create initial state
initial_state = MCTSState(...)

# Run MCTS with convergence control
search_state = SearchState(start_time=time.time(), n_max=n_max)

for iteration in range(1, n_max + 1):
    # Run MCTS iteration
    # ... (your MCTS logic here) ...

    value = mcts.best_value  # Get best value so far
    aci = compute_aci_if_needed()  # Optional

    # Update convergence state
    search_state.update(iteration, value, aci)

    # Check convergence
    should_stop, reason = controller.should_stop(search_state)
    if should_stop:
        print(f"Stopping: {reason}")
        break

    # Adjust N_max
    new_n_max, adj_reason = controller.adjust_n_max(search_state)
    if new_n_max != search_state.n_max:
        search_state.n_max = new_n_max
        mcts_config.max_iterations = new_n_max

print(f"MCTS completed in {search_state.iteration} iterations")
```

### Parallel MCTS with Gelman-Rubin

```python
from rese.phase3.convergence_controller import ConvergenceController, GelmanRubinDetector

# Create controller with Gelman-Rubin enabled
config = ConvergenceConfig(
    use_gelman_rubin=True,
    stopping_strategy='ALL'  # Require all detectors including Gelman-Rubin
)
controller = ConvergenceController(config)

# Run parallel MCTS chains
num_chains = 4
chain_histories = [[] for _ in range(num_chains)]

for chain_id in range(num_chains):
    # Run independent MCTS chain
    for iteration in range(1, n_max + 1):
        value = run_mcts_chain(chain_id, iteration)
        chain_histories[chain_id].append(value)

# Add chains to Gelman-Rubin detector
gelman_rubin = controller.gelman_rubin_detector
for chain_history in chain_histories:
    gelman_rubin.add_chain(chain_history)

# Check convergence (includes Gelman-Rubin)
result = gelman_rubin.detect(search_state)
if result.converged:
    print(f"R-hat = {result.details['r_hat']:.3f} < {config.r_hat_threshold}")
```

---

## Advanced Usage

### Custom Detector

```python
from rese.phase3.convergence_controller import ConvergenceDetector, ConvergenceResult

class CustomDetector(ConvergenceDetector):
    """Custom convergence detector"""

    def detect(self, search_state):
        # Your custom logic here
        converged = your_custom_logic(search_state)

        return ConvergenceResult(
            converged=converged,
            method=type(self),
            iteration=search_state.iteration,
            confidence=your_confidence,
            details={'custom_metric': your_metric}
        )

# Add to controller
controller = create_convergence_controller()
controller.detectors.append(CustomDetector(controller.config))
```

### Custom N_max Estimation

```python
from rese.phase3.convergence_controller import NMaxEstimator

class CustomNMaxEstimator(NMaxEstimator):
    """Custom N_max estimator"""

    def estimate_initial(self, csp=None, aci_result=None, problem_size=0):
        # Your custom logic
        n_max = your_custom_estimation(csp, aci_result, problem_size)

        # Apply bounds
        return int(np.clip(n_max,
                          self.config.min_n_max,
                          self.config.max_n_max))

# Use custom estimator
controller = create_convergence_controller()
controller.n_max_estimator = CustomNMaxEstimator(controller.config)
```

### Custom Stopping Strategy

```python
# Modify stopping combination logic
def custom_combine(controller, detector_results):
    """Custom stopping combination"""
    # Your custom logic
    converged = your_custom_combination(detector_results)
    reason = "Custom decision"

    return converged, reason

# Replace combination method
controller._combine_results = lambda results: custom_combine(controller, results)
```

---

## API Reference

### ConvergenceController

Main controller for convergence detection and N_max adjustment.

#### Methods

##### `__init__(config, aci_calculator, stage9_reporter)`

Initialize controller.

**Parameters:**
- `config` (ConvergenceConfig): Configuration
- `aci_calculator` (ACICalculator): Optional Γ₁ ACI calculator
- `stage9_reporter` (Stage9Reporter): Optional Stage 9 reporter

##### `get_n_max(csp=None, aci_result=None, problem_size=0) -> int`

Estimate initial N_max.

**Parameters:**
- `csp` (CSPInstance): Optional CSP instance
- `aci_result` (ACIResult): Optional ACI result
- `problem_size` (int): Problem size (number of variables)

**Returns:**
- `int`: Estimated N_max

##### `should_stop(search_state) -> Tuple[bool, str]`

Check if search should stop.

**Parameters:**
- `search_state` (SearchState): Current search state

**Returns:**
- `Tuple[bool, str]`: (should_stop, reason)

##### `adjust_n_max(search_state) -> Tuple[int, str]`

Adjust N_max dynamically.

**Parameters:**
- `search_state` (SearchState): Current search state

**Returns:**
- `Tuple[int, str]`: (new_n_max, reason)

### SearchState

Maintains search state for convergence monitoring.

#### Attributes

- `iteration` (int): Current iteration
- `value_history` (List[float]): History of best values
- `current_value` (float): Current best value
- `aci_history` (List[float]): History of ACI scores
- `current_aci` (float): Current ACI score
- `best_solution` (Any): Best solution found
- `last_improvement_iteration` (int): Iteration of last improvement
- `start_time` (float): Search start time
- `elapsed_time` (float): Elapsed time
- `n_max` (int): Current maximum iterations

#### Methods

##### `update(iteration, value, aci=None)`

Update search state.

**Parameters:**
- `iteration` (int): Current iteration
- `value` (float): Current value
- `aci` (float, optional): Current ACI

### ConvergenceConfig

Configuration for convergence control.

See [Configuration Guide](#configuration-guide) for all parameters.

### Convenience Functions

##### `create_convergence_controller(use_aci=True, use_dynamic_adjustment=True, verbose=False) -> ConvergenceController`

Create controller with sensible defaults.

---

## Best Practices

### 1. Choose Appropriate Configuration

**Development/Fast Iteration:**
```python
config = ConvergenceConfig(
    variance_threshold=0.01,
    stopping_strategy='ANY',
    min_iterations_before_stop=10
)
```

**Production/Quality:**
```python
config = ConvergenceConfig(
    variance_threshold=0.001,
    stopping_strategy='MAJORITY',
    min_iterations_before_stop=50
)
```

**Validation/Rigorous:**
```python
config = ConvergenceConfig(
    variance_threshold=0.0001,
    stopping_strategy='ALL',
    min_iterations_before_stop=200,
    use_gelman_rubin=True
)
```

### 2. Use ACI Integration

Always integrate with Γ₁ for best results:

```python
controller = ConvergenceController(
    config=config,
    aci_calculator=aci_calculator  # From Γ₁
)
```

### 3. Monitor Search State

Track search state for debugging:

```python
for iteration in range(1, n_max + 1):
    search_state.update(iteration, value, aci)

    # Log progress
    if iteration % 20 == 0:
        print(f"Iter {iteration}: value={value:.4f}, "
              f"ACI={aci:.3f}, N_max={search_state.n_max}")
```

### 4. Adjust Check Intervals

Balance responsiveness vs. overhead:

```python
config = ConvergenceConfig(
    check_interval=20,      # Check convergence every 20 iterations
    adjust_interval=50,     # Adjust N_max every 50 iterations
    aci_computation_interval=10  # Compute ACI every 10 iterations
)
```

### 5. Handle Edge Cases

```python
# Always check minimum iterations
if search_state.iteration >= config.min_iterations_before_stop:
    should_stop, reason = controller.should_stop(search_state)
    if should_stop:
        break

# Always respect max iterations
if search_state.iteration >= search_state.n_max:
    print("Maximum iterations reached")
    break
```

### 6. Validate Results

After search, validate convergence:

```python
# Get statistics
stats = controller.get_statistics()
print(f"Total checks: {stats['total_checks']}")
print(f"Total adjustments: {stats['total_adjustments']}")

# Check final state
print(f"Final iteration: {search_state.iteration}/{search_state.n_max}")
print(f"Converged: {should_stop}")
print(f"Reason: {reason}")
```

---

## Troubleshooting

### Issue: Never Stops

**Symptoms:** Search runs until max_iterations without converging.

**Possible Causes:**
1. Thresholds too strict
2. Problem genuinely not converging
3. Detectors not appropriate for problem type

**Solutions:**
```python
# Loosen thresholds
config = ConvergenceConfig(
    variance_threshold=0.01,  # Was 0.001
    gradient_threshold=0.01,  # Was 0.001
    stopping_strategy='ANY'   # Stop if any detector converges
)

# Or enable early stopping
config = ConvergenceConfig(
    enable_early_stopping=True,
    low_aci_threshold=0.4,
    no_improvement_iterations=50
)
```

### Issue: Stops Too Early

**Symptoms:** Search stops before finding good solution.

**Possible Causes:**
1. Thresholds too loose
2. Min_iterations too low
3. Early stopping too aggressive

**Solutions:**
```python
# Tighten thresholds
config = ConvergenceConfig(
    variance_threshold=0.0001,  # Was 0.001
    stopping_strategy='ALL',    # Require all detectors
    min_iterations_before_stop=100  # Was 50
)

# Or disable early stopping
config = ConvergenceConfig(
    enable_early_stopping=False
)
```

### Issue: N_max Too Small

**Symptoms:** Search stops before exploring enough.

**Possible Causes:**
1. Underestimation from ACI
2. Base N_max too low
3. Dynamic adjustment reducing too much

**Solutions:**
```python
# Increase base N_max
config = ConvergenceConfig(
    base_n_max=2000,  # Was 1000
    max_n_max=20000   # Was 10000
)

# Or reduce ACI weight
config = ConvergenceConfig(
    aci_weight_n_max=0.5  # Was 0.7 (more structural, less ACI)
)

# Or disable dynamic adjustment
config = ConvergenceConfig(
    use_dynamic_adjustment=False
)
```

### Issue: High Computational Cost

**Symptoms:** Convergence checking takes too long.

**Possible Causes:**
1. Computing ACI too frequently
2. Using Gelman-Rubin (expensive)
3. Check intervals too small

**Solutions:**
```python
# Reduce computation frequency
config = ConvergenceConfig(
    aci_computation_interval=50,  # Was 10
    check_interval=50,            # Was 20
    use_gelman_rubin=False        # Disable expensive detector
)
```

### Issue: ACI Computation Fails

**Symptoms:** Errors when computing ACI during search.

**Possible Causes:**
1. CSP state invalid for ACI computation
2. ACI calculator not initialized
3. Missing dependencies

**Solutions:**
```python
# Safe ACI computation with fallback
def safe_compute_aci(state, aci_calculator):
    try:
        if aci_calculator is not None:
            return aci_calculator.calculate(state).ACI
    except Exception as e:
        print(f"ACI computation failed: {e}")
    return None  # Fallback

# Use in search loop
aci = safe_compute_aci(current_state, aci_calculator)
search_state.update(iteration, value, aci)
```

---

## Performance Tips

1. **Use appropriate stopping strategy:**
   - `ANY`: Fastest, but may stop prematurely
   - `MAJORITY`: Good balance (recommended)
   - `ALL`: Slowest, but most thorough

2. **Adjust check intervals:**
   - Small intervals (10-20): More responsive, more overhead
   - Large intervals (50-100): Less overhead, less responsive

3. **Enable early stopping for intractable problems:**
   - Saves time on hopeless problems
   - Set `low_aci_threshold` appropriately (0.3-0.4)

4. **Use dynamic adjustment:**
   - Reduces iterations for easy problems
   - Increases iterations for hard problems
   - Saves 20-50% computation on average

5. **Compute ACI periodically:**
   - Every 10 iterations: Detailed monitoring
   - Every 20-50 iterations: Good balance
   - Every 100+ iterations: Minimal overhead

---

## Summary

The Convergence Controller provides intelligent, adaptive stopping for MCTS search in the RESE framework. By integrating with Γ₁ (ACI calculation) and Stage 9 (E2E validation), it optimizes computational resources while ensuring solution quality.

**Key Takeaways:**
1. Start with default configuration for most use cases
2. Integrate with Γ₁ ACI calculator for best results
3. Choose appropriate stopping strategy (ANY, MAJORITY, ALL)
4. Enable early stopping for intractable problems
5. Adjust thresholds based on problem characteristics
6. Monitor search state for debugging and validation

For more details, see:
- Research document: `rese/docs/convergence_control_research.md`
- Module implementation: `rese/phase3/convergence_controller.py`
- Unit tests: `rese/phase3/tests/test_convergence_controller.py`
