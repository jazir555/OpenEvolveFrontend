# ML-Based Gauntlet Optimizer - User Guide

Complete guide for using the ML-Based Gauntlet Optimizer component.

## Overview

The ML-Based Gauntlet Optimizer uses machine learning to automatically discover optimal gauntlet configurations for different problem domains and objectives.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Optimization Strategies](#optimization-strategies)
3. [Objectives](#objectives)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

---

## Quick Start

### Basic Usage

```python
from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
    MLBasedGauntletOptimizer,
    OptimizationStrategy,
    OptimizationObjective
)

# Create optimizer
optimizer = MLBasedGauntletOptimizer(
    strategy=OptimizationStrategy.Q_LEARNING,
    max_iterations=100
)

# Optimize for balanced performance
result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED
)

# Get optimal configuration
print(f"Best configuration: {result.best_state.to_dict()}")
print(f"Improvement: {result.improvement_percent:.1f}%")
print(f"Recommendation: {result.recommendation}")
```

### Using Factory Function

```python
from glue.adapters.gauntlet_adapter.src.ml_optimizer import create_optimizer

optimizer = create_optimizer(
    strategy="q_learning",  # or "dqn", "genetic", "bayesian"
    learning_rate=0.1,
    max_iterations=100
)
```

---

## Optimization Strategies

### Q-Learning (Default)

Model-free reinforcement learning that learns optimal actions through exploration and exploitation.

**Best for:**
- Problems with unknown dynamics
- Sequential decision making
- Medium-sized state spaces

```python
optimizer = MLBasedGauntletOptimizer(
    strategy=OptimizationStrategy.Q_LEARNING,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.1
)
```

### DQN (Deep Q-Network)

Uses neural networks to approximate Q-values for larger state spaces.

**Best for:**
- Large state spaces
- Complex problems
- When you have lots of training data

```python
optimizer = MLBasedGauntletOptimizer(
    strategy=OptimizationStrategy.DQN,
    learning_rate=0.001,
    max_iterations=200
)
```

### Genetic Algorithm

Evolutionary approach that evolves a population of configurations.

**Best for:**
- Global optimization
- Problems with many local optima
- When gradient information is unavailable

```python
optimizer = MLBasedGauntletOptimizer(
    strategy=OptimizationStrategy.GENETIC_ALGORITHM,
    max_iterations=100
)
```

### Bayesian Optimization

Probabilistic model-based optimization using Gaussian processes.

**Best for:**
- Expensive-to-evaluate functions
- Low-dimensional problems
- When you need sample efficiency

```python
optimizer = MLBasedGauntletOptimizer(
    strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
    max_iterations=50
)
```

---

## Objectives

### MAXIMIZE_ACCURACY

Prioritize solution correctness over speed or cost.

```python
result = optimizer.optimize(
    domain="finance",
    objective=OptimizationObjective.MAXIMIZE_ACCURACY
)
```

### MINIMIZE_TIME

Prioritize fast execution over accuracy.

```python
result = optimizer.optimize(
    domain="web",
    objective=OptimizationObjective.MINIMIZE_TIME
)
```

### MINIMIZE_COST

Prioritize computational efficiency.

```python
result = optimizer.optimize(
    domain="general",
    objective=OptimizationObjective.MINIMIZE_COST
)
```

### BALANCED (Default)

Balance accuracy, speed, and cost.

```python
result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED
)
```

---

## API Reference

### MLBasedGauntletOptimizer

#### Constructor

```python
MLBasedGauntletOptimizer(
    strategy: OptimizationStrategy = OptimizationStrategy.Q_LEARNING,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon: float = 0.1,
    max_iterations: int = 100
)
```

**Parameters:**
- `strategy`: Optimization algorithm to use
- `learning_rate`: How much to update Q-values per step (0.0-1.0)
- `discount_factor`: How much to value future rewards (0.0-1.0)
- `epsilon`: Initial exploration rate (0.0-1.0)
- `max_iterations`: Maximum optimization iterations

#### Methods

##### optimize()

```python
def optimize(
    self,
    domain: str,
    objective: OptimizationObjective,
    historical_data: Optional[List[Dict[str, Any]]] = None,
    initial_state: Optional[GauntletState] = None
) -> OptimizationResult
```

Optimize gauntlet configuration for given domain and objective.

---

## Examples

### Example 1: Domain-Specific Optimization

```python
from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
    MLBasedGauntletOptimizer,
    OptimizationObjective
)

optimizer = MLBasedGauntletOptimizer()

# Optimize for different domains
domains = ["code", "math", "finance", "science"]

for domain in domains:
    result = optimizer.optimize(
        domain=domain,
        objective=OptimizationObjective.BALANCED
    )

    print(f"{domain}: {result.best_state.to_dict()}")
    print(f"  Improvement: {result.improvement_percent:.1f}%")
```

### Example 2: Training from Historical Data

```python
# Load historical execution data
historical_data = [
    {
        "score": 0.75,
        "time": 30.0,
        "config": {"round1_threshold": 0.5},
        "passed": True
    },
    {
        "score": 0.82,
        "time": 35.0,
        "config": {"round1_threshold": 0.6},
        "passed": True
    },
    # ... more historical data
]

# Optimize using historical data
result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.MAXIMIZE_ACCURACY,
    historical_data=historical_data
)
```

### Example 3: Multi-Objective Optimization

```python
# Optimize for different objectives
objectives = [
    OptimizationObjective.MAXIMIZE_ACCURACY,
    OptimizationObjective.MINIMIZE_TIME,
    OptimizationObjective.MINIMIZE_COST,
    OptimizationObjective.BALANCED
]

results = {}
for obj in objectives:
    result = optimizer.optimize(
        domain="code",
        objective=obj
    )
    results[obj.value] = result

# Compare results
for obj_name, result in results.items():
    print(f"{obj_name}:")
    print(f"  Score: {result.best_score:.3f}")
    print(f"  Config: {result.best_state.to_dict()}")
```

### Example 4: Progressive Optimization

```python
# Start with quick optimization
quick_result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED,
    max_iterations=20
)

# Use result as starting point for deeper optimization
deep_result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED,
    initial_state=quick_result.best_state,
    max_iterations=200
)

print(f"Quick: {quick_result.best_score:.3f}")
print(f"Deep: {deep_result.best_score:.3f}")
```

---

## Best Practices

### 1. Choose the Right Strategy

- **Q-Learning**: Good default for most problems
- **DQN**: Use for complex, high-dimensional problems
- **Genetic**: Use when you need global optimization
- **Bayesian**: Use when evaluation is expensive

### 2. Set Appropriate Iterations

```python
# Quick exploration
optimizer = MLBasedGauntletOptimizer(max_iterations=20)

# Standard optimization
optimizer = MLBasedGauntletOptimizer(max_iterations=100)

# Deep optimization
optimizer = MLBasedGauntletOptimizer(max_iterations=500)
```

### 3. Use Historical Data

```python
# Collect historical data from actual executions
historical_data = collect_execution_history()

# Use it to inform optimization
result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED,
    historical_data=historical_data
)
```

### 4. Validate Results

```python
result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED
)

# Test the optimized configuration
test_result = test_configuration(
    config=result.best_state,
    test_cases=test_suites
)

print(f"Optimized score: {result.best_score:.3f}")
print(f"Actual test score: {test_result.score:.3f}")
```

### 5. Monitor Convergence

```python
result = optimizer.optimize(
    domain="code",
    objective=OptimizationObjective.BALANCED
)

# Check convergence history
import matplotlib.pyplot as plt

plt.plot(result.convergence_history)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Optimization Convergence')
plt.show()
```

---

## Troubleshooting

### Optimization Not Converging

**Issue**: The optimization doesn't improve the score.

**Solutions**:
1. Increase `max_iterations`
2. Try a different `strategy`
3. Adjust `learning_rate` (try 0.01 for Q-learning)
4. Provide historical data to guide the search

### Slow Optimization

**Issue**: Optimization takes too long.

**Solutions**:
1. Reduce `max_iterations`
2. Use `OptimizationStrategy.BAYESIAN_OPTIMIZATION` for sample efficiency
3. Reduce state space complexity

### Overfitting to Domain

**Issue**: Optimized config only works for specific domain.

**Solutions**:
1. Train on multiple domains
2. Use `OptimizationObjective.BALANCED`
3. Increase exploration (`epsilon` parameter)

---

## Advanced Usage

### Custom Reward Function

```python
class CustomOptimizer(MLBasedGauntletOptimizer):
    def _evaluate_configuration(self, state, domain, objective):
        # Custom evaluation logic
        score = base_evaluation(state)

        # Add custom bonuses/penalties
        if state.enable_parallel:
            score += 0.1

        return score
```

### Continuous Learning

```python
# After each gauntlet execution
optimizer.learn_from_execution(
    state=current_state,
    action=action_taken,
    reward=result.score,
    next_state=new_state,
    done=execution_complete
)
```

---

## Support

For issues or questions:
- GitHub: https://github.com/openevolve/gauntlet-optimizer/issues
- Documentation: https://docs.openevolve.org/ml-optimizer
