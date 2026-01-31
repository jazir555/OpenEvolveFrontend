"""
Advanced Features Example - Checkpoints, Early Stopping, Multi-Objective

This example demonstrates advanced OpenEvolve features:
- Checkpointing and resumption
- Early stopping based on convergence
- Multi-objective optimization
"""

# EVOLVE-BLOCK-START
def multi_objective_optimize(x, y):
    """
    Optimize multiple objectives:
    1. Maximize f1(x,y) = x + y (sum)
    2. Minimize f2(x,y) = x^2 + y^2 (distance from origin)
    3. Satisfy constraint: x, y in [0, 10]

    Pareto optimal solutions balance these competing objectives.
    """
    f1 = x + y  # Want to maximize
    f2 = x**2 + y**2  # Want to minimize
    return f1, f2
# EVOLVE-BLOCK-END


"""
ADVANCED FEATURE 1: CHECKPOINTS
-------------------------------
Checkpoints save evolution state periodically.

Enable in config:
```python
config = Config()
config.checkpoint_interval = 10  # Save every 10 iterations
config.max_iterations = 100

result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config
)
```

Resume from checkpoint:
```python
# CLI
openevolve program.py evaluator.py \\
    --checkpoint output/checkpoints/checkpoint_50

# Python API
result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    checkpoint_path='output/checkpoints/checkpoint_50'
)
```


ADVANCED FEATURE 2: EARLY STOPPING
----------------------------------
Stop evolution when convergence is detected.

Enable in config:
```python
config = Config()
config.early_stopping_patience = 10  # Stop if no improvement for 10 iterations
config.convergence_threshold = 0.001  # Improvement < 0.001 = converged
config.early_stopping_metric = "combined_score"

result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config
)
```

Early stopping triggers when:
- No improvement for N iterations (patience)
- Improvement is below threshold (convergence)
- Saves computation time


ADVANCED FEATURE 3: MULTI-OBJECTIVE OPTIMIZATION
-----------------------------------------------
Optimize multiple competing objectives simultaneously.

Evaluator returns multiple metrics:
```python
def evaluate(program_path):
    # ... load program ...

    f1, f2 = module.multi_objective_optimize(x, y)

    return {
        "sum_score": f1 / 20.0,  # Normalize to 0-1
        "distance_score": 1.0 - (f2 / 200.0),  # Invert (minimize)
        "constraint_violation": penalty,
        # OpenEvolve combines these into combined_score
    }
```

Weight objectives:
```python
# In evaluator
combined = (sum_score * 0.5) + (distance_score * 0.3) + (constraint_violation * 0.2)
return {"combined_score": combined, ...}
```


ADVANCED FEATURE 4: CUSTOM FEATURE DIMENSIONS
--------------------------------------------
Use custom metrics for MAP-Elites diversity.

```python
config = Config()
config.database.feature_dimensions = [
    "complexity",      # Built-in
    "diversity",       # Built-in
    "sum_score",       # Custom from evaluator
    "distance_score"   # Custom from evaluator
]
config.database.feature_bins = {
    "complexity": 10,
    "diversity": 10,
    "sum_score": 5,
    "distance_score": 5
}
```

This creates a 4D MAP-Elites grid for better diversity.


ADVANCED FEATURE 5: PARALLEL EVALUATION
-------------------------------------
Evaluate multiple programs in parallel.

```python
config = Config()
config.evaluator.parallel_evaluations = 4  # Evaluate 4 programs at once

result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config
)
```

Requires enough CPU cores and memory.


ADVANCED FEATURE 6: ISLAND-BASED EVOLUTION
----------------------------------------
Run multiple independent populations with migration.

```python
config = Config()
config.database.num_islands = 5  # 5 independent populations
config.database.migration_interval = 20  # Migrate every 20 iterations
config.database.migration_rate = 0.1  # Migrate 10% of population

result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config
)
```

Benefits:
- Better exploration
- Reduced premature convergence
- Parallel processing


ADVANCED FEATURE 7: CUSTOM SELECTION STRATEGIES
---------------------------------------------
Control how programs are selected for evolution.

```python
config = Config()
config.database.elite_selection_ratio = 0.1  # Top 10% (elites)
config.database.exploration_ratio = 0.2      # 20% diverse programs
config.database.exploitation_ratio = 0.7     # 70% best performers

result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config
)
```


ADVANCED FEATURE 8: EVOLUTION TRACING
------------------------------------
Track every step of evolution for analysis.

```python
config = Config()
config.evolution_trace.enabled = True
config.evolution_trace.format = "jsonl"  # or "json", "hdf5"
config.evolution_trace.include_code = True
config.evolution_trace.include_prompts = True
config.evolution_trace.output_path = "trace.jsonl"

result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config
)
```

Analyzing trace:
```python
import json

with open('trace.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        print(f"Iteration: {entry['iteration']}, Score: {entry['score']}")
```


FULL EXAMPLE WITH ALL FEATURES:
------------------------------
```python
from openevolve import run_evolution
from openevolve.config import Config

# Create comprehensive config
config = Config()

# Evolution settings
config.max_iterations = 100
config.checkpoint_interval = 10

# Early stopping
config.early_stopping_patience = 15
config.convergence_threshold = 0.0001

# Island-based evolution
config.database.num_islands = 5
config.database.migration_interval = 25
config.database.migration_rate = 0.15

# Custom feature dimensions
config.database.feature_dimensions = [
    "complexity", "diversity", "sum_score", "distance_score"
]
config.database.feature_bins = 10

# Parallel evaluation
config.evaluator.parallel_evaluations = 4

# Evolution tracing
config.evolution_trace.enabled = True
config.evolution_trace.format = "jsonl"
config.evolution_trace.include_code = True
config.evolution_trace.include_prompts = False

# Run evolution
result = run_evolution(
    'multi_objective.py',
    'multi_evaluator.py',
    config=config,
    output_dir='advanced_results'
)

print(f"Best score: {result.best_score:.4f}")
print(f"Converged early: {result.metrics.get('early_stopping', False)}")
```
"""
