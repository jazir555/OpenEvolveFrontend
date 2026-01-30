# Hybrid MCTS Framework - Quick Reference Card

## Installation

```bash
pip install numpy aiohttp pytest psutil
```

## Basic Usage

```python
from hybrid_mcts_framework import HybridMCTSEngine, HybridMCTSConfig, HybridMCTSApproach

# Create engine
config = HybridMCTSConfig(approach=HybridMCTSApproach.EVOLVED_POLICIES)
engine = HybridMCTSEngine(config)

# Search
result = await engine.search("theorem test (a : nat) : a + 0 = a")
print(f"Success: {result.success}, Fitness: {result.best_fitness}")
```

## Quick Commands

### Presets
```python
from hybrid_mcts_framework import create_framework_from_preset

engine = create_framework_from_preset("fast")      # Quick exploration
engine = create_framework_from_preset("balanced")  # Default
engine = create_framework_from_preset("thorough")  # Maximum quality
engine = create_framework_from_preset("leanaide")  # Verification focused
engine = create_framework_from_preset("research")  # Experimental
```

### Utility Functions
```python
from hybrid_mcts_framework import quick_search, thorough_search

result = await quick_search(theorem)       # Fast preset
result = await thorough_search(theorem)    # Thorough preset
```

## Three Approaches

### 1. Evolved Policies
```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLVED_POLICIES,
    policy_population_size=50,
    policy_generations=10,
)
# Best for: Medium complexity, structured problems
```

### 2. Evolutionary Nodes
```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLUTIONARY_NODES,
    node_population_size=20,
    node_evolution_generations=5,
)
# Best for: High complexity, deep search trees
```

### 3. Coevolution
```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.COEVOLUTION,
    tree_population_size=100,
    tree_generations=50,
)
# Best for: Diverse strategies, open-ended problems
```

## Advanced Features

### Adaptive Selection
```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.ADAPTIVE,
    adaptive_enabled=True,
)
# Automatically selects best approach
```

### Combined Search
```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.COMBINED,
    combination_strategy="voting",  # or "weighted", "best", "ensemble"
)
# Runs all three approaches and combines results
```

### LeanAide Integration
```python
config = HybridMCTSConfig(
    leanaide_enabled=True,
    leanaide_host="localhost",
    leanaide_port=7654,
    leanaide_verify_every=5,
)
# Formal verification with Lean
```

### Caching
```python
config = HybridMCTSConfig(
    cache_enabled=True,
    cache_max_size=10000,
)
# Cache policies, nodes, trees for faster repeated searches
```

### Monitoring
```python
engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Get monitoring report
report = engine.get_monitoring_report()
print(f"Time: {report['execution_time']:.2f}s")
print(f"Memory: {report['peak_memory_mb']:.2f}MB")
```

## Result Structure

```python
result = HybridMCTSResult(
    success=True,
    approach_used=HybridMCTSApproach.EVOLVED_POLICIES,
    best_proof="proof text",
    best_fitness=0.85,
    execution_time=12.5,
    nodes_explored=1523,
    generations_completed=10,
    policy_fitness_history=[0.5, 0.55, 0.6, ...],
    verified_count=3,
)
```

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploration_constant` | 1.414 | UCB1 exploration parameter |
| `simulations` | 100 | MCTS simulations per node |
| `max_depth` | 50 | Maximum search depth |
| `population_size` | 50 | Evolutionary population size |
| `generations` | 10 | Number of generations |
| `mutation_rate` | 0.1 | Mutation probability |
| `crossover_rate` | 0.7 | Crossover probability |
| `parallel_evaluation` | True | Enable parallel processing |
| `max_workers` | 4 | Number of parallel workers |
| `cache_enabled` | True | Enable caching |

## Workflow Integration

```python
from hybrid_mcts_framework import HybridMCTSWorkflowIntegrator

integrator = HybridMCTSWorkflowIntegrator(config)

# Solve single subproblem
solution = await integrator.solve_subproblem({
    "id": "sub_1",
    "statement": "theorem test (a : nat) : a + 0 = a",
    "domain": "algebra",
    "difficulty": "easy",
})

# Solve batch
solutions = await integrator.solve_batch(subproblems)
```

## Benchmarking

```python
from hybrid_mcts_framework import HybridBenchmark

benchmark = HybridBenchmark(config)
comparison = await benchmark.benchmark_all(theorems)

print(f"Best: {comparison.best_overall.value}")
print(f"Fastest: {comparison.fastest.value}")
```

## Testing

```bash
# Run all tests
pytest test_hybrid_mcts_framework.py -v

# Run specific test class
pytest test_hybrid_mcts_framework.py::TestHybridMCTSEngine -v

# Run with coverage
pytest test_hybrid_mcts_framework.py --cov=hybrid_mcts_framework
```

## Demos

```bash
# Run all demos
python demo_hybrid_mcts.py

# Run specific demo
python demo_hybrid_mcts.py --demo evolved

# Interactive mode
python demo_hybrid_mcts.py --interactive
```

## Troubleshooting

### Performance
```python
config = HybridMCTSConfig(
    simulations=50,              # Reduce
    parallel_evaluation=False,   # Disable parallelism
    cache_enabled=False,         # Reduce memory
)
```

### Quality
```python
config = HybridMCTSConfig(
    simulations=200,             # Increase
    early_stopping=False,        # Disable early stopping
    population_size=100,         # Larger population
)
```

### Memory
```python
config = HybridMCTSConfig(
    cache_enabled=False,
    max_workers=2,
    population_size=20,
)
```

## File Structure

```
hybrid_mcts_framework.py          # Main framework (1878 lines)
test_hybrid_mcts_framework.py     # Test suite (990 lines)
demo_hybrid_mcts.py               # Demos (532 lines)
HYBRID_MCTS_FRAMEWORK_GUIDE.md    # Full documentation
HYBRID_MCTS_QUICK_REFERENCE.md    # This file
```

## Key Classes

- `HybridMCTSConfig` - Configuration
- `HybridMCTSResult` - Results
- `HybridMCTSEngine` - Main engine
- `HybridCache` - Caching
- `HybridMCTSMonitor` - Monitoring
- `AdaptiveHybridSelector` - Adaptive selection
- `HybridMCTSWithLeanAide` - Lean integration
- `CombinedHybridMCTS` - Combined search
- `HybridBenchmark` - Benchmarking
- `HybridMCTSWorkflowIntegrator` - Workflow integration

## Import Patterns

```python
# Core
from hybrid_mcts_framework import (
    HybridMCTSEngine,
    HybridMCTSConfig,
    HybridMCTSResult,
    HybridMCTSApproach,
)

# Presets
from hybrid_mcts_framework import (
    create_framework_from_preset,
    quick_search,
    thorough_search,
)

# Integration
from hybrid_mcts_framework import (
    HybridMCTSWorkflowIntegrator,
    HybridBenchmark,
)

# Utilities
from hybrid_mcts_framework import print_result_summary
```

## Common Patterns

### Pattern 1: Simple Search
```python
engine = create_framework_from_preset("balanced")
result = await engine.search(theorem)
```

### Pattern 2: Adaptive Search
```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.ADAPTIVE,
    adaptive_enabled=True,
)
engine = HybridMCTSEngine(config)
result = await engine.search(theorem, context={"domain": "algebra"})
```

### Pattern 3: Verified Search
```python
config = HybridMCTSConfig(
    leanaide_enabled=True,
    leanaide_verify_every=5,
)
engine = HybridMCTSEngine(config)
result = await engine.search(theorem)
print(f"Verified: {result.verified_count}")
```

### Pattern 4: Batch Processing
```python
integrator = HybridMCTSWorkflowIntegrator(config)
solutions = await integrator.solve_batch(subproblems)
```

## Tips

1. **Start with presets** - Use `balanced` for most cases
2. **Enable caching** - For repeated similar problems
3. **Use adaptive** - Let framework choose best approach
4. **Monitor progress** - Check `engine.get_monitoring_report()`
5. **Save checkpoints** - For long-running searches
6. **Verify results** - Enable LeanAide when needed
7. **Benchmark first** - Test approaches before committing
8. **Tune parameters** - Adjust based on problem characteristics

## Command Line

```bash
# Search from command line
python -m hybrid_mcts_framework "theorem test (a : nat) : a + 0 = a" \
    --approach evolved_policies \
    --preset balanced \
    --output result.json
```

## Resources

- Full Guide: `HYBRID_MCTS_FRAMEWORK_GUIDE.md`
- Implementation: `hybrid_mcts_framework.py`
- Tests: `test_hybrid_mcts_framework.py`
- Demos: `demo_hybrid_mcts.py`
- Summary: `HYBRID_MCTS_IMPLEMENTATION_SUMMARY.md`

## Support

- Check the full guide for detailed documentation
- Run demos to see examples
- Run tests to verify installation
- Check logs for debugging information

---

**Version**: 1.0.0 | **Date**: 2025-12-30 | **Status**: Production Ready
