# Hybrid MCTS Framework - Complete Guide

## Overview

The Hybrid MCTS Framework provides a unified interface to three hybrid approaches that combine Monte Carlo Tree Search (MCTS) with Evolutionary Algorithms for theorem proving in Lean 4:

1. **Evolved Rollout Policies** - Uses genetic algorithms to evolve MCTS rollout policies
2. **Evolutionary MCTS Nodes** - Evolves populations of MCTS search trees
3. **Coevolving Decision Trees** - Coevolves proof strategies with problem instances

## Quick Start

### Installation

The framework requires Python 3.8+ with the following dependencies:

```bash
pip install numpy aiohttp pytest psutil
```

### Basic Usage

```python
from hybrid_mcts_framework import HybridMCTSEngine, HybridMCTSConfig, HybridMCTSApproach

# Create configuration
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLVED_POLICIES,
    simulations=100,
    max_depth=50,
)

# Create engine
engine = HybridMCTSEngine(config)

# Run search
theorem = "theorem add_comm (a b : nat) : a + b = b + a"
result = await engine.search(theorem)

# Access results
print(f"Success: {result.success}")
print(f"Best fitness: {result.best_fitness}")
print(f"Proof: {result.best_proof}")
```

### Using Presets

```python
from hybrid_mcts_framework import create_framework_from_preset, quick_search

# Quick search with fast preset
result = await quick_search("theorem test (a : nat) : a + 0 = a")

# Thorough search with maximum quality
engine = create_framework_from_preset("thorough")
result = await engine.search("theorem mul_one (a : nat) : a * 1 = a")
```

## Core Components

### 1. HybridMCTSConfig

Unified configuration for all three approaches:

```python
@dataclass
class HybridMCTSConfig:
    # Core approach selection
    approach: HybridMCTSApproach = HybridMCTSApproach.EVOLVED_POLICIES

    # Common MCTS parameters
    exploration_constant: float = 1.414  # UCB1 C parameter
    simulations: int = 100  # Simulations per node
    max_depth: int = 50  # Maximum search depth

    # Evolved Policies parameters
    policy_population_size: int = 50
    policy_generations: int = 10
    policy_adaptation_interval: int = 10

    # Evolutionary Nodes parameters
    node_population_size: int = 20
    node_evolution_generations: int = 5
    node_convergence_threshold: float = 0.01

    # Coevolution parameters
    tree_population_size: int = 100
    tree_generations: int = 50
    tree_max_depth: int = 10

    # LeanAide integration
    leanaide_enabled: bool = True
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654

    # Performance
    parallel_evaluation: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
```

### 2. HybridMCTSResult

Unified result structure:

```python
@dataclass
class HybridMCTSResult:
    success: bool
    approach_used: HybridMCTSApproach
    best_proof: Optional[str]
    best_fitness: float
    execution_time: float
    nodes_explored: int
    generations_completed: int

    # Approach-specific metrics
    policy_fitness_history: Optional[List[float]]
    node_convergence_history: Optional[Dict[str, List[float]]]
    pareto_front: Optional[List[ProofDecisionTree]]

    # Verification
    verification_results: Optional[List[Dict]]
    verified_count: int
```

### 3. HybridMCTSEngine

Main engine for all approaches:

```python
class HybridMCTSEngine:
    def __init__(self, config: HybridMCTSConfig)

    async def search(
        self,
        theorem: str,
        approach: Optional[HybridMCTSApproach] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> HybridMCTSResult

    def get_monitoring_report(self) -> Dict[str, Any]
    def save_checkpoint(self, filepath: str)
```

## Three Hybrid Approaches

### 1. Evolved Rollout Policies

Genetic evolution of MCTS rollout policies:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLVED_POLICIES,
    policy_population_size=50,
    policy_generations=10,
    policy_mutation_rate=0.1,
    policy_adaptation_interval=10,
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Access evolved policy metrics
print(f"Fitness history: {result.policy_fitness_history}")
print(f"Policy adaptations: {result.policy_adaptations}")
print(f"Final policy: {result.final_policy}")
```

**Best for:**
- Medium-complexity theorems
- Problems with clear structure
- Scenarios where policy generalization is valuable

### 2. Evolutionary MCTS Nodes

Evolution of MCTS node populations:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLUTIONARY_NODES,
    node_population_size=20,
    node_evolution_generations=5,
    node_convergence_threshold=0.01,
    node_min_samples=5,
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Access node evolution metrics
print(f"Convergence history: {result.node_convergence_history}")
print(f"Converged nodes: {result.converged_nodes}")
print(f"Total evaluations: {result.total_node_evaluations}")
```

**Best for:**
- High-complexity theorems
- Deep search trees
- Problems requiring exploration of multiple paths

### 3. Coevolving Decision Trees

Coevolution of proof strategies:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.COEVOLUTION,
    tree_population_size=100,
    tree_generations=50,
    tree_max_depth=10,
    pareto_front_size=20,
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Access coevolution metrics
print(f"Pareto front: {result.pareto_front}")
print(f"Complexity history: {result.tree_complexity_history}")
print(f"Coevolution cycles: {result.coevolution_cycles}")
```

**Best for:**
- Diverse strategy exploration
- Open-ended problems
- Scenarios requiring trade-off analysis

## Advanced Features

### Adaptive Approach Selection

Automatically select the best approach:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.ADAPTIVE,
    adaptive_enabled=True,
    adaptive_warmup_runs=3,
)

engine = HybridMCTSEngine(config)

# Framework selects best approach automatically
result = await engine.search(theorem, context={
    "domain": "algebra",
    "difficulty": "medium",
    "time_limit": 3600,
})
```

### Combined Search

Run multiple approaches and combine results:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.COMBINED,
    combination_strategy="voting",  # or "weighted", "best", "ensemble"
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Uses all three approaches and combines results
```

### LeanAide Integration

Formal verification with Lean:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLVED_POLICIES,
    leanaide_enabled=True,
    leanaide_host="localhost",
    leanaide_port=7654,
    leanaide_verify_every=5,  # Verify every N generations
    leanaide_parallel_verify=True,
    leanaide_max_concurrent=5,
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

print(f"Verified proofs: {result.verified_count}")
print(f"Verification results: {result.verification_results}")
```

### Caching

Cache policies, nodes, and evaluations:

```python
config = HybridMCTSConfig(
    cache_enabled=True,
    cache_max_size=10000,
)

engine = HybridMCTSEngine(config)

# First search - cache miss
result1 = await engine.search(theorem)

# Second search - cache hit (faster)
result2 = await engine.search(theorem)

# Check cache stats
cache_stats = engine.cache.get_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

### Monitoring

Track execution metrics:

```python
engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Get monitoring report
report = engine.get_monitoring_report()
print(f"Execution time: {report['execution_time']:.2f}s")
print(f"Generations: {report['generations_completed']}")
print(f"Peak memory: {report['peak_memory_mb']:.2f}MB")

# Access detailed metrics
monitor = engine.monitor
fitness_history = monitor.get_generation_history("best_fitness")
best_gen = monitor.get_best_generation("best_fitness")
```

### Checkpoints

Save and restore state:

```python
config = HybridMCTSConfig(
    save_checkpoints=True,
    checkpoint_interval=10,
    checkpoint_dir="./checkpoints/hybrid_mcts",
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

# Save checkpoint
engine.save_checkpoint("checkpoint.json")
```

## Configuration Presets

### Fast

Quick exploration for testing:

```python
config = HybridMCTSConfig(
    simulations=50,
    max_depth=25,
    policy_generations=5,
    leanaide_enabled=False,
)
```

### Balanced

Default for general use:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.ADAPTIVE,
    adaptive_enabled=True,
    leanaide_enabled=True,
)
```

### Thorough

Maximum quality:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.COMBINED,
    simulations=200,
    max_depth=100,
    tree_generations=100,
    leanaide_verify_every=1,
    early_stopping=False,
    max_execution_time=7200.0,
)
```

### LeanAide-Focused

Emphasize formal verification:

```python
config = HybridMCTSConfig(
    leanaide_enabled=True,
    leanaide_verify_every=3,
    leanaide_parallel_verify=True,
    leanaide_max_concurrent=10,
)
```

### Research

Experimental features:

```python
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.ADAPTIVE,
    combination_strategy="ensemble",
    save_checkpoints=True,
    checkpoint_interval=5,
    enable_monitoring=True,
)
```

## Workflow Integration

Integrate with OpenEvolve decomposition workflow:

```python
from hybrid_mcts_framework import HybridMCTSWorkflowIntegrator

config = HybridMCTSConfig()
integrator = HybridMCTSWorkflowIntegrator(config)

# Solve single subproblem
subproblem = {
    "id": "subproblem_1",
    "statement": "theorem add_zero (a : nat) : a + 0 = a",
    "domain": "algebra",
    "difficulty": "easy",
    "dependencies": [],
}

solution = await integrator.solve_subproblem(subproblem)
print(f"Success: {solution['success']}")
print(f"Proof: {solution['proof']}")

# Solve batch of subproblems
solutions = await integrator.solve_batch(subproblems)
```

## Benchmarking

Compare all approaches:

```python
from hybrid_mcts_framework import HybridBenchmark

config = HybridMCTSConfig()
benchmark = HybridBenchmark(config)

test_theorems = [
    "theorem thm1 (a : nat) : a + 0 = a",
    "theorem thm2 (a b : nat) : a + b = b + a",
    "theorem thm3 (a b c : nat) : a * (b + c) = a * b + a * c",
]

# Run all approaches
comparison = await benchmark.benchmark_all(test_theorems)

# View results
print(f"Best overall: {comparison.best_overall.value}")
print(f"Fastest: {comparison.fastest.value}")
print(f"Most reliable: {comparison.most_reliable.value}")

for recommendation in comparison.recommendations:
    print(f"Recommendation: {recommendation}")

# Save comparison report
benchmark.save_comparison_report(comparison, "comparison.json")
```

## Utility Functions

### Quick Search

```python
result = await quick_search(
    theorem="theorem test (a : nat) : a + 0 = a",
    approach="evolved_policies"
)
```

### Thorough Search

```python
result = await thorough_search(
    theorem="theorem test (a b : nat) : a + b = b + a",
    approach="combined"
)
```

### Print Results

```python
from hybrid_mcts_framework import print_result_summary

print_result_summary(result)
```

Output:
```
============================================================
Hybrid MCTS Search Results
============================================================
Approach:     evolved_policies
Success:      True
Best Fitness: 0.8500
Time:         12.45s
Generations:  10
Nodes:        1523

Proof:
------------------------------------------------------------
[Proof text here]
------------------------------------------------------------

Verified:     3 proofs
============================================================
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_hybrid_mcts_framework.py -v

# Run specific test class
pytest test_hybrid_mcts_framework.py::TestHybridMCTSEngine -v

# Run with coverage
pytest test_hybrid_mcts_framework.py --cov=hybrid_mcts_framework --cov-report=html
```

## Running Demos

```bash
# Run all demonstrations
python test_hybrid_mcts_framework.py

# This will run:
# - Basic usage demo
# - All approaches comparison
# - Adaptive selection demo
```

## File Structure

```
hybrid_mcts_framework.py          # Main framework (~1800 lines)
├── Enums and Constants
├── HybridMCTSConfig              # Unified configuration
├── HybridMCTSResult              # Unified result structure
├── HybridCache                   # Caching system
├── HybridMCTSMonitor             # Monitoring system
├── AdaptiveHybridSelector        # Adaptive approach selection
├── HybridMCTSWithLeanAide        # LeanAide integration
├── HybridBenchmark               # Benchmarking
├── CombinedHybridMCTS            # Combined search
├── HybridMCTSEngine              # Main engine
├── HybridMCTSPresets             # Configuration presets
└── HybridMCTSWorkflowIntegrator  # Workflow integration

test_hybrid_mcts_framework.py     # Test suite (~1200 lines)
├── Configuration Tests
├── Result Tests
├── Cache Tests
├── Monitor Tests
├── Selector Tests
├── LeanAide Tests
├── Engine Tests
├── Preset Tests
├── Workflow Tests
├── Combined Search Tests
├── Integration Tests
└── Performance Tests
```

## Key Features Summary

### 1. Unified Interface
- Single entry point for all three approaches
- Consistent API across all methods
- Standardized configuration and result structures

### 2. Adaptive Selection
- Automatic approach selection based on problem features
- Performance tracking and learning
- Configurable warm-up period

### 3. Combination Methods
- Voting-based combination
- Weighted averaging
- Best result selection
- Ensemble methods

### 4. Caching
- Policy caching
- Node caching
- Tree caching
- Evaluation caching
- Configurable cache size and eviction

### 5. Benchmarking
- Systematic comparison of approaches
- Statistical analysis
- Performance reports
- Automated recommendations

### 6. Workflow Integration
- OpenEvolve integration
- Subproblem solving
- Batch processing
- Parallel execution

### 7. LeanAide Integration
- Formal verification
- Parallel verification
- Fitness adjustment
- Detailed feedback

### 8. Monitoring
- Real-time metrics
- Resource tracking
- Generation history
- Execution summary

### 9. Presets
- Fast: Quick exploration
- Balanced: General use
- Thorough: Maximum quality
- LeanAide-focused: Formal verification
- Research: Experimental features

### 10. Extensibility
- Pluggable approaches
- Custom selection strategies
- Configurable combination methods
- Extensible monitoring

## Performance Considerations

### Memory Usage
- Enable caching for repeated searches
- Use `cache_max_size` to limit memory
- Monitor memory with tracking enabled

### Execution Time
- Use `fast` preset for quick results
- Enable `parallel_evaluation` for speed
- Adjust `simulations` and `generations` for quality/time trade-off

### CPU Utilization
- Set `max_workers` based on available cores
- Use `parallel_evaluation` for concurrent work
- LeanAide verification can be parallelized

### Quality vs Speed
- Lower `simulations` for faster results
- Fewer `generations` for quicker convergence
- Disable `early_stopping` for maximum quality
- Increase `population_size` for better solutions

## Troubleshooting

### Import Errors
```python
# If LeanAide client is not available
# The framework will gracefully disable LeanAide features
```

### Performance Issues
```python
# Reduce computational load
config = HybridMCTSConfig(
    simulations=50,  # Reduce from 100
    policy_generations=5,  # Reduce from 10
    parallel_evaluation=False,  # Disable parallelism
)
```

### Memory Issues
```python
# Reduce memory footprint
config = HybridMCTSConfig(
    cache_enabled=False,  # Disable cache
    population_size=20,  # Reduce population
    max_workers=2,  # Reduce workers
)
```

### Convergence Issues
```python
# Improve convergence
config = HybridMCTSConfig(
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_threshold=0.001,
    exploration_constant=1.414,  # Increase exploration
)
```

## Best Practices

1. **Start with presets**: Use `balanced` preset for most cases
2. **Enable caching**: For repeated similar problems
3. **Use adaptive selection**: Let the framework choose the best approach
4. **Monitor progress**: Enable monitoring for production use
5. **Save checkpoints**: For long-running searches
6. **Verify results**: Enable LeanAide when formal verification is needed
7. **Benchmark first**: Test all approaches before committing to one
8. **Tune parameters**: Adjust based on problem characteristics
9. **Combine approaches**: Use `combined` for robustness
10. **Track metrics**: Use monitoring to understand behavior

## Future Enhancements

- Distributed execution across multiple machines
- GPU acceleration for neural network policies
- Integration with other theorem provers
- Advanced ensemble methods
- Multi-objective optimization
- Reinforcement learning integration
- Transfer learning between domains
- Interactive search control

## References

- MCTS: Browne et al. (2012) "A Survey of Monte Carlo Tree Search Methods"
- Evolutionary Algorithms: Eiben & Smith (2015) "Introduction to Evolutionary Computing"
- Lean 4: de Moura et al. (2022) "The Lean 4 Theorem Prover"
- Coevolution: Hillis (1990) "Co-evolving parasites improve simulated evolution"

## Support

For issues, questions, or contributions, please refer to the main OpenEvolve documentation.
