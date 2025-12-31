# MDAP/MAKER + MCTS Unified Framework - Quick Reference

A quick reference guide for the MDAP/MAKER + MCTS unified framework.

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Configuration](#configuration)
4. [Approaches](#approaches)
5. [Results](#results)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Basic installation
pip install asyncio numpy

# Optional dependencies
pip install leanaide-client  # Lean 4 verification
pip install mdap-engine       # MDAP components
pip install maker-engine      # MAKER components
```

---

## Basic Usage

### Minimal Example

```python
from mdap_maker_mcts_unified import MDAPMAKERMCTSEngine, MDAPMAKERMCTSConfig

async def search():
    config = MDAPMAKERMCTSConfig(num_agents=5, simulations=100)
    engine = MDAPMAKERMCTSEngine(config)
    result = await engine.search("theorem thm (n : Nat) : n + 0 = n := by")
    return result

# Run
result = asyncio.run(search())
print(f"Success: {result.success}, Fitness: {result.best_fitness}")
```

### Using Presets

```python
from mdap_maker_mcts_unified import MDAPMCTSPresets

# Fast (quick results)
config = MDAPMCTSPresets.fast()

# Balanced (recommended)
config = MDAPMCTSPresets.balanced()

# Thorough (maximum quality)
config = MDAPMCTSPresets.thorough()

# Experimental (try all)
config = MDAPMCTSPresets.experimental()
```

---

## Configuration

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `approach` | MCTSApproach | EVOLVED_POLICIES | Which approach to use |
| `num_agents` | int | 5 | Number of MDAP agents |
| `simulations` | int | 100 | MCTS simulations per node |
| `max_depth` | int | 50 | Maximum search depth |
| `enable_decomposition` | bool | True | Enable task decomposition |
| `leanaide_enabled` | bool | True | Enable LeanAide verification |
| `enable_caching` | bool | True | Enable result caching |
| `parallel_evaluation` | bool | True | Enable parallel evaluation |

### MDAP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voting_strategy` | str | "first_k_ahead" | MAKER voting strategy |
| `k_ahead` | int | 3 | Votes needed for consensus |
| `consensus_threshold` | float | 0.75 | Consensus agreement threshold |
| `agent_reliability_threshold` | float | 0.6 | Minimum agent reliability |
| `decomposition_depth` | int | 3 | Maximum decomposition depth |

### Approach-Specific Parameters

#### Evolved Policies

```python
evolved_policy = EvolvedPolicyConfig(
    population_size=50,    # Policy population
    generations=10,        # Evolution generations
    mutation_rate=0.1,     # Mutation probability
    crossover_rate=0.7     # Crossover probability
)
```

#### Evolutionary Nodes

```python
evolutionary_node = EvolutionaryNodeConfig(
    population_per_node=20,      # Population at each node
    max_generations_per_node=5,  # Generations per node
    sequence_length=5,           # Action sequence length
    mutation_rate=0.15           # Mutation probability
)
```

#### Coevolution

```python
coevolution = CoevolutionConfig(
    tree_population=30,         # Number of trees
    coevolution_generations=15, # Coevolution generations
    tree_depth=4,               # Maximum tree depth
    mutation_rate=0.2           # Mutation probability
)
```

---

## Approaches

### 1. Evolved Policies (EVOLVED_POLICIES)

Evolve rollout policies using genetic algorithms.

**Best for:**
- General theorem proving
- Well-defined problem domains
- When speed is important

**Key features:**
- Fast convergence
- Low memory usage
- Good for simple to medium problems

```python
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.EVOLVED_POLICIES,
    evolved_policy=EvolvedPolicyConfig(
        population_size=50,
        generations=10
    )
)
```

### 2. Evolutionary Nodes (EVOLUTIONARY_NODES)

Evolve action sequences at each MCTS node.

**Best for:**
- Structured domains (algebra, analysis)
- Problems requiring tactical reasoning
- Medium to high complexity

**Key features:**
- Rich exploration
- Adaptive per-node evolution
- Good for multi-step proofs

```python
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.EVOLUTIONARY_NODES,
    evolutionary_node=EvolutionaryNodeConfig(
        population_per_node=20,
        max_generations_per_node=5
    )
)
```

### 3. Coevolution (COEVOLUTION)

Coevolve decision trees with competitive evaluation.

**Best for:**
- Highly complex theorems
- Problems requiring deep reasoning
- When quality is more important than speed

**Key features:**
- Sophisticated search
- Competitive coevolution
- Best for hard problems

```python
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.COEVOLUTION,
    coevolution=CoevolutionConfig(
        tree_population=30,
        coevolution_generations=15
    )
)
```

### 4. Adaptive (ADAPTIVE)

Automatically select the best approach.

**Best for:**
- Unknown problem domains
- Mixed complexity problems
- Production systems

```python
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.ADAPTIVE
)
```

### 5. Combined (COMBINED)

Run all approaches and combine results.

**Best for:**
- Maximum quality requirements
- Benchmarking
- Research and evaluation

**Warning:** Slowest option (runs 3x approaches)

```python
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.COMBINED,
    combined_search=True
)
```

---

## Results

### Accessing Results

```python
result = await engine.search(theorem)

# Basic
result.success          # bool
result.best_proof       # str or None
result.best_fitness     # float
result.execution_time   # float (seconds)

# MDAP metrics
result.consensus_score     # float or None
result.agreement_level     # float or None
result.agent_results       # List[AgentResult] or None

# Verification
result.verification_result.is_valid      # bool
result.verification_result.verification_time  # float

# Approach-specific
result.policy_metrics    # PolicyMetrics or None
result.node_metrics      # NodeMetrics or None
result.tree_metrics      # TreeMetrics or None
```

### Checking Success

```python
if result.success:
    print(f"Found proof with fitness {result.best_fitness}")

    if result.verification_result and result.verification_result.is_valid:
        print("Proof verified by LeanAide!")
    elif result.consensus_score and result.consensus_score > 0.8:
        print("High consensus among agents")
    else:
        print("Proof found but not verified")
else:
    print(f"Failed: {result.error_message}")
```

### Iterating Agent Results

```python
if result.agent_results:
    for agent_result in result.agent_results:
        print(f"{agent_result.agent_id}:")
        print(f"  Fitness: {agent_result.fitness:.3f}")
        print(f"  Confidence: {agent_result.confidence:.3f}")
        print(f"  Reasoning: {agent_result.reasoning}")
```

### Serialization

```python
# Save result
import json

result_dict = result.to_dict()
with open('result.json', 'w') as f:
    json.dump(result_dict, f, indent=2)

# Load result
with open('result.json', 'r') as f:
    loaded_dict = json.load(f)

result = MDAPMAKERMCTSResult.from_dict(loaded_dict)
```

---

## Common Patterns

### Pattern 1: Retry with Different Approach

```python
async def search_with_retry(theorem):
    approaches = [
        MCTSApproach.EVOLVED_POLICIES,
        MCTSApproach.EVOLUTIONARY_NODES,
        MCTSApproach.COEVOLUTION
    ]

    for approach in approaches:
        config = MDAPMAKERMCTSConfig(approach=approach)
        engine = MDAPMAKERMCTSEngine(config)
        result = await engine.search(theorem)

        if result.success and result.best_fitness > 0.8:
            return result

    return None  # All failed
```

### Pattern 2: Parallel Multi-Approach Search

```python
async def search_parallel(theorem):
    config = MDAPMAKERMCTSConfig(
        approach=MCTSApproach.COMBINED,
        num_agents=5
    )
    engine = MDAPMAKERMCTSEngine(config)
    return await engine.search(theorem)
```

### Pattern 3: Batch Processing

```python
async def search_batch(theorems):
    config = MDAPMAKERMCTSConfig(num_agents=5)
    engine = MDAPMAKERMCTSEngine(config)

    tasks = [engine.search(thm) for thm in theorems]
    results = await asyncio.gather(*tasks)
    return results
```

### Pattern 4: Benchmarking

```python
async def benchmark():
    config = MDAPMAKERMCTSConfig(num_agents=5, simulations=50)
    benchmark = MDAPMCTSBenchmark(config)

    theorems = ["thm1", "thm2", "thm3"]
    report = await benchmark.benchmark_all(theorems)

    print(f"Best approach: {report.comparison['best_success_rate']['approach']}")
    return report
```

### Pattern 5: Workflow Integration

```python
async def solve_subproblem(subproblem):
    integrator = MDAPMCTSWorkflowIntegrator(config)
    solution = await integrator.solve_with_mdap_mcts(subproblem)

    if solution.quality_metrics.get('verification'):
        return solution
    return None
```

### Pattern 6: Custom Cache

```python
async def search_with_custom_cache(theorem):
    cache = MDAPMCTSCache(max_size=1000)
    engine = MDAPMAKERMCTSEngine(config, cache=cache)

    result = await engine.search(theorem)

    # Check cache stats
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")

    return result
```

### Pattern 7: Monitoring

```python
async def search_with_monitoring(theorem):
    monitor = MDAPMCTSMonitor()
    engine = MDAPMAKERMCTSEngine(config, monitor=monitor)

    result = await engine.search(theorem)

    # Get execution summary
    summary = monitor.get_summary()
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    print(f"Total evaluations: {summary['total_agent_evaluations']}")

    return result
```

### Pattern 8: Adaptive Selection

```python
async def search_adaptive(theorem):
    selector = MDAPAdaptiveSelector()

    # Select best approach
    approach = selector.select_approach(theorem, available_agents=5)

    # Use selected approach
    config = MDAPMAKERMCTSConfig(approach=approach)
    engine = MDAPMAKERMCTSEngine(config)
    result = await engine.search(theorem)

    # Record for learning
    selector.record_result(theorem, approach, result.success)

    return result
```

---

## Troubleshooting

### Problem: Import Warnings

```
WARNING: MDAP engine not available
WARNING: MAKER engine not available
```

**Solution:** Install missing dependencies
```bash
pip install mdap-engine maker-engine
```

### Problem: Out of Memory

**Solution:** Reduce cache and parallelism
```python
config = MDAPMAKERMCTSConfig(
    cache_size=1000,
    max_workers=2,
    enable_decomposition=False
)
```

### Problem: Too Slow

**Solution:** Reduce computation
```python
config = MDAPMAKERMCTSConfig(
    simulations=50,      # Reduce from 100
    max_depth=25,        # Reduce from 50
    num_agents=3,        # Reduce from 5
    leanaide_enabled=False  # Disable verification
)
```

### Problem: Low Success Rate

**Solution:** Increase quality settings
```python
config = MDAPMAKERMCTSConfig(
    simulations=200,     # Increase simulations
    max_depth=100,       # Increase depth
    num_agents=7,        # More agents
    enable_decomposition=True,  # Enable decomposition
    consensus_threshold=0.8      # Higher consensus
)
```

### Problem: All Approaches Failing

**Solution:** Check theorem validity
```python
# Verify theorem is well-formed
theorem = "theorem example (n : Nat) : n + 0 = n := by"

# Check with simple approach first
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.EVOLVED_POLICIES,
    simulations=10  # Quick test
)
```

---

## Performance Comparison

| Approach | Speed | Memory | Quality | Best For |
|----------|-------|--------|---------|----------|
| Evolved Policies | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | General use |
| Evolutionary Nodes | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Structured domains |
| Coevolution | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex problems |
| Adaptive | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Unknown domains |
| Combined | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Maximum quality |

---

## Configuration Quick Reference

### Fast (Quick Results)
```python
config = MDAPMAKERMCTSConfig(
    num_agents=3,
    simulations=50,
    max_depth=30,
    enable_decomposition=False,
    leanaide_enabled=False
)
```

### Balanced (Recommended)
```python
config = MDAPMAKERMCTSConfig(
    num_agents=5,
    simulations=100,
    max_depth=50,
    enable_decomposition=True,
    leanaide_enabled=True
)
```

### Thorough (Maximum Quality)
```python
config = MDAPMAKERMCTSConfig(
    num_agents=7,
    simulations=200,
    max_depth=100,
    enable_decomposition=True,
    decomposition_depth=5,
    leanaide_enabled=True,
    consensus_threshold=0.8
)
```

---

## Tips

1. **Start with presets:** Use `MDAPMCTSPresets.balanced()` for good defaults
2. **Enable caching:** Speeds up repeated searches
3. **Use parallel evaluation:** Faster on multi-core systems
4. **Monitor consensus:** Low consensus indicates disagreement
5. **Verify proofs:** LeanAide ensures correctness
6. **Decompose complex problems:** Break into subtasks
7. **Try adaptive selection:** Let the framework choose
8. **Benchmark first:** Find best approach for your domain

---

## See Also

- [Full Documentation](MDAP_MAKER_MCTS_README.md)
- [Demo Script](demo_mdap_maker_mcts_unified.py)
- [Source Code](mdap_maker_mcts_unified.py)
- [MDAP/MAKER Paper](https://arxiv.org/abs/2511.09030)
