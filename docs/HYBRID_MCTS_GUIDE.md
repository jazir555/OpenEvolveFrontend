# Hybrid MCTS-Evolution User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Approach Selection Guide](#approach-selection-guide)
3. [Configuration Guide](#configuration-guide)
4. [Performance Tuning](#performance-tuning)
5. [Best Practices](#best-practices)
6. [Common Pitfalls](#common-pitfalls)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)
9. [FAQ](#faq)

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/openevolve/openevolve-frontend.git
cd openevolve-frontend

# Install dependencies
pip install -r requirements.txt

# Install hybrid MCTS module
pip install -e .
```

### Quick Start Example

```python
import asyncio
from hybrid_mcts import (
    HybridMCTSEngine,
    HybridMCTSPresets,
    HybridMCTSApproach
)

async def main():
    # Create configuration
    config = HybridMCTSPresets.balanced()

    # Initialize engine
    engine = HybridMCTSEngine(config)

    # Search for proof
    theorem = "For all natural numbers n, n + 0 = n"
    result = await engine.search(theorem)

    # Check result
    if result.success:
        print(f"Proof found!")
        print(result.best_proof.lean_code)
    else:
        print("Proof not found")

if __name__ == "__main__":
    asyncio.run(main())
```

### Basic Usage Patterns

#### Pattern 1: Use Preset

```python
from hybrid_mcts import HybridMCTSPresets, HybridMCTSEngine

# Use preset configuration
config = HybridMCTSPresets.fast()
engine = HybridMCTSEngine(config)
result = await engine.search(theorem)
```

#### Pattern 2: Custom Configuration

```python
from hybrid_mcts import HybridMCTSConfig, HybridMCTSEngine

# Custom configuration
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.EVOLVED_POLICIES,
    population_size=100,
    generations=30,
    mcts_simulations=500,
    enable_caching=True
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)
```

#### Pattern 3: Adaptive Selection

```python
from hybrid_mcts import (
    HybridMCTSEngine,
    AdaptiveHybridSelector,
    HybridMCTSPresets
)

# Let framework choose best approach
config = HybridMCTSPresets.balanced()
config.enable_adaptive_selection = True

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

print(f"Used approach: {result.approach_used}")
```

---

## Approach Selection Guide

### Decision Tree

```
START
  │
  ├─ Do you have training data?
  │   ├─ Yes ──► Go to A
  │   └─ No ──► Use Evolutionary Nodes
  │
A ├─ Are problems similar?
  │   ├─ Yes ──► Go to B
  │   └─ No ──► Use Evolutionary Nodes
  │
B ├─ Is inference speed critical?
  │   ├─ Yes ──► Use Evolved Policies
  │   └─ No ──► Go to C
  │
C ├─ Is problem complexity high?
  │   ├─ Yes ──► Go to D
  │   └─ No ──► Use Evolved Policies
  │
D ├─ Is search space large?
  │   ├─ Yes ──► Use Coevolution
  │   └─ No ──► Use Evolutionary Nodes
```

### Approach Comparison Matrix

| Feature | Evolved Policies | Evolutionary Nodes | Coevolution |
|---------|------------------|-------------------|-------------|
| **Training Required** | Yes (offline) | No | Optional |
| **Inference Speed** | Very Fast | Medium | Slow |
| **Proof Quality** | Good | Very Good | Excellent |
| **Memory Usage** | Low | Medium | High |
| **Best For** | Similar problems, repeated use | Complex proofs | Domain adaptation |
| **Worst For** | One-off problems | Simple proofs | Time-critical |

### Detailed Approach Profiles

#### Evolved Policies

**When to Use**:
- Have many similar theorems
- Need fast inference
- Can afford offline training
- Problems are in same domain

**When NOT to Use**:
- Single theorem
- No training data available
- Very different problems

**Example Use Cases**:
- Arithmetic theorems (commutativity, associativity)
- Algebraic simplifications
- Pattern-matching proofs

#### Evolutionary Nodes

**When to Use**:
- Complex proofs with many decisions
- Problem structure unknown
- Can handle medium compute
- Need adaptability

**When NOT to Use**:
- Very simple proofs
- Extremely time-critical
- Limited compute resources

**Example Use Cases**:
- Induction proofs with cases
- Proof by contradiction
- Multi-step existential proofs

#### Coevolution

**When to Use**:
- Exploring new domain
- Research applications
- Quality > speed
- Can afford extended compute

**When NOT to Use**:
- Production time constraints
- Simple problems
- Limited compute

**Example Use Cases**:
- Novel mathematical domains
- Competition problems
- Learning domain-specific strategies

### Feature-Based Selection

```python
from hybrid_mcts import AdaptiveHybridSelector

selector = AdaptiveHybridSelector()

# Extract features automatically
features = selector.extract_features(theorem)
# Features:
# - complexity: estimated difficulty
# - depth: expected proof depth
# - domain: arithmetic, algebra, etc.
# - novelty: similarity to training data

# Get recommendation
approach = selector.select_approach(theorem, features)
print(f"Recommended: {approach}")
```

---

## Configuration Guide

### Evolved Policies Configuration

#### Key Parameters

```python
config = HybridMCTSConfig(
    # Approach
    approach=HybridMCTSApproach.EVOLVED_POLICIES,

    # Policy Training (Offline)
    policy_training_generations=50,    # Higher = better policy
    policy_population_size=30,         # 20-50 recommended
    policy_mutation_rate=0.15,         # 0.1-0.2 typical

    # MCTS (Online)
    mcts_simulations=1000,              # Fewer needed with good policy
    mcts_time_budget=60.0,
    mcts_exploration_constant=1.414,

    # Performance
    enable_caching=True,                # Cache learned policies
    cache_size=1000
)
```

#### Parameter Tuning Guide

**Policy Population Size**:
- **Small (10-20)**: Fast training, lower quality
- **Medium (30-50)**: Balanced (recommended)
- **Large (100+)**: Best quality, slow training

**Training Generations**:
- **Quick (10-20)**: Prototype, baseline
- **Standard (30-50)**: Production use
- **Thorough (100+)**: Best quality, research

**Mutation Rate**:
- **Low (0.05-0.1)**: Fine-tuning good policies
- **Medium (0.1-0.2)**: Standard evolution
- **High (0.2-0.4)**: Exploring, avoiding local optima

### Evolutionary Nodes Configuration

#### Key Parameters

```python
config = HybridMCTSConfig(
    # Approach
    approach=HybridMCTSApproach.EVOLUTIONARY_NODES,

    # Node Evolution
    node_population_size=10,            # Sequences per node
    node_evolution_frequency=5,         # Evolve every N visits
    sequence_length_range=(3, 10),      # Min/max sequence length

    # Evolution
    population_size=50,                 # Overall population
    generations=20,                     # Not directly used
    mutation_rate=0.1,
    crossover_rate=0.8,

    # MCTS
    mcts_simulations=1000,
    mcts_exploration_constant=1.414
)
```

#### Parameter Tuning Guide

**Node Population Size**:
- **Small (5-10)**: Lower memory, faster
- **Medium (10-20)**: Balanced (recommended)
- **Large (20+)**: Better exploration, more memory

**Evolution Frequency**:
- **Frequent (3-5)**: More adaptation, more compute
- **Medium (5-10)**: Balanced (recommended)
- **Rare (15+)**: Less adaptation, faster

**Sequence Length Range**:
- **Short (2-5)**: Fast rollouts, less depth
- **Medium (5-10)**: Balanced (recommended)
- **Long (10-20)**: Deep exploration, slower

### Coevolution Configuration

#### Key Parameters

```python
config = HybridMCTSConfig(
    # Approach
    approach=HybridMCTSApproach.COEVOLUTION,

    # Populations
    tree_population_size=20,            # Number of trees
    evaluator_population_size=15,       # Number of evaluators
    coevolution_generations=30,         # Coevolution rounds

    # Evaluation
    evaluation_simulations=500,         # MC sims per evaluation
    mcts_simulations=500,               # Lower for coevolution

    # Evolution
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism_count=2
)
```

#### Parameter Tuning Guide

**Population Sizes**:
- **Trees**: 15-30 recommended
- **Evaluators**: 10-20 recommended
- **Ratio**: 3:2 (trees:evaluators) works well

**Coevolution Generations**:
- **Quick (10-20)**: Initial exploration
- **Standard (30-50)**: Production use
- **Extended (100+)**: Research, best quality

**Evaluation Simulations**:
- **Low (100-300)**: Fast evaluation
- **Medium (500-1000)**: Balanced (recommended)
- **High (2000+)**: Accurate evaluation, slow

### Unified Framework Configuration

```python
config = HybridMCTSConfig(
    # Let framework choose
    approach=HybridMCTSApproach.ADAPTIVE,

    # Adaptive Selection
    enable_adaptive_selection=True,
    adaptive_window_size=10,            # Track last N results
    switch_threshold=0.3,               # Switch if 30% better

    # Fallback parameters
    population_size=50,
    generations=20,
    mcts_simulations=1000
)
```

---

## Performance Tuning

### Population Size Guidelines

```
Problem Size         │ Population Size
─────────────────────┼─────────────────
Simple (< 10 steps)  │ 20-30
Medium (10-50 steps) │ 50-100
Complex (> 50 steps) │ 100-200
```

### Generations Guidelines

```
Use Case             │ Generations
─────────────────────┼─────────────
Quick prototype      │ 10-20
Standard run         │ 30-50
Thorough search      │ 50-100
Research/quality     │ 100+
```

### Exploration vs Exploitation

#### Tuning the Balance

```python
# Exploration-focused (find diverse solutions)
config = HybridMCTSConfig(
    mcts_exploration_constant=2.0,     # High exploration
    mutation_rate=0.2,                 # More mutation
    selection_method=SelectionMethod.TOURNAMENT
)

# Exploitation-focused (refine known solutions)
config = HybridMCTSConfig(
    mcts_exploration_constant=1.0,     # Low exploration
    mutation_rate=0.05,                # Less mutation
    selection_method=SelectionMethod.RANK
)

# Balanced (recommended)
config = HybridMCTSConfig(
    mcts_exploration_constant=1.414,   # sqrt(2)
    mutation_rate=0.1,
    selection_method=SelectionMethod.TOURNAMENT
)
```

### Caching Strategies

```python
# Enable caching for repeated problems
config = HybridMCTSConfig(
    enable_caching=True,
    cache_size=1000,                   # Cache entries
)

# Clear cache periodically
engine = HybridMCTSEngine(config)
# ... use engine ...
engine.clear_cache()

# Pre-warm cache with known proofs
engine.warm_cache(known_theorems)
```

### Parallel Execution

```python
# Configure parallel workers
config = HybridMCTSConfig(
    max_workers=4,                     # Parallel workers
    mcts_parallel_simulations=4,       # Parallel MCTS rollouts
)

# Batch processing
theorems = load_theorem_corpus()
results = await engine.batch_search(
    theorems,
    parallel=True
)
```

### Resource Allocation

```python
# CPU-bound
config = HybridMCTSConfig(
    max_workers=cpu_count(),           # Use all CPUs
    mcts_parallel_simulations=cpu_count()
)

# Memory-constrained
config = HybridMCTSConfig(
    node_population_size=5,            # Reduce memory
    enable_caching=False               # Disable cache
)

# Time-critical
config = HybridMCTSConfig(
    mcts_time_budget=10.0,             # 10 second limit
    mcts_simulations=100,              # Fewer iterations
    approach=HybridMCTSApproach.EVOLVED_POLICIES  # Fastest
)
```

---

## Best Practices

### 1. Start Simple

```python
# BAD: Start with complex configuration
config = HybridMCTSConfig(
    approach=HybridMCTSApproach.COMBINED,
    tree_population_size=100,
    coevolution_generations=200,
    # ... many complex parameters
)

# GOOD: Start with preset
config = HybridMCTSPresets.balanced()
# Tune based on results
```

### 2. Use Presets as Starting Point

```python
# Choose appropriate preset
if time_critical:
    config = HybridMCTSPresets.fast()
elif quality_critical:
    config = HybridMCTSPresets.thorough()
else:
    config = HybridMCTSPresets.balanced()

# Then customize specific parameters
config.mcts_time_budget = my_time_limit
```

### 3. Monitor Convergence

```python
from hybrid_mCTS import PolicyEvolutionEngine

engine = PolicyEvolutionEngine(config)

def track_progress(generation, metrics):
    print(f"Gen {generation}: Best={metrics.best_fitness:.3f}")
    if metrics.stagnation_count > 10:
        print("Warning: Stagnation detected!")

best_policy = await engine.evolve_policies(
    test_theorems=training_set,
    progress_callback=track_progress
)
```

### 4. Cache Learned Policies

```python
# Train once, use many times
engine = PolicyEvolutionEngine(config)
policy = await engine.evolve_policies(training_theorems)

# Save for reuse
engine.save_policy(policy, "domain_policy.json")

# Load for inference
loaded_policy = PolicyEvolutionEngine.load_policy("domain_policy.json")
mcts = EvolvedPolicyMCTS(loaded_policy, mcts_config)
```

### 5. Verify with LeanAide

```python
config = HybridMCTSConfig(
    leanaide_enabled=True,             # Enable verification
    leanaide_host="localhost",
    leanaide_port=7654
)

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

if result.verification_success:
    print("Formally verified!")
```

### 6. Use Adaptive Selection for Unknown Domains

```python
# When unsure, let framework decide
config = HybridMCTSPresets.balanced()
config.enable_adaptive_selection = True

engine = HybridMCTSEngine(config)
result = await engine.search(theorem)

print(f"Auto-selected: {result.approach_used}")
```

### 7. Batch Process Similar Theorems

```python
# Group similar theorems
theorems_by_domain = group_by_domain(all_theorems)

# Process each batch with optimized config
for domain, theorems in theorems_by_domain.items():
    if domain == "arithmetic":
        config = HybridMCTSPresets.fast()
    else:
        config = HybridMCTSPresets.balanced()

    engine = HybridMCTSEngine(config)
    results = await engine.batch_search(theorems)
```

### 8. Handle Failures Gracefully

```python
try:
    result = await engine.search(theorem)
except SearchTimeout:
    logger.warning("Search timed out, using fallback")
    result = await fallback_search(theorem)
except PopulationDiversityError:
    logger.warning("Diversity lost, reinitializing")
    engine.reset()
    result = await engine.search(theorem)
```

### 9. Track and Log Metrics

```python
import logging

logging.basicConfig(level=logging.INFO)

config = HybridMCTSConfig(
    log_level="INFO",
    log_metrics=True,
    enable_progress_tracking=True
)

# Results will include detailed metrics
result = await engine.search(theorem)
print(result.get_summary())
```

### 10. Validate on Test Set

```python
# Train on training set
policy = await engine.evolve_policies(training_set)

# Validate on test set
test_results = []
for theorem in test_set:
    result = await engine.search(theorem)
    test_results.append(result.success)

success_rate = sum(test_results) / len(test_results)
print(f"Test success rate: {success_rate:.2%}")
```

---

## Common Pitfalls

### 1. Overfitting to Training Set

**Problem**: Policy works well on training data but fails on new problems

**Symptoms**:
- High training success, low test success
- Policy exploits training artifacts

**Solutions**:
```python
# Use diverse training set
training_set = load_diverse_theorems()

# Add regularization
config.mutation_rate = 0.15  # Maintain exploration

# Validate on held-out set
validation_set = load_validation_theorems()
```

### 2. Premature Convergence

**Problem**: Population converges to local optimum

**Symptoms**:
- Fitness plateaus early
- Low diversity
- Similar solutions across population

**Solutions**:
```python
# Increase mutation rate
config.mutation_rate = 0.2

# Add diversity maintenance
config.diversity_maintenance = True

# Use tournament selection (maintains diversity)
config.selection_method = SelectionMethod.TOURNAMENT
config.tournament_size = 3
```

### 3. Insufficient Population Diversity

**Problem**: Population loses diversity, evolution stalls

**Symptoms**:
- All individuals similar
- No improvement over generations
- High population similarity

**Solutions**:
```python
# Increase population size
config.population_size = 100

# Enable diversity mechanisms
config.diversity_maintenance = True
config.crowding_distance = True

# Inject random individuals
population.inject_random(rate=0.1)
```

### 4. Ignoring Formal Verification

**Problem**: Generated proofs look good but are invalid

**Symptoms**:
- High confidence, but LeanAide fails
- Tactic applications fail
- Type mismatches

**Solutions**:
```python
# Always enable LeanAide
config.leanaide_enabled = True
config.leanaide_host = "localhost"

# Verify before accepting
if not result.verification_success:
    # Retry or fallback
    result = await fallback_search(theorem)
```

### 5. Wrong Approach Selection

**Problem**: Using suboptimal approach for problem

**Symptoms**:
- Poor performance
- Excessive compute
- Poor quality results

**Solutions**:
```python
# Use adaptive selection
config.approach = HybridMCTSApproach.ADAPTIVE

# Or manually analyze problem
selector = AdaptiveHybridSelector()
approach = selector.select_approach(theorem)
config.approach = approach
```

### 6. Over-Tuning Parameters

**Problem**: Spending too much time tuning

**Symptoms**:
- Many failed experiments
- Inconsistent results
- Diminishing returns

**Solutions**:
```python
# Start with presets
config = HybridMCTSPresets.balanced()

# Change one parameter at a time
config.mcts_simulations = 1500  # Only this

# Test on representative sample
test_theorems = get_representative_sample()
```

### 7. Memory Issues with Large Trees

**Problem**: Evolutionary nodes use too much memory

**Symptoms**:
- Out of memory errors
- Slow performance
- System swapping

**Solutions**:
```python
# Reduce node population
config.node_population_size = 5

# Limit tree depth
config.sequence_length_range = (3, 7)

# Disable caching
config.enable_caching = False
```

### 8. Slow Convergence

**Problem**: Evolution takes too long

**Symptoms**:
- Many generations without improvement
- Slow fitness increase
- Wasted compute

**Solutions**:
```python
# Increase selection pressure
config.tournament_size = 5
config.elitism_count = 5

# Adjust mutation
config.mutation_rate = 0.15  # Optimal range

# Use adaptive parameters
config.adaptive_parameters = True
```

---

## Troubleshooting

### Problem: Low Success Rate

**Symptoms**:
- Most searches fail
- Low win rates
- Poor proof quality

**Diagnosis**:
```python
# Check configuration
config.validate()

# Analyze problem
selector = AdaptiveHybridSelector()
features = selector.extract_features(theorem)
print(f"Complexity: {features['complexity']}")
print(f"Domain: {features['domain']}")

# Check convergence
metrics = engine.get_metrics()
print(f"Best fitness: {metrics['best_fitness']}")
```

**Solutions**:
1. Increase population size
2. Increase generations
3. Try different approach
4. Enable adaptive selection
5. Improve training data quality

### Problem: Slow Convergence

**Symptoms**:
- Fitness increases slowly
- Many generations to plateau
- Inefficient search

**Solutions**:
```python
# Increase selection pressure
config.tournament_size = 5

# Adjust mutation rate
config.mutation_rate = 0.15

# Use elitism
config.elitism_count = 3

# Enable adaptive parameters
config.adaptive_parameters = True
```

### Problem: Memory Issues

**Symptoms**:
- Out of memory errors
- System swapping
- Performance degradation

**Solutions**:
```python
# Reduce memory usage
config.node_population_size = 5
config.enable_caching = False
config.tree_population_size = 15

# Limit tree depth
config.sequence_length_range = (3, 7)

# Use evolved policies (lowest memory)
config.approach = HybridMCTSApproach.EVOLVED_POLICIES
```

### Problem: Time Budget Exceeded

**Symptoms**:
- Searches timeout
- Incomplete results
- Poor time utilization

**Solutions**:
```python
# Reduce iterations
config.mcts_simulations = 500

# Use faster approach
config.approach = HybridMCTSApproach.EVOLVED_POLICIES

# Parallelize
config.max_workers = cpu_count()
config.mcts_parallel_simulations = 4

# Early termination
config.early_termination = True
```

### Problem: Poor Proof Quality

**Symptoms**:
- Proofs found but low quality
- Excessive tactic use
- Inelegant proofs

**Solutions**:
```python
# Enable LeanAide verification
config.leanaide_enabled = True

# Use coevolution for quality
config.approach = HybridMCTSApproach.COEVOLUTION

# Add quality metrics
config.evaluation_criteria = ["success", "elegance", "length"]

# Increase search time
config.mcts_simulations = 2000
```

### Problem: Inconsistent Results

**Symptoms**:
- High variance between runs
- Different results with same config
- Unreliable performance

**Solutions**:
```python
# Set random seed
config.seed = 42

# Increase population
config.population_size = 100

# Use adaptive approach
config.approach = HybridMCTSApproach.ADAPTIVE

# Multiple runs, take best
results = []
for i in range(5):
    result = await engine.search(theorem)
    results.append(result)
best = max(results, key=lambda r: r.best_fitness)
```

---

## Advanced Topics

### Transfer Learning

```python
# Learn policy in source domain
source_engine = PolicyEvolutionEngine(config)
source_policy = await source_engine.evolve_policies(
    source_domain_theorems
)

# Transfer to target domain
target_policy = source_policy.copy()

# Fine-tune on target domain
target_engine = PolicyEvolutionEngine(config)
target_policy = await target_engine.evolve_policies(
    target_domain_theorems,
    initial_population=[target_policy]
)
```

### Multi-Objective Optimization

```python
# Optimize multiple objectives
from hybrid_mcts import MultiObjectiveCoevolution

multi = MultiObjectiveCoevolution(
    objectives=["success", "speed", "elegance"],
    objective_weights=[0.5, 0.3, 0.2]
)

pareto_front = await multi.coevolve_multi_objective(
    test_theorems=theorems,
    generations=50
)

# Select from Pareto front
for solution in pareto_front:
    print(f"Success: {solution.success_score}")
    print(f"Speed: {solution.speed_score}")
    print(f"Elegance: {solution.elegance_score}")
```

### Domain Adaptation

```python
# Adapt to new domain
from hybrid_mcts import DomainAdapter

adapter = DomainAdapter(
    source_policy=learned_policy,
    target_domain="new_domain"
)

adapted_policy = adapter.adapt(
    adaptation_data=few_target_theorems,
    adaptation_generations=10
)
```

### Ensemble Methods

```python
# Combine multiple approaches
combined = CombinedHybridMCTS(
    approaches=[
        HybridMCTSApproach.EVOLVED_POLICIES,
        HybridMCTSApproach.EVOLUTIONARY_NODES
    ],
    combination_method="weighted"
)

# Learn optimal weights
combined.learn_weights(training_theorems)

# Use ensemble
result = await combined.search_combined(theorem)
```

### Distributed Evolution

```python
# Distribute across machines
from hybrid_mcts import DistributedEvolution

distributed = DistributedEvolution(
    config=config,
    worker_urls=[
        "worker1.example.com",
        "worker2.example.com",
        "worker3.example.com"
    ]
)

best_policy = await distributed.evolve_distributed(
    test_theorems=theorems,
    generations=100
)
```

---

## FAQ

### General Questions

**Q: What is hybrid MCTS-evolution?**
A: It combines Monte Carlo Tree Search (directed search) with evolutionary algorithms (population-based optimization) to leverage strengths of both.

**Q: Which approach should I use?**
A: Start with `HybridMCTSPresets.balanced()` with adaptive selection enabled. The framework will choose the best approach automatically.

**Q: How long does training take?**
A: Depends on configuration:
- Fast preset: minutes
- Balanced: 10-30 minutes
- Thorough: hours

**Q: Can I use hybrid MCTS without training data?**
A: Yes, use Evolutionary Nodes approach which doesn't require pre-training.

### Performance Questions

**Q: Why is my search so slow?**
A: Common causes:
- Too many simulations: reduce `mcts_simulations`
- Large population: reduce `population_size`
- Disabled parallelization: enable with `max_workers`

**Q: How can I speed up inference?**
A: Use Evolved Policies approach with pre-trained policy. It's the fastest for inference.

**Q: What's the best configuration for quality?**
A: Use `HybridMCTSPresets.thorough()` which uses Coevolution approach.

**Q: How do I balance speed and quality?**
A: Use `HybridMCTSPresets.balanced()` and adjust `mcts_simulations` based on time budget.

### Technical Questions

**Q: How does adaptive selection work?**
A: It tracks performance of each approach and switches when one significantly outperforms others.

**Q: Can I save and load learned policies?**
A: Yes, use `save_policy()` and `load_policy()` methods.

**Q: How do I integrate with LeanAide?**
A: Set `leanaide_enabled=True` and provide connection details in config.

**Q: What's the difference between the three approaches?**
A: See [Approach Comparison Matrix](#approach-comparison-matrix) above.

### Troubleshooting Questions

**Q: Why is my population losing diversity?**
A: Reduce selection pressure, increase mutation rate, or enable diversity maintenance.

**Q: Why are my results inconsistent?**
A: Set random seed, increase population size, or use ensemble methods.

**Q: Why is memory usage so high?**
A: Reduce node population size, disable caching, or use Evolved Policies approach.

**Q: Why is verification failing?**
A: Ensure LeanAide is running, check connection settings, verify theorem is valid.

### Best Practice Questions

**Q: Should I use presets or custom config?**
A: Start with presets, then customize specific parameters as needed.

**Q: How many training theorems do I need?**
A: Minimum 10-20, ideal 50-100 for evolved policies.

**Q: Should I enable caching?**
A: Yes for repeated similar problems, no for one-off searches.

**Q: How often should I retrain policies?**
A: Retrain when:
- New domain encountered
- Performance degrades
- New theorems are significantly different

### Integration Questions

**Q: How do I integrate with my workflow?**
A: See [HYBRID_MCTS_INTEGRATION.md](./HYBRID_MCTS_INTEGRATION.md)

**Q: Can I use hybrid MCTS with other solvers?**
A: Yes, use CombinedHybridMCTS or custom integration.

**Q: How do I monitor progress?**
A: Use progress_callback parameter and enable logging.

**Q: Can I run multiple searches in parallel?**
A: Yes, use `batch_search()` with `parallel=True`.

### Advanced Questions

**Q: Can I use custom fitness functions?**
A: Yes, implement custom evaluator and pass to evolution engine.

**Q: How do I do transfer learning?**
A: Load policy from source domain, fine-tune on target domain.

**Q: Can I use neural networks?**
A: Yes, implement custom policy network as RolloutPolicyGenome.

**Q: How do I scale to clusters?**
A: Use DistributedEvolution class for distributed training.

---

**Document Version**: 1.0
**Last Updated**: 2025-12-30
**Author**: OpenEvolve Frontend Team
**Related Docs**:
- [HYBRID_MCTS_ARCHITECTURE.md](./HYBRID_MCTS_ARCHITECTURE.md)
- [HYBRID_MCTS_API.md](./HYBRID_MCTS_API.md)
- [HYBRID_MCTS_EXAMPLES.md](./HYBRID_MCTS_EXAMPLES.md)
