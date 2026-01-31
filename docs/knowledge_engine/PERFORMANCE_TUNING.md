# Performance Tuning Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Table of Contents

- [Performance Characteristics](#performance-characteristics)
- [Benchmarking Your Problems](#benchmarking-your-problems)
- [Optimization Strategies](#optimization-strategies)
- [Resource Management](#resource-management)
- [Scaling Considerations](#scaling-considerations)
- [Profiling and Monitoring](#profiling-and-monitoring)

---

## Performance Characteristics

### Evaluation Cost Spectrum

| Cost Level | Evaluations/Hour | Recommended Mode | Typical Domains |
|------------|------------------|------------------|-----------------|
| **Very Low** | >1000 | Standard | Web design, UI optimization |
| **Low** | 100-1000 | QD, Standard | Pharma, molecular design |
| **Medium** | 10-100 | MO, Adversarial | Trading, signal processing |
| **High** | 1-10 | PES | Finance, engineering |
| **Very High** | <1 | PES + Planning | Science, experiments |

### Convergence Speed by Mode

| Mode | Generations to Converge | Best For |
|------|------------------------|----------|
| **PES** | 10-30 (fastest) | Expensive evaluations |
| **Standard GA** | 50-100 | Cheap evaluations |
| **QD** | 100-200 | Exploration |
| **MO** | 100-150 | Trade-off analysis |
| **Adversarial** | 150-200 | Robustness |

### Sample Efficiency

| Domain | System | Mode | Evaluations Needed | Sample Efficiency |
|--------|--------|------|-------------------|-------------------|
| Finance | LoongFlow | PES | 30 | 60% fewer vs baseline |
| Trading | OpenEvolve | Adversarial | 100 | Similar baseline |
| Science | Hybrid | PES+QD | 20 | 60% fewer vs baseline |
| Engineering | Hybrid | PES+Adv | 80 | 40% fewer vs baseline |
| Pharma | OpenEvolve | QD | 200 | Similar baseline |
| Web | OpenEvolve | Standard | 500 | Similar baseline |

---

## Benchmarking Your Problems

### Running Benchmarks

```python
from openevolve.unified import benchmark

# Benchmark across modes
results = await benchmark(
    problem="Your problem description",
    domain="your_domain",
    modes=["pes", "qd", "mo", "adversarial", "standard"],
    max_evaluations=50,
    num_runs=3  # For statistical significance
)

# Analyze results
for mode, metrics in results.items():
    print(f"{mode}:")
    print(f"  Best fitness: {metrics['best_fitness']:.3f}")
    print(f"  Avg fitness: {metrics['avg_fitness']:.3f}")
    print(f"  Evaluations: {metrics['evaluations']}")
    print(f"  Time: {metrics['time']:.1f}s")
```

### Interpreting Benchmark Results

```python
# Example output:
"""
pes:
  Best fitness: 0.875
  Avg fitness: 0.842
  Evaluations: 28
  Time: 120s

qd:
  Best fitness: 0.831
  Avg fitness: 0.795
  Evaluations: 50
  Time: 180s

standard:
  Best fitness: 0.789
  Avg fitness: 0.742
  Evaluations: 50
  Time: 150s
"""

# Analysis:
# - PES is fastest and best quality → Use PES
# - 28 evaluations vs 50 = 44% fewer
# - 0.875 fitness vs 0.789 = 11% better
```

### Choosing Based on Benchmarks

```python
# If PES wins
if results['pes']['best_fitness'] >= best_other_fitness * 0.95:
    if results['pes']['evaluations'] < min_other_evaluations * 0.8:
        recommendation = "pes"  # Clear winner
        reason = "60% fewer evals with comparable quality"

# If MO wins (multiple objectives)
if len(objectives) > 1 and results['mo']['pareto_diversity'] > 0.8:
    recommendation = "mo"
    reason = "Diverse Pareto front needed"

# If Adversarial wins (safety-critical)
if safety_critical and results['adversarial']['robustness_score'] > 0.9:
    recommendation = "adversarial"
    reason = "Robustness critical"
```

---

## Optimization Strategies

### Strategy 1: Reduce Evaluations (Expensive Problems)

**When:** Evaluations take >1 minute each

```python
config = UnifiedEvolutionConfig(
    evolution_mode="pes",  # Directed search
    enable_planning=True,  # Use knowledge
    enable_memory=True,  # Learn from past
    early_stopping=True,  # Stop when confident
    early_stop_threshold=0.9,

    # Aggressive settings
    max_evaluations=30,
    population_size=50  # Smaller population
)
```

**Expected Improvement:** 60% fewer evaluations

### Strategy 2: Parallelize (Multi-Core Machines)

**When:** Have multiple cores available

```python
config = UnifiedEvolutionConfig(
    evolution_mode="standard",  # Easy to parallelize
    max_workers=8,  # Use 8 cores
    num_islands=4,  # 4 independent islands
    island_migration=True,  # Exchange best solutions

    population_size=200,  # Larger population
    max_evaluations=500
)
```

**Expected Improvement:** Near-linear speedup with cores

### Strategy 3: Use Knowledge (Repeated Problems)

**When:** Solving similar problems repeatedly

```python
config = UnifiedEvolutionConfig(
    enable_knowledge_engine=True,  # Extract knowledge
    extract_knowledge=True,
    use_past_solutions=True,  # Warm start
    query_similar_runs=True,  # Learn from past

    # Memory retrieval
    memory_top_k=5,  # Use top 5 similar solutions
    memory_similarity_threshold=0.8
)
```

**Expected Improvement:** 30-50% faster convergence

### Strategy 4: Hybrid Approach (Complex Problems)

**When:** Problem has multiple phases

```python
# Phase 1: Quick exploration
result1 = await evolve(
    problem=problem,
    domain=domain,
    evolution_mode="pes",
    max_evaluations=30
)

# Phase 2: Refine best solutions
result2 = await evolve(
    problem=f"Refine: {result1['best_solution']}",
    domain=domain,
    evolution_mode="mo",
    initial_solutions=result1['archive'],
    max_evaluations=50
)

# Phase 3: Validate
result3 = await evolve(
    problem=problem,
    domain=domain,
    evolution_mode="adversarial",
    initial_solution=result2['best_solution'],
    max_evaluations=30
)
```

**Expected Improvement:** Better solutions, same total budget

### Strategy 5: Adaptive Budget (Unknown Problems)

**When:** Don't know optimal evaluation budget

```python
# Start with small budget
result = await evolve(
    problem=problem,
    domain=domain,
    max_evaluations=20
)

# Check convergence
if result['converged']:
    print(f"Converged early! Only {result['evaluations']} needed")
else:
    # Continue with more budget
    result = await evolve(
        problem=problem,
        domain=domain,
        initial_solution=result['best_solution'],
        max_evaluations=100
    )
```

---

## Resource Management

### Memory Management

**Problem:** Large archives/memory consume RAM

```python
# Limit archive size
config = UnifiedEvolutionConfig(
    evolution_mode="qd",
    archive_size=1000,  # Limit archive entries
    archive_pruning=True,  # Prune low-fitness entries
    max_memory_gb=4  # Limit memory usage
)
```

### CPU Management

**Problem:** Too many workers slow down system

```python
# Adaptive worker count
import psutil

cpu_count = psutil.cpu_count()
config = UnifiedEvolutionConfig(
    max_workers=max(1, cpu_count - 2),  # Leave 2 cores free
    evaluation_timeout=300  # 5 minute timeout per eval
)
```

### Disk Management

**Problem:** Knowledge graph grows large

```python
# Prune old artifacts
config = UnifiedEvolutionConfig(
    knowledge_retention_days=30,  # Keep 30 days
    knowledge_pruning=True,  # Enable pruning
    max_knowledge_size_gb=10  # Limit graph size
)
```

---

## Scaling Considerations

### Problem Size Scaling

| Problem Size | Recommended Config | Expected Time |
|--------------|-------------------|---------------|
| Small (<10 params) | `population_size=50`, `max_evals=100` | Minutes |
| Medium (10-50 params) | `population_size=100`, `max_evals=500` | Hours |
| Large (>50 params) | `population_size=200`, `max_evals=1000` | Days |

### Evaluation Budget Scaling

```python
# Calculate needed budget
def estimate_budget(problem_complexity, evaluation_cost):
    if evaluation_cost == "high":
        # Use PES for efficiency
        return 30
    elif evaluation_cost == "medium":
        # Use standard GA
        return 100
    else:  # low
        # Use QD for exploration
        return 500
```

### Distributed Scaling

```python
from openevolve.unified import DistributedEvolutionEngine

engine = DistributedEvolutionEngine(
    workers=[
        "worker1.example.com",
        "worker2.example.com",
        "worker3.example.com",
        "worker4.example.com"
    ],
    knowledge_engine=ke  # Shared knowledge
)

result = await engine.evolve(
    problem=problem,
    domain=domain,
    distributed=True,
    max_evaluations=1000
)

# Expected: Near-linear speedup with workers
```

---

## Profiling and Monitoring

### Enable Profiling

```python
from openevolve.unified import profile

# Profile evolution run
profile_result = await profile(
    problem=problem,
    domain=domain,
    max_evaluations=50
)

# See where time is spent
print(profile_result['timing'])
"""
{
  "evolution": 120.5,  # seconds
  "evaluation": 100.2,  # 83% of time
  "gauntlet": 15.3,  # 13% of time
  "knowledge_extraction": 5.0  # 4% of time
}
"""

# If evaluation dominates → optimize evaluation function
# If gauntlet dominates → reduce gauntlet rounds
# If knowledge dominates → disable knowledge engine
```

### Real-Time Monitoring

```python
from openevolve.unified import evolve, ProgressCallback

class MyCallback(ProgressCallback):
    def on_iteration(self, iteration, best_fitness, population):
        print(f"Iter {iteration}: Best = {best_fitness:.3f}")

    def on_evaluation(self, evaluation_id, fitness):
        print(f"Eval {evaluation_id}: Fitness = {fitness:.3f}")

result = await evolve(
    problem=problem,
    domain=domain,
    callbacks=[MyCallback()]
)
```

### Performance Metrics Dashboard

```python
# Track metrics over time
metrics = {
    "fitness_over_time": [],
    "evaluations_over_time": [],
    "convergence_rate": []
}

# After each run
update_dashboard(metrics)

# Visualize
import matplotlib.pyplot as plt

plt.plot(metrics['fitness_over_time'])
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Convergence Over Time')
plt.show()
```

---

## Optimization Checklist

### Before Optimization

- [ ] Run benchmarks to determine optimal mode
- [ ] Check evaluation cost (time per evaluation)
- [ ] Estimate required budget
- [ ] Set realistic timeouts

### During Optimization

- [ ] Monitor convergence rate
- [ ] Check resource usage (CPU, memory, disk)
- [ ] Log performance metrics
- [ ] Adjust parameters if needed

### After Optimization

- [ ] Analyze results vs baseline
- [ ] Check for overfitting
- [ ] Validate on test data
- [ ] Document findings

---

**End of Performance Tuning Guide**

For more information, see:
- [Unified Evolution Engine Guide](UNIFIED_EVOLUTION_ENGINE_GUIDE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
