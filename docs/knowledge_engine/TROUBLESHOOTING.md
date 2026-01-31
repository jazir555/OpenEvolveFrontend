# Troubleshooting Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Table of Contents

- [Common Issues](#common-issues)
- [Error Messages](#error-messages)
- [Performance Issues](#performance-issues)
- [Domain-Specific Issues](#domain-specific-issues)
- [Debugging Techniques](#debugging-techniques)

---

## Common Issues

### Issue 1: Slow Convergence

**Symptoms:**
- Takes many iterations to converge
- Fitness plateau early
- No improvement over time

**Diagnosis:**
```python
# Check convergence rate
result = await evolve(problem=problem, domain=domain)

if result['iterations'] > result['max_iterations'] * 0.9:
    print("Warning: Slow convergence detected")

if result['improvement_rate'] < 0.001:
    print("Warning: Fitness plateau detected")
```

**Solutions:**

1. **Enable PES mode (if not already)**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    evolution_mode="pes",  # Directed search
    enable_planning=True,
    enable_memory=True
)
```

2. **Enable early stopping**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    early_stopping=True,
    early_stop_threshold=0.9
)
```

3. **Use knowledge from past runs**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    enable_knowledge_engine=True,
    use_past_solutions=True
)
```

4. **Adjust population size**
```python
# Smaller population for faster convergence
result = await evolve(
    problem=problem,
    domain=domain,
    population_size=50  # vs default 100
)
```

---

### Issue 2: Poor Solution Quality

**Symptoms:**
- Fitness score is low
- Solution doesn't meet requirements
- Worse than baseline

**Diagnosis:**
```python
result = await evolve(problem=problem, domain=domain)

baseline = get_baseline_performance()
if result['fitness'] < baseline * 0.9:
    print("Warning: Solution quality below baseline")

# Check gauntlet results
if not result['gauntlet_results']['passed']:
    print("Warning: Solution failed gauntlet")
```

**Solutions:**

1. **Increase evaluation budget**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    max_evaluations=200  # vs default 100
)
```

2. **Check constraints are realistic**
```python
# Are constraints too tight?
constraints = {
    "max_cost": 1000,  # Is this achievable?
    "min_quality": 0.95  # Is this realistic?
}

# Try relaxing constraints
```

3. **Try different mode**
```python
for mode in ["pes", "qd", "mo", "adversarial"]:
    result = await evolve(
        problem=problem,
        domain=domain,
        evolution_mode=mode
    )
    print(f"{mode}: {result['fitness']}")
```

4. **Enable gauntlet for quality assurance**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    enable_gauntlet=True
)
```

---

### Issue 3: Out of Memory

**Symptoms:**
- `MemoryError` exception
- Process killed by OOM killer
- System slows down

**Diagnosis:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024

if memory_mb > 1000:  # > 1GB
    print("Warning: High memory usage")
```

**Solutions:**

1. **Reduce archive size**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    archive_size=500  # vs default 1000
)
```

2. **Reduce population size**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    population_size=50  # vs default 100
)
```

3. **Disable QD if not needed**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    evolution_mode="pes"  # No archive needed
)
```

4. **Enable archive pruning**
```python
config = UnifiedEvolutionConfig(
    archive_pruning=True,
    archive_keep_ratio=0.5  # Keep top 50%
)
```

---

### Issue 4: Knowledge Engine Errors

**Symptoms:**
- `KnowledgeEngineError` exception
- Timeouts when querying knowledge
- Failed to store artifacts

**Diagnosis:**
```python
# Test knowledge engine connection
from openevolve.unified.knowledge import test_connection

status = test_connection()
if not status['neo4j']:
    print("Warning: Neo4j not responding")
if not status['qdrant']:
    print("Warning: Qdrant not responding")
```

**Solutions:**

1. **Check services are running**
```bash
# Check Neo4j
docker ps | grep neo4j
curl http://localhost:7474

# Check Qdrant
docker ps | grep qdrant
curl http://localhost:6333
```

2. **Restart services**
```bash
docker restart neo4j
docker restart qdrant
```

3. **Disable knowledge engine if not critical**
```python
result = await evolve(
    problem=problem,
    domain=domain,
    enable_knowledge_engine=False  # Disable
)
```

4. **Increase timeouts**
```python
config = UnifiedEvolutionConfig(
    knowledge_query_timeout=30,  # 30 seconds
    knowledge_store_timeout=30
)
```

---

## Error Messages

### "No convergence after N iterations"

**Meaning:** Algorithm didn't find optimal solution within budget

**Solutions:**
```python
# Increase max_iterations
result = await evolve(
    problem=problem,
    domain=domain,
    max_iterations=200  # vs default 100
)

# Or relax convergence_threshold
result = await evolve(
    problem=problem,
    domain=domain,
    convergence_threshold=0.01  # vs default 0.001
)
```

### "Gauntlet failed: Round X"

**Meaning:** Solution failed quality check in round X

**Diagnosis:**
```python
result = await evolve(problem=problem, domain=domain)

gauntlet = result['gauntlet_results']
print(f"Round 1 (LoongFlow): {gauntlet['loongflow_score']}")
print(f"Round 2 (Red Team): {gauntlet['red_team_score']}")
print(f"Round 3 (Gold Team): {gauntlet['gold_team_score']}")
```

**Solutions:**
```python
# Improve problem definition
problem = """
More detailed problem description with:
- Clear objectives
- Specific constraints
- Success criteria
- Examples of good solutions
"""

result = await evolve(
    problem=problem,
    domain=domain,
    enable_gauntlet=True
)
```

### "Knowledge engine query timeout"

**Meaning:** Knowledge graph query took too long

**Solutions:**
```python
# Reduce query complexity
result = await evolve(
    problem=problem,
    domain=domain,
    knowledge_query_limit=5,  # vs default 10
    knowledge_similarity_threshold=0.9  # vs default 0.7
)

# Or increase timeout
config = UnifiedEvolutionConfig(
    knowledge_query_timeout=60  # 60 seconds
)
```

### "Evaluation function failed"

**Meaning:** Custom evaluation function raised exception

**Diagnosis:**
```python
def safe_evaluation(solution, problem):
    try:
        return my_evaluation(solution, problem)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return -float('inf')  # Penalize bad solutions

result = await evolve(
    problem=problem,
    domain=domain,
    evaluation_function=safe_evaluation
)
```

---

## Performance Issues

### Issue: Evaluation Bottleneck

**Symptoms:** 80%+ time spent in evaluation

**Diagnosis:**
```python
from openevolve.unified import profile

profile_result = await profile(problem=problem, domain=domain)
print(profile_result['timing'])
```

**Solutions:**
```python
# 1. Enable caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_evaluation(solution_hash, problem):
    return my_evaluation(solution, problem)

# 2. Parallelize evaluations
config = UnifiedEvolutionConfig(
    max_workers=8,
    parallel_evaluation=True
)

# 3. Use PES for fewer evaluations
config = UnifiedEvolutionConfig(
    evolution_mode="pes",
    enable_planning=True
)
```

### Issue: Gauntlet Overhead

**Symptoms:** Gauntlet takes >20% of total time

**Solutions:**
```python
# 1. Reduce gauntlet rounds
config = UnifiedEvolutionConfig(
    gauntlet_rounds=["loongflow"]  # Only round 1
)

# 2. Increase score thresholds
config = UnifiedEvolutionConfig(
    gauntlet_thresholds={
        "loongflow": 0.7,  # vs default 0.5
        "red_team": 0.8,  # vs default 0.7
        "gold_team": 0.95  # vs default 0.9
    }
)

# 3. Disable gauntlet (not recommended for production)
config = UnifiedEvolutionConfig(
    enable_gauntlet=False
)
```

---

## Domain-Specific Issues

### Finance: Overfitting to Historical Data

**Symptoms:** Great backtest, poor live performance

**Solutions:**
```python
# 1. Walk-forward validation
result = await evolve(
    problem=problem,
    domain="finance",
    validation_method="walk_forward",
    num_folds=5
)

# 2. Simpler strategies
config = UnifiedEvolutionConfig(
    complexity_penalty=True,
    max_parameters=10
)

# 3. Regularization
config = UnifiedEvolutionConfig(
    l2_regularization=0.01,
    early_stopping=True
)
```

### Trading: High Correlation to Benchmark

**Symptoms:** Low alpha, high beta

**Solutions:**
```python
# Add market neutral constraint
result = await evolve(
    problem=problem,
    domain="trading",
    constraints={
        "market_neutral": True,
        "max_beta": 0.3
    }
)
```

### Science: Too Many Experiments

**Symptoms:** Budget exhausted before optimization

**Solutions:**
```python
# Use PES for efficiency
result = await evolve(
    problem=problem,
    domain="science",
    evolution_mode="pes",
    max_evaluations=20,  # Limited budget
    enable_planning=True
)
```

---

## Debugging Techniques

### Enable Verbose Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

result = await evolve(
    problem=problem,
    domain=domain,
    verbose=True
)
```

### Profile Execution

```python
from openevolve.unified import profile

profile_result = await profile(
    problem=problem,
    domain=domain,
    max_evaluations=50
)

# See where time is spent
for phase, duration in profile_result['timing'].items():
    print(f"{phase}: {duration:.1f}s ({duration/sum(profile_result['timing'].values())*100:.1f}%)")
```

### Visualize Evolution

```python
from openevolve.unified import visualize

# Generate convergence plot
visualize(result, save_path="convergence.png")

# Generate Pareto front (if MO)
if result['pareto_front']:
    visualize(result, plot_type="pareto", save_path="pareto.png")

# Generate archive heatmap (if QD)
if result['archive']:
    visualize(result, plot_type="archive", save_path="archive.png")
```

### Export Debug Information

```python
result = await evolve(problem=problem, domain=domain)

# Export to JSON
import json

with open("debug_result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

# Export to CSV
import pandas as pd

if result['pareto_front']:
    df = pd.DataFrame(result['pareto_front'])
    df.to_csv("pareto_front.csv", index=False)
```

---

## When to Ask for Help

### Check Knowledge Engine First

```python
# Query for similar problems
similar = await query_knowledge(
    query="Similar problems with poor convergence",
    domain=domain,
    limit=10
)

# Get recommendations
recommendations = await get_recommendations(
    problem_type="optimization",
    symptoms=["slow_convergence", "poor_quality"]
)
```

### Consult Documentation

- [Unified Evolution Engine Guide](UNIFIED_EVOLUTION_ENGINE_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Domain Guides](domains/)
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)

### Community Support

- **GitHub Issues** - Bug reports and feature requests
- **Stack Overflow** - Tag questions with `openevolve`
- **Discord** - Real-time chat with community

### Provide Debug Information

When asking for help, include:

1. **Problem description**
```python
print(problem)
```

2. **Configuration**
```python
print(config.to_dict())
```

3. **Error messages**
```python
import traceback
traceback.print_exc()
```

4. **System information**
```python
import platform, psutil

print(f"Python: {platform.python_version()}")
print(f"OS: {platform.system()}")
print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"Cores: {psutil.cpu_count()}")
```

---

**End of Troubleshooting Guide**

For more information, see:
- [Unified Evolution Engine Guide](UNIFIED_EVOLUTION_ENGINE_GUIDE.md)
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)
