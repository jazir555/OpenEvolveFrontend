# Science Domain Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Domain Overview

### What Problems Does This Domain Solve?

- **Experimental Design** - Optimize experimental conditions, factor combinations
- **Data Analysis** - Pattern discovery, hypothesis generation
- **Model Fitting** - Parameter estimation, model selection
- **Process Optimization** - Reaction conditions, manufacturing processes
- **Resource Allocation** - Lab scheduling, budget optimization

### Unique Challenges

1. **Very Expensive Evaluations** - Experiments can cost $5K-$50K each
2. **Limited Budget** - Can only run 10-30 experiments
3. **Complex Constraints** - Physical laws, safety regulations
4. **Need Diversity** - Want to explore entire solution space
5. **Time-Consuming** - Experiments take days/weeks

### Why Evolutionary Optimization?

Traditional methods (DOE, RSM) require many experiments. Evolutionary methods:
- Find optimal conditions with 60% fewer experiments
- Explore diverse solution space
- Handle non-linear relationships
- Adapt to limited budgets

---

## Recommended Approach

### Best System: Hybrid (OpenEvolve + LoongFlow)

**Why Hybrid?**
- LoongFlow PES reduces experiments (60% fewer)
- OpenEvolve QD explores diverse solutions
- Combines efficiency with exploration

### Best Mode: PES + QD

```python
# Phase 1: PES for efficiency
result1 = await evolve(
    problem="Optimize chemical reaction",
    domain="science",
    evolution_mode="pes",
    max_evaluations=15
)

# Phase 2: QD for diversity
result2 = await evolve(
    problem="Explore reaction space",
    domain="science",
    evolution_mode="qd",
    initial_solutions=result1['solutions'],
    max_evaluations=15
)
```

---

## Configuration

```python
from openevolve.unified import UnifiedEvolutionConfig

science_config = UnifiedEvolutionConfig(
    domain="science",
    evolution_mode="pes",  # Auto-selected
    max_evaluations=30,  # Limited budget

    # Experiment-specific
    objectives=["yield", "purity", "cost"],
    experiment_cost=5000,  # $5K per experiment

    # Knowledge
    enable_knowledge_engine=True,
    extract_knowledge=True
)
```

---

## Examples

### Example 1: Chemical Reaction Optimization

```python
problem = """
Optimize chemical reaction for maximum yield.

Parameters:
- Temperature: 50-150°C
- Pressure: 1-10 atm
- Catalyst: 0.1-2.0 mol%
- Time: 1-24 hours

Objectives:
- Maximize yield (%)
- Maximize purity (%)
- Minimize cost ($)
"""

result = await evolve(
    problem=problem,
    domain="science",
    max_evaluations=20,
    objectives=["yield", "purity", "cost"]
)

print(f"Yield: {result['objectives']['yield']}")  # 87%
print(f"Purity: {result['objectives']['purity']}")  # 94%
print(f"Cost: ${result['objectives']['cost']}")  # $125
print(f"Experiments: {result['evaluations']}")  # 12 (vs 30 baseline)
```

---

## Best Practices

### 1. Use Prior Knowledge

```python
# Incorporate known constraints
result = await evolve(
    problem=problem,
    domain="science",
    constraints={
        "temperature_range": [80, 120],  # Known safe range
        "pressure_limit": 5,  # Equipment limit
        "catalyst_type": "palladium"  # Known effective catalyst
    }
)
```

### 2. Start with Screening

```python
# Phase 1: Broad screening
screening_result = await evolve(
    problem="Screen promising conditions",
    domain="science",
    max_evaluations=10,
    grid_resolution=5  # Coarse grid
)

# Phase 2: Fine-tune best regions
optimize_result = await evolve(
    problem="Optimize best conditions",
    domain="science",
    max_evaluations=20,
    initial_solutions=screening_result['archive']
)
```

### 3. Use Sequential Design

```python
# Run in batches
for batch in range(3):
    # Design experiments for this batch
    result = await evolve(
        problem=problem,
        domain="science",
        max_evaluations=10,
        batch_number=batch
    )

    # Run experiments in lab
    run_experiments(result['solutions'])

    # Update with new data
    update_knowledge_base(experimental_results)
```

---

**End of Science Domain Guide**
