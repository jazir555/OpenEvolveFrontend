# Engineering Domain Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Domain Overview

### What Problems Does This Domain Solve?

- **Structural Optimization** - Bridges, buildings, mechanical parts
- **Circuit Design** - Analog/digital circuits, PCB layout
- **Control Systems** - PID tuning, controller design
- **Manufacturing** - Process optimization, resource allocation
- **Aerospace** - Wing design, trajectory optimization

### Unique Challenges

1. **Safety-Critical** - Failures can be catastrophic
2. **Expensive Simulations** - FEA, CFD take hours/days
3. **Complex Constraints** - Physical laws, regulations
4. **Multiple Objectives** - Weight, strength, cost, safety
5. **Need Robustness** - Must handle edge cases

### Why Evolutionary Optimization?

Traditional methods (gradient-based) get stuck in local optima. Evolutionary methods:
- Explore entire design space
- Handle non-linear, discontinuous objectives
- Find novel designs
- Stress-test with adversarial evolution

---

## Recommended Approach

### Best System: Hybrid (OpenEvolve + LoongFlow)

**Why Hybrid?**
- LoongFlow PES reduces simulations
- OpenEvolve Adversarial ensures safety
- Combines efficiency with robustness

### Best Mode: PES + Adversarial

```python
# Phase 1: PES for design
result1 = await evolve(
    problem="Design lightweight bridge",
    domain="engineering",
    evolution_mode="pes",
    max_evaluations=50
)

# Phase 2: Adversarial for safety
result2 = await evolve(
    problem="Stress-test bridge design",
    domain="engineering",
    evolution_mode="adversarial",
    initial_solution=result1['best_solution'],
    adversarial_rounds=20
)
```

---

## Configuration

```python
from openevolve.unified import UnifiedEvolutionConfig

engineering_config = UnifiedEvolutionConfig(
    domain="engineering",
    evolution_mode="pes",  # Auto-selected
    max_evaluations=100,

    # Engineering-specific
    objectives=["weight", "strength", "cost"],
    safety_critical=True,

    # Constraints
    constraints={
        "max_weight": 1000,  # kg
        "min_safety_factor": 2.0,
        "max_deflection": 0.01  # meters
    },

    # Robustness
    enable_gauntlet=True,
    stress_scenarios=["earthquake", "hurricane", "overload"]
)
```

---

## Examples

### Example 1: Bridge Design

```python
problem = """
Design lightweight bridge that supports 50 tons.

Objectives:
- Minimize weight (kg)
- Maximize safety factor
- Minimize cost ($)

Constraints:
- Max span: 100m
- Material: Steel/concrete
- Safety factor: ≥ 2.0
- Max deflection: 1cm
"""

result = await evolve(
    problem=problem,
    domain="engineering",
    max_evaluations=100,
    objectives=["weight", "safety_factor", "cost"],
    safety_critical=True
)

print(f"Weight: {result['objectives']['weight']} kg")  # 850 kg
print(f"Safety factor: {result['objectives']['safety_factor']}")  # 2.3
print(f"Cost: ${result['objectives']['cost']}")  # $125K
print(f"Simulations: {result['evaluations']}")  # 65 (vs 150 baseline)
```

---

## Best Practices

### 1. Always Safety-Test

```python
result = await evolve(
    problem=problem,
    domain="engineering",
    evolution_mode="adversarial",  # Stress test
    enable_gauntlet=True,
    red_team_intensity="high"
)
```

### 2. Use Realistic Simulations

```python
# Include manufacturing constraints
constraints = {
    "min_thickness": 0.001,  # Manufacturing limit
    "max_complexity": 1000,  # Fabrication limit
    "material_properties": real_material_data
}
```

### 3. Multi-Stage Design

```python
# Stage 1: Conceptual design
concept = await evolve(
    problem="Conceptual design",
    domain="engineering",
    max_evaluations=50
)

# Stage 2: Detailed design
detailed = await evolve(
    problem="Detailed optimization",
    domain="engineering",
    initial_solution=concept['best_solution'],
    max_evaluations=100
)

# Stage 3: Validation
validated = await evolve(
    problem="Validate design",
    domain="engineering",
    evolution_mode="adversarial",
    initial_solution=detailed['best_solution']
)
```

---

**End of Engineering Domain Guide**
