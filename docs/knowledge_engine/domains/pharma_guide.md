# Pharma Domain Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Domain Overview

### What Problems Does This Domain Solve?

- **Drug Discovery** - Molecular optimization, lead compound selection
- **Formulation** - Drug delivery systems, dosage optimization
- **Clinical Trials** - Patient stratification, trial design
- **Manufacturing** - Process optimization, quality control
- **Pharmacology** - ADME prediction, toxicity screening

### Unique Challenges

1. **High Dimensionality** - Millions of molecular descriptors
2. **Complex Constraints** - Chemical validity, synthetic accessibility
3. **Multiple Objectives** - Efficacy, toxicity, solubility, cost
4. **Long Evaluation Times** - Docking, MD simulations
5. **Regulatory Requirements** - FDA guidelines, safety standards

### Why Evolutionary Optimization?

Traditional methods (HTS, QSAR) are expensive and slow. Evolutionary methods:
- Explore diverse chemical space
- Optimize multiple properties simultaneously
- Find novel scaffolds
- Reduce synthesis and testing costs

---

## Recommended Approach

### Best System: OpenEvolve

**Why?**
- Quality Diversity (MAP-Elites) explores diverse chemical space
- Multi-objective optimization for competing properties
- Can handle high-dimensional molecular descriptors

### Best Mode: QD (Quality Diversity)

**Why QD?**
- Want diverse molecular candidates
- Explore entire chemical space
- Find novel scaffolds
- Multiple backup options

---

## Configuration

```python
from openevolve.unified import UnifiedEvolutionConfig

pharma_config = UnifiedEvolutionConfig(
    domain="pharma",
    evolution_mode="qd",  # Quality Diversity
    max_evaluations=200,

    # Molecular objectives
    objectives=["binding_affinity", "toxicity", "solubility"],
    feature_dimensions=["mw", "logp", "hbd", "hba"],
    grid_resolution=10,

    # Constraints
    constraints={
        "molecular_weight": [150, 500],
        "logp": [-2, 5],
        "hbd": [0, 10],
        "hba": [0, 10]
    }
)
```

---

## Examples

### Example 1: Drug Lead Optimization

```python
problem = """
Optimize drug candidate for target binding.

Objectives:
- Maximize binding affinity (pIC50)
- Minimize toxicity (LD50)
- Maximize solubility (logS)

Constraints:
- MW: 150-500 Da
- LogP: -2 to 5
- HBD: 0-10
- HBA: 0-10
- Synthetic accessibility: ≤ 6
"""

result = await evolve(
    problem=problem,
    domain="pharma",
    evolution_mode="qd",
    max_evaluations=200,
    objectives=["binding_affinity", "toxicity", "solubility"]
)

# Get diverse candidates from archive
for cell_key, solution in result['archive'].items():
    print(f"MW: {cell_key[0]:.1f}, LogP: {cell_key[1]:.1f}")
    print(f"Affinity: {solution['binding_affinity']:.2f}")
    print(f"SMILES: {solution['smiles']}")
    print("---")
```

---

## Best Practices

### 1. Use Drug-Likeness Filters

```python
# Apply Lipinski's Rule of Five
constraints = {
    "molecular_weight": [150, 500],
    "logp": [-0.5, 5],
    "hbd": [0, 5],
    "hba": [0, 10]
}

# Apply Veber's rules
constraints["rotatable_bonds"] = [0, 10]
constraints["polar_surface_area"] = [0, 140]
```

### 2. Multi-Stage Optimization

```python
# Stage 1: High-throughput virtual screening
hts_result = await evolve(
    problem="Screen large library",
    domain="pharma",
    max_evaluations=1000,
    evaluation_function="docking_score"
)

# Stage 2: MM-GBSA refinement
refined_result = await evolve(
    problem="Refine top candidates",
    domain="pharma",
    initial_solutions=hts_result['archive'],
    max_evaluations=100,
    evaluation_function="mmgbsa_score"
)

# Stage 3: MD validation
validated_result = await evolve(
    problem="Validate with MD",
    domain="pharma",
    initial_solutions=refined_result['archive'],
    max_evaluations=20,
    evaluation_function="md_stability"
)
```

---

**End of Pharma Domain Guide**
