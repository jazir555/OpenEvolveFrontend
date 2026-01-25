# Γ₁ ACI System - Quick Reference

**Fast lookup for common operations**

---

## Import

```python
from rese.gamma1 import ACICalculator
from rese.gamma1.core.csp_models import (
    CSPInstance, Variable, Constraint,
    create_test_csp, create_tree_csp, create_dense_csp
)
from rese.gamma1.signal.signal_extractor import SignalExtractor
```

---

## Basic Usage

```python
# Create calculator
calc = ACICalculator()

# Calculate ACI
result = calc.calculate(csp)

# Access results
print(f"ACI: {result.ACI:.3f}")
print(f"Entropy (H): {result.components['disorder_entropy']:.3f}")
print(f"Coherence (C): {result.components['causal_coherence']:.3f}")
print(f"Solvability (S): {result.components['solvability_index']:.3f}")
```

---

## ACI Interpretation

| ACI Range | Category | Solver Strategy |
|-----------|----------|----------------|
| > 0.8 | Highly Tractable | Backtracking + Forward Checking |
| 0.6 - 0.8 | Tractable | Constraint Propagation (AC-3/AC-4) |
| 0.4 - 0.6 | Challenging | Monte Carlo Tree Search |
| 0.2 - 0.4 | Intractable | Specialized Solver |
| < 0.2 | Provably Intractable | Reformulate or Approximate |

---

## CSP Factory Functions

```python
# Random CSP
csp = create_test_csp(n_variables=10, domain_size=5, n_constraints=8)

# Tree CSP (easy, high ACI)
csp = create_tree_csp(n_variables=10, domain_size=5)

# Dense CSP (hard, low ACI)
csp = create_dense_csp(n_variables=10, domain_size=5, constraint_density=0.8)
```

---

## Custom CSP Creation

```python
# Variables
vars = [
    Variable(name="x", domain=[1, 2, 3]),
    Variable(name="y", domain=[1, 2, 3])
]

# Constraints
constraints = [
    Constraint(
        variables=["x", "y"],
        allowed_tuples={(1, 1), (2, 2)}
    )
]

# CSP Instance
csp = CSPInstance(variables=vars, constraints=constraints)
```

---

## Signal Extraction

```python
# Multiple instances
results = [calc.calculate(csp) for csp in csps]
times = [1.0 if solvable else float('inf') for solvable in solvables]

# Extract signal
extractor = SignalExtractor()
quality = extractor.extract_signal(results, times)

print(f"Correlation: {quality.correlation:.3f}")
print(f"Accuracy: {quality.accuracy:.3f}")
```

---

## Stage Integration

### Stage 3 (Monte Carlo)
```python
if result.ACI > 0.7:
    use_smart_sampling()
elif result.ACI > 0.4:
    use_hybrid_sampling()
else:
    use_random_sampling()
```

### Stage 6 (Error Analysis)
```python
H = result.components['disorder_entropy']
C = result.components['causal_coherence']

if H > 0.7:
    add_constraints()
elif C < 0.3:
    restructure_constraints()
```

### Stage 9 (Convergence)
```python
if result.ACI > 0.8 and progress > 0.1:
    continue_search()
elif result.ACI < 0.3 and progress < 0.01:
    abort_search()
```

---

## Common Patterns

### Batch Processing
```python
results = []
for csp in csp_list:
    result = calc.calculate(csp)
    results.append(result)

# Sort by ACI
results.sort(key=lambda r: r.ACI, reverse=True)
```

### Filter by Difficulty
```python
easy_results = [r for r in results if r.ACI > 0.7]
hard_results = [r for r in results if r.ACI < 0.4]
```

### Get Statistics
```python
import numpy as np
aci_scores = [r.ACI for r in results]
print(f"Mean: {np.mean(aci_scores):.3f}")
print(f"Std: {np.std(aci_scores):.3f}")
```

---

## Performance Tips

1. **Enable caching** for repeated calculations
2. **Use tree CSPs** when possible
3. **Batch process** multiple CSPs
4. **Adapt strategy** based on ACI

---

## Testing

```bash
# Run all tests
python -m pytest rese/tests/gamma1/test_aci_complete.py -v

# Run specific category
python -m pytest rese/tests/gamma1/test_aci_complete.py::TestACICalculator -v
```

---

## Troubleshooting

**Issue:** ACI is NaN
**Fix:** Check CSP has valid variables, domains, and constraints

**Issue:** Low confidence
**Fix:** Add more variables or constraints

**Issue:** Slow calculation
**Fix:** Enable caching or reduce problem size

---

## File Locations

- **Core:** `rese/gamma1/core/`
- **Signal:** `rese/gamma1/signal/`
- **Tests:** `rese/tests/gamma1/`
- **Docs:** `rese/docs/gamma1_*.md`

---

**For detailed information, see:** `GAMMA1_IMPLEMENTATION_COMPLETE.md`
